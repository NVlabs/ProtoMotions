# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BaseEnv.step``, ``reset`` and ``build_env`` as declared phase lists, the
advance owners, the wiring report.

On the real ``BaseEnv`` over the fake sensor simulator
(``tests/support/sensor_simulator.py``), with and without a reference motion.
"""

import inspect
import io
import logging
import textwrap
import tokenize

import pytest
import torch

pytest.importorskip("mujoco")  # the fake simulator is MuJoCo-backed kinematics

from protomotions.envs.base_env import build as build_module  # noqa: E402
from protomotions.envs.base_env.build import BUILD_PHASES  # noqa: E402
from protomotions.envs.randomness import Randomness  # noqa: E402
from protomotions.envs.base_env.config import RecoveryResetConfig  # noqa: E402
from protomotions.envs.base_env.env import (  # noqa: E402
    RESET_PHASES,
    STEP_PHASES,
    BaseEnv,
)
from protomotions.envs.base_env.mdp_managers import ObservationManager  # noqa: E402
from protomotions.envs.base_env.owners import (  # noqa: E402
    ContactMemory,
    ControlsState,
)
from protomotions.envs.base_env.reset import (  # noqa: E402
    RecoveryBank,
    ReferenceMotion,
    ResetPerturbation,
    ResetPose,
    SpawnPlacer,
)
from protomotions.envs.control.manager import ControlManager  # noqa: E402
from protomotions.envs.control.mimic_control import MimicControlConfig  # noqa: E402
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer  # noqa: E402
from protomotions.envs.sensors.runtime import SensorRuntime  # noqa: E402
from protomotions.envs.motion_manager.config import (  # noqa: E402
    MimicMotionManagerConfig,
)
from protomotions.tests.parity.chain_motions import build_chain_motion_lib  # noqa: E402
from protomotions.tests.support.sensor_simulator import (  # noqa: E402
    FlatSensorTestTerrain,
    SensorTestSimulator,
    make_sensor_test_env,
)


def _mimic_env(realign=True, **kwargs):
    """The sensor test env tracking one synthetic chain clip."""
    return make_sensor_test_env(
        motion_lib=build_chain_motion_lib(1),
        motion_manager=MimicMotionManagerConfig(
            realign_motion_with_humanoid_on_each_step=realign
        ),
        control_components={"mimic": MimicControlConfig()},
        **kwargs,
    )


class _PhaseLog:
    """Labels of the spied calls ``step`` makes ITSELF, in order.

    A call a phase's collaborator makes while a spied call is running (the
    fake's ``get_root_state`` delegates to ``get_robot_state``; the report
    phase reads the state inside ``_raw_state_extras``) is nested and is not a
    phase, so it is not recorded.
    """

    def __init__(self):
        self.calls = []
        self._depth = 0

    def spy(self, label, target, name):
        original = getattr(target, name)

        def spy(*args, **kwargs):
            if self._depth == 0:
                self.calls.append(label)
            self._depth += 1
            try:
                return original(*args, **kwargs)
            finally:
                self._depth -= 1

        setattr(target, name, spy)


def test_step_runs_the_declared_phases_in_order():
    env = _mimic_env()
    try:
        env.reset()
        log = _PhaseLog()
        log.spy("act", env._action_stage, "act")
        log.spy("physics", env.simulator, "step")
        log.spy("clock", env._episode, "tick")
        log.spy("read", env.simulator, "get_robot_state")
        log.spy("sense", env.sensors, "step")
        log.spy("record", env._state_history, "push")
        for owner in env._advance_owners:
            log.spy(f"advance:{owner.name}", owner, "advance")
        log.spy("publish", env, "_publish")
        log.spy("observe", env._observations, "compute")
        log.spy("score:rewards", env._rewards, "compute")
        log.spy("score:terminations", env._terminations, "compute")
        log.spy("report:raw", env, "_raw_state_extras")
        log.spy("report:contacts", env._contact_memory, "update")
        log.spy("keyboard", env, "consume_reset_request")

        env.step(torch.zeros(env.num_envs, 2))

        advance = [f"advance:{owner.name}" for owner in env._advance_owners]
    finally:
        env.close()
    assert advance == ["advance:motion", "advance:controls", "advance:spawn"]
    # The score phase runs the two managers (the controls' checks are entries
    # of the termination table); the report phase ends with the contact memory.
    expansion = {
        "advance": advance,
        "score": ["score:rewards", "score:terminations"],
        "report": ["report:raw", "report:contacts"],
    }
    expected = [
        label for phase in STEP_PHASES for label in expansion.get(phase, [phase])
    ]
    assert log.calls == expected
    assert BaseEnv.step.__doc__ and "STEP_PHASES" in BaseEnv.step.__doc__


def test_construction_computes_components_from_the_all_env_context(monkeypatch):
    """The constructor's first observations are the dynamic components from
    the cached all-env context (the publishers see None: the compiled path);
    the terrain cache keeps its initial values until the first reset, which
    computes its rows from a subset context, as it always did."""
    seen = []
    compute_components = ObservationManager.compute_components

    def recording(self, ctx, env_ids):
        seen.append((ctx.env_ids, env_ids.clone()))
        compute_components(self, ctx, env_ids)

    monkeypatch.setattr(ObservationManager, "compute_components", recording)
    # Height maps of ones: the zero-initialised cache shows whether it was
    # ever computed.
    monkeypatch.setattr(
        FlatSensorTestTerrain,
        "get_height_maps",
        lambda self, root_state, env_ids: torch.ones(len(env_ids), 1),
    )

    env = make_sensor_test_env()
    try:
        ((ctx_env_ids, rows),) = seen
        assert ctx_env_ids is None and torch.equal(rows, torch.arange(3))
        assert env._current_context is not None
        assert env._current_context.env_ids is None  # still cached, all envs
        assert env.get_obs()["actor"].shape[0] == 3  # the components were written

        env.reset()

        assert len(seen) == 2 and torch.equal(seen[1][0], torch.arange(3))
        assert env._current_context is None
    finally:
        env.close()


def test_advance_owners_are_static_and_declared_once():
    plain = make_sensor_test_env()
    try:
        names = [owner.name for owner in plain._advance_owners]
        assert names == ["motion", "controls", "spawn"]
        motion, controls, spawn = plain._advance_owners
        assert isinstance(motion, ReferenceMotion) and motion is plain._motion
        assert isinstance(spawn, SpawnPlacer) and spawn is plain._spawn
        # One owner on both the snapshot and the advance list; the spawn placer
        # too (its offsets are snapshotted).
        assert controls is plain._controls
        assert isinstance(plain._controls, ControlsState)
        assert any(owner is plain._controls for owner in plain._snapshot_owners)
        assert any(owner is plain._spawn for owner in plain._snapshot_owners)
        # Without a motion manager the two motion entries do nothing.
        assert plain.reference_motion.motion_manager is None
        offsets = plain._spawn.respawn_root_offset.clone()
        motion.advance(None)
        spawn.advance(None)
        assert torch.equal(plain._spawn.respawn_root_offset, offsets)
    finally:
        plain.close()


def test_motion_entries_read_the_current_manager_and_survive_a_swap():
    env = _mimic_env()
    try:
        motion, controls, spawn = env._advance_owners
        assert controls is env._controls

        # The clock ticks the manager installed NOW: swap it under the same
        # assignment and the replacement is what advances.
        before = env.reference_motion.motion_manager.motion_times.clone()
        motion.advance(None)
        torch.testing.assert_close(
            env.reference_motion.motion_manager.motion_times, before + env.dt
        )

        class _Replacement:
            ticks = 0

            def post_physics_step(self):
                self.ticks += 1

        original_manager = env.reference_motion.motion_manager
        env.reference_motion.motion_manager = _Replacement()  # on the owner itself
        assert env.reference_motion is env._motion
        motion.advance(None)
        assert env.reference_motion.motion_manager.ticks == 1
        torch.testing.assert_close(original_manager.motion_times, before + env.dt)
        env.reference_motion.motion_manager = original_manager

        # A manager without a clock (the base MotionManager) is left alone.
        env.reference_motion.motion_manager = object()
        motion.advance(None)
        env.reference_motion.motion_manager = original_manager

        # The re-alignment re-anchors every clip's XY on the robot's root ...
        env._spawn.respawn_root_offset[:, :2] = 100.0
        spawn.advance(None)
        ref = env.reference_motion.motion_lib.get_motion_state(
            env.reference_motion.motion_manager.motion_ids,
            env.reference_motion.motion_manager.motion_times,
        )
        root_xy = env.simulator.get_root_state().root_pos[:, :2]
        torch.testing.assert_close(
            env._spawn.respawn_root_offset[:, :2],
            root_xy - ref.rigid_body_pos[:, 0, :2],
        )
        # ... only when the manager's config asks for it.
        manager_config = env.reference_motion.motion_manager.config
        manager_config.realign_motion_with_humanoid_on_each_step = False
        env._spawn.respawn_root_offset[:, :2] = 100.0
        spawn.advance(None)
        assert torch.all(env._spawn.respawn_root_offset[:, :2] == 100.0)
        manager_config.realign_motion_with_humanoid_on_each_step = True

        # install_motion_lib (the live shard switch) installs a new manager
        # under the same static list; nothing is rebuilt.
        env.install_motion_lib(build_chain_motion_lib(1))
        assert env._advance_owners == (motion, controls, spawn)
        assert env.reference_motion.motion_manager is not original_manager
        assert env._motion.motion_manager is env.reference_motion.motion_manager
    finally:
        env.close()


def test_no_realign_env_keeps_its_offsets_through_a_step():
    env = _mimic_env(realign=False)
    try:
        env.reset()
        offsets = env._spawn.respawn_root_offset.clone()
        env.step(torch.zeros(env.num_envs, 2))
        assert torch.equal(env._spawn.respawn_root_offset, offsets)
    finally:
        env.close()


def test_reset_runs_the_declared_phases_in_order():
    env = _mimic_env(
        recovery_reset=RecoveryResetConfig(recovery_prob=1.0, fall_sim_steps=2)
    )
    try:
        env.reset()
        log = _PhaseLog()
        log.spy("assign", env._motion, "ref_reset_envs")
        log.spy("pose", env._pose, "compute")
        log.spy("recover", env._recovery, "swap_in")
        log.spy("perturb", env._perturbation, "apply")
        log.spy("write", env.simulator, "reset_envs")
        for owner in env._row_owners:
            log.spy(f"clear:{owner.name}", owner, "reset_rows")
        log.spy("read", env.simulator, "get_robot_state")
        log.spy("sense", env.sensors, "reset")
        log.spy("history", env._state_history, "reset")
        log.spy("controls", env.control_manager, "reset_rows")
        log.spy("observe", env, "observe")

        env.reset(torch.tensor([0, 2]))

        clear = [f"clear:{owner.name}" for owner in env._row_owners]
    finally:
        env.close()
    assert clear == ["clear:episode", "clear:contact_memory", "clear:action_stage"]
    expansion = {"clear": clear}
    expected = [
        label for phase in RESET_PHASES for label in expansion.get(phase, [phase])
    ]
    assert log.calls == expected
    assert BaseEnv.reset.__doc__ and "RESET_PHASES" in BaseEnv.reset.__doc__


def _comment_block_heads(fn):
    """The first word of each full-line comment block in ``fn``'s body.

    A block is a run of full-line comments on consecutive lines; its head is
    the first word of its first line, a trailing colon stripped (``# advance:
    motion clock, ...`` reads ``advance``). The docstring is one string token,
    so nothing inside it counts.
    """
    source = textwrap.dedent(inspect.getsource(fn))
    heads = []
    previous_row = None
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.COMMENT or not tok.line.lstrip().startswith("#"):
            continue
        row = tok.start[0]
        if previous_row != row - 1:
            words = tok.string.lstrip("#").split()  # a bare "#" heads as ""
            heads.append(words[0].rstrip(":") if words else "")
        previous_row = row
    return tuple(heads)


@pytest.mark.parametrize(
    "method, phases",
    [(BaseEnv.step, STEP_PHASES), (BaseEnv.reset, RESET_PHASES)],
    ids=["step", "reset"],
)
def test_body_comments_are_the_declared_phases_in_order(method, phases):
    """The source reads as the phase list.

    The spies above prove the collaborators run in the declared order; this
    pins the narration: the body of ``step()`` is one ``# <phase>`` comment
    block per phase, in ``STEP_PHASES``' order, and ``reset()``'s likewise
    against ``RESET_PHASES``. A reader of env.py can take the comments as the
    list; a phase added, renamed or moved in one place and not the other fails
    here, and so does a stray standalone comment (an explanation belongs on
    its phase's comment block, as the multi-line ones already are).
    """
    assert _comment_block_heads(method) == phases


def test_build_env_runs_the_documented_phases_in_order(monkeypatch):
    """Spies on the collaborators each construction phase creates or calls;
    their order must be ``BUILD_PHASES``' (the module docstring says why)."""
    calls = []

    def spy_init(cls, label):
        original = cls.__init__

        def __init__(self, *args, **kwargs):
            calls.append(label)
            original(self, *args, **kwargs)

        monkeypatch.setattr(cls, "__init__", __init__)

    def spy_method(cls, name, label):
        original = getattr(cls, name)

        def method(self, *args, **kwargs):
            calls.append(label)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(cls, name, method)

    spy_init(Randomness, "randomness")
    spy_method(ReferenceMotion, "install", "motion")
    spy_init(ContactMemory, "owners")
    spy_init(SensorRuntime, "sensors")
    spy_init(ControlManager, "controls")
    spy_method(SensorTestSimulator, "_initialize_with_markers", "markers")
    spy_method(build_module, "bind_managers", "managers")
    spy_init(StateHistoryBuffer, "history")
    spy_method(build_module, "declare_owners", "lists")
    spy_method(build_module, "validate_bindings", "check")
    spy_method(RecoveryBank, "generate", "recovery")
    spy_method(ObservationManager, "compute_components", "observe")
    env = _mimic_env(
        recovery_reset=RecoveryResetConfig(recovery_prob=1.0, fall_sim_steps=2)
    )
    try:
        kinematic_info = env.robot_config.kinematic_info
        recovery, pose, perturbation = env._recovery, env._pose, env._perturbation
    finally:
        env.close()
    # The spied collaborators, in the order they were created or called. The
    # robot_model phase has no collaborator to spy (it moves the kinematic
    # tensors in place); the helpers built after it hold those same tensors.
    assert calls == [phase for phase in BUILD_PHASES if phase != "robot_model"]
    assert recovery.dof_limits_lower is kinematic_info.dof_limits_lower
    assert perturbation.dof_limits_upper is kinematic_info.dof_limits_upper
    assert isinstance(pose, ResetPose) and isinstance(perturbation, ResetPerturbation)


def test_wiring_report_names_the_four_lists_and_is_logged_once(caplog):
    with caplog.at_level(logging.INFO, logger="protomotions.envs.base_env.build"):
        env = _mimic_env()
    try:
        report = build_module.wiring_report(env)
        publish_owner_names = [owner.name for owner in env._publish_owners]
        snapshot_owner_names = [owner.name for owner in env._snapshot_owners]
    finally:
        env.close()

    lines = report.splitlines()
    assert lines[0] == "BaseEnv wiring"
    assert lines[1] == "  advance owners (step order): motion, controls, spawn"
    assert lines[2] == "  publish owners (context producers):"
    publish_lines = [line for line in lines if line.startswith("    ")]
    producers = [line.split(":")[0].strip() for line in publish_lines]
    # The publish owners, then each control component under its own name.
    assert producers == publish_owner_names + ["control/mimic"]
    by_producer = dict(zip(producers, publish_lines))
    assert by_producer["episode"].endswith("1 names (progress_buf)")
    assert by_producer["terrain"].endswith("0 names")  # flat test terrain
    assert "current.dof_pos" in by_producer["state"]
    assert by_producer["state"].endswith(", ...)")
    assert "mimic." in by_producer["control/mimic"]
    assert lines[-2] == (
        "  row owners (reset clears): episode, contact_memory, action_stage"
    )
    assert lines[-1] == "  snapshot owners (restore order): " + ", ".join(
        snapshot_owner_names
    )
    # The constructor logs the same block, once, through the build module's
    # logger.
    logged = [record for record in caplog.records if "BaseEnv wiring" in record.message]
    assert len(logged) == 1 and logged[0].levelno == logging.INFO
    assert logged[0].message == report
