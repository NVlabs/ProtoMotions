# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The control contract, pinned across every in-tree control.

A control is built from its config and a ``TaskInputs`` and holds no env,
simulator, motion library or manager; it never resets its own rows and never
writes the episode clock; its episode-ending checks are rules the termination
manager runs beside the recipe's, a truncating one resetting without failing;
the viewer's reset key is a request mask.
"""

import importlib
import pkgutil
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import protomotions
from protomotions.components.motion_lib import MotionLib
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.base_env.episode_tensors import EpisodeTensors
from protomotions.envs.control.base import ControlComponent
from protomotions.envs.control.external_kinematic_control import (
    ExternalKinematicControl,
    ExternalKinematicControlConfig,
)
from protomotions.envs.control.gpc_sft_reference_target_control import (
    GPCSFTReferenceTargetControl,
    GPCSFTReferenceTargetControlConfig,
)
from protomotions.envs.control.kinematic_replay_control import (
    KinematicReplayControl,
    KinematicReplayControlConfig,
)
from protomotions.envs.control.manager import ControlManager, TeleportQueue
from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControl,
    MaskedMimicControlConfig,
)
from protomotions.envs.control.masked_mimic_prompt import GoalCanvas
from protomotions.envs.control.masked_mimic_tasks.controller import (
    MaskedMimicTaskControl,
    MaskedMimicTaskControlConfig,
)
from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
from protomotions.envs.control.path_follower_control import (
    PathFollowerControl,
    PathFollowerControlConfig,
)
from protomotions.envs.control.scene_boxes_control import (
    SceneBoxesControl,
    SceneBoxesControlConfig,
)
from protomotions.envs.control.steering_control import (
    SteeringControl,
    SteeringControlConfig,
)
from protomotions.envs.control.target_control import TargetControl, TargetControlConfig
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.tests.support.fake_reference import fake_reference
from protomotions.tests.support.task_inputs import make_robot, make_task_inputs

CONTROL_DIR = Path(protomotions.__file__).resolve().parent / "envs" / "control"
HANDLES = (BaseEnv, Simulator, MotionLib, MotionManager)


def _reference(num_envs=2):
    motion = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        motion_manager=SimpleNamespace(
            motion_ids=torch.zeros(num_envs, dtype=torch.long),
            motion_times=torch.zeros(num_envs),
            get_done_tracks=lambda env_ids=None: torch.zeros(
                num_envs, dtype=torch.bool
            ),
        ),
        motion_lib=SimpleNamespace(
            get_motion_length=lambda ids: torch.full_like(ids, 5.0, dtype=torch.float)
        ),
    )
    return fake_reference(motion)


def _canvas():
    return GoalCanvas(
        future_steps=(1,),
        num_masked_future_steps=5,
        conditionable_bodies=("anchor", "hand"),
        anchor_body="anchor",
        anchor_pose_submasking=True,
        has_local_stream=True,
        style_frames=None,
        has_velocity_command=True,
    )


def every_in_tree_control():
    """One instance of every in-tree control over ``make_task_inputs``."""
    robot = make_robot(
        ("anchor", "hand", "head"),
        trackable_bodies_subset=["anchor", "hand"],
        head_body_name="head",
        hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
        semantic_forward_axis_xy=(1.0, 0.0),
    )
    inputs = make_task_inputs(2, robot=robot, reference=_reference(2), dt=0.02)
    return [
        SteeringControl(SteeringControlConfig(), inputs),
        TargetControl(TargetControlConfig(), inputs),
        SceneBoxesControl(SceneBoxesControlConfig(), inputs),
        GPCSFTReferenceTargetControl(GPCSFTReferenceTargetControlConfig(), inputs),
        MimicControl(MimicControlConfig(), inputs),
        MaskedMimicControl(MaskedMimicControlConfig(), inputs),
        PathFollowerControl(PathFollowerControlConfig(), inputs),
        KinematicReplayControl(KinematicReplayControlConfig(), inputs),
        ExternalKinematicControl(ExternalKinematicControlConfig(), inputs),
        MaskedMimicTaskControl(
            MaskedMimicTaskControlConfig(canvas=_canvas(), reach_body_name="hand"),
            inputs,
        ),
    ]


def test_every_in_tree_control_holds_no_env_simulator_or_motion_handle():
    controls = every_in_tree_control()
    assert len(controls) == 10
    for control in controls:
        for name, value in vars(control).items():
            assert name != "env", type(control).__name__
            assert not isinstance(value, HANDLES), (type(control).__name__, name)
    # The build-time walk agrees.
    manager = ControlManager({}, controls[0].inputs, TeleportQueue())
    manager.components = {type(c).__name__: c for c in controls}
    manager.check_holds_no_handles()


# The attributes the registry walk does not descend into: the config and the
# env-provided inputs (their tensors are the env's). The registry itself is
# not a helper (``_is_helper`` excludes it), so it needs no entry here, and
# a helper attribute that happens to be called ``state`` stays visible.
_WALK_SKIPS = frozenset({"config", "inputs"})


def _is_helper(value) -> bool:
    """An object of this code base a control holds (a command source, a path
    generator, a target stream), but not a control (a back-reference), a
    registry or a config."""
    return (
        not torch.is_tensor(value)
        and hasattr(value, "__dict__")
        and type(value).__module__.startswith("protomotions")
        and not isinstance(value, (ControlComponent, EpisodeTensors))
        and not type(value).__name__.endswith("Config")
    )


def bare_per_env_tensors(obj, num_envs: int, depth: int = 2, path: str = ""):
    """Every tensor attribute with a ``num_envs`` first dimension on ``obj``,
    inside its dict / list / tuple attributes, and on its helper objects up to
    ``depth`` levels down (a helper held in a dict or a list counts, so a
    "dict of streams" is walked), as ``(path, tensor)``. This is what the
    ``__setattr__`` guard cannot see beyond the control's own attributes."""
    seen = set()

    def per_env(tensor) -> bool:
        return (
            torch.is_tensor(tensor) and tensor.ndim >= 1 and tensor.shape[0] == num_envs
        )

    def leaf(item, here, remaining):
        # An item of a dict / list / tuple attribute: a per-env tensor, or a
        # helper to walk one level down.
        if per_env(item):
            yield here, item
        elif remaining > 0 and _is_helper(item):
            yield from walk(item, f"{here}.", remaining - 1)

    def walk(node, prefix, remaining):
        if id(node) in seen:
            return
        seen.add(id(node))
        for name, value in vars(node).items():
            if name in _WALK_SKIPS:
                continue
            here = f"{prefix}{name}"
            if per_env(value):
                yield here, value
            elif isinstance(value, dict):
                for key, item in value.items():
                    yield from leaf(item, f"{here}[{key!r}]", remaining)
            elif isinstance(value, (list, tuple)):
                for index, item in enumerate(value):
                    yield from leaf(item, f"{here}[{index}]", remaining)
            elif remaining > 0 and _is_helper(value):
                yield from walk(value, f"{here}.", remaining - 1)

    yield from walk(obj, path, depth)


def test_every_per_env_tensor_of_every_control_is_in_its_registry():
    """The control's own tensors (the guard's scope) and the ones inside its
    helpers two levels down (beyond it) are all registered: a child registry
    makes a helper's tensors the control's."""
    for control in every_in_tree_control():
        num_envs = control.inputs.num_envs
        found = list(bare_per_env_tensors(control, num_envs))
        for path, tensor in found:
            assert control.state.owns(tensor), (type(control).__name__, path)


def test_the_registry_walk_sees_a_bare_tensor_inside_a_helper():
    """The walk reaches what the guard cannot: a per-env tensor on a helper
    object, in a container, and two levels down; a back-reference to the
    control, the config and the inputs are not descended into."""

    class _Helper:
        pass

    control = every_in_tree_control()[0]
    num_envs = control.inputs.num_envs
    inner = _Helper()
    inner.deep = torch.zeros(num_envs, 2)
    helper = _Helper()
    helper.bare = torch.zeros(num_envs, 3)
    helper.rows = {"a": torch.zeros(num_envs)}
    helper.inner = inner
    helper.back = control  # a back-reference, as the command sources hold
    helper.scalar = 3.0  # not a tensor: outside the walk, as documented
    control.__dict__["probe"] = helper  # bypass the guard, as a helper would

    paths = {path for path, _ in bare_per_env_tensors(control, num_envs)}
    assert {"probe.bare", "probe.rows['a']", "probe.inner.deep"} <= paths
    assert not any(path.startswith("probe.back") for path in paths)
    registered = [
        p for p, t in bare_per_env_tensors(control, num_envs) if control.state.owns(t)
    ]
    assert "probe.bare" not in registered


def test_every_in_tree_control_is_every_control_under_envs_control():
    """The list the contract tests run over is complete: every
    ``ControlComponent`` subclass defined in a module under ``envs/control``
    (the ``masked_mimic_tasks`` package included) is instantiated once by
    ``every_in_tree_control``, and nothing else is."""
    import protomotions.envs.control as package

    defined = set()
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(info.name)
        for value in vars(module).values():
            if (
                isinstance(value, type)
                and issubclass(value, ControlComponent)
                and value is not ControlComponent
                and value.__module__ == module.__name__
            ):
                defined.add(value)
    listed = {type(control) for control in every_in_tree_control()}
    assert listed == defined, {
        "missing from every_in_tree_control": sorted(
            c.__name__ for c in defined - listed
        ),
        "not under envs/control": sorted(c.__name__ for c in listed - defined),
    }


def test_the_registry_walk_sees_a_helper_held_in_a_container():
    """A "dict of streams" or a list of sources: the helper inside the
    container is walked like a direct helper attribute, and a container
    item that is neither a tensor nor a helper is left alone."""

    class _Helper:
        pass

    control = every_in_tree_control()[0]
    num_envs = control.inputs.num_envs
    stream = _Helper()
    stream.rows = torch.zeros(num_envs, 2)
    holder = _Helper()
    holder.streams = {"left": stream}
    holder.sources = [_Helper(), 2.0]
    holder.sources[0].bare = torch.zeros(num_envs)
    holder.state = torch.zeros(num_envs)  # a helper attribute named state
    control.__dict__["probe"] = holder

    paths = {path for path, _ in bare_per_env_tensors(control, num_envs)}
    assert {
        "probe.streams['left'].rows",
        "probe.sources[0].bare",
        "probe.state",
    } <= paths


def _control_sources():
    return [p for p in CONTROL_DIR.rglob("*.py") if p.name != "__init__.py"]


def test_no_control_resets_its_own_rows():
    """``reset_rows`` is the orchestrator's call: a schedule change inside an
    episode goes through ``resample``."""
    offenders = [
        str(path.relative_to(CONTROL_DIR))
        for path in _control_sources()
        if "self.reset_rows(" in path.read_text()
    ]
    assert offenders == []


def test_no_control_writes_the_episode_clock():
    """The clock is read-only for a control (``EpisodeClock``); the animation
    players restart it through ``TaskInputs.restart_episode``."""
    write = re.compile(r"clock\.progress_buf(\[[^\]]*\])?\s*(=[^=]|\+=|-=|\*=|/=)")
    offenders = [
        str(path.relative_to(CONTROL_DIR))
        for path in _control_sources()
        if write.search(path.read_text())
    ]
    assert offenders == []
    assert write.search("self.inputs.clock.progress_buf[ids] = 0")  # the pattern bites


def test_base_class_has_no_step_reset_or_check_pair():
    for name in ("step", "reset", "check_resets_and_terminations", "restore_state"):
        assert not hasattr(ControlComponent, name), name
    for name in ("reset_rows", "resample", "advance", "publish", "terminations"):
        assert hasattr(ControlComponent, name), name


# ---------------------------------------------------------------------------
# The merged termination table on the fake scenarios
# ---------------------------------------------------------------------------


def _record(scenario_name, steps=12, prepare=None):
    from protomotions.tests.parity.scenarios import (
        SCENARIOS,
        build_env,
        eager_mdp_kernels,
    )

    scenario = SCENARIOS[scenario_name]
    with eager_mdp_kernels():
        built = build_env(scenario)
        env = built.env
        try:
            env.reset()
            if prepare is not None:
                prepare(env)
            seen = []
            for _ in range(steps):
                _, _, done, terminated, extras = env.step(
                    torch.zeros(env.num_envs, env.robot_config.number_of_actions)
                )
                seen.append(
                    (
                        done.clone(),
                        terminated.clone(),
                        {
                            k: v.clone()
                            for k, v in extras.items()
                            if k.startswith("termination/")
                        },
                        env._episode.progress_buf.clone(),
                    )
                )
                done_ids = done.nonzero(as_tuple=False).flatten()
                if len(done_ids):
                    env.reset(done_ids)
            table = env._terminations.components
        finally:
            env.close()
    return table, seen


def test_fake_target_table_mixes_recipe_and_control_checks():
    """fake3_target: the recipe's terminations and the target control's three
    checks are one table; each flag is logged under its own name and every
    flag resets AND terminates (none is a truncation)."""
    pytest.importorskip("mujoco")
    table, seen = _record("fake3_target")
    assert {"target/fall", "target/gap", "target/stuck"} <= set(table)
    assert not any(rule.truncation for rule in table.values())
    for done, terminated, flags, _ in seen:
        assert set(flags) == {f"termination/{name}" for name in table}
        fired = torch.zeros_like(done)
        for flag in flags.values():
            fired |= flag.bool()
        assert torch.all(fired <= done)  # a flag resets ...
        assert torch.all(fired <= terminated)  # ... and terminates


def test_kinematic_replay_plays_the_reference_through_the_env():
    """Playback smoke: with the replay beside the mimic control on the fake
    mimic scenario, every step teleports every env onto the placed reference
    pose (through the teleport queue, after every control advanced) and
    restarts its clock, so the time limit never fires. (The recipe's tracking
    check still compares the pre-teleport measurement; that is the step's
    ``read`` phase and not this control's concern.)"""
    pytest.importorskip("mujoco")
    from protomotions.tests.parity import scenarios

    original = scenarios._fake_mimic_task

    def with_replay(num_motions):
        task = original(num_motions)
        task.controls["replay"] = KinematicReplayControlConfig()
        return task

    scenarios._fake_mimic_task = with_replay
    try:
        with scenarios.eager_mdp_kernels():
            env = scenarios.build_env(scenarios.SCENARIOS["fake3_mimic"]).env
            try:
                env.reset()
                for _ in range(6):
                    env.step(
                        torch.zeros(env.num_envs, env.robot_config.number_of_actions)
                    )
                    assert torch.all(env._episode.progress_buf == 0)
                    reference = env.reference
                    placed = reference.poses_at(None, reference.times)
                    root = env.simulator.get_robot_state().rigid_body_pos[:, 0]
                    torch.testing.assert_close(root, placed.rigid_body_pos[:, 0])
            finally:
                env.close()
    finally:
        scenarios._fake_mimic_task = original


@pytest.mark.parametrize("bootstrap", [True, False])
def test_fake_mimic_clip_done_is_a_truncation_only_under_bootstrap(bootstrap):
    """fake3_mimic: the clip-done check resets the env; it terminates only
    without ``bootstrap_on_episode_end``."""
    pytest.importorskip("mujoco")
    from protomotions.tests.parity import scenarios

    original = scenarios._fake_mimic_task

    def patched(num_motions):
        task = original(num_motions)
        task.controls["mimic"].bootstrap_on_episode_end = bootstrap
        return task

    def near_the_clip_end(env):
        # Pin every env a few steps before its clip ends, so the check fires
        # inside the horizon whatever the motion sampler drew.
        ids, _ = env.reference_motion.motion_manager.get_assignment()
        lengths = env.reference_motion.motion_lib.get_motion_length(ids)
        env.reference_motion.motion_manager.set_assignment(
            ids, (lengths - 3 * env.dt).clamp_min(0.0)
        )

    scenarios._fake_mimic_task = patched
    try:
        table, seen = _record("fake3_mimic", steps=45, prepare=near_the_clip_end)
    finally:
        scenarios._fake_mimic_task = original
    assert table["mimic/clip_done"].truncation is bootstrap
    clip_done_fired = False
    for done, terminated, flags, progress in seen:
        clip_done = flags["termination/mimic/clip_done"].bool()
        others = torch.zeros_like(done)
        for name, flag in flags.items():
            if name != "termination/mimic/clip_done":
                others |= flag.bool()
        assert torch.all(clip_done <= done)
        if bootstrap:
            # A row that ended only by the clip is reset but not terminated.
            only_clip = clip_done & ~others
            assert not bool(terminated[only_clip].any())
        else:
            assert torch.all(clip_done <= terminated)
        clip_done_fired |= bool(clip_done.any())
    assert clip_done_fired


def test_termination_table_refuses_a_name_used_by_the_recipe_and_a_control():
    """``build.termination_table``: the recipe and a control naming the same
    entry is a ``ValueError`` that names the clash, not a silent override (and
    not an ``assert``, which ``python -O`` strips)."""
    from protomotions.envs.base_env.build import termination_table

    inputs = every_in_tree_control()[0].inputs
    target = TargetControl(TargetControlConfig(enable_gap_termination=True), inputs)
    manager = ControlManager({}, inputs, TeleportQueue())
    manager.components = {"target": target}
    rules = manager.terminations()
    assert "target/gap" in rules
    env = SimpleNamespace(
        config=SimpleNamespace(
            termination_components={"target/gap": rules["target/gap"]}
        ),
        control_manager=manager,
    )
    with pytest.raises(ValueError, match="target/gap"):
        termination_table(env)


def test_control_manager_names_a_control_whose_advance_returns_nothing():
    """``ControlManager.advance``: a control whose ``advance`` returns None
    instead of its logging dict is refused by name, not with an
    ``AttributeError`` on ``NoneType``."""
    inputs = every_in_tree_control()[0].inputs
    steering = SteeringControl(SteeringControlConfig(), inputs)
    manager = ControlManager({}, inputs, TeleportQueue())
    manager.components = {"steering": steering}
    steering.advance = lambda state: None  # a misbehaving control
    with pytest.raises(TypeError, match="'steering'.advance returned NoneType"):
        manager.advance(None)
