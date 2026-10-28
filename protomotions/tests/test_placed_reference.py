# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The placed reference query, the clip assignment in the env snapshot and the
one sampling-weights writer (handover card C1)."""

from __future__ import annotations

import os
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import protomotions
import protomotions.envs.control as control_package
from protomotions.envs.base_env.reset.motion import ReferenceMotion
from protomotions.envs.base_env.reset.placed_reference import (
    NO_CLIPS,
    NO_DONE_TRACKS,
)
from protomotions.envs.motion_manager.config import (
    MimicMotionManagerConfig,
    MotionManagerConfig,
)
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    RobotState,
    StateConversion,
)
from protomotions.tests.support.fake_reference import fake_reference

CPU = torch.device("cpu")


class _Lib:
    """Three clips; body 0 of a pose encodes (clip id, time, 0) so a placed pose
    can be read back."""

    def __init__(self, lengths=(2.0, 3.0, 4.0)):
        self.motion_lengths = torch.tensor(lengths)
        self.motion_weights = torch.ones(len(lengths))
        self.motion_file = "lib"
        self.queries = []

    def num_motions(self):
        return self.motion_lengths.numel()

    def get_motion_length(self, motion_ids):
        return self.motion_lengths[motion_ids]

    def get_motion_positions(self, motion_ids, motion_times):
        return self.get_motion_state(motion_ids, motion_times).rigid_body_pos

    def get_motion_state(self, motion_ids, motion_times):
        self.queries.append((motion_ids.clone(), motion_times.clone()))
        n = motion_ids.shape[0]
        body0 = torch.stack((motion_ids.float(), motion_times, torch.zeros(n)), -1)
        rot = torch.zeros(n, 2, 4)
        rot[..., 3] = 1
        return RobotState(
            state_conversion=StateConversion.COMMON,
            rigid_body_pos=torch.stack((body0, torch.ones(n, 3)), dim=1),
            rigid_body_rot=rot,
            rigid_body_vel=torch.full((n, 2, 3), 7.0),
            rigid_body_ang_vel=torch.full((n, 2, 3), 8.0),
            dof_pos=motion_times[:, None].clone(),
            dof_vel=torch.full((n, 1), 9.0),
        )


# One XY offset per env, and a z lift that depends on the pose's time, so a
# placement taken from one slot is distinguishable from one taken per pose.
ENV_XY = torch.tensor([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])


def _placement(positions, env_ids):
    rows = torch.arange(3) if env_ids is None else env_ids
    offset = torch.zeros_like(positions)
    offset[:, :, :2] = ENV_XY[rows][:, None, :]
    offset[:, :, 2] = 100.0 * positions[:, 0, 1][:, None]
    return offset


def _env(manager_config=None, lib=None, with_manager=True):
    lib = _Lib() if lib is None else lib
    env = SimpleNamespace(num_envs=3, device=CPU, motion_lib=lib, motion_manager=None)
    if with_manager:
        config = MimicMotionManagerConfig(init_start_prob=0.0)
        if manager_config is not None:
            config = manager_config
        cls = (
            MimicMotionManager
            if isinstance(config, MimicMotionManagerConfig)
            else MotionManager
        )
        env.motion_manager = cls(config, 3, 0.1, CPU, lib, generator=None)
        env.motion_manager.set_assignment(
            torch.tensor([0, 1, 2]), torch.tensor([0.5, 1.0, 1.5])
        )
    return env


def _reference(env, **kwargs):
    return fake_reference(env, placement_offset=_placement, **kwargs)


def test_ids_times_and_lengths_read_the_installed_manager_and_library():
    """ids and times are the manager's own tensors; lengths are the library's."""
    env = _env()
    reference = _reference(env)
    assert reference.has_clips
    assert reference.ids is env.motion_manager.motion_ids
    assert reference.times is env.motion_manager.motion_times
    assert torch.equal(
        reference.lengths(torch.tensor([2, 0])), torch.tensor([4.0, 2.0])
    )


def test_poses_at_places_every_pose_on_its_own_env():
    """poses_at adds each row's placement to each pose; the clip comes from the
    row's assignment, the time from the argument."""
    reference = _reference(_env())
    state = reference.poses_at(torch.tensor([2, 0]), torch.tensor([0.25, 0.75]))
    expected = torch.tensor([[32.0, 0.25, 25.0], [10.0, 0.75, 75.0]])
    assert torch.allclose(state.rigid_body_pos[:, 0], expected)
    assert torch.allclose(state.rigid_body_pos[:, 1, :2], ENV_XY[[2, 0]] + 1.0)


def test_raw_poses_are_not_placed():
    """raw returns the library's poses in the clip's frame."""
    reference = _reference(_env())
    state = reference.raw(torch.tensor([1]), torch.tensor([0.5]))
    assert torch.equal(state.rigid_body_pos[0, 0], torch.tensor([1.0, 0.5, 0.0]))
    positions = reference.raw_positions(torch.tensor([1]), torch.tensor([0.5]))
    assert torch.equal(positions, state.rigid_body_pos)


def test_window_places_every_slot_from_the_first():
    """Every slot of a window shares the placement of slot 0 (the mimic
    convention), from one library query over the flattened window."""
    env = _env()
    reference = _reference(env)
    times = torch.tensor([[0.5, 1.0, 1.5], [0.2, 0.4, 0.6]])
    window = reference.window(torch.tensor([1, 0]), times)
    assert len(env.motion_lib.queries) == 1
    assert torch.equal(env.motion_lib.queries[0][0], torch.tensor([1, 1, 1, 0, 0, 0]))
    # x: the env's XY; y: the slot's own time; z: slot 0's time, for every slot.
    assert torch.allclose(window.pos[0, :, 0, 0], torch.full((3,), 21.0))
    assert torch.allclose(window.pos[0, :, 0, 1], times[0])
    assert torch.allclose(window.pos[0, :, 0, 2], torch.full((3,), 50.0))
    assert torch.allclose(window.pos[1, :, 0, 2], torch.full((3,), 20.0))
    assert window.rot.shape == (2, 3, 2, 4)
    assert window.vel.shape == window.ang_vel.shape == (2, 3, 2, 3)
    assert window.dof_pos.shape == window.dof_vel.shape == (2, 3, 1)
    assert torch.equal(window.times, times)


def test_current_and_window_is_one_query_with_clamped_future_times():
    """The current frame and the window come from ONE library query over the
    concatenation; future times are clamped to the clip length; the current
    frame is placed per pose and the window from its first slot."""
    env = _env()
    reference = _reference(env)
    env.motion_lib.queries.clear()
    current, window = reference.current_and_window(None, torch.tensor([0.5, 3.0]))
    assert len(env.motion_lib.queries) == 1
    ids, times = env.motion_lib.queries[0]
    assert torch.equal(ids, torch.tensor([0, 1, 2, 0, 0, 1, 1, 2, 2]))
    expected_future = torch.tensor([[1.0, 2.0], [1.5, 3.0], [2.0, 4.0]])
    assert torch.allclose(times[3:], expected_future.reshape(-1))
    assert torch.allclose(window.times, expected_future)
    assert torch.allclose(
        current.rigid_body_pos[:, 0, 0], ENV_XY[:, 0] + torch.arange(3)
    )
    assert torch.allclose(
        current.rigid_body_pos[:, 0, 2], torch.tensor([50.0, 100.0, 150.0])
    )
    # The window's lift comes from its slot 0 (time 1.0, 1.5, 2.0), not from the
    # current frame's time.
    assert torch.allclose(window.pos[:, 1, 0, 2], torch.tensor([100.0, 150.0, 200.0]))


def test_scene_poses_at_is_none_without_scenes_and_at_rest_with_them():
    """Without scenes there are no object poses; with them the objects are at rest."""
    env = _env()
    assert _reference(env).scene_poses_at(torch.arange(3), torch.zeros(3)) is None

    class _Scenes:
        def __init__(self):
            self.calls = []

        def num_scenes(self):
            return 2

        def get_scene_pose(self, env_ids, times, offset):
            self.calls.append((env_ids.clone(), times.clone(), offset))
            return ObjectState(
                state_conversion=StateConversion.COMMON,
                root_pos=torch.ones(len(env_ids), 1, 3),
                root_rot=torch.zeros(len(env_ids), 1, 4),
                root_vel=torch.ones(len(env_ids), 1, 3),
                root_ang_vel=torch.ones(len(env_ids), 1, 3),
            )

    scenes = _Scenes()
    reference = _reference(env, scene_lib=scenes, ref_object_respawn_offset=0.3)
    objects = reference.scene_poses_at(torch.tensor([1]), torch.tensor([0.7]))
    assert scenes.calls[0][2] == 0.3
    assert torch.equal(objects.root_vel, torch.zeros(1, 1, 3))
    assert torch.equal(objects.root_ang_vel, torch.zeros(1, 1, 3))


def test_spawn_translation_is_the_evaluator_arithmetic():
    """spawn_translation() subtracts the spawn-only height bump from z of an
    owned copy: the mimic evaluator's inverse of the spawn offset."""
    offsets = torch.tensor([[1.0, 2.0, 0.35], [3.0, 4.0, 1.05], [5.0, 6.0, 0.05]])
    reference = _reference(_env(), respawn_root_offset=offsets, ref_respawn_offset=0.05)
    assert torch.allclose(
        reference.spawn_translation(torch.tensor([1, 2])),
        torch.tensor([[3.0, 4.0, 1.0], [5.0, 6.0, 0.0]]),
    )
    assert torch.equal(offsets[1], torch.tensor([3.0, 4.0, 1.05]))  # not written
    assert reference.spawn_translation(None).shape == (3, 3)


def test_writers_go_to_the_manager_in_place():
    """set_assignment, assign, resample and advance_time write the manager's
    tensors in place."""
    env = _env()
    reference = _reference(env)
    ids, times = env.motion_manager.motion_ids, env.motion_manager.motion_times
    reference.set_assignment(torch.tensor([2, 2, 2]), torch.tensor([0.1, 0.2, 0.3]))
    reference.assign(torch.tensor([0]), times=torch.tensor([0.9]))
    reference.advance_time(0.1)
    assert (
        env.motion_manager.motion_ids is ids
        and env.motion_manager.motion_times is times
    )
    assert torch.equal(ids, torch.tensor([2, 2, 2]))
    assert torch.allclose(times, torch.tensor([1.0, 0.3, 0.4]))
    torch.manual_seed(0)
    reference.resample(torch.tensor([1]))
    assert 0 <= int(ids[1]) < 3 and torch.equal(ids[[0, 2]], torch.tensor([2, 2]))


def test_without_clips_every_assignment_read_raises():
    """Without an installed manager has_clips is False and ids, times and the
    writers raise naming the reason."""
    reference = _reference(_env(with_manager=False))
    assert not reference.has_clips
    with pytest.raises(RuntimeError, match=re.escape(NO_CLIPS)):
        reference.ids
    with pytest.raises(RuntimeError, match=re.escape(NO_CLIPS)):
        reference.advance_time(0.1)


def test_done_needs_a_manager_with_done_tracks():
    """done reads the mimic manager's clip ends; a base manager has none."""
    mimic = _reference(_env())
    assert torch.equal(mimic.done(), torch.tensor([False, False, False]))
    assert mimic.done(torch.tensor([2])).shape == (1,)
    base = _reference(_env(MotionManagerConfig(init_start_prob=0.0)))
    with pytest.raises(RuntimeError, match=re.escape(NO_DONE_TRACKS)):
        base.done()


def test_controls_never_assign_into_the_reference_tensors():
    """No control writes the query's ids/times, indexed (``reference.times[...]
    = ...``), augmented (``reference.times += dt``) or rebound; the writers
    (assign, set_assignment, advance_time) are the API."""
    pattern = re.compile(r"reference\.(ids|times)(\[[^\]]*\])?\s*(\+|-|\*|/)?=[^=]")
    offenders = []
    for path in sorted(Path(control_package.__file__).parent.rglob("*.py")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert offenders == []


def test_reference_motion_snapshot_round_trips_the_assignment():
    """ReferenceMotion saves the clip assignment as owned clones and loads it
    back in place; without a manager the snapshot is None and loading None is
    a no-op."""
    env = _env()
    motion = ReferenceMotion(None, None, None, 3, 0.1, CPU, 0, None)
    motion.motion_lib, motion.motion_manager = env.motion_lib, env.motion_manager
    snapshot = motion.save_state()
    ids, times = env.motion_manager.motion_ids, env.motion_manager.motion_times
    assert snapshot[0] is not ids and torch.equal(snapshot[0], ids)
    ids[:] = 7
    times[:] = 9.0
    motion.load_state(snapshot)
    assert torch.equal(ids, torch.tensor([0, 1, 2]))
    assert torch.allclose(times, torch.tensor([0.5, 1.0, 1.5]))
    motion.motion_manager = None
    assert motion.save_state() is None
    motion.load_state(None)


def test_reference_motion_refuses_a_snapshot_from_another_library():
    """A clip assignment saved against one library is not written over another:
    a manager that appeared or disappeared since the save, or an id outside
    the installed library (a live shard switch in between), is refused."""
    env = _env()
    motion = ReferenceMotion(None, None, None, 3, 0.1, CPU, 0, None)
    motion.motion_lib, motion.motion_manager = env.motion_lib, env.motion_manager
    snapshot = motion.save_state()
    with pytest.raises(
        ValueError, match="names clip 7, but the installed motion library has 3 clips"
    ):
        motion.load_state((torch.tensor([0, 7, 2]), snapshot[1]))
    with pytest.raises(RuntimeError, match="one has clips and the other has none"):
        motion.load_state(None)
    motion.motion_manager = None
    with pytest.raises(RuntimeError, match="one has clips and the other has none"):
        motion.load_state(snapshot)


def _manager_with_ineligible_clips():
    # Clip 0 is too short for the reset window (0.5 + 0.1 s), clip 2 is excluded.
    lib = _Lib(lengths=(0.2, 2.0, 3.0, 4.0))
    config = MotionManagerConfig(
        init_start_prob=0.0, exclude_motion_ids=[2], sample_time_truncate_s=0.5
    )
    return MotionManager(config, 3, 0.1, CPU, lib, generator=None)


def test_set_sampling_weights_clamps_eligible_clips_and_keeps_the_rest_at_zero():
    """The one writer clamps the eligible clips to [floor, ceiling] and keeps a
    too-short or excluded clip at 0 whatever it is given, from construction on."""
    manager = _manager_with_ineligible_clips()
    assert torch.equal(manager.motion_weights, torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.equal(manager._eligible, torch.tensor([False, True, False, True]))
    manager.set_sampling_weights(torch.full((4,), 5.0), floor=0.5, ceiling=2.0)
    assert torch.equal(manager.motion_weights, torch.tensor([0.0, 2.0, 0.0, 2.0]))
    manager.set_sampling_weights(torch.full((4,), 0.1), floor=0.5, ceiling=float("inf"))
    assert torch.equal(manager.motion_weights, torch.tensor([0.0, 0.5, 0.0, 0.5]))
    with pytest.raises(ValueError, match=r"shape \(4,\), got \(3,\)"):
        manager.set_sampling_weights(torch.ones(3), floor=0.0, ceiling=1.0)
    with pytest.raises(ValueError, match="0 <= floor <= ceiling"):
        manager.set_sampling_weights(torch.ones(4), floor=2.0, ceiling=1.0)


def test_loaded_weights_pass_through_the_writer():
    """A resumed checkpoint's weights cannot revive a too-short or excluded clip."""
    manager = _manager_with_ineligible_clips()
    manager.load_state_dict(
        {"motion_file_name": "lib", "motion_weights": torch.ones(4)}
    )
    assert torch.equal(manager.motion_weights, torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_fixed_clip_ids_outside_the_library_name_the_env_and_the_clip():
    """A scene paired to a clip the library lacks fails at manager construction."""
    with pytest.raises(
        ValueError,
        match="environment 1 is paired to clip 2, but the loaded motion library has 2 clips",
    ):
        MotionManager(
            MotionManagerConfig(init_start_prob=0.0),
            3,
            0.1,
            CPU,
            _Lib(lengths=(2.0, 3.0)),
            fixed_motion_ids_per_env=torch.tensor([0, 2, -1]),
            generator=None,
        )


REPO_ROOT = Path(protomotions.__file__).resolve().parents[1]


@pytest.mark.parity_slow
@pytest.mark.skipif(
    os.environ.get("PROTOMOTIONS_PARITY_SLOW") != "1",
    reason="MuJoCo playback build; set PROTOMOTIONS_PARITY_SLOW=1",
)
def test_kinematic_playback_example_replays_through_the_query_on_mujoco(monkeypatch):
    """examples/env_kinematic_playback.py runs 50 steps on MuJoCo: the kinematic
    replay's resample, assign, advance_time, done, raw, placement_offset and
    scene_poses_at paths over a real library, manager and spawn placer."""
    from protomotions.envs.base_env.env import BaseEnv

    steps = []
    original_step = BaseEnv.step

    def counted_step(self, actions):
        if len(steps) == 50:
            raise KeyboardInterrupt  # the example's loop ends on it and closes the env
        steps.append(1)
        return original_step(self, actions)

    monkeypatch.setattr(BaseEnv, "step", counted_step)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "env_kinematic_playback.py",
            "--robot-name",
            "smpl",
            "--simulator",
            "mujoco",
            "--num-envs",
            "1",
            "--motion-file",
            str(REPO_ROOT / "examples/data/smpl_humanoid_sit_armchair.motion"),
            "--experiment-path",
            str(REPO_ROOT / "examples/experiments/mimic/mlp.py"),
            "--headless",
            "--cpu-only",
        ],
    )
    runpy.run_path(
        str(REPO_ROOT / "examples/env_kinematic_playback.py"), run_name="__main__"
    )
    assert len(steps) == 50
