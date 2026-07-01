# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight control-component lifecycle tests."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.control.external_kinematic_control import (
    ExternalKinematicControl,
    ExternalKinematicControlConfig,
)
from protomotions.envs.control.kinematic_replay_control import (
    KinematicReplayControl,
    KinematicReplayControlConfig,
)
from protomotions.envs.control.masked_mimic_control import (
    FixedBodyCondition,
    MaskedMimicControl,
    MaskedMimicControlConfig,
)
from protomotions.envs.control.path_follower_control import (
    PathFollowerControl,
    PathFollowerControlConfig,
)
from protomotions.envs.control.steering_control import (
    SteeringControl,
    SteeringControlConfig,
)
from protomotions.envs.control.target_control import (
    KeyboardTargetCommandSourceConfig,
    RandomTargetCommandSourceConfig,
    TargetControl,
    TargetControlConfig,
    TargetKeyBindingConfig,
)
from protomotions.simulator.base_simulator.user_interface import UserInterface
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
    RobotState,
    StateConversion,
)


def _identity_quat(*shape):
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def _robot_state(root_pos, root_rot=None, num_bodies=None, dof_dim=1, contacts=None):
    root_pos = torch.as_tensor(root_pos, dtype=torch.float32)
    if root_pos.ndim == 1:
        root_pos = root_pos.unsqueeze(0)
    num_envs = root_pos.shape[0]
    num_bodies = num_bodies or 1
    body_pos = root_pos[:, None, :].repeat(1, num_bodies, 1)
    body_rot = _identity_quat(num_envs, num_bodies)
    if root_rot is not None:
        body_rot[:, 0] = torch.as_tensor(root_rot, dtype=torch.float32)
    return RobotState(
        state_conversion=StateConversion.COMMON,
        rigid_body_pos=body_pos,
        rigid_body_rot=body_rot,
        rigid_body_vel=torch.zeros(num_envs, num_bodies, 3),
        rigid_body_ang_vel=torch.zeros(num_envs, num_bodies, 3),
        rigid_body_contacts=(
            torch.zeros(num_envs, num_bodies, dtype=torch.bool)
            if contacts is None
            else contacts
        ),
        dof_pos=torch.zeros(num_envs, dof_dim),
        dof_vel=torch.zeros(num_envs, dof_dim),
    )


def _reset_state(num_envs=2):
    return ResetState(
        state_conversion=StateConversion.COMMON,
        root_pos=torch.zeros(num_envs, 3),
        root_rot=_identity_quat(num_envs),
        root_vel=torch.zeros(num_envs, 3),
        root_ang_vel=torch.zeros(num_envs, 3),
        dof_pos=torch.zeros(num_envs, 1),
        dof_vel=torch.zeros(num_envs, 1),
    )


def _object_state(root_pos, root_rot=None, root_vel=None):
    root_pos = torch.as_tensor(root_pos, dtype=torch.float32)
    num_envs, num_objects = root_pos.shape[:2]
    return ObjectState(
        state_conversion=StateConversion.COMMON,
        root_pos=root_pos,
        root_rot=(
            _identity_quat(num_envs, num_objects)
            if root_rot is None
            else torch.as_tensor(root_rot, dtype=torch.float32)
        ),
        root_vel=(
            torch.zeros(num_envs, num_objects, 3)
            if root_vel is None
            else torch.as_tensor(root_vel, dtype=torch.float32)
        ),
        root_ang_vel=torch.zeros(num_envs, num_objects, 3),
    )


class _FakeSimulator:
    def __init__(self, root_pos, *, num_bodies=1, headless=False):
        self.headless = headless
        self.user_interface = UserInterface()
        self.decimation = 2
        self.config = SimpleNamespace(sim=SimpleNamespace(fps=20))
        self.reset_calls = []
        self.robot_state = _robot_state(root_pos, num_bodies=num_bodies)
        self.object_state = None

    def get_root_state(self, env_ids=None):
        root = SimpleNamespace(
            root_pos=self.robot_state.root_pos,
            root_rot=self.robot_state.root_rot,
        )
        if env_ids is None:
            return root
        return SimpleNamespace(
            root_pos=root.root_pos[env_ids],
            root_rot=root.root_rot[env_ids],
        )

    def get_robot_state(self, env_ids=None):
        if env_ids is None:
            return self.robot_state
        return self.robot_state[env_ids]

    def get_bodies_state(self):
        return self.robot_state

    def get_object_root_state(self):
        return self.object_state

    def reset_envs(self, reset_state, object_state, env_ids):
        self.reset_calls.append((reset_state, object_state, env_ids.clone()))

class _FlatTerrain:
    def __init__(self, ground=0.0):
        self.ground = ground

    def get_ground_heights(self, positions):
        return torch.full((positions.shape[0], 1), self.ground, device=positions.device)


def test_target_control_fixed_target_lifecycle_context_and_markers():
    simulator = _FakeSimulator([[1.0, 2.0, 0.4], [3.0, 4.0, 0.8]], headless=False)
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.1,
        progress_buf=torch.tensor([5, 6]),
        reset_buf=torch.tensor([True, False]),
        terminate_buf=torch.tensor([False, False]),
        terrain=_FlatTerrain(ground=0.25),
        simulator=simulator,
    )
    control = TargetControl(
        TargetControlConfig(
            command_source=RandomTargetCommandSourceConfig(
                fixed_target_position=(2.5, -1.5),
            ),
            tar_proximity_threshold=0.4,
            enable_fall_termination=False,
        ),
        env,
    )

    control.reset(torch.tensor([0, 1]))
    ctx = SimpleNamespace()
    control.populate_context(ctx)
    markers = control.get_markers_state()
    reset_buf, terminate_buf = control.check_resets_and_terminations()

    assert torch.allclose(
        ctx.target.tar_pos,
        torch.tensor([[2.5, -1.5, 0.25], [2.5, -1.5, 0.25]]),
    )
    assert ctx.target.tar_proximity_threshold == 0.4
    assert torch.isinf(control._tar_change_time).all()
    assert torch.equal(reset_buf, torch.zeros(2, dtype=torch.bool))
    assert torch.equal(terminate_buf, torch.zeros(2, dtype=torch.bool))
    assert control.create_visualization_markers(headless=True) == {}
    assert len(control.create_visualization_markers(headless=False)["target_markers"].markers) == 1
    assert torch.allclose(markers["target_markers"].translation[:, 0, 2], torch.tensor([0.35, 0.35]))

    simulator.headless = True
    assert control.get_markers_state() == {}


def test_target_control_resamples_on_schedule_and_clamps_bounds():
    torch.manual_seed(5)
    simulator = _FakeSimulator([[4.0, -4.0, 1.0], [0.0, 0.0, 1.0]])
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.5,
        progress_buf=torch.tensor([4, 0]),
        reset_buf=torch.tensor([False, True]),
        terminate_buf=torch.tensor([False, False]),
        terrain=_FlatTerrain(ground=0.4),
        simulator=simulator,
    )
    control = TargetControl(
        TargetControlConfig(
            command_source=RandomTargetCommandSourceConfig(
                tar_change_time_min=1.0,
                tar_change_time_max=1.0,
                tar_dist_max=100.0,
                target_bounds=(-1.0, 1.0, -2.0, 2.0),
            ),
            enable_fall_termination=False,
        ),
        env,
    )
    control._tar_change_time[:] = torch.tensor([1.0, 100.0])
    control._tar_pos[:] = torch.tensor([[9.0, 9.0, 9.0], [0.25, 0.5, 0.4]])

    control.step()
    changed_target = control._tar_pos[0].clone()
    unchanged_target = control._tar_pos[1].clone()

    assert -1.0 <= changed_target[0] <= 1.0
    assert -2.0 <= changed_target[1] <= 2.0
    assert changed_target[2] == 0.4
    assert torch.allclose(unchanged_target, torch.tensor([0.25, 0.5, 0.4]))
    assert torch.allclose(control._tar_change_time, torch.tensor([3.0, 100.0]))


def test_target_control_keyboard_source_uses_registered_keys_for_active_env_only():
    simulator = _FakeSimulator([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]])
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.1,
        progress_buf=torch.tensor([0, 0]),
        reset_buf=torch.tensor([False, False]),
        terminate_buf=torch.tensor([False, False]),
        terrain=_FlatTerrain(ground=0.0),
        simulator=simulator,
    )
    control = TargetControl(
        TargetControlConfig(
            command_source=KeyboardTargetCommandSourceConfig(
                key_bindings=(
                    TargetKeyBindingConfig(
                        key="W",
                        action="move_forward",
                        delta_xy=(0.0, 1.0),
                        description="Move target forward",
                    ),
                ),
                fail_if_headless=False,
            ),
            enable_fall_termination=False,
        ),
        env,
    )
    control._tar_pos[:, :2] = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    simulator.user_interface.active_env_id = 1

    simulator.user_interface.handle_key_event("W", pressed=True)
    control.step()

    assert torch.allclose(control._tar_pos[:, :2], torch.tensor([[1.0, 1.0], [2.0, 3.0]]))


def test_target_control_keyboard_source_rejects_headless_simulator_by_default():
    simulator = _FakeSimulator([[0.0, 0.0, 1.0]], headless=True)
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        dt=0.1,
        progress_buf=torch.tensor([0]),
        reset_buf=torch.tensor([False]),
        terminate_buf=torch.tensor([False]),
        terrain=_FlatTerrain(ground=0.0),
        simulator=simulator,
    )

    with pytest.raises(RuntimeError, match="headless"):
        TargetControl(
            TargetControlConfig(
                command_source=KeyboardTargetCommandSourceConfig(
                    key_bindings=(
                        TargetKeyBindingConfig(
                            key="W",
                            action="move_forward",
                            delta_xy=(0.0, 1.0),
                            description="Move target forward",
                        ),
                    )
                ),
                enable_fall_termination=False,
            ),
            env,
        )


def test_target_control_gap_and_stuck_termination_respect_grace_period():
    simulator = _FakeSimulator([[0.0, 0.0, 0.1], [1.0, 0.0, 0.1]])
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.1,
        progress_buf=torch.tensor([3, 20]),
        reset_buf=torch.tensor([False, False]),
        terminate_buf=torch.tensor([False, False]),
        terrain=_FlatTerrain(ground=0.0),
        simulator=simulator,
    )
    control = TargetControl(
        TargetControlConfig(
            enable_fall_termination=False,
            enable_gap_termination=True,
            gap_termination_threshold=0.5,
            enable_stuck_termination=True,
            stuck_window_frames=4,
            stuck_movement_threshold=0.2,
            stuck_height_threshold=0.4,
            reset_grace_period=5,
        ),
        env,
    )
    control._last_support_root_height[:] = 1.0
    control._root_pos_history[:] = simulator.robot_state.root_pos.unsqueeze(1)

    reset_buf, terminate_buf = control.check_resets_and_terminations()

    assert reset_buf is not terminate_buf
    assert torch.equal(reset_buf, torch.tensor([False, True]))
    assert torch.equal(terminate_buf, torch.tensor([False, True]))

    reset_buf[1] = False
    assert terminate_buf[1]


def test_target_control_fall_termination_combines_contact_height_and_grace():
    contacts = torch.tensor(
        [
            [True, False],
            [True, False],
            [False, True],
        ]
    )
    simulator = _FakeSimulator(
        [[0.0, 0.0, 0.05], [1.0, 0.0, 0.05], [2.0, 0.0, 0.05]],
        num_bodies=2,
    )
    simulator.robot_state = _robot_state(
        [[0.0, 0.0, 0.05], [1.0, 0.0, 0.05], [2.0, 0.0, 0.05]],
        num_bodies=2,
        contacts=contacts,
    )
    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        dt=0.1,
        progress_buf=torch.tensor([2, 10, 10]),
        reset_buf=torch.tensor([False, False, False]),
        terminate_buf=torch.tensor([False, False, False]),
        terrain=_FlatTerrain(ground=0.0),
        simulator=simulator,
        non_termination_contact_body_ids=torch.tensor([1]),
    )
    control = TargetControl(
        TargetControlConfig(
            enable_fall_termination=True,
            fall_termination_height=0.2,
            reset_grace_period=5,
        ),
        env,
    )

    reset_buf, terminate_buf = control.check_resets_and_terminations()

    assert torch.equal(reset_buf, torch.tensor([False, True, False]))
    assert torch.equal(terminate_buf, torch.tensor([False, True, False]))


def test_steering_control_reset_step_context_and_marker_state_are_deterministic():
    simulator = _FakeSimulator([[0.0, 0.0, 0.9], [2.0, 0.0, 0.9]], headless=False)
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        progress_buf=torch.tensor([0, 0]),
        reset_buf=torch.tensor([True, False]),
        terminate_buf=torch.tensor([False, False]),
        simulator=simulator,
    )
    control = SteeringControl(
        SteeringControlConfig(
            heading_change_steps_min=2,
            heading_change_steps_max=3,
            random_heading_probability=0.0,
            standard_heading_change=0.0,
            standard_speed_change=0.0,
            stop_probability=0.0,
            enable_rand_facing=False,
        ),
        env,
    )

    control.reset(torch.tensor([0, 1]))
    ctx = SimpleNamespace()
    control.populate_context(ctx)

    assert torch.equal(control._heading_change_steps, torch.tensor([2, 2]))
    assert torch.allclose(ctx.steering.tar_dir, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    assert torch.allclose(ctx.steering.tar_face_dir, ctx.steering.tar_dir)
    assert torch.allclose(ctx.steering.tar_speed, torch.ones(2))
    assert torch.allclose(ctx.steering.prev_root_pos, simulator.robot_state.root_pos)

    simulator.robot_state.rigid_body_pos[:, 0, 0] += torch.tensor([0.5, 1.0])
    env.progress_buf[:] = torch.tensor([1, 2])
    control.step()

    assert torch.allclose(control._prev_root_pos, torch.tensor([[0.0, 0.0, 0.9], [3.0, 0.0, 0.9]]))
    assert torch.allclose(control._curr_root_pos, torch.tensor([[0.5, 0.0, 0.9], [3.0, 0.0, 0.9]]))
    assert torch.equal(control._heading_change_steps, torch.tensor([2, 4]))
    marker_state = control.get_markers_state()
    assert set(marker_state) == {"movement_markers", "facing_markers"}
    assert torch.allclose(
        marker_state["movement_markers"].translation[:, 0],
        torch.tensor([[1.5, 0.0, 0.9], [4.0, 0.0, 0.9]]),
    )


def test_steering_control_empty_reset_has_no_task_termination_and_declares_markers():
    simulator = _FakeSimulator([[0.0, 0.0, 0.9]], headless=False)
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        progress_buf=torch.tensor([123]),
        reset_buf=torch.tensor([False]),
        terminate_buf=torch.tensor([False]),
        simulator=simulator,
    )
    control = SteeringControl(SteeringControlConfig(), env)
    original_speed = control._tar_speed.clone()

    control.reset(torch.tensor([], dtype=torch.long))
    reset_buf, terminate_buf = control.check_resets_and_terminations()
    markers = control.create_visualization_markers(headless=False)

    assert torch.equal(control._tar_speed, original_speed)
    assert torch.equal(reset_buf, torch.tensor([False]))
    assert torch.equal(terminate_buf, torch.tensor([False]))
    assert set(markers) == {"movement_markers", "facing_markers"}
    assert markers["movement_markers"].type == "arrow"
    assert markers["facing_markers"].markers[0].size == "regular"


def test_steering_control_randomized_reset_stops_and_suppresses_headless_markers():
    torch.manual_seed(13)
    simulator = _FakeSimulator(
        [[0.0, 0.0, 0.9], [2.0, 0.0, 0.9], [4.0, 0.0, 0.9]],
        headless=True,
    )
    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        progress_buf=torch.tensor([7, 8, 9]),
        reset_buf=torch.tensor([False, False, False]),
        terminate_buf=torch.tensor([False, True, False]),
        simulator=simulator,
    )
    control = SteeringControl(
        SteeringControlConfig(
            tar_speed_min=0.25,
            tar_speed_max=1.75,
            heading_change_steps_min=3,
            heading_change_steps_max=6,
            random_heading_probability=1.0,
            standard_heading_change=0.0,
            standard_speed_change=0.0,
            stop_probability=1.0,
            enable_rand_facing=True,
        ),
        env,
    )

    control.reset(torch.tensor([0, 1, 2]))
    ctx = SimpleNamespace()
    control.populate_context(ctx)

    assert torch.allclose(ctx.steering.tar_speed, torch.zeros(3))
    assert torch.allclose(torch.linalg.norm(ctx.steering.tar_dir, dim=-1), torch.ones(3))
    assert torch.allclose(
        torch.linalg.norm(ctx.steering.tar_face_dir, dim=-1), torch.ones(3)
    )
    assert control._heading_change_steps[1] < control._heading_change_steps[0]
    assert control.create_visualization_markers(headless=True) == {}
    assert control.get_markers_state() == {}


def test_path_follower_reset_context_and_marker_samples(monkeypatch):
    import protomotions.envs.control.path_follower_control as path_module

    class _FakePathGenerator:
        instances = []

        def __init__(self, config, device, num_envs, episode_dur, height_conditioned):
            self.reset_calls = []
            _FakePathGenerator.instances.append(self)

        def reset(self, env_ids, head_position):
            self.reset_calls.append((env_ids.clone(), head_position.clone()))

        def calc_pos(self, env_ids, times):
            env_ids = env_ids.float()
            return torch.stack((env_ids, times, env_ids + times), dim=-1)

    monkeypatch.setattr(path_module, "PathGenerator", _FakePathGenerator)
    simulator = _FakeSimulator([[1.0, 2.0, 1.5], [3.0, 4.0, 1.7]], num_bodies=3)
    simulator.robot_state.rigid_body_pos[:, 2] = torch.tensor(
        [[1.5, 2.5, 2.0], [3.5, 4.5, 2.4]]
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.5,
        progress_buf=torch.tensor([2, 4]),
        config=SimpleNamespace(max_episode_length=20),
        terrain=_FlatTerrain(ground=0.2),
        simulator=simulator,
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(body_names=["root", "spine", "head"]),
            common_naming_to_robot_body_names={"head_body_name": ["head"]},
        ),
    )
    config = PathFollowerControlConfig(num_traj_samples=3, traj_sample_timestep=0.25)
    config.path_generator.height_conditioned = False
    control = PathFollowerControl(config, env)

    control.reset(torch.tensor([0, 1]))
    ctx = SimpleNamespace()
    control.populate_context(ctx)
    markers = control.get_markers_state()

    reset_ids, reset_head_pos = control.path_generator.reset_calls[0]
    assert torch.equal(reset_ids, torch.tensor([0, 1]))
    assert torch.allclose(reset_head_pos[:, 2], torch.tensor([1.3, 1.5]))
    assert control.head_body_id == 2
    assert ctx.path.height_conditioned is False
    assert torch.allclose(ctx.path.tar_pos, torch.tensor([[0.0, 1.0, 1.0], [1.0, 2.0, 3.0]]))
    assert torch.allclose(ctx.path.head_pos[:, 2], torch.tensor([0.5, 0.7]))
    assert ctx.path.traj_samples.shape == (2, 3, 3)
    assert torch.allclose(markers["path_markers"].translation[..., 2], torch.full((2, 3), 1.0))
    assert torch.equal(
        markers["path_markers"].orientation,
        torch.zeros(2, 3, 4),
    )


def test_path_follower_terminates_on_distance_and_height_only_after_min_progress(monkeypatch):
    import protomotions.envs.control.path_follower_control as path_module

    class _StaticPathGenerator:
        def __init__(self, config, device, num_envs, episode_dur, height_conditioned):
            pass

        def reset(self, env_ids, head_position):
            pass

        def calc_pos(self, env_ids, times):
            return torch.zeros(len(env_ids), 3)

    monkeypatch.setattr(path_module, "PathGenerator", _StaticPathGenerator)
    simulator = _FakeSimulator([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], num_bodies=2)
    simulator.robot_state.rigid_body_pos[:, 0] = torch.tensor(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    simulator.robot_state.rigid_body_pos[:, 1] = torch.tensor(
        [[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]]
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.1,
        progress_buf=torch.tensor([10, 11]),
        config=SimpleNamespace(max_episode_length=20),
        terrain=_FlatTerrain(ground=0.0),
        simulator=simulator,
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(body_names=["root", "head"]),
            common_naming_to_robot_body_names={"head_body_name": ["head"]},
        ),
    )
    config = PathFollowerControlConfig(
        enable_path_termination=True,
        fail_dist=1.0,
        fail_height_dist=0.5,
    )
    config.path_generator.height_conditioned = True
    control = PathFollowerControl(config, env)

    reset_buf, terminate_buf = control.check_resets_and_terminations()

    assert torch.equal(reset_buf, torch.tensor([False, False]))
    assert torch.equal(terminate_buf, torch.tensor([False, True]))
    assert control.create_visualization_markers(headless=True) == {}


def test_path_follower_disabled_termination_does_not_query_path(monkeypatch):
    import protomotions.envs.control.path_follower_control as path_module

    class _NoQueryPathGenerator:
        def __init__(self, config, device, num_envs, episode_dur, height_conditioned):
            pass

        def calc_pos(self, env_ids, times):
            raise AssertionError("disabled termination should not query path targets")

    monkeypatch.setattr(path_module, "PathGenerator", _NoQueryPathGenerator)
    simulator = _FakeSimulator([[0.0, 0.0, 0.0]], num_bodies=2)
    simulator.robot_state.rigid_body_pos[:, 1] = torch.tensor([[100.0, 0.0, 100.0]])
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        dt=0.1,
        progress_buf=torch.tensor([100]),
        config=SimpleNamespace(max_episode_length=20),
        terrain=_FlatTerrain(ground=0.0),
        simulator=simulator,
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(body_names=["root", "head"]),
            common_naming_to_robot_body_names={"head_body_name": ["head"]},
        ),
    )
    config = PathFollowerControlConfig(enable_path_termination=False)
    config.path_generator.height_conditioned = True
    control = PathFollowerControl(config, env)

    reset_buf, terminate_buf = control.check_resets_and_terminations()

    assert torch.equal(reset_buf, torch.tensor([False]))
    assert torch.equal(terminate_buf, torch.tensor([False]))


def test_path_follower_height_conditioned_markers_and_default_queries(monkeypatch):
    import protomotions.envs.control.path_follower_control as path_module

    class _RecordingPathGenerator:
        instances = []

        def __init__(self, config, device, num_envs, episode_dur, height_conditioned):
            del config, device, num_envs, episode_dur
            self.height_conditioned = height_conditioned
            self.reset_calls = []
            self.calc_calls = []
            _RecordingPathGenerator.instances.append(self)

        def reset(self, env_ids, head_position):
            self.reset_calls.append((env_ids.clone(), head_position.clone()))

        def calc_pos(self, env_ids, times):
            self.calc_calls.append((env_ids.clone(), times.clone()))
            return torch.stack((env_ids.float() + 0.5, times, times + 1.0), dim=-1)

    monkeypatch.setattr(path_module, "PathGenerator", _RecordingPathGenerator)
    simulator = _FakeSimulator([[0.0, 0.0, 0.5], [1.0, 1.0, 0.25]], num_bodies=2)
    simulator.robot_state.rigid_body_pos[:, 1] = torch.tensor(
        [[0.0, 0.0, 1.5], [1.0, 1.0, 2.25]]
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.5,
        progress_buf=torch.tensor([2, 4]),
        config=SimpleNamespace(max_episode_length=20),
        terrain=_FlatTerrain(ground=0.25),
        simulator=simulator,
        robot_config=SimpleNamespace(
            kinematic_info=SimpleNamespace(body_names=["root", "head"]),
            common_naming_to_robot_body_names={"head_body_name": ["head"]},
        ),
    )
    config = PathFollowerControlConfig(num_traj_samples=2, traj_sample_timestep=0.5)
    config.path_generator.height_conditioned = True
    control = PathFollowerControl(config, env)

    control.reset(torch.tensor([], dtype=torch.long))
    assert control.path_generator.reset_calls == []

    control.reset(torch.tensor([0, 1]))
    reset_ids, reset_head_pos = control.path_generator.reset_calls[0]
    assert torch.equal(reset_ids, torch.tensor([0, 1]))
    torch.testing.assert_close(reset_head_pos[:, 2], torch.tensor([0.25, 0.0]))
    assert control.step() is None
    torch.testing.assert_close(
        control.calc_target_pos(),
        torch.tensor([[0.5, 1.0, 2.0], [1.5, 2.0, 3.0]]),
    )
    torch.testing.assert_close(
        control.get_head_position()[:, 2], torch.tensor([1.0, 2.0])
    )

    marker_cfg = control.create_visualization_markers(headless=False)
    marker_state = control.get_markers_state()

    assert len(marker_cfg["path_markers"].markers) == 2
    torch.testing.assert_close(
        marker_state["path_markers"].translation[..., 2],
        torch.tensor([[2.25, 2.75], [3.25, 3.75]]),
    )
    assert control.create_visualization_markers(headless=True) == {}

    simulator.headless = True
    assert control.get_markers_state() == {}


def test_external_kinematic_control_applies_one_pending_pose_and_clears_buffers():
    simulator = _FakeSimulator([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        progress_buf=torch.tensor([9, 8]),
        reset_buf=torch.tensor([True, False]),
        terminate_buf=torch.tensor([False, True]),
        simulator=simulator,
    )
    control = ExternalKinematicControl(ExternalKinematicControlConfig(), env)
    pending_pose = _reset_state(num_envs=2)

    control.set_next_pose(pending_pose)
    control.step()
    assert simulator.reset_calls == []

    control.reset(torch.tensor([0, 1]))
    control.set_next_pose(pending_pose)
    control.step()
    reset_buf, terminate_buf = control.check_resets_and_terminations()
    ctx = SimpleNamespace(untouched=True)
    control.populate_context(ctx)

    reset_state, object_state, env_ids = simulator.reset_calls[0]
    assert reset_state is pending_pose
    assert object_state is None
    assert torch.equal(env_ids, torch.tensor([0, 1]))
    assert torch.equal(env.progress_buf, torch.tensor([0, 0]))
    assert torch.equal(env.reset_buf, torch.tensor([0, 0]))
    assert torch.equal(env.terminate_buf, torch.tensor([0, 0]))
    assert control._next_pose is None
    assert torch.equal(reset_buf, torch.zeros(2, dtype=torch.bool))
    assert torch.equal(terminate_buf, torch.zeros(2, dtype=torch.bool))
    assert ctx.untouched is True


def test_kinematic_replay_advances_motion_resamples_done_and_offsets_states():
    simulator = _FakeSimulator([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    reset_key = simulator.user_interface.register_key(
        "R", owner="env", description="Reset all environments"
    )
    simulator.user_interface.handle_key_event("R", pressed=True)

    class _MotionManager:
        def __init__(self):
            self.motion_ids = torch.tensor([3, 4])
            self.motion_times = torch.tensor([0.5, 0.8])
            self.sample_calls = []

        def sample_motions(self, env_ids):
            self.sample_calls.append(env_ids.clone())

        def get_done_tracks(self):
            return torch.tensor([False, True])

    class _MotionLib:
        def get_motion_state(self, motion_ids, motion_times):
            num_envs = len(motion_ids)
            return RobotState(
                state_conversion=StateConversion.COMMON,
                rigid_body_pos=torch.stack(
                    (
                        torch.stack((motion_ids.float(), motion_times, torch.zeros(num_envs)), dim=-1),
                        torch.ones(num_envs, 3),
                    ),
                    dim=1,
                ),
                rigid_body_rot=_identity_quat(num_envs, 2),
                rigid_body_vel=torch.ones(num_envs, 2, 3) * 7.0,
                rigid_body_ang_vel=torch.ones(num_envs, 2, 3) * 8.0,
                dof_pos=motion_times.unsqueeze(-1),
                dof_vel=torch.ones(num_envs, 1) * 9.0,
            )

    class _SceneLib:
        def __init__(self):
            self.pose_queries = []

        def get_scene_pose(self, env_ids, motion_times, offset):
            self.pose_queries.append((env_ids.clone(), motion_times.clone(), offset))
            return _object_state(torch.zeros(len(env_ids), 1, 3), root_vel=torch.ones(len(env_ids), 1, 3))

        def num_scenes(self):
            return 1

    motion_manager = _MotionManager()
    scene_lib = _SceneLib()
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        progress_buf=torch.tensor([5, 6]),
        reset_buf=torch.tensor([True, False]),
        terminate_buf=torch.tensor([False, True]),
        simulator=simulator,
        motion_manager=motion_manager,
        motion_lib=_MotionLib(),
        scene_lib=scene_lib,
        config=SimpleNamespace(ref_object_respawn_offset=0.3),
        consume_reset_request=lambda: simulator.user_interface.consume_key_press(
            reset_key
        ),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda root, *args: torch.ones(root.shape[0], 1, 3),
    )
    control = KinematicReplayControl(KinematicReplayControlConfig(), env)

    control.step()
    reset_state, object_state, env_ids = simulator.reset_calls[0]

    assert torch.equal(motion_manager.sample_calls[0], torch.tensor([0, 1]))
    assert torch.equal(motion_manager.sample_calls[1], torch.tensor([1]))
    assert torch.allclose(motion_manager.motion_times, torch.tensor([0.1, 0.1]))
    assert torch.equal(env_ids, torch.tensor([0, 1]))
    assert torch.allclose(reset_state.root_pos, torch.tensor([[4.0, 1.1, 1.0], [5.0, 1.1, 1.0]]))
    assert torch.equal(reset_state.dof_vel, torch.zeros(2, 1))
    assert torch.equal(reset_state.root_vel, torch.zeros(2, 3))
    assert torch.equal(object_state.root_vel, torch.zeros(2, 1, 3))
    assert torch.allclose(object_state.root_pos, torch.ones(2, 1, 3))
    assert torch.equal(env.progress_buf, torch.tensor([0, 0]))
    assert not simulator.user_interface.was_pressed(reset_key)


def test_kinematic_replay_without_scenes_leaves_object_pose_unoffset():
    simulator = _FakeSimulator([[0.0, 0.0, 0.0]])

    class _MotionManager:
        motion_ids = torch.tensor([2])
        motion_times = torch.tensor([0.4])

        def __init__(self):
            self.sample_calls = []

        def sample_motions(self, env_ids):
            self.sample_calls.append(env_ids.clone())

        def get_done_tracks(self):
            return torch.tensor([False])

    class _MotionLib:
        def get_motion_state(self, motion_ids, motion_times):
            return RobotState(
                state_conversion=StateConversion.COMMON,
                rigid_body_pos=torch.tensor([[[2.0, motion_times[0].item(), 0.0]]]),
                rigid_body_rot=_identity_quat(1, 1),
                rigid_body_vel=torch.ones(1, 1, 3),
                rigid_body_ang_vel=torch.ones(1, 1, 3),
                dof_pos=torch.ones(1, 1),
                dof_vel=torch.ones(1, 1),
            )

    class _SceneLib:
        def get_scene_pose(self, env_ids, motion_times, offset):
            del env_ids, motion_times, offset
            return _object_state(torch.ones(1, 1, 3) * 5.0)

        def num_scenes(self):
            return 0

    motion_manager = _MotionManager()
    env = SimpleNamespace(
        num_envs=1,
        device=torch.device("cpu"),
        progress_buf=torch.tensor([7]),
        reset_buf=torch.tensor([True]),
        terminate_buf=torch.tensor([True]),
        simulator=simulator,
        motion_manager=motion_manager,
        motion_lib=_MotionLib(),
        scene_lib=_SceneLib(),
        config=SimpleNamespace(ref_object_respawn_offset=0.0),
        consume_reset_request=lambda: False,
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda root, *args: torch.ones(root.shape[0], 1, 3),
    )
    control = KinematicReplayControl(KinematicReplayControlConfig(), env)

    control.step()
    reset_state, object_state, env_ids = simulator.reset_calls[0]

    assert motion_manager.sample_calls == []
    assert torch.equal(env_ids, torch.tensor([0]))
    assert torch.allclose(reset_state.root_pos, torch.tensor([[3.0, 1.5, 1.0]]))
    assert torch.allclose(object_state.root_pos, torch.ones(1, 1, 3) * 5.0)
    assert torch.equal(env.progress_buf, torch.tensor([0]))
    assert torch.equal(env.reset_buf, torch.tensor([False]))
    assert torch.equal(env.terminate_buf, torch.tensor([False]))


def test_masked_mimic_fixed_masks_context_and_colored_marker_state():
    class _MotionLib:
        def __init__(self):
            self.motion_lengths = torch.tensor([2.0, 2.0])

        def get_motion_length(self, motion_ids):
            return self.motion_lengths[motion_ids]

        def get_motion_state(self, motion_ids, motion_times):
            num = len(motion_ids)
            pos = torch.zeros(num, 3, 3)
            pos[:, :, 0] = motion_ids.float().unsqueeze(-1)
            pos[:, :, 1] = motion_times.unsqueeze(-1)
            pos[:, :, 2] = torch.arange(3).float()
            return RobotState(
                state_conversion=StateConversion.COMMON,
                rigid_body_pos=pos,
                rigid_body_rot=_identity_quat(num, 3),
                rigid_body_vel=torch.zeros(num, 3, 3),
                rigid_body_ang_vel=torch.zeros(num, 3, 3),
                dof_pos=motion_times.unsqueeze(-1),
                dof_vel=torch.zeros(num, 1),
            )

    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.1,
        simulator=SimpleNamespace(headless=False),
        motion_manager=SimpleNamespace(
            motion_ids=torch.tensor([0, 1]),
            motion_times=torch.tensor([0.2, 0.2]),
            get_done_tracks=lambda: torch.tensor([False, False]),
        ),
        motion_lib=_MotionLib(),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda ref: torch.ones(ref.shape[0], 1, 3),
        robot_config=SimpleNamespace(
            anchor_body_index=0,
            trackable_bodies_subset=["hand", "foot"],
            mimic_small_marker_bodies={"foot"},
            kinematic_info=SimpleNamespace(
                body_names=["root", "hand", "foot"],
                hinge_axes_map={1: torch.tensor([[0.0, 0.0, 1.0]])},
            ),
        ),
    )
    control = MaskedMimicControl(
        MaskedMimicControlConfig(
            future_steps=1,
            num_masked_future_steps=2,
            fixed_conditioning=[FixedBodyCondition("hand", 1)],
            repeat_mask_probability=0.0,
            visible_target_pose_prob=1.0,
        ),
        env,
    )

    control.target_times[:] = torch.tensor([[1.4, 1.8], [0.6, 1.4]])
    control.masked_mimic_target_poses_masks[:] = torch.tensor(
        [[True, False], [True, False]]
    )
    control.masked_mimic_target_bodies_masks[:] = torch.tensor(
        [
            [True, False, False, False, False, False, False, False],
            [True, False, False, False, False, False, False, False],
        ]
    )
    control._initialized = True

    marker_cfg = control.create_visualization_markers(headless=False)
    markers = control.get_markers_state()
    ctx = SimpleNamespace()
    control.populate_context(ctx)

    assert [marker.size for marker in marker_cfg["body_markers_blue"].markers] == [
        "regular",
        "small",
    ]
    assert set(markers) == {
        "body_markers_blue",
        "body_markers_yellow",
        "body_markers_red",
    }
    assert torch.allclose(markers["body_markers_blue"].translation[0, 0], torch.tensor([1.0, 2.4, 2.0]))
    assert torch.all(markers["body_markers_blue"].translation[1] > 50.0)
    assert torch.allclose(markers["body_markers_yellow"].translation[1, 0], torch.tensor([2.0, 1.6, 2.0]))
    assert torch.all(markers["body_markers_red"].translation > 50.0)
    assert ctx.masked_mimic.ref_pos.shape == (2, 2, 3, 3)
    assert torch.allclose(ctx.masked_mimic.time_offsets, torch.tensor([[1.2, 1.6], [0.4, 1.2]]))
    assert torch.equal(ctx.masked_mimic.target_poses_masks, control.masked_mimic_target_poses_masks)
    assert torch.equal(ctx.masked_mimic.target_bodies_masks, control.masked_mimic_target_bodies_masks)

    env.simulator.headless = True
    assert control.get_markers_state() == {}


def test_masked_mimic_reset_initializes_future_times_and_can_hide_all_targets():
    class _MotionLib:
        motion_lengths = torch.tensor([1.0, 1.5])

        def get_motion_length(self, motion_ids):
            return self.motion_lengths[motion_ids]

        def get_motion_state(self, motion_ids, motion_times):
            num = len(motion_ids)
            return RobotState(
                state_conversion=StateConversion.COMMON,
                rigid_body_pos=torch.zeros(num, 3, 3),
                rigid_body_rot=_identity_quat(num, 3),
                rigid_body_vel=torch.zeros(num, 3, 3),
                rigid_body_ang_vel=torch.zeros(num, 3, 3),
                dof_pos=torch.zeros(num, 1),
                dof_vel=torch.zeros(num, 1),
            )

    motion_manager = SimpleNamespace(
        motion_ids=torch.tensor([0, 1]),
        motion_times=torch.tensor([0.2, 1.4]),
        get_done_tracks=lambda: torch.tensor([False, False]),
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.1,
        simulator=SimpleNamespace(headless=True),
        motion_manager=motion_manager,
        motion_lib=_MotionLib(),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda ref: torch.zeros(ref.shape[0], 1, 3),
        robot_config=SimpleNamespace(
            anchor_body_index=0,
            trackable_bodies_subset=["hand", "foot"],
            mimic_small_marker_bodies=set(),
            kinematic_info=SimpleNamespace(
                body_names=["root", "hand", "foot"],
                hinge_axes_map={},
            ),
        ),
    )
    control = MaskedMimicControl(
        MaskedMimicControlConfig(
            future_steps=1,
            num_masked_future_steps=3,
            visible_target_pose_prob=0.0,
            repeat_mask_probability=0.0,
        ),
        env,
    )

    torch.manual_seed(23)
    control.reset(torch.tensor([0, 1]))

    assert control._initialized is True
    assert control.target_times.shape == (2, 3)
    assert torch.all(control.target_times >= motion_manager.motion_times.unsqueeze(-1))
    assert torch.all(control.target_times <= torch.tensor([[1.0], [1.5]]))
    assert torch.equal(
        control.masked_mimic_target_poses_masks,
        torch.zeros(2, 3, dtype=torch.bool),
    )
    assert torch.equal(
        control.masked_mimic_target_bodies_masks,
        torch.zeros(2, 3 * 2 * 2, dtype=torch.bool),
    )


def test_masked_mimic_body_mask_sampling_honors_max_and_small_body_probabilities():
    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        dt=0.1,
        simulator=SimpleNamespace(headless=True),
        motion_manager=SimpleNamespace(
            motion_ids=torch.zeros(4, dtype=torch.long),
            motion_times=torch.zeros(4),
            get_done_tracks=lambda: torch.zeros(4, dtype=torch.bool),
        ),
        motion_lib=SimpleNamespace(motion_lengths=torch.ones(1)),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda ref: torch.zeros(ref.shape[0], 1, 3),
        robot_config=SimpleNamespace(
            anchor_body_index=0,
            trackable_bodies_subset=["a", "b", "c", "d", "e"],
            mimic_small_marker_bodies=set(),
            kinematic_info=SimpleNamespace(
                body_names=["root", "a", "b", "c", "d", "e"],
                hinge_axes_map={},
            ),
        ),
    )
    max_control = MaskedMimicControl(
        MaskedMimicControlConfig(
            num_masked_future_steps=1,
            force_max_conditioned_bodies_prob=1.0,
            force_small_num_conditioned_bodies_prob=0.0,
        ),
        env,
    )

    torch.manual_seed(29)
    max_masks = max_control._sample_new_body_masks(4).view(4, 5, 2)
    assert torch.equal(max_masks.any(dim=-1), torch.ones(4, 5, dtype=torch.bool))

    small_control = MaskedMimicControl(
        MaskedMimicControlConfig(
            num_masked_future_steps=1,
            force_max_conditioned_bodies_prob=0.0,
            force_small_num_conditioned_bodies_prob=1.0,
        ),
        env,
    )

    torch.manual_seed(31)
    small_masks = small_control._sample_new_body_masks(4).view(4, 5, 2)
    active_counts = small_masks.any(dim=-1).sum(dim=-1)
    assert torch.all(active_counts >= 1)
    assert torch.all(active_counts <= 3)


def test_masked_mimic_shifts_masks_resamples_nonempty_and_inherits_clip_termination():
    class _MotionLib:
        def __init__(self):
            self.motion_lengths = torch.tensor([2.0, 3.0])

        def get_motion_length(self, motion_ids):
            return self.motion_lengths[motion_ids]

        def get_motion_state(self, motion_ids, motion_times):
            num = len(motion_ids)
            return RobotState(
                state_conversion=StateConversion.COMMON,
                rigid_body_pos=torch.zeros(num, 3, 3),
                rigid_body_rot=_identity_quat(num, 3),
                rigid_body_vel=torch.zeros(num, 3, 3),
                rigid_body_ang_vel=torch.zeros(num, 3, 3),
                dof_pos=torch.zeros(num, 1),
                dof_vel=torch.zeros(num, 1),
            )

    motion_manager = SimpleNamespace(
        motion_ids=torch.tensor([0, 1]),
        motion_times=torch.tensor([0.5, 0.5]),
        get_done_tracks=lambda: torch.tensor([False, True]),
    )
    env = SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        dt=0.1,
        simulator=SimpleNamespace(headless=True),
        motion_manager=motion_manager,
        motion_lib=_MotionLib(),
        get_spawn_to_ref_pose_offset_with_terrain_height_correction=lambda ref: torch.zeros(ref.shape[0], 1, 3),
        robot_config=SimpleNamespace(
            anchor_body_index=0,
            trackable_bodies_subset=["hand", "foot"],
            mimic_small_marker_bodies=set(),
            kinematic_info=SimpleNamespace(
                body_names=["root", "hand", "foot"],
                hinge_axes_map={},
            ),
        ),
    )
    control = MaskedMimicControl(
        MaskedMimicControlConfig(
            bootstrap_on_episode_end=False,
            future_steps=1,
            num_masked_future_steps=3,
            repeat_mask_probability=1.0,
            visible_target_pose_prob=1.0,
            fixed_conditioning=[FixedBodyCondition("hand", 0)],
        ),
        env,
    )
    control.target_times[:] = torch.tensor([[0.4, 0.8, 1.0], [0.4, 1.0, 1.4]])
    masks = control.masked_mimic_target_bodies_masks.view(2, 3, 2, 2)
    masks.zero_()
    masks[0, 1, 1, 0] = True
    masks[0, 2, 1, 1] = True
    masks[1, 0, 1, 0] = True
    control.masked_mimic_target_poses_masks[:] = masks.any(dim=-1).any(dim=-1)
    control._initialized = True

    torch.manual_seed(17)
    control.step()
    reset_buf, terminate_buf = control.check_resets_and_terminations()

    shifted = control.masked_mimic_target_bodies_masks.view(2, 3, 2, 2)
    assert torch.equal(shifted[0, 0], torch.tensor([[False, False], [True, False]]))
    assert torch.equal(shifted[0, 1], torch.tensor([[False, False], [False, True]]))
    assert torch.equal(shifted[0, 2], torch.tensor([[False, False], [False, True]]))
    assert torch.equal(shifted[1, 0], torch.zeros(2, 2, dtype=torch.bool))
    assert torch.equal(shifted[1, 2], torch.tensor([[True, False], [False, False]]))
    assert torch.equal(
        control.masked_mimic_target_poses_masks,
        shifted.any(dim=-1).any(dim=-1),
    )
    assert control.target_times[0, 0] == 0.8
    assert control.target_times[0, -1] > motion_manager.motion_times[0]
    assert torch.equal(reset_buf, torch.tensor([False, True]))
    assert torch.equal(terminate_buf, torch.tensor([False, True]))
    assert control.create_visualization_markers(headless=True) == {}
    assert control.get_markers_state() == {}
