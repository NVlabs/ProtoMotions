# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for BaseEnv lifecycle helpers with lightweight fakes."""

from types import SimpleNamespace

import pytest
import torch

import protomotions.envs.base_env.env as base_env_module
from protomotions.envs.base_env.config import EnvConfig, RecoveryResetConfig
from protomotions.envs.base_env.env import BaseEnv
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
)
from protomotions.simulator.base_simulator.user_interface import UserInterface
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
    RobotState,
    StateConversion,
)
from protomotions.envs.obs.observation_noise import NoisyObservations
from protomotions.utils.rotations import calc_heading_quat_inv


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def _robot_state(
    num_envs: int = 3,
    num_bodies: int = 2,
    num_dofs: int = 2,
    base: float = 0.0,
    contacts: bool = True,
) -> RobotState:
    rigid_body_pos = (
        torch.arange(num_envs * num_bodies * 3, dtype=torch.float).reshape(
            num_envs, num_bodies, 3
        )
        + base
    )
    rigid_body_rot = _identity_quat(num_envs, num_bodies)
    rigid_body_vel = torch.ones(num_envs, num_bodies, 3) * (base + 1.0)
    rigid_body_ang_vel = torch.ones(num_envs, num_bodies, 3) * (base + 2.0)
    rigid_body_contacts = None
    if contacts:
        rigid_body_contacts = (
            torch.arange(num_envs * num_bodies).reshape(num_envs, num_bodies) % 2 == 0
        )
    return RobotState(
        state_conversion=StateConversion.COMMON,
        fps=30.0,
        dof_pos=torch.arange(num_envs * num_dofs, dtype=torch.float).reshape(
            num_envs, num_dofs
        )
        + base,
        dof_vel=torch.ones(num_envs, num_dofs) * (base + 3.0),
        dof_forces=torch.ones(num_envs, num_dofs) * (base + 4.0),
        rigid_body_pos=rigid_body_pos,
        rigid_body_rot=rigid_body_rot,
        rigid_body_vel=rigid_body_vel,
        rigid_body_ang_vel=rigid_body_ang_vel,
        rigid_body_contacts=rigid_body_contacts,
        rigid_body_contact_forces=torch.ones(num_envs, num_bodies, 3)
        * (base + 5.0),
        local_rigid_body_rot=rigid_body_rot.clone(),
    )


def _reset_state(num_envs: int = 3, num_dofs: int = 2, base: float = 0.0):
    return ResetState(
        state_conversion=StateConversion.COMMON,
        root_pos=torch.arange(num_envs * 3, dtype=torch.float).reshape(num_envs, 3)
        + base,
        root_rot=_identity_quat(num_envs),
        root_vel=torch.ones(num_envs, 3) * (base + 1.0),
        root_ang_vel=torch.ones(num_envs, 3) * (base + 2.0),
        dof_pos=torch.ones(num_envs, num_dofs) * (base + 3.0),
        dof_vel=torch.ones(num_envs, num_dofs) * (base + 4.0),
    )


def _object_state(num_envs: int = 3, num_objects: int = 1, base: float = 0.0):
    return ObjectState(
        state_conversion=StateConversion.COMMON,
        root_pos=torch.arange(num_envs * num_objects * 3, dtype=torch.float).reshape(
            num_envs, num_objects, 3
        )
        + base,
        root_rot=_identity_quat(num_envs, num_objects),
        root_vel=torch.ones(num_envs, num_objects, 3) * (base + 1.0),
        root_ang_vel=torch.ones(num_envs, num_objects, 3) * (base + 2.0),
        contact_forces=torch.ones(num_envs, num_objects, 3) * (base + 3.0),
    )


class _Terrain:
    def __init__(self, flat: bool = False):
        self.config = SimpleNamespace()
        self.num_height_points = 2
        self.height_points = torch.zeros(3, self.num_height_points, 3)
        self.height_samples = torch.zeros(1, 1)
        self.flat = flat
        self.sample_calls = []
        self.height_queries = []

    def is_flat(self):
        return self.flat

    def sample_valid_locations(self, num_envs, sample_flat=False):
        self.sample_calls.append((num_envs, sample_flat))
        values = torch.arange(num_envs * 2, dtype=torch.float).reshape(num_envs, 2)
        return values + torch.tensor([10.0, 20.0])

    def find_terrain_height_for_max_below_body(self, rigid_body_pos):
        self.height_queries.append(rigid_body_pos.clone())
        return torch.arange(rigid_body_pos.shape[0], dtype=torch.float) + 0.25

    def get_ground_heights(self, root_pos):
        return torch.ones(root_pos.shape[0], 1, device=root_pos.device) * 0.5

    def get_height_maps(self, root_state, _: object, return_all_dims: bool = False):
        del root_state, return_all_dims
        values = torch.arange(3 * self.num_height_points * 3, dtype=torch.float)
        return values.reshape(3, self.num_height_points, 3)


class _SceneLib:
    def __init__(
        self,
        num_envs: int = 3,
        scenes: int = 0,
        objects: int = 1,
        humanoid_motion_ids=None,
    ):
        self._num_envs = num_envs
        self._scenes = scenes
        self.num_objects_per_scene = objects if scenes > 0 else 0
        self.humanoid_motion_ids = humanoid_motion_ids

    def num_scenes(self):
        return self._scenes

    def get_scene_positions(self, terrain, device):
        del terrain
        values = torch.arange(self._num_envs * 3, dtype=torch.float, device=device)
        return values.reshape(self._num_envs, 3) + 100.0

    def get_default_object_state(self, device):
        del device
        return _object_state(self._num_envs)

    def get_scene_pose(self, env_ids, motion_times, respawn_offset=0.0):
        del respawn_offset
        state = _object_state(len(env_ids), base=50.0)
        state.root_pos[:, :, 0] = env_ids.float().view(-1, 1)
        state.root_pos[:, :, 1] = motion_times.float().view(-1, 1)
        return state

    def get_humanoid_motion_ids(self):
        return self.humanoid_motion_ids


class _MotionLib:
    def __init__(
        self,
        num_motions: int = 0,
        num_envs: int = 3,
        num_bodies: int = 2,
        num_dofs: int = 2,
        contacts: bool = True,
    ):
        self._num_motions = num_motions
        self.motion_lengths = torch.ones(max(num_motions, 1)) * 0.35
        self.motion_file = "/tmp/unit_motion.npy"
        self.num_envs = num_envs
        self.num_bodies = num_bodies
        self.num_dofs = num_dofs
        self.contacts = contacts
        self.smooth_windows = []
        self.queries = []

    def num_motions(self):
        return self._num_motions

    def smooth_contacts(self, window):
        self.smooth_windows.append(window)

    def get_motion_state(self, motion_ids, motion_times):
        self.queries.append((motion_ids.clone(), motion_times.clone()))
        state = _robot_state(
            len(motion_ids),
            num_bodies=self.num_bodies,
            num_dofs=self.num_dofs,
            base=10.0,
            contacts=self.contacts,
        )
        state.rigid_body_pos[:, 0, 0] = motion_ids.float()
        state.rigid_body_pos[:, 0, 1] = motion_times.float()
        state.dof_pos = torch.zeros(len(motion_ids), self.num_dofs)
        state.dof_pos[:, :2] = torch.stack(
            [motion_ids.float(), motion_times.float()], dim=-1
        )[:, : self.num_dofs]
        return state


class _FakeSimulator:
    def __init__(self, num_envs: int = 3, headless: bool = True):
        self.num_envs = num_envs
        self.num_act = 2
        self.dt = 0.05
        self.headless = headless
        self.user_interface = UserInterface()
        self.config = SimpleNamespace(
            _target_="fake.IsaacGym",
            domain_randomization=SimpleNamespace(observation_noise=None),
        )
        self.state = _robot_state(num_envs)
        self.reset_calls = []
        self.steps = []
        self.initialized_markers = None
        self.closed = False

    def is_simulation_running(self):
        return True

    def get_default_robot_reset_state(self):
        return _reset_state(self.num_envs)

    def _initialize_with_markers(self, markers):
        self.initialized_markers = markers

    def get_robot_state(self, env_ids=None):
        return self.state if env_ids is None else self.state[env_ids]

    def get_root_state(self):
        return self.state

    def step(self, actions, markers_callback=None):
        self.steps.append((actions.clone(), markers_callback))
        if markers_callback is not None:
            self.last_markers = markers_callback()

    def reset_envs(self, new_states, new_object_states=None, env_ids=None):
        self.reset_calls.append((new_states, new_object_states, env_ids.clone()))

    def get_current_actions(self):
        return torch.ones(self.num_envs, self.num_act) * 0.7

    def get_object_root_state(self, env_ids=None):
        state = _object_state(self.num_envs, base=80.0)
        return state if env_ids is None else state[env_ids]

    def close(self):
        self.closed = True


class _ControlManager:
    def __init__(self):
        self.components = {}
        self.step_calls = 0
        self.reset_calls = []
        self.populate_calls = []
        self.loaded_state = None
        self.reset_buf = torch.tensor([False, False, False])
        self.terminate_buf = torch.tensor([False, False, False])

    def get_markers_state(self):
        return {"control_marker": object()}

    def create_visualization_markers(self, headless):
        del headless
        return {"control_marker": VisualizationMarkerConfig(markers=[MarkerConfig()])}

    def step(self):
        self.step_calls += 1

    def reset(self, env_ids):
        self.reset_calls.append(env_ids.clone())

    def check_resets_and_terminations(self):
        return self.reset_buf.clone(), self.terminate_buf.clone()

    def populate_context(self, ctx):
        self.populate_calls.append(ctx)


class _ObsCallback:
    def __init__(self, obs):
        self.obs = obs
        self.compute_calls = []

    def get_obs(self):
        return {k: v.clone() for k, v in self.obs.items()}

    def compute_observations(self, env_ids):
        self.compute_calls.append(env_ids.clone())


class _ComponentExecutor:
    def __init__(self, env):
        self.env = env
        self.observations = {}
        self.rewards = {}
        self.terminations = {}

    def execute_all(self, components, ctx):
        del ctx
        if components is self.env.config.observation_components:
            return self.observations
        if components is self.env.config.reward_components:
            return self.rewards
        if components is self.env.config.termination_components:
            return self.terminations
        return {}


class _StateHistory:
    def __init__(self, num_envs=3, history_steps=2, store_noisy=False):
        self.num_history_steps = history_steps
        self.store_noisy = store_noisy
        self.actions = torch.ones(num_envs, history_steps + 1, 2)
        self.processed_actions = torch.ones(num_envs, history_steps + 1, 2) * 2.0
        self.historical_rigid_body_pos = torch.ones(num_envs, history_steps, 2, 3)
        self.historical_rigid_body_rot = _identity_quat(num_envs, history_steps, 2)
        self.historical_rigid_body_vel = torch.ones(num_envs, history_steps, 2, 3) * 2.0
        self.historical_rigid_body_ang_vel = (
            torch.ones(num_envs, history_steps, 2, 3) * 3.0
        )
        self.historical_dof_pos = torch.ones(num_envs, history_steps, 2)
        self.historical_dof_vel = torch.ones(num_envs, history_steps, 2) * 4.0
        self.historical_ground_heights = torch.ones(num_envs, history_steps) * 0.5
        self.historical_actions = self.actions[:, 1:]
        self.historical_processed_actions = self.processed_actions[:, 1:]
        self.historical_body_contacts = torch.ones(
            num_envs, history_steps, 1, dtype=torch.bool
        )
        self.historical_root_pos = self.historical_rigid_body_pos[:, :, 0]
        self.historical_root_rot = self.historical_rigid_body_rot[:, :, 0]
        self.historical_root_ang_vel = self.historical_rigid_body_ang_vel[:, :, 0]
        self.historical_anchor_pos = self.historical_rigid_body_pos[:, :, 0]
        self.historical_anchor_rot = self.historical_rigid_body_rot[:, :, 0]
        self.historical_anchor_vel = self.historical_rigid_body_vel[:, :, 0]
        self.historical_anchor_ang_vel = self.historical_rigid_body_ang_vel[:, :, 0]
        self.noisy_historical_rigid_body_pos = self.historical_rigid_body_pos + 0.1
        self.noisy_historical_rigid_body_rot = self.historical_rigid_body_rot.clone()
        self.noisy_historical_rigid_body_vel = self.historical_rigid_body_vel + 0.1
        self.noisy_historical_rigid_body_ang_vel = (
            self.historical_rigid_body_ang_vel + 0.1
        )
        self.noisy_historical_dof_pos = self.historical_dof_pos + 0.1
        self.noisy_historical_dof_vel = self.historical_dof_vel + 0.1
        self.noisy_historical_ground_heights = self.historical_ground_heights + 0.1
        self.noisy_historical_root_pos = self.noisy_historical_rigid_body_pos[:, :, 0]
        self.noisy_historical_root_rot = self.noisy_historical_rigid_body_rot[:, :, 0]
        self.noisy_historical_root_ang_vel = (
            self.noisy_historical_rigid_body_ang_vel[:, :, 0]
        )
        self.noisy_historical_anchor_pos = self.noisy_historical_rigid_body_pos[:, :, 0]
        self.noisy_historical_anchor_rot = self.noisy_historical_rigid_body_rot[:, :, 0]
        self.rotate_calls = []
        self.single_reset_calls = []
        self.state_reset_calls = []
        self.loaded_state = None

    def rotate_and_update(self, **kwargs):
        self.rotate_calls.append(kwargs)

    def reset_from_single_state(self, **kwargs):
        self.single_reset_calls.append(kwargs)

    def reset_from_states(self, **kwargs):
        self.state_reset_calls.append(kwargs)

    def save_state(self):
        return {"history": torch.tensor([1.0])}

    def load_state(self, state):
        self.loaded_state = state


class _MotionManager:
    def __init__(self):
        self.motion_ids = torch.tensor([0, 1, 0])
        self.motion_times = torch.tensor([0.15, 0.25, 0.35])
        self.config = SimpleNamespace(realign_motion_with_humanoid_on_each_step=False)
        self.sample_calls = []
        self.loaded_state = None

    def sample_motions(self, env_ids):
        self.sample_calls.append(env_ids.clone())

    def post_physics_step(self):
        self.post_physics_called = True

    def get_state_dict(self):
        return {"motion_ids": self.motion_ids.clone()}

    def load_state_dict(self, state):
        self.loaded_state = state


def _robot_config():
    return SimpleNamespace(
        kinematic_info=SimpleNamespace(
            num_bodies=2,
            num_dofs=2,
            body_names=["root", "foot"],
            dof_limits_lower=torch.tensor([-1.0, -1.0]),
            dof_limits_upper=torch.tensor([1.0, 1.0]),
            to=lambda device: None,
        ),
        contact_bodies=["root"],
        non_termination_contact_bodies=["foot"],
        anchor_body_index=0,
        number_of_actions=2,
        reset_noise=None,
    )


def _make_env(
    *,
    num_envs: int = 3,
    scenes: int = 0,
    motions: int = 0,
    headless: bool = True,
    history: bool = False,
):
    env = object.__new__(BaseEnv)
    env.device = torch.device("cpu")
    env.config = EnvConfig(
        max_episode_length=3,
        reset_grace_period=0,
        ref_respawn_offset=0.05,
    )
    env.config.observation_components = {}
    env.config.reward_components = {}
    env.config.termination_components = {}
    env.config.scene_obs.enabled = scenes > 0
    env.robot_config = _robot_config()
    env.terrain = _Terrain(flat=False)
    env.scene_lib = _SceneLib(num_envs=num_envs, scenes=scenes)
    env.motion_lib = _MotionLib(num_motions=motions, num_envs=num_envs)
    env.simulator = _FakeSimulator(num_envs=num_envs, headless=headless)
    env._key_bindings = env.simulator.user_interface.scope("env")
    env._key_bindings.register("R", "reset", "Reset all environments")
    env.num_envs = num_envs
    env.max_episode_length = env.config.max_episode_length
    env.dt = env.simulator.dt
    env.rew_buf = torch.zeros(num_envs)
    env.reset_buf = torch.ones(num_envs, dtype=torch.bool)
    env.terminate_buf = torch.ones(num_envs, dtype=torch.bool)
    env.progress_buf = torch.zeros(num_envs, dtype=torch.long)
    env.respawn_root_offset = torch.zeros(num_envs, 3)
    env.odom_scale = torch.ones(num_envs)
    env.odom_yaw_cos_sin = torch.zeros(num_envs, 2)
    env.odom_yaw_cos_sin[:, 0] = 1.0
    # Mirrors BaseEnv.__init__: the odometer's episode-start reference pose.
    env.odom_start_xy = torch.zeros(num_envs, 2)
    env.odom_start_heading_inv = torch.zeros(num_envs, 4)
    env.odom_start_heading_inv[:, 3] = 1.0
    env.prev_contact_force_magnitudes = torch.zeros(num_envs, 2)
    env._current_raw_action = torch.zeros(num_envs, 2)
    env._current_processed_action = torch.zeros(num_envs, 2)
    env._current_context = None
    env._current_noisy_obs = None
    env._action_config_device_ready = False
    env.skip_height_correction = False
    env.state_history = _StateHistory(num_envs) if history else None
    env.control_manager = _ControlManager()
    env.motion_manager = _MotionManager() if motions > 0 else None
    env.terrain_obs_cb = _ObsCallback({"terrain": torch.ones(num_envs, 1)})
    env.scene_obs_cb = _ObsCallback({"scene": torch.ones(num_envs, 2)})
    env._component_manager = _ComponentExecutor(env)
    env._observation_buffer = {}
    env.extras = {}
    return env


def test_recovery_reset_config_defaults_to_disabled():
    config = RecoveryResetConfig()

    assert config.recovery_prob == pytest.approx(0.0)
    assert config.fall_sim_steps == 150


def test_base_env_recovery_reset_is_noop_for_legacy_config():
    env = _make_env()
    del env.config.recovery_reset
    env._fall_reset_states = None
    new_states = _reset_state()
    original = new_states.clone()

    returned_states, recovery_mask = BaseEnv._apply_recovery_reset_states(
        env, torch.arange(env.num_envs), new_states
    )

    assert not recovery_mask.any()
    assert torch.equal(returned_states.root_pos, original.root_pos)
    assert torch.equal(returned_states.dof_pos, original.dof_pos)


def test_base_env_generates_grounded_fall_reset_states():
    env = _make_env()
    env.config.recovery_reset = RecoveryResetConfig(
        recovery_prob=1.0,
        fall_sim_steps=2,
    )
    env._fall_reset_states = None
    default_state = env.default_reset_state.clone()
    snapshot = {"saved": True}
    restored = []
    env.save_state = lambda: snapshot
    env.restore_state = restored.append

    torch.manual_seed(7)
    BaseEnv._generate_fall_reset_states(env)

    assert restored == [snapshot]
    assert env.terrain.sample_calls == [(env.num_envs, True)]
    assert len(env.simulator.reset_calls) == 1
    initial_state, object_state, env_ids = env.simulator.reset_calls[0]
    expected_xy = torch.tensor(
        [[10.0, 21.0], [12.0, 23.0], [14.0, 25.0]]
    )
    assert torch.equal(env_ids, torch.arange(env.num_envs))
    assert object_state is env.default_object_state
    assert torch.equal(initial_state.root_pos[:, :2], expected_xy)
    assert torch.allclose(
        initial_state.root_pos[:, 2], default_state.root_pos[:, 2] + 0.5
    )
    assert torch.allclose(
        initial_state.root_rot.norm(dim=-1), torch.ones(env.num_envs)
    )
    assert torch.count_nonzero(initial_state.root_vel) == 0
    assert torch.count_nonzero(initial_state.root_ang_vel) == 0
    assert torch.count_nonzero(initial_state.dof_vel) == 0

    assert len(env.simulator.steps) == 2
    first_targets = env.simulator.steps[0][0]
    assert torch.equal(env.simulator.steps[1][0], first_targets)
    assert torch.all(first_targets >= env.robot_config.kinematic_info.dof_limits_lower)
    assert torch.all(first_targets <= env.robot_config.kinematic_info.dof_limits_upper)

    assert env._fall_reset_states is not None
    assert torch.count_nonzero(env._fall_reset_states.root_vel) == 0
    assert torch.count_nonzero(env._fall_reset_states.root_ang_vel) == 0
    assert torch.count_nonzero(env._fall_reset_states.dof_vel) == 0


def test_base_env_samples_recovery_reset_states_by_probability(monkeypatch):
    env = _make_env()
    env.config.recovery_reset = RecoveryResetConfig(
        recovery_prob=0.5,
        fall_sim_steps=2,
    )
    env.respawn_root_offset[:, 2] = torch.tensor([1.0, 2.0, 3.0])
    env._fall_reset_states = _reset_state(base=100.0)
    env._fall_reset_states.dof_pos = torch.tensor(
        [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
    )
    env._fall_reset_states.root_rot = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    new_states = _reset_state()
    original = new_states.clone()

    monkeypatch.setattr(
        base_env_module.torch,
        "rand",
        lambda count, device=None: torch.tensor(
            [0.2, 0.8, 0.1], device=device
        )[:count],
    )

    def fake_randint(low, high, size, device=None):
        assert (low, high, size) == (0, 3, (2,))
        return torch.tensor([2, 0], device=device)

    monkeypatch.setattr(base_env_module.torch, "randint", fake_randint)

    returned_states, recovery_mask = BaseEnv._apply_recovery_reset_states(
        env,
        torch.arange(env.num_envs),
        new_states,
    )

    assert torch.equal(recovery_mask, torch.tensor([True, False, True]))
    assert torch.equal(returned_states.root_pos[:, :2], original.root_pos[:, :2])
    assert torch.equal(
        returned_states.root_pos[:, 2],
        torch.tensor([109.0, 5.0, 105.0]),
    )
    assert torch.equal(
        returned_states.root_rot[[0, 2]],
        env._fall_reset_states.root_rot[[2, 0]],
    )
    assert torch.equal(
        returned_states.dof_pos[[0, 2]],
        env._fall_reset_states.dof_pos[[2, 0]],
    )
    assert torch.equal(returned_states.root_pos[1], original.root_pos[1])
    assert torch.equal(returned_states.root_rot[1], original.root_rot[1])
    assert torch.equal(returned_states.dof_pos[1], original.dof_pos[1])
    assert torch.count_nonzero(returned_states.root_vel[[0, 2]]) == 0
    assert torch.count_nonzero(returned_states.root_ang_vel[[0, 2]]) == 0
    assert torch.count_nonzero(returned_states.dof_vel[[0, 2]]) == 0


def test_base_env_recovery_resets_use_current_state_for_history():
    env = _make_env(motions=2, history=True)
    env.config.recovery_reset = RecoveryResetConfig(recovery_prob=1.0)
    env._apply_recovery_reset_states = lambda env_ids, reset_states: (
        reset_states,
        torch.tensor([False, True, False], device=env.device),
    )

    BaseEnv.reset(env, env_ids=torch.arange(env.num_envs))

    assert torch.equal(
        env.state_history.single_reset_calls[-1]["env_ids"], torch.tensor([1])
    )
    assert torch.equal(
        env.state_history.state_reset_calls[-1]["env_ids"], torch.tensor([0, 2])
    )


def test_base_env_scene_surface_context_ignores_objects_without_pointclouds():
    env = _make_env(num_envs=2, scenes=1)

    def _unexpected_pointcloud_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("object pointclouds should not be read when disabled")

    env.scene_lib.get_scene_neutral_pointcloud = _unexpected_pointcloud_call
    env.scene_lib.get_per_object_valid_mask = _unexpected_pointcloud_call

    context = env._build_scene_surface_context()

    assert context.object_pos.shape == (2, 0, 3)
    assert context.object_rot.shape == (2, 0, 4)
    assert context.neutral_pointclouds.shape == (2, 0, 0, 3)
    assert context.object_valid_mask.shape == (2, 0)


def test_base_env_build_context_selects_subset_without_caching_it():
    env = _make_env(num_envs=3, scenes=0, history=True)
    env.control_manager.populate_context = lambda ctx: setattr(ctx, "mimic", object())
    env.odom_start_xy = torch.tensor([[0.0, 0.0], [10.0, 20.0], [30.0, 40.0]])
    env_ids = torch.tensor([2, 0])

    context = env._build_context(env_ids)

    assert torch.equal(context.env_ids, env_ids)
    assert torch.equal(context.current.root_pos, env.simulator.state.root_pos[env_ids])
    assert context.noisy is context.current
    assert torch.equal(
        context.historical.actions,
        env.state_history.historical_actions[env_ids],
    )
    assert context.terrain.height_points.shape[0] == len(env_ids)
    assert context.scene.object_pos.shape == (len(env_ids), 0, 3)
    assert torch.equal(context.progress_buf, env.progress_buf[env_ids])
    assert torch.equal(context.odom_start_xy, env.odom_start_xy[env_ids])
    assert context.odom_disp_start_clean.shape == (len(env_ids), 2)
    assert env._current_context is None


def test_process_observations_accepts_subset_and_multidimensional_outputs():
    env = _make_env(num_envs=3)
    env_ids = torch.tensor([0, 2])
    values = torch.arange(12, dtype=torch.float).reshape(2, 2, 3)
    env.config.observation_components = {"grid": object()}
    env._component_manager.observations = {"grid": values}

    env._process_observations(
        context=SimpleNamespace(env_ids=env_ids),
        env_ids=env_ids,
    )

    assert env._observation_buffer["grid"].shape == (3, 2, 3)
    assert torch.equal(env._observation_buffer["grid"][env_ids], values)
    assert torch.equal(env._observation_buffer["grid"][1], torch.zeros(2, 3))


def test_process_observations_preserves_global_order_for_reordered_full_batch():
    env = _make_env(num_envs=3)
    env_ids = torch.tensor([2, 0, 1])
    values = torch.tensor([[10.0], [20.0], [30.0]])
    env.config.observation_components = {"ordered": object()}
    env._component_manager.observations = {"ordered": values}

    env._process_observations(
        context=SimpleNamespace(env_ids=None),
        env_ids=env_ids,
    )

    assert torch.equal(env._observation_buffer["ordered"], values)


def test_process_observations_rejects_rows_that_do_not_match_context():
    env = _make_env(num_envs=3)
    env_ids = torch.tensor([2, 0])
    env.config.observation_components = {"bad": object()}
    env._component_manager.observations = {"bad": torch.zeros(3, 1)}

    with pytest.raises(ValueError, match="bad.*2 rows.*got 3"):
        env._process_observations(
            context=SimpleNamespace(env_ids=env_ids),
            env_ids=env_ids,
        )


def test_base_env_close_releases_ui_handles_and_component_resources():
    env = _make_env()
    component_scope = env.simulator.user_interface.scope("dummy_component")
    component_scope.register("K", "toggle", "Toggle dummy component")

    class _ClosableComponent:
        def close(self):
            component_scope.unregister_all()

    env.control_manager.components["dummy"] = _ClosableComponent()
    assert set(env.simulator.user_interface.registered_keys) == {"R", "K"}

    BaseEnv.close(env)

    assert env.simulator.closed is True
    assert env.simulator.user_interface.registered_keys == {}
    assert env._key_bindings is None

    # Repeated close should not try to unregister stale env handles.
    BaseEnv.close(env)
    assert env.simulator.user_interface.registered_keys == {}


def test_base_env_processing_helpers_update_buffers_and_logs():
    env = _make_env(scenes=1)
    action = torch.ones(3, 2)
    env.config.action_config = {
        "fn": lambda action, gain, bias: {"processed_action": action * gain + bias},
        "gain": torch.tensor(2.0),
        "bias": 1.0,
    }

    processed = BaseEnv._process_action(env, action, context=object())

    assert torch.equal(processed["processed_action"], torch.ones(3, 2) * 3.0)
    assert env._action_config_device_ready is True
    env.config.action_config = None
    assert BaseEnv._process_action(env, action, context=object()) == {
        "processed_action": action
    }
    assert BaseEnv.get_action_size(env) == 2
    assert BaseEnv.is_simulation_running(env) is True

    env.config.observation_components = {"dynamic": object()}
    env._component_manager.observations = {
        "dynamic": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    }
    env._process_observations(context=object(), env_ids=torch.tensor([0, 2]))
    assert torch.equal(env._observation_buffer["dynamic"][0], torch.tensor([1.0, 2.0]))
    assert torch.equal(env._observation_buffer["dynamic"][1], torch.zeros(2))
    assert torch.equal(env._observation_buffer["dynamic"][2], torch.tensor([5.0, 6.0]))

    obs = BaseEnv.get_obs(env)
    assert set(obs) == {"terrain", "scene", "dynamic"}
    obs["dynamic"][0, 0] = -99.0
    assert env._observation_buffer["dynamic"][0, 0] == 1.0

    BaseEnv.compute_observations(env, torch.tensor([1]), context=object())
    assert torch.equal(env.terrain_obs_cb.compute_calls[-1], torch.tensor([1]))
    assert torch.equal(env.scene_obs_cb.compute_calls[-1], torch.tensor([1]))
    with pytest.raises(ValueError, match="context is required"):
        BaseEnv.compute_observations(env, context=None)

    env.config.reward_components = {"alive": {"weight": 2.0}}
    env._component_manager.rewards = {"alive": torch.tensor([1.0, 2.0, 3.0])}
    BaseEnv.compute_reward(env, context=object())
    assert torch.equal(env.rew_buf, torch.tensor([2.0, 4.0, 6.0]))
    assert torch.equal(env.extras["total_env_reward"], env.rew_buf)
    assert BaseEnv.get_has_reset_grace(env) is None

    env.config.reset_grace_period = 2
    env.progress_buf = torch.tensor([1, 2, 3])
    assert torch.equal(
        BaseEnv.get_has_reset_grace(env), torch.tensor([True, True, False])
    )

    env.progress_buf = torch.tensor([0, 3, 1])
    env.control_manager.reset_buf = torch.tensor([True, False, False])
    env.control_manager.terminate_buf = torch.tensor([False, False, True])
    env.config.termination_components = {"fell": {"terminate_on_true": True}}
    env._component_manager.terminations = {"fell": torch.tensor([False, True, False])}
    reset, terminated = BaseEnv.check_resets_and_terminations(env, context=object())
    assert torch.equal(reset, torch.tensor([True, True, False]))
    assert torch.equal(terminated, torch.tensor([False, True, True]))
    assert torch.equal(env.extras["termination/fell"], torch.tensor([0.0, 1.0, 0.0]))

    BaseEnv.on_epoch_end(env, current_epoch=4)
    BaseEnv.user_reset(env)
    assert torch.equal(env.progress_buf, torch.full((3,), 100000000000))


def test_base_env_initialize_observations_populates_all_envs():
    env = _make_env(scenes=1)
    env.config.observation_components = {"dynamic": object()}
    env._component_manager.observations = {
        "dynamic": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    }

    BaseEnv._initialize_observations(env)

    assert torch.equal(
        env._observation_buffer["dynamic"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    assert env.control_manager.populate_calls[-1] is env._current_context


def test_base_env_cached_body_ids_and_clean_context_without_history():
    env = _make_env(history=False)
    env.robot_config.non_termination_contact_bodies = "all"

    ctx = BaseEnv.context.fget(env)

    assert torch.equal(env.contact_body_ids, torch.tensor([0]))
    assert torch.equal(ctx.non_termination_contact_body_ids, torch.tensor([0, 1]))
    assert ctx.historical is None
    assert ctx.noisy_historical is None
    assert ctx.previous_action is None
    assert ctx.previous_processed_action is None
    assert ctx.current_processed_action is env._current_processed_action
    assert BaseEnv.context.fget(env) is ctx
    assert env.control_manager.populate_calls[-1] is ctx


def test_base_env_spawn_offsets_cover_scene_and_non_scene_paths():
    env = _make_env()
    env_ids = torch.tensor([0, 2])
    ref_state = _robot_state()[env_ids]

    BaseEnv.update_respawn_root_offset_by_env_ids(
        env, env_ids, ref_state=ref_state, sample_flat=True
    )

    expected_xy = torch.tensor([[10.0, 21.0], [12.0, 23.0]]) - ref_state.root_pos[
        :, :2
    ]
    assert env.terrain.sample_calls[-1] == (2, True)
    assert torch.equal(env.respawn_root_offset[env_ids, :2], expected_xy)
    assert torch.allclose(env.respawn_root_offset[env_ids, 2], torch.tensor([0.3, 1.3]))

    offset = BaseEnv.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
        env, torch.zeros(3, 2, 3)
    )
    assert torch.equal(
        offset[:, :, :2],
        env.respawn_root_offset[:, None, :2].expand_as(offset[:, :, :2]),
    )
    assert torch.allclose(offset[:, :, 2], torch.tensor([[0.25], [1.25], [2.25]]))

    env.skip_height_correction = True
    no_z_offset = BaseEnv.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
        env, torch.zeros(3, 2, 3), env_ids=torch.tensor([0, 1, 2])
    )
    assert torch.equal(no_z_offset[:, :, 2], torch.zeros(3, 2))

    scene_env = _make_env(scenes=3)
    BaseEnv.update_respawn_root_offset_by_env_ids(scene_env, env_ids, ref_state=ref_state)
    assert torch.equal(
        scene_env.respawn_root_offset[env_ids, :2],
        torch.tensor([[100.0, 101.0], [106.0, 107.0]]),
    )
    assert torch.allclose(
        scene_env.respawn_root_offset[env_ids, 2], torch.tensor([0.3, 1.3])
    )
    scene_mask, non_scene_mask = BaseEnv.get_scene_non_scene_mask(scene_env, env_ids)
    assert torch.equal(scene_mask, torch.tensor([True, True]))
    assert torch.equal(non_scene_mask, torch.tensor([False, False]))


def test_base_env_reset_state_construction_and_ref_selection():
    env = _make_env(scenes=2, motions=2)
    env_ids = torch.tensor([0, 1])

    default_state, default_objects = BaseEnv.compute_default_reset_state(env, env_ids)
    assert torch.equal(
        default_state.root_pos,
        env.simulator.get_default_robot_reset_state()[env_ids].root_pos
        + env.respawn_root_offset[env_ids],
    )
    assert torch.equal(
        default_objects.root_pos,
        env.scene_lib.get_default_object_state(env.device)[env_ids].root_pos
        + env.respawn_root_offset[env_ids].unsqueeze(1),
    )

    motion_ids = torch.tensor([1, 0])
    motion_times = torch.tensor([0.2, 0.3])
    ref_state, ref_objects = BaseEnv.compute_ref_reset_state(
        env, env_ids, motion_ids, motion_times
    )
    assert torch.equal(ref_state.dof_pos, torch.stack([motion_ids.float(), motion_times], dim=-1))
    assert torch.equal(ref_objects.root_vel, torch.zeros_like(ref_objects.root_pos))
    assert torch.equal(ref_objects.root_ang_vel, torch.zeros_like(ref_objects.root_pos))

    empty_env = _make_env(motions=0)
    ref_env_ids, ids, times = BaseEnv._get_ref_reset_envs(
        empty_env, env_ids, force_default_mask=None
    )
    assert ref_env_ids.numel() == 0
    assert ids is None
    assert times is None

    force_default_mask = torch.tensor([True, False])
    ref_env_ids, ids, times = BaseEnv._get_ref_reset_envs(
        env, env_ids, force_default_mask=force_default_mask
    )
    assert torch.equal(ref_env_ids, torch.tensor([1]))
    assert torch.equal(env.motion_manager.sample_calls[-1], torch.tensor([1]))
    assert torch.equal(ids, env.motion_manager.motion_ids[ref_env_ids])
    assert torch.equal(times, env.motion_manager.motion_times[ref_env_ids])

    sample_count = len(env.motion_manager.sample_calls)
    ref_env_ids, ids, times = BaseEnv._get_ref_reset_envs(
        env,
        env_ids,
        force_default_mask=None,
        disable_motion_resample=True,
    )
    assert torch.equal(ref_env_ids, env_ids)
    assert len(env.motion_manager.sample_calls) == sample_count
    assert torch.equal(ids, env.motion_manager.motion_ids[env_ids])
    assert torch.equal(times, env.motion_manager.motion_times[env_ids])

    with pytest.raises(AssertionError, match="force_default_mask length"):
        BaseEnv._get_ref_reset_envs(
            env, env_ids, force_default_mask=torch.tensor([True])
        )


def test_base_env_motion_validation_rejects_body_count_mismatch():
    env = _make_env(motions=2)
    env.motion_lib = _MotionLib(num_motions=2, num_bodies=3, num_dofs=2)

    with pytest.raises(ValueError, match="3 bodies"):
        BaseEnv._validate_motion_lib_compatibility(env)


def test_base_env_reset_flow_mixes_default_and_reference_resets():
    env = _make_env(scenes=2, motions=2, history=True)
    env.config.odom_scale_range = (2.0, 2.0)
    env.config.odom_yaw_range_deg = 0.0

    obs, info = BaseEnv.reset(
        env,
        env_ids=[0, 1, 2],
        force_default_mask=torch.tensor([True, False, True]),
    )

    assert info == {}
    assert set(obs) == {"terrain", "scene"}
    reset_state, object_state, reset_ids = env.simulator.reset_calls[-1]
    assert torch.equal(reset_ids, torch.tensor([0, 1, 2]))
    assert reset_state.root_pos.shape == (3, 3)
    assert object_state.root_pos.shape == (3, 1, 3)
    assert torch.equal(env.control_manager.reset_calls[-1], torch.tensor([0, 1, 2]))
    assert torch.equal(env.progress_buf, torch.zeros(3, dtype=torch.long))
    assert torch.equal(env.reset_buf, torch.zeros(3, dtype=torch.bool))
    assert torch.equal(env.terminate_buf, torch.zeros(3, dtype=torch.bool))
    assert torch.equal(env._current_raw_action, torch.zeros(3, 2))
    assert torch.equal(env._current_processed_action, torch.zeros(3, 2))
    assert torch.equal(env.odom_scale, torch.full((3,), 2.0))
    assert torch.equal(env.odom_yaw_cos_sin, torch.tensor([[1.0, 0.0]]).repeat(3, 1))
    # The odometer's start pose is re-sampled from the post-reset robot state, so
    # displacement-from-start reads zero on the first step of a new episode.
    anchor_idx = env.robot_config.anchor_body_index
    reset_state = env.simulator.get_robot_state()
    assert torch.equal(env.odom_start_xy, reset_state.rigid_body_pos[:, anchor_idx, :2])
    expected_heading_inv = calc_heading_quat_inv(
        reset_state.rigid_body_rot[:, anchor_idx], w_last=True
    )
    assert torch.allclose(env.odom_start_heading_inv, expected_heading_inv)
    assert env.state_history.single_reset_calls
    assert env.state_history.state_reset_calls

    empty_obs, empty_info = BaseEnv.reset(env, env_ids=[])
    assert empty_info == {}
    assert set(empty_obs) == {"terrain", "scene"}


def test_base_env_reference_only_reset_skips_default_state_construction(monkeypatch):
    env = _make_env(scenes=2, motions=2, history=False)

    def fail_default(*args, **kwargs):
        raise AssertionError("reference-only reset should not build default states")

    monkeypatch.setattr(env, "compute_default_reset_state", fail_default)

    BaseEnv.reset(env, env_ids=[0, 2])

    reset_state, object_state, reset_ids = env.simulator.reset_calls[-1]
    assert torch.equal(reset_ids, torch.tensor([0, 2]))
    assert reset_state.root_pos.shape[0] == 2
    assert object_state.root_pos.shape[0] == 2


def test_compute_ref_reset_state_does_not_mutate_motion_state():
    env = _make_env(scenes=1, motions=1, history=False)
    source_state = _robot_state(num_envs=1, base=3.0)
    original_root_pos = source_state.root_pos.clone()
    env.motion_lib.get_motion_state = lambda motion_ids, motion_times: source_state

    BaseEnv.compute_ref_reset_state(
        env,
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0.1]),
    )

    assert torch.equal(source_state.root_pos, original_root_pos)


def test_base_env_reset_all_defaults_can_apply_reset_noise_without_ref_resample(
    monkeypatch,
):
    env = _make_env(motions=0, scenes=0, history=False)
    env.robot_config.reset_noise = SimpleNamespace(
        dof_pos_noise=0.0,
        dof_vel_noise=0.0,
        root_pos_noise=0.0,
        root_rot_noise=0.0,
        root_vel_noise=0.0,
        root_ang_vel_noise=0.0,
    )
    noise_calls = []

    def fake_apply_reset_noise(**kwargs):
        noise_calls.append(kwargs)

    monkeypatch.setattr(base_env_module, "apply_reset_noise", fake_apply_reset_noise)

    obs, info = BaseEnv.reset(env, sample_flat=True)
    reset_state, object_state, reset_ids = env.simulator.reset_calls[-1]

    assert info == {}
    assert set(obs) == {"terrain"}
    assert torch.equal(reset_ids, torch.tensor([0, 1, 2]))
    assert reset_state.root_pos.shape == (3, 3)
    assert object_state.root_pos.shape[0] == 3
    assert len(noise_calls) == 1
    assert noise_calls[0]["reset_state"] is reset_state
    assert env.terrain.sample_calls[-1] == (3, True)
    assert env.motion_manager is None


def test_base_env_reset_updates_noisy_cache_subset(monkeypatch):
    env = _make_env(history=True)
    env.robot_config.anchor_body_index = 1
    env.simulator.state.rigid_body_pos[:, 0, 0] = 10.0
    env.simulator.state.rigid_body_pos[:, 1, 0] = 20.0
    env.terrain.get_ground_heights = lambda positions: positions[:, :1] + 0.5
    ground_height_calls = []
    env._current_noisy_obs = NoisyObservations(
        rigid_body_pos=torch.zeros(3, 2, 3),
        rigid_body_rot=_identity_quat(3, 2),
        rigid_body_vel=torch.zeros(3, 2, 3),
        rigid_body_ang_vel=torch.zeros(3, 2, 3),
        dof_pos=torch.zeros(3, 2),
        dof_vel=torch.zeros(3, 2),
        root_rot=_identity_quat(3),
        root_local_ang_vel=torch.zeros(3, 3),
        anchor_rot=_identity_quat(3),
        anchor_local_ang_vel=torch.zeros(3, 3),
        ground_heights=torch.zeros(3),
    )

    def fake_apply_observation_noise(**kwargs):
        ground_height_calls.append(kwargs["ground_heights"].clone())
        assert "env_ids" not in kwargs
        n = kwargs["robot_state"].root_pos.shape[0]
        return NoisyObservations(
            rigid_body_pos=torch.ones(n, 2, 3) * 9.0,
            rigid_body_rot=_identity_quat(n, 2),
            rigid_body_vel=torch.ones(n, 2, 3) * 8.0,
            rigid_body_ang_vel=torch.ones(n, 2, 3) * 7.0,
            dof_pos=torch.ones(n, 2) * 6.0,
            dof_vel=torch.ones(n, 2) * 5.0,
            root_rot=_identity_quat(n),
            root_local_ang_vel=torch.ones(n, 3) * 4.0,
            anchor_rot=_identity_quat(n),
            anchor_local_ang_vel=torch.ones(n, 3) * 3.0,
            ground_heights=torch.ones(n) * 2.0,
        )

    monkeypatch.setattr(
        base_env_module, "apply_observation_noise", fake_apply_observation_noise
    )

    BaseEnv.reset(env, env_ids=torch.tensor([0, 2]), force_default_mask=None)

    assert torch.equal(env._current_noisy_obs.dof_pos[0], torch.tensor([6.0, 6.0]))
    assert torch.equal(env._current_noisy_obs.dof_pos[1], torch.zeros(2))
    assert torch.equal(env._current_noisy_obs.dof_pos[2], torch.tensor([6.0, 6.0]))
    assert torch.equal(
        env._current_noisy_obs.ground_heights, torch.tensor([2.0, 0.0, 2.0])
    )
    assert torch.equal(ground_height_calls[0], torch.tensor([20.5, 20.5]))


def test_base_env_reset_state_history_from_current_and_reference_states():
    env = _make_env(motions=2, history=True)
    env_ids = torch.tensor([0, 1, 2])
    ref_env_ids = torch.tensor([1, 2])
    motion_ids = torch.tensor([0, 1])
    motion_times = torch.tensor([0.05, 0.4])

    BaseEnv._reset_state_history(
        env,
        env_ids=env_ids,
        current_state_history_mask=torch.tensor([True, False, False]),
        ref_env_ids=ref_env_ids,
        motion_ids=motion_ids,
        motion_times=motion_times,
    )

    single = env.state_history.single_reset_calls[-1]
    assert torch.equal(single["env_ids"], torch.tensor([0]))
    assert torch.equal(single["body_contacts"], torch.tensor([[True]]))

    historical = env.state_history.state_reset_calls[-1]
    assert torch.equal(historical["env_ids"], ref_env_ids)
    assert historical["rigid_body_pos"].shape == (2, 3, 2, 3)
    assert historical["ground_heights"].shape == (2, 3)
    assert historical["body_contacts"].shape == (2, 3, 1)
    _, queried_times = env.motion_lib.queries[-1]
    assert torch.equal(queried_times, torch.tensor([0.05, 0.0, 0.0, 0.35, 0.35, 0.3]))

    env_no_contacts = _make_env(motions=2, history=True)
    env_no_contacts.motion_lib.contacts = False
    BaseEnv._reset_state_history(
        env_no_contacts,
        env_ids=torch.tensor([1]),
        current_state_history_mask=torch.tensor([False]),
        ref_env_ids=torch.tensor([1]),
        motion_ids=torch.tensor([0]),
        motion_times=torch.tensor([0.1]),
    )
    assert not env_no_contacts.state_history.state_reset_calls[-1]["body_contacts"].any()


def test_base_env_ground_height_uses_configured_anchor_body():
    env = _make_env(history=False)
    env.robot_config.anchor_body_index = 1
    env.simulator.state.rigid_body_pos[:, 0, 0] = 10.0
    env.simulator.state.rigid_body_pos[:, 1, 0] = 20.0
    env.terrain.get_ground_heights = lambda positions: positions[:, :1] + 0.5

    context = BaseEnv._build_global_context(env)

    expected = torch.full((3,), 20.5)
    assert torch.equal(context.ground_heights, expected)
    assert torch.equal(context.noisy_ground_heights, expected)


def test_base_env_history_ground_height_uses_configured_anchor_body():
    env = _make_env(history=True)
    env.robot_config.anchor_body_index = 1
    env.simulator.state.rigid_body_pos[:, 0, 0] = 10.0
    env.simulator.state.rigid_body_pos[:, 1, 0] = 20.0
    env.terrain.get_ground_heights = lambda positions: positions[:, :1] + 0.5

    BaseEnv._reset_state_history(
        env,
        env_ids=torch.arange(3),
        current_state_history_mask=torch.ones(3, dtype=torch.bool),
        ref_env_ids=torch.empty(0, dtype=torch.long),
        motion_ids=None,
        motion_times=None,
    )

    assert torch.equal(
        env.state_history.single_reset_calls[-1]["ground_heights"],
        torch.full((3,), 20.5),
    )


def test_base_env_post_physics_step_builds_context_and_raw_extras():
    env = _make_env(motions=2, history=True)
    env.robot_config.anchor_body_index = 1
    env.simulator.state.rigid_body_pos[:, 0, 0] = 10.0
    env.simulator.state.rigid_body_pos[:, 1, 0] = 20.0
    env.terrain.get_ground_heights = lambda positions: positions[:, :1] + 0.5
    env.motion_manager.config.realign_motion_with_humanoid_on_each_step = True
    observations = []
    rewards = []
    resets = []

    env.compute_observations = lambda context: observations.append(context)
    env.compute_reward = lambda context: rewards.append(context)

    def fake_check(context):
        resets.append(context)
        return torch.tensor([False, True, False]), torch.tensor([False, False, True])

    env.check_resets_and_terminations = fake_check

    BaseEnv.post_physics_step(env)

    assert torch.equal(env.progress_buf, torch.ones(3, dtype=torch.long))
    assert env.state_history.rotate_calls
    assert torch.equal(
        env.state_history.rotate_calls[-1]["ground_heights"],
        torch.full((3,), 20.5),
    )
    assert env.motion_manager.post_physics_called is True
    assert env.control_manager.step_calls == 1
    assert observations and rewards and resets
    assert torch.equal(env.reset_buf, torch.tensor([False, True, False]))
    assert torch.equal(env.terminate_buf, torch.tensor([False, False, True]))
    assert torch.equal(env.extras["terminate"], env.terminate_buf)
    assert "raw/rigid_body_pos" in env.extras
    assert torch.equal(
        env.prev_contact_force_magnitudes,
        torch.norm(env.simulator.state.rigid_body_contact_forces, dim=-1),
    )
    assert env.control_manager.populate_calls[-1] is env._current_context
    assert env._current_context.previous_action.shape == (3, 2)


def test_base_env_post_physics_step_uses_noisy_history_cache(monkeypatch):
    env = _make_env(history=True)
    env.state_history.store_noisy = True

    def fake_apply_observation_noise(**kwargs):
        n = kwargs["robot_state"].rigid_body_pos.shape[0]
        return NoisyObservations(
            rigid_body_pos=torch.ones(n, 2, 3) * 11.0,
            rigid_body_rot=_identity_quat(n, 2),
            rigid_body_vel=torch.ones(n, 2, 3) * 12.0,
            rigid_body_ang_vel=torch.ones(n, 2, 3) * 13.0,
            dof_pos=torch.ones(n, 2) * 14.0,
            dof_vel=torch.ones(n, 2) * 15.0,
            root_rot=_identity_quat(n),
            root_local_ang_vel=torch.ones(n, 3) * 16.0,
            anchor_rot=_identity_quat(n),
            anchor_local_ang_vel=torch.ones(n, 3) * 17.0,
            ground_heights=torch.ones(n) * 18.0,
        )

    monkeypatch.setattr(
        base_env_module, "apply_observation_noise", fake_apply_observation_noise
    )
    env.compute_observations = lambda context: None
    env.compute_reward = lambda context: None
    env.check_resets_and_terminations = lambda context: (
        torch.zeros(3, dtype=torch.bool),
        torch.zeros(3, dtype=torch.bool),
    )

    BaseEnv.post_physics_step(env)

    rotate_kwargs = env.state_history.rotate_calls[-1]
    assert torch.equal(rotate_kwargs["noisy_dof_pos"], torch.ones(3, 2) * 14.0)
    assert torch.equal(
        rotate_kwargs["noisy_ground_heights"], torch.ones(3) * 18.0
    )
    assert torch.equal(env._current_context.noisy.dof_pos, torch.ones(3, 2) * 14.0)
    assert torch.equal(env._current_context.noisy_ground_heights, torch.ones(3) * 18.0)


def test_base_env_step_processes_action_and_honors_user_reset():
    env = _make_env()
    env.config.action_config = {
        "fn": lambda action: {"processed_action": action + 1.0},
    }
    env.simulator.user_interface.handle_key_event("R", pressed=True)
    env._current_context = object()
    env._current_noisy_obs = object()
    post_calls = []
    user_reset_calls = []
    env.post_physics_step = lambda: post_calls.append(True)
    env.user_reset = lambda: user_reset_calls.append(True)
    env.get_obs = lambda: {"done": torch.ones(1)}
    env.rew_buf[:] = torch.tensor([1.0, 2.0, 3.0])
    env.reset_buf[:] = torch.tensor([False, True, False])
    env.terminate_buf[:] = torch.tensor([False, False, True])

    obs, rewards, resets, terminated, extras = BaseEnv.step(env, torch.ones(3, 2))

    assert obs == {"done": torch.ones(1)}
    assert rewards is env.rew_buf
    assert resets is env.reset_buf
    assert terminated is env.terminate_buf
    assert extras == {}
    assert torch.equal(env._current_raw_action, torch.ones(3, 2))
    assert torch.equal(env._current_processed_action, torch.ones(3, 2) * 2.0)
    assert torch.equal(env.simulator.steps[-1][0], torch.ones(3, 2) * 2.0)
    assert env.simulator.steps[-1][1] == env.get_markers_state
    assert post_calls == [True]
    assert user_reset_calls == [True]


def test_base_env_motion_manager_markers_state_save_restore_and_close(monkeypatch):
    env = _make_env(scenes=2, motions=2, headless=False, history=True)
    env.config.show_terrain_markers = True

    class _ConstructedMotionManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        base_env_module,
        "get_class",
        lambda target: _ConstructedMotionManager,
    )
    env.scene_lib.humanoid_motion_ids = [1, 0, 1]
    BaseEnv.create_motion_manager(env)
    assert torch.equal(
        env.motion_manager.kwargs["fixed_motion_ids_per_env"], torch.tensor([1, 0, 1])
    )
    assert env.motion_manager.kwargs["num_envs"] == 3

    env.motion_manager = _MotionManager()
    markers = BaseEnv.create_visualization_markers(env, headless=False)
    assert "terrain_markers" in markers
    assert "control_marker" in markers
    assert BaseEnv.create_visualization_markers(env, headless=True) == {}

    marker_state = BaseEnv.get_markers_state(env)
    assert "terrain_markers" in marker_state
    assert marker_state["terrain_markers"].translation.shape == (3, 2, 3)
    assert "control_marker" in marker_state

    env.simulator.headless = True
    assert BaseEnv.get_markers_state(env) == {}
    env.simulator.headless = False

    state = BaseEnv.get_state_dict(env)
    assert torch.equal(state["motion_manager"]["motion_ids"], torch.tensor([0, 1, 0]))
    BaseEnv.load_state_dict(env, {"motion_manager": {"loaded": True}})
    assert env.motion_manager.loaded_state == {"loaded": True}
    assert BaseEnv.get_task_id(env) == "unit_motion.npy"

    env.motion_manager = None
    assert BaseEnv.get_state_dict(env) == {}
    BaseEnv.load_state_dict(env, {})
    assert BaseEnv.get_task_id(env) == "null"

    env.motion_manager = _MotionManager()
    noisy = NoisyObservations(
        rigid_body_pos=torch.ones(3, 2, 3),
        rigid_body_rot=_identity_quat(3, 2),
        rigid_body_vel=torch.ones(3, 2, 3) * 2.0,
        rigid_body_ang_vel=torch.ones(3, 2, 3) * 3.0,
        dof_pos=torch.ones(3, 2),
        dof_vel=torch.ones(3, 2) * 4.0,
        root_rot=_identity_quat(3),
        root_local_ang_vel=torch.ones(3, 3),
        anchor_rot=_identity_quat(3),
        anchor_local_ang_vel=torch.ones(3, 3) * 5.0,
        ground_heights=torch.ones(3),
    )
    env._current_noisy_obs = noisy
    env.progress_buf = torch.tensor([1, 2, 3])
    env.reset_buf = torch.tensor([False, True, False])
    env.terminate_buf = torch.tensor([False, False, True])
    env.respawn_root_offset[:] = 3.0
    env.odom_start_xy[:] = torch.tensor([[1.0, 2.0]])
    env.odom_start_heading_inv[:] = torch.tensor([[0.0, 0.0, 0.5, 0.866]])
    snapshot = BaseEnv.save_state(env)
    assert "state_history" in snapshot
    assert "object_state" in snapshot
    assert snapshot["_current_noisy_obs"] is not noisy
    # The odometer start pose has to round-trip: it is the reference the
    # displacement-from-start reading is measured against, so losing it across a
    # save/restore would silently teleport the believed position.
    assert torch.equal(snapshot["odom_start_xy"], env.odom_start_xy)
    assert snapshot["odom_start_xy"] is not env.odom_start_xy
    assert torch.equal(snapshot["odom_start_heading_inv"], env.odom_start_heading_inv)

    env.progress_buf.zero_()
    env.odom_start_xy.zero_()
    env.odom_start_heading_inv.zero_()
    BaseEnv.restore_state(env, snapshot)
    assert torch.equal(env.progress_buf, torch.tensor([1, 2, 3]))
    assert torch.equal(env.odom_start_xy, torch.tensor([[1.0, 2.0]]).repeat(3, 1))
    assert torch.equal(
        env.odom_start_heading_inv,
        torch.tensor([[0.0, 0.0, 0.5, 0.866]]).repeat(3, 1),
    )
    assert torch.equal(env.state_history.loaded_state["history"], torch.tensor([1.0]))
    assert env.simulator.steps[-1][1] is None
    assert env._current_context is None

    env.simulator.config._target_ = "fake.Mujoco"
    step_count = len(env.simulator.steps)
    BaseEnv.restore_state(env, snapshot)
    assert len(env.simulator.steps) == step_count

    BaseEnv.close(env)
    assert env.simulator.closed is True


def test_base_env_constructor_initializes_lightweight_dependencies(monkeypatch):
    class _ConstructedMotionManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        base_env_module,
        "get_class",
        lambda target: _ConstructedMotionManager,
    )
    monkeypatch.setattr(BaseEnv, "_initialize_observations", lambda self: None)

    config = EnvConfig(
        max_episode_length=7,
        num_state_history_steps=2,
        ref_contact_smooth_window=3,
    )
    config.observation_components = {}
    config.reward_components = {}
    config.termination_components = {}
    robot_config = _robot_config()
    terrain = _Terrain(flat=True)
    simulator = _FakeSimulator(num_envs=2)
    scene_lib = _SceneLib(num_envs=2, scenes=0)
    motion_lib = _MotionLib(num_motions=2, num_envs=2)

    env = BaseEnv(
        config,
        robot_config,
        torch.device("cpu"),
        terrain,
        simulator,
        scene_lib,
        motion_lib,
    )

    assert env.num_envs == 2
    assert env.max_episode_length == 7
    assert env.skip_height_correction is True
    assert env.dt == simulator.dt
    assert torch.equal(env.reset_buf, torch.ones(2, dtype=torch.bool))
    assert env.prev_contact_force_magnitudes.shape == (2, 2)
    assert motion_lib.smooth_windows == [3]
    assert env.motion_manager.kwargs["num_envs"] == 2
    assert env.state_history.num_history_steps == 2
    assert simulator.initialized_markers == {}

    no_motion_config = EnvConfig(
        max_episode_length=5,
        num_state_history_steps=0,
        ref_contact_smooth_window=3,
    )
    no_motion_config.observation_components = {}
    no_motion_config.reward_components = {}
    no_motion_config.termination_components = {}
    no_motion_robot_config = _robot_config()
    no_motion_terrain = _Terrain(flat=False)
    no_motion_simulator = _FakeSimulator(num_envs=2)
    no_motion_scene_lib = _SceneLib(num_envs=2, scenes=0)
    no_motion_lib = _MotionLib(num_motions=0, num_envs=2)

    no_motion_env = BaseEnv(
        no_motion_config,
        no_motion_robot_config,
        torch.device("cpu"),
        no_motion_terrain,
        no_motion_simulator,
        no_motion_scene_lib,
        no_motion_lib,
    )

    assert no_motion_env.motion_manager is None
    assert no_motion_env.state_history is None
    assert no_motion_lib.smooth_windows == []
    assert no_motion_simulator.initialized_markers == {}


def test_install_motion_lib_uses_prebuilt_candidate_without_constructing(monkeypatch):
    env = object.__new__(BaseEnv)
    candidate = _MotionLib(num_motions=2, num_envs=2)
    env.motion_lib = _MotionLib(num_motions=1, num_envs=2)
    env.motion_manager = "old-manager"
    env.config = SimpleNamespace(ref_contact_smooth_window=3)
    env._current_context = {"stale": True}
    events = []

    env._validate_motion_lib_compatibility = lambda motion_lib: events.append(
        ("validate", motion_lib)
    )

    def create_motion_manager():
        events.append(("manager", env.motion_lib))
        env.motion_manager = "new-manager"

    env.create_motion_manager = create_motion_manager

    class _MustNotConstructMotionLib:
        def __init__(self, *args, **kwargs):
            raise AssertionError("install_motion_lib must not construct MotionLib")

    monkeypatch.setattr(
        base_env_module, "MotionLib", _MustNotConstructMotionLib, raising=False
    )

    assert BaseEnv.install_motion_lib(env, candidate) is None
    assert candidate.smooth_windows == [3]
    assert env.motion_lib is candidate
    assert env.motion_manager == "new-manager"
    assert env._current_context is None
    assert events == [("validate", candidate), ("manager", candidate)]


def test_install_motion_lib_keeps_current_state_when_candidate_is_incompatible():
    env = object.__new__(BaseEnv)
    old_motion_lib = _MotionLib(num_motions=1, num_envs=2)
    env.motion_lib = old_motion_lib
    env.motion_manager = "old-manager"
    env.config = SimpleNamespace(ref_contact_smooth_window=3)
    env._current_context = {"current": True}
    candidate = _MotionLib(num_motions=2, num_envs=2)
    env._validate_motion_lib_compatibility = lambda motion_lib: (_ for _ in ()).throw(
        ValueError("incompatible")
    )

    with pytest.raises(ValueError, match="incompatible"):
        BaseEnv.install_motion_lib(env, candidate)

    assert env.motion_lib is old_motion_lib
    assert env.motion_manager == "old-manager"
    assert env._current_context == {"current": True}
    assert candidate.smooth_windows == []


def test_install_motion_lib_clears_manager_for_empty_prebuilt_library():
    env = object.__new__(BaseEnv)
    env.motion_lib = _MotionLib(num_motions=1, num_envs=2)
    env.motion_manager = "old-manager"
    env.config = SimpleNamespace(ref_contact_smooth_window=3)
    env._current_context = {"stale": True}

    BaseEnv.install_motion_lib(env, _MotionLib(num_motions=0, num_envs=2))

    assert env.motion_manager is None
    assert env._current_context is None


def test_apply_motion_weights_to_scene_weights_loads_expected_checkpoint(tmp_path):
    assert BaseEnv.apply_motion_weights_to_scene_weights(None, "motion.npy", torch.device("cpu")) is None
    assert BaseEnv.apply_motion_weights_to_scene_weights(str(tmp_path), None, torch.device("cpu")) is None
    assert (
        BaseEnv.apply_motion_weights_to_scene_weights(
            str(tmp_path), "/tmp/missing.npy", torch.device("cpu")
        )
        is None
    )

    checkpoint = tmp_path / "env_clip.npy.ckpt"
    torch.save({"motion_manager": {"motion_weights": torch.tensor([0.25, 0.75])}}, checkpoint)
    weights = BaseEnv.apply_motion_weights_to_scene_weights(
        str(tmp_path), "/tmp/clip.npy", torch.device("cpu")
    )
    assert weights == [0.25, 0.75]

    no_manager = tmp_path / "env_no_manager.npy.ckpt"
    torch.save({"other": True}, no_manager)
    assert (
        BaseEnv.apply_motion_weights_to_scene_weights(
            str(tmp_path), "/tmp/no_manager.npy", torch.device("cpu")
        )
        is None
    )

    bad_checkpoint = tmp_path / "env_bad.npy.ckpt"
    bad_checkpoint.write_text("not a torch checkpoint")
    assert (
        BaseEnv.apply_motion_weights_to_scene_weights(
            str(tmp_path), "/tmp/bad.npy", torch.device("cpu")
        )
        is None
    )
