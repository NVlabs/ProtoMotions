# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for BaseEnv lifecycle helpers with lightweight fakes."""

from types import SimpleNamespace

import pytest
import torch

import protomotions.envs.base_env.env as env_module
import protomotions.envs.base_env.reset.motion as motion_module
import protomotions.envs.base_env.reset.perturbation as perturbation_module
import protomotions.envs.base_env.reset.recovery as recovery_module
from protomotions.envs.action.config import ActionConfig
from protomotions.envs.base_env.build import bind_managers, declare_owners
from protomotions.envs.base_env.config import EnvConfig, RecoveryResetConfig
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.motion_manager.motion_manager import MotionManager
from protomotions.envs.base_env.markers import Markers
from protomotions.envs.base_env.mdp_managers import ObservationManager
from protomotions.envs.base_env.publishers import HistorySampler
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.randomness import Randomness
from protomotions.envs.rewards.meta import RewardMeta
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.envs.base_env.owners import (
    ActionStage,
    ContactMemory,
    EpisodeBuffers,
)
from protomotions.envs.base_env.reset import (
    RecoveryBank,
    ReferenceMotion,
    ResetPerturbation,
    ResetPose,
    SpawnPlacer,
)
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
)
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
    RobotState,
    StateConversion,
)
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer
from protomotions.simulator.base_simulator.user_interface import UserInterface


def _scaled_action(action, gain, bias):
    return {"processed_action": action * gain + bias}


def _plus_one_action(action):
    return {"processed_action": action + 1.0}


def _grad_scaled_action(action, scale):
    return {"processed_action": action * scale}


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
    )


class _Terrain:
    def __init__(self, flat: bool = False):
        # A height field comes with its cell size (``terrain.horizontal_scale``).
        self.config = SimpleNamespace(horizontal_scale=0.1)
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
        # No pointcloud API on this double, so the publisher must not read one.
        self.has_pointclouds = False

    def num_scenes(self):
        return self._scenes

    def get_per_object_valid_mask(self, env_ids):
        return torch.ones(len(env_ids), self.num_objects_per_scene, dtype=torch.bool)

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
    save_backend_state = Simulator.save_backend_state
    restore_backend_state = Simulator.restore_backend_state

    def __init__(self, num_envs: int = 3, headless: bool = True):
        self.num_envs = num_envs
        self.num_act = 2
        self.dt = 0.05
        self.headless = headless
        self.user_interface = UserInterface()
        self.config = SimpleNamespace(
            _target_="fake.IsaacGym",
            domain_randomization=None,
        )
        from protomotions.simulator.base_simulator.contacts import ContactLayout

        self.contacts_enabled = True
        self.contact_layout = ContactLayout.build(["root", "foot"], 0, 0)
        self.has_complete_contact_labels = True
        self.state = _robot_state(num_envs)
        self.current_actions = torch.ones(num_envs, self.num_act) * 0.7
        self.reset_calls = []
        self.steps = []
        self.initialized_markers = None
        self.closed = False
        # No push randomization, like the real Simulator without a push config.
        self.push_enabled = False
        self.has_push_randomization = False

    def set_push_enabled(self, enabled):
        if enabled:
            raise ValueError("fake simulator has no push randomization")

    def is_simulation_running(self):
        return True

    def get_default_robot_reset_state(self):
        return _reset_state(self.num_envs)

    def _initialize_with_markers(self, markers):
        self.initialized_markers = markers

    def get_robot_state(self, env_ids=None):
        return self.state if env_ids is None else self.state[env_ids]

    def get_contact_forces(self, env_ids=None):
        forces = torch.zeros(self.num_envs, 2, 3, 3)
        forces[:, :, 0, 2] = self.state.rigid_body_contacts.float()
        return forces if env_ids is None else forces[env_ids]

    def get_root_state(self):
        return self.state

    def step(self, actions, markers_callback=None):
        self.steps.append((actions.clone(), markers_callback))
        if markers_callback is not None:
            self.last_markers = markers_callback()

    def reset_envs(self, new_states, new_object_states=None, env_ids=None):
        self.reset_calls.append((new_states, new_object_states, env_ids.clone()))
        # The real Simulator zeroes its live action tensor in place.
        self.current_actions[env_ids] = 0.0

    def get_current_actions(self):
        return self.current_actions

    def save_state(self):
        return {"current_actions": self.current_actions.clone()}

    def restore_state(self, state):
        self.current_actions.copy_(state["current_actions"])

    def get_object_root_state(self, env_ids=None):
        state = _object_state(self.num_envs, base=80.0)
        return state if env_ids is None else state[env_ids]

    def close(self):
        self.closed = True


class _FakeControl:
    """A control component publishing nothing under no namespace; it records
    the rows ``BaseEnv._publish`` asked it for on its manager."""

    namespaces = ()
    config = None

    def __init__(self, manager):
        self.manager = manager

    def publish(self, env_ids):
        self.manager.publish_calls.append(env_ids)
        return {}

    def close(self):
        pass


class _Teleports:
    def __init__(self):
        self.applied = []

    def apply(self, simulator):
        self.applied.append(simulator)


class _ControlManager:
    def __init__(self):
        self.components = {"fake": _FakeControl(self)}
        self.step_calls = 0
        self.advance_states = []
        self.reset_calls = []
        self.publish_calls = []
        self.teleports = _Teleports()

    def get_markers_state(self, state):
        return {"control_marker": object()}

    def create_visualization_markers(self, headless):
        del headless
        return {"control_marker": VisualizationMarkerConfig(markers=[MarkerConfig()])}

    def advance(self, state):
        self.step_calls += 1
        self.advance_states.append(state)
        return {}

    def reset_rows(self, env_ids, state):
        self.reset_calls.append(env_ids.clone())

    def terminations(self):
        return {}

    def save_state(self):
        return {"task": self.task.clone()} if hasattr(self, "task") else {}

    def load_state(self, state):
        if "task" in state:
            self.task.copy_(state["task"])


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
        # The termination manager unwraps its rules into a new dict per call.
        if components.keys() == self.env.config.termination_components.keys():
            return self.terminations
        return {}


def _seed_history(env):
    """Re-seed the real history from the fake simulator's current state."""
    env._state_history.reset(
        torch.arange(env.num_envs),
        env._history_sampler.sample(env.simulator.state, None),
    )


def _privileged_history(env):
    return env._state_history.capture().states["privileged"]


class _MotionManager:
    # The clip-assignment API is the real one over this fake's tensors: the
    # env snapshot saves and restores the assignment through it.
    get_assignment = MotionManager.get_assignment
    set_assignment = MotionManager.set_assignment

    def __init__(self):
        self.num_envs = 3
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
        contact_bodies="all",
        non_termination_contact_bodies=["foot"],
        anchor_body_index=0,
        number_of_actions=2,
        reset_noise=None,
    )


def _recovery_bank(env) -> RecoveryBank:
    """The recovery bank over the env's CURRENT config, as ``build_env``
    builds it. The bank captures the config (the cache is generated from it
    at build), so a test that needs another ``config.recovery_reset`` passes
    it to ``_make_env`` or rebuilds the bank with this; the perturbation
    reads ``robot_config.reset_noise`` live and needs no rebuild."""
    kinematic_info = env.robot_config.kinematic_info
    return RecoveryBank(
        getattr(env.config, "recovery_reset", None),
        env.simulator,
        env.terrain,
        kinematic_info.dof_limits_lower,
        kinematic_info.dof_limits_upper,
        env.robot_config.number_of_actions,
        env.num_envs,
        env.device,
        None,
    )


def _markers(env, *, headless: bool) -> Markers:
    """The markers collaborator over the fake env's pieces, as build_env builds it."""
    return Markers(
        env.control_manager,
        env.terrain,
        env.simulator.get_root_state,
        env.simulator.get_robot_state,
        show_terrain_markers=env.config.show_terrain_markers,
        headless=headless,
        num_envs=env.num_envs,
        device=env.device,
    )


def _make_env(
    *,
    num_envs: int = 3,
    scenes: int = 0,
    motions: int = 0,
    headless: bool = True,
    history: bool = False,
    simulator=None,
    action_config=None,
    recovery_reset=None,
):
    """A bare BaseEnv over fakes: the owners and the reset chain's helpers,
    then the collaborators, then the managers and the four owner lists bound
    the way ``build_env`` binds them. ``recovery_reset`` replaces the config's
    default before the recovery bank captures it."""
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
    env.config.action_config = action_config
    if recovery_reset is not None:
        env.config.recovery_reset = recovery_reset
    env.robot_config = _robot_config()
    env.terrain = _Terrain(flat=False)
    env.scene_lib = _SceneLib(num_envs=num_envs, scenes=scenes)
    env.simulator = (
        _FakeSimulator(num_envs=num_envs, headless=headless)
        if simulator is None
        else simulator
    )
    env._key_bindings = env.simulator.user_interface.scope("env")
    env._key_bindings.register("R", "reset", "Reset all environments")
    env.num_envs = num_envs
    env.dt = env.simulator.dt
    env._current_context = None
    env._advancing = False
    env._perturbations_enabled = True
    # The reference motion is filled directly (no install), so the fakes
    # stand in for the library and the manager.
    env._motion = ReferenceMotion(
        env.config.motion_manager,
        env.robot_config,
        env.scene_lib,
        num_envs,
        env.dt,
        env.device,
        env.config.ref_contact_smooth_window,
        None,
    )
    env._motion.motion_lib = _MotionLib(num_motions=motions, num_envs=num_envs)
    env._motion.motion_manager = _MotionManager() if motions > 0 else None
    env._episode = EpisodeBuffers(num_envs, env.device, env.config.max_episode_length)
    env._contact_memory = ContactMemory(num_envs, 2, env.device)
    env._action_stage = ActionStage(action_config, num_envs, 2, env.device)
    env._spawn = SpawnPlacer(
        env.terrain,
        env.scene_lib,
        env._motion,
        env.simulator,
        env.config.ref_respawn_offset,
        env.config.skip_correct_terrain_height_on_flat,
        num_envs,
        env.device,
    )
    env._recovery = _recovery_bank(env)
    env._pose = ResetPose(
        env.simulator,
        env.scene_lib,
        env._motion,
        env._spawn,
        env.config.ref_object_respawn_offset,
        env.device,
    )
    kinematic_info = env.robot_config.kinematic_info
    env._perturbation = ResetPerturbation(
        env.robot_config,
        kinematic_info.dof_limits_lower,
        kinematic_info.dof_limits_upper,
        None,
    )
    env.randomness = Randomness(0, env.device)
    # No sensor suite: the history holds privileged state only.
    env.sensors = None
    env._history_sampler = HistorySampler(
        env.terrain, env.robot_config, env.simulator, env._action_stage
    )
    # The fake binds nothing; the test names the ring's depth directly.
    env._state_history = StateHistoryBuffer(
        2 if history else 0,
        env._history_sampler.sample(env.simulator.state, None),
    )
    env.control_manager = _ControlManager()
    env._markers = _markers(env, headless=env.simulator.headless)
    env._component_manager = _ComponentExecutor(env)
    env.extras = {}
    # The managers hold the config's component dicts: a test that rebinds
    # ``env.config.<group>_components`` calls ``bind_managers`` again. The
    # owner lists come after them (the observation cache holds the
    # observation manager's dict).
    bind_managers(env)
    declare_owners(env)
    return env


def test_recovery_reset_config_defaults_to_disabled():
    config = RecoveryResetConfig()

    assert config.recovery_prob == pytest.approx(0.0)
    assert config.fall_sim_steps == 150


def test_base_env_recovery_reset_is_noop_for_legacy_config():
    env = _make_env()
    del env.config.recovery_reset
    env._recovery = _recovery_bank(env)
    new_states = _reset_state()
    original = new_states.clone()

    assert env._recovery.enabled is False
    returned_states, recovery_mask = env._recovery.swap_in(
        torch.arange(env.num_envs), new_states, env._spawn.respawn_root_offset
    )

    assert not recovery_mask.any()
    assert torch.equal(returned_states.root_pos, original.root_pos)
    assert torch.equal(returned_states.dof_pos, original.dof_pos)


def test_base_env_generates_grounded_fall_reset_states():
    env = _make_env(
        recovery_reset=RecoveryResetConfig(recovery_prob=1.0, fall_sim_steps=2)
    )
    default_state = env._pose.default_reset_state.clone()
    snapshot = {"saved": True}
    restored = []

    torch.manual_seed(7)
    env._recovery.generate(
        env._pose.default_reset_state,
        env._pose.default_object_state,
        save_state=lambda: snapshot,
        restore_state=restored.append,
    )

    assert restored == [snapshot]
    assert env.terrain.sample_calls == [(env.num_envs, True)]
    assert len(env.simulator.reset_calls) == 1
    initial_state, object_state, env_ids = env.simulator.reset_calls[0]
    expected_xy = torch.tensor([[10.0, 21.0], [12.0, 23.0], [14.0, 25.0]])
    assert torch.equal(env_ids, torch.arange(env.num_envs))
    assert object_state is env._pose.default_object_state
    assert torch.equal(initial_state.root_pos[:, :2], expected_xy)
    assert torch.allclose(
        initial_state.root_pos[:, 2], default_state.root_pos[:, 2] + 0.5
    )
    assert torch.allclose(initial_state.root_rot.norm(dim=-1), torch.ones(env.num_envs))
    assert torch.count_nonzero(initial_state.root_vel) == 0
    assert torch.count_nonzero(initial_state.root_ang_vel) == 0
    assert torch.count_nonzero(initial_state.dof_vel) == 0

    assert len(env.simulator.steps) == 2
    first_targets = env.simulator.steps[0][0]
    assert torch.equal(env.simulator.steps[1][0], first_targets)
    assert torch.all(first_targets >= env.robot_config.kinematic_info.dof_limits_lower)
    assert torch.all(first_targets <= env.robot_config.kinematic_info.dof_limits_upper)

    fall_states = env._recovery.fall_states
    assert fall_states is not None
    assert torch.count_nonzero(fall_states.root_vel) == 0
    assert torch.count_nonzero(fall_states.root_ang_vel) == 0
    assert torch.count_nonzero(fall_states.dof_vel) == 0


def test_base_env_samples_recovery_reset_states_by_probability(monkeypatch):
    env = _make_env(
        recovery_reset=RecoveryResetConfig(recovery_prob=0.5, fall_sim_steps=2)
    )
    env._spawn.respawn_root_offset[:, 2] = torch.tensor([1.0, 2.0, 3.0])
    fall_states = env._recovery.fall_states = _reset_state(base=100.0)
    fall_states.dof_pos = torch.tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])
    fall_states.root_rot = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    new_states = _reset_state()
    original = new_states.clone()

    monkeypatch.setattr(
        recovery_module.torch,
        "rand",
        lambda count, device=None, generator=None: torch.tensor(
            [0.2, 0.8, 0.1], device=device
        )[:count],
    )

    def fake_randint(low, high, size, device=None, generator=None):
        assert (low, high, size) == (0, 3, (2,))
        return torch.tensor([2, 0], device=device)

    monkeypatch.setattr(recovery_module.torch, "randint", fake_randint)

    returned_states, recovery_mask = env._recovery.swap_in(
        torch.arange(env.num_envs), new_states, env._spawn.respawn_root_offset
    )

    assert torch.equal(recovery_mask, torch.tensor([True, False, True]))
    assert torch.equal(returned_states.root_pos[:, :2], original.root_pos[:, :2])
    assert torch.equal(
        returned_states.root_pos[:, 2],
        torch.tensor([109.0, 5.0, 105.0]),
    )
    assert torch.equal(returned_states.root_rot[[0, 2]], fall_states.root_rot[[2, 0]])
    assert torch.equal(returned_states.dof_pos[[0, 2]], fall_states.dof_pos[[2, 0]])
    assert torch.equal(returned_states.root_pos[1], original.root_pos[1])
    assert torch.equal(returned_states.root_rot[1], original.root_rot[1])
    assert torch.equal(returned_states.dof_pos[1], original.dof_pos[1])
    assert torch.count_nonzero(returned_states.root_vel[[0, 2]]) == 0
    assert torch.count_nonzero(returned_states.root_ang_vel[[0, 2]]) == 0
    assert torch.count_nonzero(returned_states.dof_vel[[0, 2]]) == 0


def test_base_env_every_reset_kind_seeds_history_from_the_post_reset_state():
    env = _make_env(
        motions=2,
        history=True,
        recovery_reset=RecoveryResetConfig(recovery_prob=1.0),
    )
    env._recovery.swap_in = lambda env_ids, reset_states, respawn_root_offset: (
        reset_states,
        torch.tensor([False, True, False], device=env.device),
    )
    env._state_history.push(env._history_sampler.sample(_robot_state(base=100.0), None))

    BaseEnv.reset(env, env_ids=torch.arange(env.num_envs))

    # Reference, default and recovery resets alike repeat the measured frame;
    # no reference-clip prehistory is copied in, and no clip query happens.
    privileged = _privileged_history(env)
    for slot in range(env._state_history.num_history_steps + 1):
        assert torch.equal(
            privileged.rigid_body_pos[:, slot], env.simulator.state.rigid_body_pos
        )
    assert torch.equal(env._state_history.capture().raw_actions, torch.zeros(3, 3, 2))
    assert all(
        times.numel() == 0 or times.ndim == 1
        for _, times in env.reference_motion.motion_lib.queries
    )


def test_base_env_scene_surface_context_publishes_objects_without_pointclouds():
    """A scene library with objects but no computed pointclouds publishes the
    three object names and nothing for the pointclouds (the ``pointclouds``
    requirement is unmet, so the name is absent rather than zero-width)."""
    env = _make_env(num_envs=2, scenes=1)

    def _unexpected_pointcloud_call(*args, **kwargs):
        del args, kwargs
        raise AssertionError("object pointclouds should not be read when disabled")

    env.scene_lib.get_scene_neutral_pointcloud = _unexpected_pointcloud_call

    context = env._publish()

    objects = env.scene_lib.num_objects_per_scene
    assert context["scene.object_pos"].shape == (2, objects, 3)
    assert context["scene.object_rot"].shape == (2, objects, 4)
    assert context["scene.object_valid_mask"].shape == (2, objects)
    assert "scene.neutral_pointclouds" not in context


def test_base_env_publish_selects_subset_without_caching_it():
    env = _make_env(num_envs=3, scenes=0, history=True)
    env.control_manager.components["fake"].publish = lambda env_ids: {
        "mimic.anchor_idx": 0
    }
    env._action_stage.raw[:] = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    env._state_history.push(env._history_sampler.sample(env.simulator.state, None))
    env_ids = torch.tensor([2, 0])

    context = env._publish(env_ids)

    assert torch.equal(context.env_ids, env_ids)
    assert context["env_ids"] is context.env_ids
    assert torch.equal(
        context["current.root_pos"], env.simulator.state.root_pos[env_ids]
    )
    # Without a sensor suite there are no sensor sources, not privileged aliases.
    assert not any(key.startswith(("noisy.", "clean_historical.")) for key in context)
    assert torch.equal(context["current_action"], env._action_stage.raw[env_ids])
    assert torch.equal(context["previous_action"], torch.zeros(2, 2))
    assert torch.equal(context["historical.raw_actions"], torch.zeros(2, 2, 2))
    assert torch.equal(
        context["raw_action_history"][:, 0], env._action_stage.raw[env_ids]
    )
    assert context["terrain.height_points"].shape[0] == len(env_ids)
    assert not any(name.startswith("scene.") for name in context)  # no objects
    assert torch.equal(context["progress_buf"], env.progress_buf[env_ids])
    assert context["mimic.anchor_idx"] == 0
    assert env._current_context is None


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
    action = torch.ones(3, 2)
    env = _make_env(
        scenes=1,
        action_config=ActionConfig(
            fn=_scaled_action, params={"gain": torch.tensor(2.0), "bias": 1.0}
        ),
    )

    processed = env._action_stage.act(action)

    assert torch.equal(processed, torch.ones(3, 2) * 3.0)
    assert torch.equal(env._action_stage.raw, action)
    assert torch.equal(env._action_stage.processed, processed)
    assert env._action_stage.device_ready is True
    # Without an action config the stage hands the caller's tensor through.
    assert ActionStage(None, 3, 2, env.device).act(action) is action
    assert BaseEnv.is_simulation_running(env) is True

    # The managers are bound to the config's component dicts (one shared
    # component manager); the fakes' kernels return canned tensors. The build
    # resolves body selections first, so an entry is a component, not a
    # placeholder.
    env.config.observation_components = {"dynamic": MdpComponent(lambda: None, {})}
    env.config.reward_components = {
        "alive": MdpComponent(lambda: None, {}, meta=RewardMeta(weight=2.0))
    }
    env.config.termination_components = {
        "fell": TerminationRule(MdpComponent(lambda: None, {}))
    }
    bind_managers(env)
    assert env._rewards.component_manager is env._component_manager
    assert env._terminations.component_manager is env._component_manager

    env._component_manager.observations = {
        "dynamic": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    }
    rows = torch.tensor([0, 2])
    env._observations.compute(SimpleNamespace(env_ids=None), rows)
    buffer = env._observations.buffer
    assert torch.equal(buffer["dynamic"][0], torch.tensor([1.0, 2.0]))
    assert torch.equal(buffer["dynamic"][1], torch.zeros(2))
    assert torch.equal(buffer["dynamic"][2], torch.tensor([5.0, 6.0]))

    obs = BaseEnv.get_obs(env)
    assert set(obs) == {"dynamic"}
    obs["dynamic"][0, 0] = -99.0
    assert buffer["dynamic"][0, 0] == 1.0

    env._component_manager.rewards = {"alive": torch.tensor([1.0, 2.0, 3.0])}
    reward, reward_logging = env._rewards.compute(object(), env.progress_buf)
    assert torch.equal(reward, torch.tensor([2.0, 4.0, 6.0]))
    assert reward_logging["total_env_reward"] is reward
    assert env._rewards.grace_mask(env.progress_buf) is None

    env.config.reset_grace_period = 2
    bind_managers(env)
    assert torch.equal(
        env._rewards.grace_mask(torch.tensor([1, 2, 3])),
        torch.tensor([True, True, False]),
    )

    env.progress_buf[:] = torch.tensor([0, 3, 1])
    env._component_manager.terminations = {"fell": torch.tensor([False, True, False])}
    # The step's score phase hands over the pending reset requests.
    reset, terminated, term_logging = env._terminations.compute(
        object(),
        env.progress_buf,
        env.max_episode_length,
        torch.tensor([True, False, False]),
    )
    # Timeout (env 1 at max_episode_length 3) | reset request (env 0) | kernel
    # (env 1); only the kernel terminates.
    assert torch.equal(reset, torch.tensor([True, True, False]))
    assert torch.equal(terminated, torch.tensor([False, True, False]))
    assert torch.equal(term_logging["termination/fell"], torch.tensor([0.0, 1.0, 0.0]))

    BaseEnv.on_epoch_end(env, current_epoch=4)


def test_base_env_observe_fills_every_env_from_a_subset_context():
    env = _make_env(scenes=1)
    env.config.observation_components = {"dynamic": MdpComponent(lambda: None, {})}
    bind_managers(env)
    env._component_manager.observations = {
        "dynamic": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    }
    env._current_context = object()
    all_env_ids = torch.arange(3)

    BaseEnv.observe(env, all_env_ids)

    assert torch.equal(
        env._observations.buffer["dynamic"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
    )
    # The explicit row tensor reaches the publishers even for every env (the
    # compiled path is reserved for the step's all-env context), and the cached
    # all-env context is dropped, not replaced.
    assert torch.equal(env.control_manager.publish_calls[-1], all_env_ids)
    assert env._current_context is None


def test_base_env_cached_body_ids_and_clean_context_without_history():
    env = _make_env(history=False)
    # The static publisher took the id tensors when the owners were bound, so
    # a changed robot config needs the cached ids dropped and a re-bind.
    env.robot_config.non_termination_contact_bodies = "all"
    del env.non_termination_contact_body_ids
    declare_owners(env)

    ctx = BaseEnv.context.fget(env)

    assert torch.equal(env.contact_body_ids, torch.tensor([0, 1]))
    assert torch.equal(ctx["non_termination_contact_body_ids"], torch.tensor([0, 1]))
    # With zero history there is one retained slot: the current state and the
    # command that produced it, but no earlier command.
    assert not any(key.startswith("historical.") for key in ctx)
    assert "previous_processed_action" not in ctx
    assert "previous_action" not in ctx
    assert torch.equal(ctx["current_action"], env._action_stage.raw)
    assert torch.equal(ctx["current_processed_action"], env._action_stage.processed)
    assert BaseEnv.context.fget(env) is ctx
    assert env.control_manager.publish_calls[-1] is None


def test_base_env_spawn_offsets_cover_scene_and_non_scene_paths():
    env = _make_env()
    env_ids = torch.tensor([0, 2])
    ref_state = _robot_state()[env_ids]

    env._spawn.sample_spawn(env_ids, ref_state=ref_state, sample_flat=True)

    expected_xy = torch.tensor([[10.0, 21.0], [12.0, 23.0]]) - ref_state.root_pos[:, :2]
    assert env.terrain.sample_calls[-1] == (2, True)
    assert torch.equal(env._spawn.respawn_root_offset[env_ids, :2], expected_xy)
    assert torch.allclose(
        env._spawn.respawn_root_offset[env_ids, 2], torch.tensor([0.3, 1.3])
    )

    offset = env._spawn.ref_pose_offset(torch.zeros(3, 2, 3))
    assert torch.equal(
        offset[:, :, :2],
        env._spawn.respawn_root_offset[:, None, :2].expand_as(offset[:, :, :2]),
    )
    assert torch.allclose(offset[:, :, 2], torch.tensor([[0.25], [1.25], [2.25]]))

    env._spawn.skip_height_correction = True
    no_z_offset = env._spawn.ref_pose_offset(
        torch.zeros(3, 2, 3), env_ids=torch.tensor([0, 1, 2])
    )
    assert torch.equal(no_z_offset[:, :, 2], torch.zeros(3, 2))

    scene_env = _make_env(scenes=3)
    scene_env._spawn.sample_spawn(env_ids, ref_state=ref_state)
    assert torch.equal(
        scene_env._spawn.respawn_root_offset[env_ids, :2],
        torch.tensor([[100.0, 101.0], [106.0, 107.0]]),
    )
    assert torch.allclose(
        scene_env._spawn.respawn_root_offset[env_ids, 2], torch.tensor([0.3, 1.3])
    )
    scene_mask, non_scene_mask = scene_env._spawn.scene_non_scene_mask(env_ids)
    assert torch.equal(scene_mask, torch.tensor([True, True]))
    assert torch.equal(non_scene_mask, torch.tensor([False, False]))


def test_base_env_reset_state_construction_and_ref_selection():
    env = _make_env(scenes=2, motions=2)
    env_ids = torch.tensor([0, 1])

    default_state, default_objects = env._pose.default(env_ids)
    assert torch.equal(
        default_state.root_pos,
        env.simulator.get_default_robot_reset_state()[env_ids].root_pos
        + env._spawn.respawn_root_offset[env_ids],
    )
    assert torch.equal(
        default_objects.root_pos,
        env.scene_lib.get_default_object_state(env.device)[env_ids].root_pos
        + env._spawn.respawn_root_offset[env_ids].unsqueeze(1),
    )

    motion_ids = torch.tensor([1, 0])
    motion_times = torch.tensor([0.2, 0.3])
    ref_state, ref_objects = env._pose.reference(env_ids, motion_ids, motion_times)
    assert torch.equal(
        ref_state.dof_pos, torch.stack([motion_ids.float(), motion_times], dim=-1)
    )
    assert torch.equal(ref_objects.root_vel, torch.zeros_like(ref_objects.root_pos))
    assert torch.equal(ref_objects.root_ang_vel, torch.zeros_like(ref_objects.root_pos))

    empty_env = _make_env(motions=0)
    ref_env_ids, ids, times = empty_env._motion.ref_reset_envs(
        env_ids, force_default_mask=None
    )
    assert ref_env_ids.numel() == 0
    assert ids is None
    assert times is None

    force_default_mask = torch.tensor([True, False])
    ref_env_ids, ids, times = env._motion.ref_reset_envs(
        env_ids, force_default_mask=force_default_mask
    )
    assert torch.equal(ref_env_ids, torch.tensor([1]))
    assert torch.equal(
        env.reference_motion.motion_manager.sample_calls[-1], torch.tensor([1])
    )
    assert torch.equal(ids, env.reference_motion.motion_manager.motion_ids[ref_env_ids])
    assert torch.equal(
        times, env.reference_motion.motion_manager.motion_times[ref_env_ids]
    )

    sample_count = len(env.reference_motion.motion_manager.sample_calls)
    ref_env_ids, ids, times = env._motion.ref_reset_envs(
        env_ids, force_default_mask=None, disable_motion_resample=True
    )
    assert torch.equal(ref_env_ids, env_ids)
    assert len(env.reference_motion.motion_manager.sample_calls) == sample_count
    assert torch.equal(ids, env.reference_motion.motion_manager.motion_ids[env_ids])
    assert torch.equal(times, env.reference_motion.motion_manager.motion_times[env_ids])

    with pytest.raises(AssertionError, match="force_default_mask length"):
        env._motion.ref_reset_envs(env_ids, force_default_mask=torch.tensor([True]))


def test_base_env_motion_validation_rejects_body_count_mismatch():
    env = _make_env(motions=2)
    candidate = _MotionLib(num_motions=2, num_bodies=3, num_dofs=2)

    with pytest.raises(ValueError, match="3 bodies"):
        env._motion.validate_compatibility(candidate)


def test_base_env_reset_flow_mixes_default_and_reference_resets():
    env = _make_env(scenes=2, motions=2, history=True)
    env._state_history.push(env._history_sampler.sample(_robot_state(base=100.0), None))

    obs, info = BaseEnv.reset(
        env,
        env_ids=[0, 1, 2],
        force_default_mask=torch.tensor([True, False, True]),
    )

    assert info == {}
    assert set(obs) == set()  # the fake binds no observation
    reset_state, object_state, reset_ids = env.simulator.reset_calls[-1]
    assert torch.equal(reset_ids, torch.tensor([0, 1, 2]))
    assert reset_state.root_pos.shape == (3, 3)
    assert object_state.root_pos.shape == (3, 1, 3)
    assert torch.equal(env.control_manager.reset_calls[-1], torch.tensor([0, 1, 2]))
    assert torch.equal(env.progress_buf, torch.zeros(3, dtype=torch.long))
    assert torch.equal(env.reset_buf, torch.zeros(3, dtype=torch.bool))
    assert torch.equal(env.terminate_buf, torch.zeros(3, dtype=torch.bool))
    assert torch.equal(env._action_stage.raw, torch.zeros(3, 2))
    assert torch.equal(env._action_stage.processed, torch.zeros(3, 2))
    # Every slot holds the post-reset state regardless of reset kind.
    privileged = _privileged_history(env)
    for slot in range(3):
        assert torch.equal(privileged.dof_pos[:, slot], env.simulator.state.dof_pos)

    empty_obs, empty_info = BaseEnv.reset(env, env_ids=[])
    assert empty_info == {}
    assert set(empty_obs) == set()


def test_base_env_reference_only_reset_skips_default_state_construction(monkeypatch):
    env = _make_env(scenes=2, motions=2, history=False)

    def fail_default(*args, **kwargs):
        raise AssertionError("reference-only reset should not build default states")

    monkeypatch.setattr(env._pose, "default", fail_default)

    BaseEnv.reset(env, env_ids=[0, 2])

    reset_state, object_state, reset_ids = env.simulator.reset_calls[-1]
    assert torch.equal(reset_ids, torch.tensor([0, 2]))
    assert reset_state.root_pos.shape[0] == 2
    assert object_state.root_pos.shape[0] == 2


def test_reset_pose_reference_does_not_mutate_motion_state():
    env = _make_env(scenes=1, motions=1, history=False)
    source_state = _robot_state(num_envs=1, base=3.0)
    original_root_pos = source_state.root_pos.clone()
    env.reference_motion.motion_lib.get_motion_state = (
        lambda motion_ids, motion_times: source_state
    )

    env._pose.reference(torch.tensor([0]), torch.tensor([0]), torch.tensor([0.1]))

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
    # No rebuild: the perturbation reads robot_config.reset_noise on apply.
    noise_calls = []

    def fake_apply_reset_noise(**kwargs):
        noise_calls.append(kwargs)

    monkeypatch.setattr(
        perturbation_module, "apply_reset_noise", fake_apply_reset_noise
    )

    obs, info = BaseEnv.reset(env, sample_flat=True)
    reset_state, object_state, reset_ids = env.simulator.reset_calls[-1]

    assert info == {}
    assert set(obs) == set()  # the fake binds no observation
    assert torch.equal(reset_ids, torch.tensor([0, 1, 2]))
    assert reset_state.root_pos.shape == (3, 3)
    assert object_state.root_pos.shape[0] == 3
    assert len(noise_calls) == 1
    assert noise_calls[0]["reset_state"] is reset_state
    assert env.terrain.sample_calls[-1] == (3, True)
    assert env.reference_motion.motion_manager is None


def test_base_env_partial_reset_repeats_current_state_and_preserves_other_rows():
    env = _make_env(motions=2, history=True)
    before = _privileged_history(env).dof_pos.clone()
    env.simulator.state = _robot_state(base=100.0)

    BaseEnv.reset(env, env_ids=torch.tensor([0, 2]), force_default_mask=None)

    after = _privileged_history(env).dof_pos
    for slot in range(3):
        assert torch.equal(after[[0, 2], slot], env.simulator.state.dof_pos[[0, 2]])
    assert torch.equal(after[1], before[1])


def test_base_env_ground_height_uses_configured_anchor_body():
    env = _make_env(history=False)
    env.robot_config.anchor_body_index = 1
    env.simulator.state.rigid_body_pos[:, 0, 0] = 10.0
    env.simulator.state.rigid_body_pos[:, 1, 0] = 20.0
    env.terrain.get_ground_heights = lambda positions: positions[:, :1] + 0.5
    _seed_history(env)

    context = BaseEnv._publish(env)

    assert torch.equal(context["ground_heights"], torch.full((3,), 20.5))
    assert context["current.ground_heights"] is context["ground_heights"]


def test_base_env_history_ground_height_uses_configured_anchor_body():
    env = _make_env(history=True)
    env.robot_config.anchor_body_index = 1
    env.simulator.state.rigid_body_pos[:, 0, 0] = 10.0
    env.simulator.state.rigid_body_pos[:, 1, 0] = 20.0
    env.terrain.get_ground_heights = lambda positions: positions[:, :1] + 0.5

    _seed_history(env)

    assert torch.equal(
        _privileged_history(env).ground_heights, torch.full((3, 3), 20.5)
    )


def test_base_env_step_advances_everything_and_builds_context_and_raw_extras():
    env = _make_env(motions=2, history=True)
    env.robot_config.anchor_body_index = 1
    env.simulator.state.rigid_body_pos[:, 0, 0] = 10.0
    env.simulator.state.rigid_body_pos[:, 1, 0] = 20.0
    env.terrain.get_ground_heights = lambda positions: positions[:, :1] + 0.5
    # The advance list is static; the spawn placer re-aligns when the
    # manager it reads asks for it.
    manager_config = env.reference_motion.motion_manager.config
    manager_config.realign_motion_with_humanoid_on_each_step = True
    # Two termination rules: a truncating one (reset only) and a failure.
    env.config.termination_components = {
        "clip": TerminationRule(MdpComponent(lambda: None, {}), truncation=True),
        "fell": TerminationRule(MdpComponent(lambda: None, {})),
    }
    bind_managers(env)
    env._component_manager.terminations = {
        "clip": torch.tensor([False, True, False]),
        "fell": torch.tensor([False, False, True]),
    }

    obs, rewards, resets, terminated, extras = BaseEnv.step(env, torch.zeros(3, 2))

    assert [owner.name for owner in env._advance_owners] == [
        "motion",
        "controls",
        "spawn",
    ]
    assert torch.equal(env.progress_buf, torch.ones(3, dtype=torch.long))
    privileged = _privileged_history(env)
    assert torch.equal(privileged.ground_heights[:, 0], torch.full((3,), 20.5))
    assert torch.equal(privileged.ground_heights[:, 1], torch.full((3,), 0.5))
    assert env.reference_motion.motion_manager.post_physics_called is True
    assert env.control_manager.step_calls == 1
    # The re-alignment queried the clip once and re-anchored every row's XY.
    assert len(env.reference_motion.motion_lib.queries) == 1
    ref_root_xy = torch.stack(
        [
            env.reference_motion.motion_manager.motion_ids.float(),
            env.reference_motion.motion_manager.motion_times,
        ],
        dim=-1,
    )
    assert torch.equal(
        env._spawn.respawn_root_offset[:, :2],
        env.simulator.state.root_pos[:, :2] - ref_root_xy,
    )
    assert torch.equal(env.reset_buf, torch.tensor([False, True, True]))
    assert torch.equal(env.terminate_buf, torch.tensor([False, False, True]))
    assert torch.equal(extras["termination/clip"], torch.tensor([0.0, 1.0, 0.0]))
    assert rewards is env.rew_buf and resets is env.reset_buf
    assert terminated is env.terminate_buf
    assert extras["terminate"] is env.terminate_buf
    assert torch.equal(extras["total_env_reward"], torch.zeros(3))
    assert "raw/rigid_body_pos" in extras
    assert set(obs) == set()  # the fake binds no observation
    assert torch.equal(
        env._contact_memory.prev_contact_force_magnitudes,
        env.simulator.get_contact_forces().norm(dim=-1).amax(dim=-1),
    )
    assert env.control_manager.publish_calls[-1] is None  # the all-env context
    assert env._current_context["previous_action"].shape == (3, 2)


def test_base_env_context_is_not_readable_while_the_control_components_advance():
    """Reading ``env.context`` during the advance phase would show the
    reference before the per-step re-alignment, so the property raises there
    and works again once the phase is over, also when an advance owner fails."""
    env = _make_env(motions=2, history=True)
    seen = {}

    def advance_reading_context(state):
        env.control_manager.step_calls += 1
        with pytest.raises(
            RuntimeError, match="not valid while the control components advance"
        ):
            env.context
        seen["advancing"] = env._advancing
        return {}

    env.control_manager.advance = advance_reading_context
    BaseEnv.step(env, torch.zeros(3, 2))
    assert seen == {"advancing": True}
    assert env._advancing is False
    assert env.context is env._current_context  # the publish phase built it

    def failing_advance(state):
        raise ValueError("boom")

    env.control_manager.advance = failing_advance
    with pytest.raises(ValueError, match="boom"):
        BaseEnv.step(env, torch.zeros(3, 2))
    assert env._advancing is False
    assert env.context is not None


def test_base_env_step_pushes_one_sample_with_its_command():
    env = _make_env(history=True)
    env.simulator.state = _robot_state(base=100.0)

    BaseEnv.step(env, torch.full((3, 2), 7.0))

    captured = env._state_history.capture()
    assert set(captured.states) == {"privileged"}
    assert torch.equal(
        captured.states["privileged"].dof_pos[:, 0], env.simulator.state.dof_pos
    )
    assert torch.equal(
        captured.states["privileged"].dof_pos[:, 1], _robot_state().dof_pos
    )
    assert torch.equal(captured.raw_actions[:, 0], torch.full((3, 2), 7.0))
    assert torch.equal(captured.raw_actions[:, 1], torch.zeros(3, 2))
    assert torch.equal(env._current_context["current_action"], torch.full((3, 2), 7.0))
    assert torch.equal(env._current_context["previous_action"], torch.zeros(3, 2))


def test_base_env_step_processes_action_and_honors_user_reset():
    env = _make_env(action_config=ActionConfig(fn=_plus_one_action, params={}))
    env.simulator.user_interface.handle_key_event("R", pressed=True)
    env._current_context = object()

    obs, rewards, resets, terminated, extras = BaseEnv.step(env, torch.ones(3, 2))

    assert set(obs) == set()  # the fake binds no observation
    assert rewards is env.rew_buf
    assert resets is env.reset_buf
    assert terminated is env.terminate_buf
    assert extras["terminate"] is env.terminate_buf
    assert torch.equal(env._action_stage.raw, torch.ones(3, 2))
    assert torch.equal(env._action_stage.processed, torch.ones(3, 2) * 2.0)
    assert torch.equal(env.simulator.steps[-1][0], torch.ones(3, 2) * 2.0)
    assert env.simulator.steps[-1][1] == env._markers.state
    # The key is consumed after the checks: this step's flags are untouched,
    # the request mask is set for every env (the clock is not touched) and the
    # request fires at the next step's check, where it is taken.
    assert not resets.any()
    assert env._episode.reset_request.all()
    assert torch.equal(env.progress_buf, torch.ones(3, dtype=torch.long))
    assert BaseEnv.consume_reset_request(env) is False
    env._episode.max_episode_length = 100  # so only the request can reset
    _, _, resets, terminated, _ = BaseEnv.step(env, torch.ones(3, 2))
    assert resets.all() and not terminated.any()
    assert not env._episode.reset_request.any()


def test_restore_state_puts_back_everything_the_first_post_eval_step_reads():
    # After evaluation the trainer continues the interrupted episodes and only
    # recomputes observations for envs that were already done, so the first
    # training step reads these caches, last step's contact forces, the
    # simulator's action history and the control state as they are left by
    # the write-back.
    env = _make_env(scenes=1)
    # The observation cache owner holds this dict object: fill it, don't rebind.
    buffer = env._observations.buffer
    buffer["max_coords_obs"] = torch.arange(6.0).view(3, 2)
    env._contact_memory.prev_contact_force_magnitudes[:] = torch.tensor([[0.0, 410.0]])
    env._action_stage.raw[:] = 0.4
    env._action_stage.processed[:] = 0.2
    env.control_manager.task = torch.tensor([1.0, 2.0, 3.0])
    saved_actions = env.simulator.current_actions.clone()
    before = BaseEnv.get_obs(env)

    snapshot = BaseEnv.save_state(env)

    # Evaluation resets every env (zeroing the simulator's live action tensor
    # in place), adds or overwrites every cache, and samples a new task.
    env.simulator.reset_envs(None, None, torch.arange(env.num_envs))
    buffer["max_coords_obs"].fill_(-1.0)
    buffer["eval_only"] = torch.ones(3, 1)
    env._contact_memory.prev_contact_force_magnitudes.zero_()
    env._action_stage.raw.zero_()
    env._action_stage.processed.zero_()
    env.control_manager.task.zero_()

    BaseEnv.restore_state(env, snapshot)

    after = BaseEnv.get_obs(env)
    assert after.keys() == before.keys()
    for name, value in before.items():
        assert torch.equal(after[name], value), name
    assert torch.equal(
        env._contact_memory.prev_contact_force_magnitudes[0], torch.tensor([0.0, 410.0])
    )
    assert torch.equal(env._action_stage.raw, torch.full((3, 2), 0.4))
    assert torch.equal(env._action_stage.processed, torch.full((3, 2), 0.2))
    assert torch.equal(env.control_manager.task, torch.tensor([1.0, 2.0, 3.0]))
    # Reset kinematics are refreshed by the backend: restoring must not advance
    # the saved pose. Its action state must still survive evaluation's resets.
    assert env.simulator.steps == []
    assert torch.equal(env.simulator.current_actions, saved_actions)
    assert env._current_context is None

    # The restored buffers are copies: changing them must not reach the snapshot.
    buffer["max_coords_obs"].fill_(5.0)
    assert torch.equal(
        snapshot["obs_cache"]["max_coords_obs"], torch.arange(6.0).view(3, 2)
    )


def test_forwarders_alias_the_owner_tensors_and_cannot_be_rebound():
    env = _make_env()
    owned = {
        "progress_buf": env._episode.progress_buf,
        "reset_buf": env._episode.reset_buf,
        "terminate_buf": env._episode.terminate_buf,
        "rew_buf": env._episode.rew_buf,
    }
    # Every getter hands out the owner's tensor itself, not a copy or a view.
    for name, tensor in owned.items():
        assert getattr(env, name) is tensor, name

    # The buffers are written in place; every alias taken earlier keeps
    # seeing the one buffer. Rebinding the name is refused (no setter): the
    # owner's tensor IS the buffer, and the env's own code reads the owner.
    alias = env.progress_buf
    env.progress_buf[:] = torch.tensor([4, 5, 6])
    assert alias is env.progress_buf
    assert torch.equal(alias, torch.tensor([4, 5, 6]))
    for name, tensor in owned.items():
        with pytest.raises(AttributeError):
            setattr(env, name, torch.zeros_like(tensor))
        assert getattr(env, name) is tensor, name
    assert torch.equal(env.progress_buf, torch.tensor([4, 5, 6]))

    # The scalar forwarder keeps its setter: the curriculum assigns it.
    env.max_episode_length = 9
    assert env._episode.max_episode_length == 9


def test_base_env_motion_manager_markers_state_save_restore_and_close(monkeypatch):
    env = _make_env(scenes=2, motions=2, headless=False, history=True)
    env.config.show_terrain_markers = True

    class _ConstructedMotionManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        motion_module,
        "get_class",
        lambda target: _ConstructedMotionManager,
    )
    env.scene_lib.humanoid_motion_ids = [1, 0, 1]
    env.reference_motion.motion_manager = env._motion.create_manager()
    assert torch.equal(
        env.reference_motion.motion_manager.kwargs["fixed_motion_ids_per_env"],
        torch.tensor([1, 0, 1]),
    )
    assert env.reference_motion.motion_manager.kwargs["num_envs"] == 3

    env.reference_motion.motion_manager = _MotionManager()
    # The markers collaborator merges the terrain height-map markers with the
    # control components'; headless, both calls are empty.
    markers = _markers(env, headless=False)
    assert "terrain_markers" in markers.configs()
    assert "control_marker" in markers.configs()
    assert _markers(env, headless=True).configs() == {}

    marker_state = markers.state()
    assert "terrain_markers" in marker_state
    assert marker_state["terrain_markers"].translation.shape == (3, 2, 3)
    assert "control_marker" in marker_state
    assert _markers(env, headless=True).state() == {}

    state = BaseEnv.get_state_dict(env)
    assert torch.equal(state["motion_manager"]["motion_ids"], torch.tensor([0, 1, 0]))
    BaseEnv.load_state_dict(env, {"motion_manager": {"loaded": True}})
    assert env.reference_motion.motion_manager.loaded_state == {"loaded": True}
    assert BaseEnv.get_task_id(env) == "unit_motion.npy"

    env.reference_motion.motion_manager = None
    assert BaseEnv.get_state_dict(env) == {}
    BaseEnv.load_state_dict(env, {})
    assert BaseEnv.get_task_id(env) == "null"

    env.reference_motion.motion_manager = _MotionManager()
    env.progress_buf[:] = torch.tensor([1, 2, 3])
    env.reset_buf[:] = torch.tensor([False, True, False])
    env.terminate_buf[:] = torch.tensor([False, False, True])
    env._spawn.respawn_root_offset[:] = 3.0
    env._action_stage.raw[:] = 7.0
    env._state_history.push(env._history_sampler.sample(_robot_state(base=100.0), None))
    saved_history = env._state_history.capture()
    snapshot = BaseEnv.save_state(env)
    # One entry per snapshot owner, in restore order; the sensors owner is
    # absent without a sensor suite, and objects ride with the physical state.
    assert list(snapshot) == [owner.name for owner in env._snapshot_owners]
    assert "history" in snapshot
    assert "object_state" in snapshot["sim_physical"]
    assert "sensors" not in snapshot

    env.progress_buf.zero_()
    env._action_stage.raw.zero_()
    env._state_history.push(env._history_sampler.sample(_robot_state(base=200.0), None))
    step_count = len(env.simulator.steps)
    BaseEnv.restore_state(env, snapshot)

    assert torch.equal(env.progress_buf, torch.tensor([1, 2, 3]))
    assert torch.equal(env._action_stage.raw, torch.full((3, 2), 7.0))
    restored = env._state_history.capture()
    assert torch.equal(
        restored.states["privileged"].dof_pos,
        saved_history.states["privileged"].dof_pos,
    )
    assert torch.equal(restored.raw_actions, saved_history.raw_actions)
    # Restore writes the backend without advancing physics on any backend.
    assert len(env.simulator.steps) == step_count
    assert env.simulator.reset_calls[-1][2].tolist() == [0, 1, 2]
    # Main restores cached observations directly; context is rebuilt on demand.
    assert env._current_context is None

    BaseEnv.close(env)
    assert env.simulator.closed is True


class _ContactCachingSimulator(_FakeSimulator):
    """Fake backend that remembers contacts and records the call order."""

    def __init__(self, events, **kwargs):
        super().__init__(**kwargs)
        self.events = events

    def flush_contact_cache(self):
        self.events.append("flush_contact_cache")

    def reset_envs(self, new_states, new_object_states=None, env_ids=None):
        self.events.append("reset_envs")
        super().reset_envs(new_states, new_object_states, env_ids)


def test_restore_state_flushes_contacts_before_writing_state_back():
    # PhysX keeps each robot's contacts from the last substep, and a state write
    # does not clear them. If the flush ran after the write-back (or not at all),
    # robots restored into a grounded pose could be launched on the next substep.
    events = []
    env = _make_env(scenes=1, simulator=_ContactCachingSimulator(events, num_envs=3))
    env.simulator.config._target_ = "fake.IsaacLab"
    snapshot = BaseEnv.save_state(env)

    BaseEnv.restore_state(env, snapshot)

    assert events == ["flush_contact_cache", "reset_envs"]
    new_states, new_object_states, env_ids = env.simulator.reset_calls[-1]
    assert isinstance(new_states, ResetState)
    physical = snapshot["sim_physical"]
    for name in (
        "root_pos",
        "root_rot",
        "root_vel",
        "root_ang_vel",
        "dof_pos",
        "dof_vel",
    ):
        torch.testing.assert_close(
            getattr(new_states, name), getattr(physical["robot_state"], name)
        )
    assert new_object_states is physical["object_state"]
    assert torch.equal(env_ids, torch.arange(env.num_envs))


def test_restore_state_works_with_a_simulator_that_cannot_flush():
    env = _make_env()
    assert not hasattr(env.simulator, "flush_contact_cache")
    env.simulator.config._target_ = "fake.IsaacLab"
    env.progress_buf[:] = torch.tensor([4, 5, 6])
    snapshot = BaseEnv.save_state(env)
    env.progress_buf.zero_()

    BaseEnv.restore_state(env, snapshot)

    assert len(env.simulator.reset_calls) == 1
    restored = env.simulator.reset_calls[0][0]
    assert isinstance(restored, ResetState)
    for name in (
        "root_pos",
        "root_rot",
        "root_vel",
        "root_ang_vel",
        "dof_pos",
        "dof_vel",
    ):
        torch.testing.assert_close(
            getattr(restored, name),
            getattr(snapshot["sim_physical"]["robot_state"], name),
        )
    assert torch.equal(env.progress_buf, torch.tensor([4, 5, 6]))


def test_restore_state_restores_solver_history_after_reset():
    class SolverHistorySimulator(_ContactCachingSimulator):
        def __init__(self, events, **kwargs):
            super().__init__(events, **kwargs)
            self.warmstart = torch.arange(6.0).reshape(3, 2)

        def save_backend_state(self):
            return {"warmstart": self.warmstart.clone()}

        def reset_envs(self, *args, **kwargs):
            super().reset_envs(*args, **kwargs)
            self.warmstart.zero_()

        def restore_backend_state(self, state):
            self.events.append("restore_backend_state")
            self.warmstart.copy_(state["warmstart"])

    events = []
    env = _make_env(simulator=SolverHistorySimulator(events, num_envs=3))
    env.simulator.config._target_ = "fake.Newton"
    expected = env.simulator.warmstart.clone()
    snapshot = BaseEnv.save_state(env)
    env.simulator.warmstart.fill_(-99.0)

    BaseEnv.restore_state(env, snapshot)

    assert events == ["flush_contact_cache", "reset_envs", "restore_backend_state"]
    assert torch.equal(env.simulator.warmstart, expected)
    env.simulator.warmstart.zero_()
    assert torch.equal(
        snapshot["sim_physical"]["simulator_backend_state"]["warmstart"], expected
    )


def test_default_backend_snapshot_is_empty_and_rejects_foreign_state():
    sim = _FakeSimulator()
    assert sim.save_backend_state() == {}
    sim.restore_backend_state({})
    with pytest.raises(ValueError, match="backend snapshot"):
        sim.restore_backend_state({"unexpected": torch.ones(1)})


def test_base_env_constructor_initializes_lightweight_dependencies(monkeypatch):
    class _ConstructedMotionManager:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        motion_module,
        "get_class",
        lambda target: _ConstructedMotionManager,
    )
    # The spawn placer reads the manager's config when it advances.
    _ConstructedMotionManager.config = property(lambda self: self.kwargs["config"])
    # Construction computes the dynamic components only (the real component
    # manager, with no components, on the cached all-env context); record
    # what it was handed. The fakes' terrain cache cannot run on this
    # simulator double, and construction must not ask it to.
    component_calls = []
    compute_components = ObservationManager.compute_components

    def recording_compute_components(self, ctx, env_ids):
        component_calls.append((ctx, env_ids.clone()))
        compute_components(self, ctx, env_ids)

    monkeypatch.setattr(
        ObservationManager, "compute_components", recording_compute_components
    )

    config = EnvConfig(
        max_episode_length=7,
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
    assert env._spawn.skip_height_correction is True
    assert env.dt == simulator.dt
    assert torch.equal(env.reset_buf, torch.ones(2, dtype=torch.bool))
    assert env._contact_memory.prev_contact_force_magnitudes.shape == (2, 2)
    assert motion_lib.smooth_windows == [3]
    assert env.reference_motion.motion_lib is motion_lib is env._motion.motion_lib
    assert env.reference_motion.motion_manager.kwargs["num_envs"] == 2
    assert env.reference_motion.motion_manager is env._motion.motion_manager
    assert [owner.name for owner in env._advance_owners] == [
        "motion",
        "controls",
        "spawn",
    ]
    assert env._recovery.fall_states is None  # recovery_prob is 0 by default
    # Nothing in this config reads the past, so the ring keeps no past slot.
    assert env._state_history.num_history_steps == 0
    assert simulator.initialized_markers == {}
    # The first observations came from the all-env context (the publishers
    # saw None), for every row, and that context stays cached.
    assert len(component_calls) == 1
    ((ctx, rows),) = component_calls
    assert ctx.env_ids is None and torch.equal(rows, torch.arange(2))
    assert env._current_context is ctx

    no_motion_config = EnvConfig(
        max_episode_length=5,
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

    assert no_motion_env.reference_motion.motion_manager is None
    assert no_motion_env._state_history.num_history_steps == 0
    assert no_motion_env.sensors is None
    assert no_motion_lib.smooth_windows == []
    assert no_motion_simulator.initialized_markers == {}


def _reference_motion(num_envs: int = 2) -> ReferenceMotion:
    """A reference motion over the fakes, holding an installed library."""
    motion = ReferenceMotion(
        EnvConfig().motion_manager,
        _robot_config(),
        _SceneLib(num_envs=num_envs),
        num_envs,
        0.05,
        torch.device("cpu"),
        contact_smooth_window=3,
        generator=None,
    )
    motion.motion_lib = _MotionLib(num_motions=1, num_envs=num_envs)
    motion.motion_manager = "old-manager"
    return motion


def test_install_uses_prebuilt_candidate_without_constructing():
    motion = _reference_motion()
    candidate = _MotionLib(num_motions=2, num_envs=2)
    events = []

    motion.validate_compatibility = lambda motion_lib: events.append(
        ("validate", motion_lib)
    )

    def create_manager():
        events.append(("manager", motion.motion_lib))
        return "new-manager"

    motion.create_manager = create_manager

    assert motion.install(candidate) is None
    assert candidate.smooth_windows == [3]
    assert motion.motion_lib is candidate
    assert motion.motion_manager == "new-manager"
    assert events == [("validate", candidate), ("manager", candidate)]


def test_install_keeps_current_state_when_candidate_is_incompatible():
    motion = _reference_motion()
    old_motion_lib = motion.motion_lib
    candidate = _MotionLib(num_motions=2, num_envs=2)
    motion.validate_compatibility = lambda motion_lib: (_ for _ in ()).throw(
        ValueError("incompatible")
    )

    with pytest.raises(ValueError, match="incompatible"):
        motion.install(candidate)

    assert motion.motion_lib is old_motion_lib
    assert motion.motion_manager == "old-manager"
    assert candidate.smooth_windows == []


def test_install_clears_manager_for_empty_prebuilt_library():
    motion = _reference_motion()

    motion.install(_MotionLib(num_motions=0, num_envs=2))

    assert motion.motion_manager is None


def test_env_install_motion_lib_delegates_and_reruns_the_two_way_check(monkeypatch):
    env = _make_env(motions=2)
    events = []
    env._motion.install = lambda motion_lib: events.append(("install", motion_lib))
    # The binding check is the build module's; the env calls the name it
    # imported.
    monkeypatch.setattr(
        env_module, "validate_bindings", lambda env: events.append(("check",))
    )
    env._current_context = {"stale": True}
    candidate = _MotionLib(num_motions=2)

    BaseEnv.install_motion_lib(env, candidate)

    assert events == [("install", candidate), ("check",)]
    assert env._current_context is None
    # The env reaches the reference motion as its owner, not through copies.
    assert env.reference_motion is env._motion


def test_base_env_step_refuses_a_graph_carrying_action():
    """The gradient boundary, stated instead of repaired (item 29, P22 PR 4).

    step() records the raw and processed actions with ``buffer[:] = x``; a
    ``copy_`` from a tensor that carries a graph would turn the persistent
    buffer into a graph node and pin the whole action graph for its lifetime
    (the interactive-eval GPU leak). The stage used to ``.detach()`` silently;
    it now refuses the action, because every in-tree caller steps under
    ``torch.no_grad()`` and a graph-carrying action means one forgot to. The
    buffers stay untouched and grad-free; a detached action steps as before.
    """
    env = _make_env(action_config=ActionConfig(fn=_plus_one_action, params={}))
    before = {
        name: getattr(env._action_stage, name).clone() for name in ("raw", "processed")
    }

    action = torch.ones(3, 2, requires_grad=True) * 2.0
    assert action.grad_fn is not None  # the action genuinely carries a graph
    with pytest.raises(ValueError, match="carries an autograd graph"):
        BaseEnv.step(env, action)
    for name in ("raw", "processed"):
        buf = getattr(env._action_stage, name)
        assert torch.equal(buf, before[name]), name  # nothing was stored
        assert not buf.requires_grad and buf.grad_fn is None, name

    BaseEnv.step(env, action.detach())
    for name in ("raw", "processed"):
        buf = getattr(env._action_stage, name)
        assert not buf.requires_grad and buf.grad_fn is None, name
    assert torch.equal(env._action_stage.raw, torch.full((3, 2), 2.0))
    assert torch.equal(env._action_stage.processed, torch.full((3, 2), 3.0))


def test_base_env_step_refuses_an_action_function_that_carries_a_graph():
    """The same boundary, seen from the action function: a parameter that
    requires grad turns a grad-free action into a graph-carrying command, which
    the stage refuses before storing either buffer."""
    scale = torch.ones(2, requires_grad=True)
    env = _make_env(
        action_config=ActionConfig(fn=_grad_scaled_action, params={"scale": scale})
    )
    before = {
        name: getattr(env._action_stage, name).clone() for name in ("raw", "processed")
    }
    action = torch.full((3, 2), 2.0)
    with pytest.raises(ValueError, match="action function's output carries"):
        BaseEnv.step(env, action)
    for name in ("raw", "processed"):
        buf = getattr(env._action_stage, name)
        assert torch.equal(buf, before[name]), name  # nothing was stored
        assert not buf.requires_grad and buf.grad_fn is None, name
