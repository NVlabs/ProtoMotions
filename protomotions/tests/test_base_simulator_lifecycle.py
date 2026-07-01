# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for base simulator lifecycle logic using a fake backend."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.robot_configs.base import ControlType
from protomotions.simulator.base_simulator.config import (
    ActionNoiseDomainRandomizationConfig,
    CenterOfMassDomainRandomizationConfig,
    DomainRandomizationConfig,
    FrictionDomainRandomizationConfig,
    MarkerState,
    ProjectileConfig,
    PushDomainRandomizationConfig,
    SimBodyOrdering,
    SimParams,
    SimulatorConfig,
    VisualizationMarkerConfig,
)
from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
    RobotState,
    RootOnlyState,
    StateConversion,
)


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def _robot_config(control_type=ControlType.BUILT_IN_PD):
    dof_names = ["joint_a", "joint_b"]
    return SimpleNamespace(
        control=SimpleNamespace(
            control_type=control_type,
            control_info={
                "joint_a": SimpleNamespace(
                    stiffness=10.0,
                    damping=1.0,
                    effort_limit=5.0,
                ),
                "joint_b": SimpleNamespace(
                    stiffness=20.0,
                    damping=2.0,
                    effort_limit=6.0,
                ),
            },
        ),
        kinematic_info=SimpleNamespace(
            num_bodies=2,
            num_dofs=2,
            dof_names=dof_names,
            body_names=["root", "foot"],
            dof_limits_lower=torch.tensor([-1.0, -2.0]),
            dof_limits_upper=torch.tensor([1.0, 2.0]),
        ),
        number_of_actions=2,
        default_dof_pos=torch.tensor([0.25, -0.5]),
        default_root_height=0.9,
    )


def _sim_config(**overrides):
    domain_randomization = DomainRandomizationConfig(
        action_noise=ActionNoiseDomainRandomizationConfig(
            action_noise_range=(0.0, 0.1),
            dof_indices=[0],
        ),
        friction=FrictionDomainRandomizationConfig(body_indices=[0]),
        center_of_mass=CenterOfMassDomainRandomizationConfig(
            com_range={"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z": (0.0, 0.3)},
            body_indices=[0],
        ),
        push=PushDomainRandomizationConfig(
            push_interval_range=(0.1, 0.2),
            max_linear_velocity=(0.1, 0.0, 0.0),
            max_angular_velocity=(0.0, 0.0, 0.2),
        ),
    )
    kwargs = {
        "_target_": "unused.FakeSimulator",
        "w_last": True,
        "headless": True,
        "num_envs": 2,
        "sim": SimParams(fps=60, decimation=2),
        "experiment_name": "unit",
        "domain_randomization": domain_randomization,
        "projectile": ProjectileConfig(
            num_projectiles=2,
            cube_half_size_range=(0.1, 0.2),
            speed_range=(1.0, 1.0),
            spawn_distance_range=(1.0, 1.0),
            spawn_height_range=(0.5, 0.5),
            direction_noise_std=0.0,
            hide_delay=0.05,
            hide_z=-3.0,
        ),
        "pd_target_max_accel": 0.25,
    }
    kwargs.update(overrides)
    return SimulatorConfig(**kwargs)


class _FakeSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        self.created = False
        self.pd_targets = []
        self.torques = []
        self.projectile_sets = []
        self.root_impulses = []
        self.reset_calls = []
        self.marker_updates = []
        self.physics_steps = 0
        super().__init__(*args, **kwargs)

    def _create_simulation(self):
        self.created = True

    def _apply_root_velocity_impulse(self, linear_velocity, angular_velocity, env_ids):
        self.root_impulses.append((linear_velocity.clone(), angular_velocity.clone(), env_ids.clone()))

    def _create_projectiles(self, config):
        self.created_projectile_config = config

    def _set_projectile_root_states(
        self,
        proj_indices,
        positions,
        rotations_xyzw,
        velocities,
        ang_velocities,
        env_ids,
    ):
        self.projectile_sets.append(
            (
                proj_indices.clone(),
                positions.clone(),
                rotations_xyzw.clone(),
                velocities.clone(),
                ang_velocities.clone(),
                env_ids.clone(),
            )
        )

    def _get_projectile_positions_rotations(self):
        return torch.ones(self.num_envs, 2, 3), _identity_quat(self.num_envs, 2)

    def _set_simulator_env_state(self, new_states, new_object_states=None, env_ids=None):
        self.reset_calls.append((new_states, new_object_states, env_ids.clone()))

    def _physics_step(self):
        self.physics_steps += 1
        self._apply_control()

    def _get_sim_body_ordering(self):
        return SimBodyOrdering(
            body_names=["foot", "root"],
            dof_names=["joint_b", "joint_a"],
        )

    def _get_simulator_root_state(self, env_ids=None):
        root_pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        root_rot = _identity_quat(2)
        root_vel = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        root_ang_vel = torch.zeros(2, 3)
        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
        return RootOnlyState(
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_bodies_state(self, env_ids=None):
        rigid_body_pos = torch.arange(2 * 2 * 3, dtype=torch.float).reshape(2, 2, 3)
        rigid_body_rot = _identity_quat(2, 2)
        rigid_body_vel = torch.ones(2, 2, 3)
        rigid_body_ang_vel = torch.ones(2, 2, 3) * 2.0
        if env_ids is not None:
            rigid_body_pos = rigid_body_pos[env_ids]
            rigid_body_rot = rigid_body_rot[env_ids]
            rigid_body_vel = rigid_body_vel[env_ids]
            rigid_body_ang_vel = rigid_body_ang_vel[env_ids]
        return RobotState(
            rigid_body_pos=rigid_body_pos,
            rigid_body_rot=rigid_body_rot,
            rigid_body_vel=rigid_body_vel,
            rigid_body_ang_vel=rigid_body_ang_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_forces(self, env_ids=None):
        forces = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        if env_ids is not None:
            forces = forces[env_ids]
        return RobotState(dof_forces=forces, state_conversion=StateConversion.SIMULATOR)

    def _get_simulator_dof_state(self, env_ids=None):
        dof_pos = torch.tensor([[0.0, 0.5], [1.0, 1.5]])
        dof_vel = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        if env_ids is not None:
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
        return RobotState(
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_dof_limits_for_verification(self):
        return torch.tensor([-2.0, -1.0]), torch.tensor([2.0, 1.0])

    def _get_simulator_bodies_contact_buf(self, env_ids=None):
        forces = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
                [[0.0, 0.0, 0.2], [0.0, 0.0, 0.0]],
            ]
        )
        if env_ids is not None:
            forces = forces[env_ids]
        return RobotState(
            rigid_body_contact_forces=forces,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_root_state(self, env_ids=None):
        root_pos = torch.ones(2, 1, 3)
        root_rot = _identity_quat(2, 1)
        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
        return ObjectState(
            root_pos=root_pos,
            root_rot=root_rot,
            state_conversion=StateConversion.SIMULATOR,
        )

    def _get_simulator_object_contact_buf(self, env_ids=None):
        forces = torch.ones(2, 1, 3)
        if env_ids is not None:
            forces = forces[env_ids]
        return ObjectState(contact_forces=forces, state_conversion=StateConversion.SIMULATOR)

    def _apply_simulator_pd_targets(self, pd_targets):
        self.pd_targets.append(pd_targets.clone())

    def _apply_simulator_torques(self, torques):
        self.torques.append(torques.clone())

    def _write_viewport_to_file(self, file_name):
        self.last_viewport_file = file_name

    def _init_camera(self):
        self.camera_initialized = True

    def _update_simulator_markers(self, markers_state=None):
        self.marker_updates.append(markers_state)


def _sim(scene_objects=0, control_type=ControlType.BUILT_IN_PD, **config_overrides):
    return _FakeSimulator(
        config=_sim_config(**config_overrides),
        robot_config=_robot_config(control_type),
        terrain=None,
        device=torch.device("cpu"),
        scene_lib=SimpleNamespace(num_objects_per_scene=scene_objects),
    )


def test_simulator_initializes_finalize_setup_and_basic_getters():
    sim = _sim(scene_objects=0)
    assert sim.created is False
    assert sim.is_simulation_running() is True
    assert "L" in sim.user_interface.registered_keys
    assert "O" in sim.user_interface.registered_keys
    with pytest.raises(ValueError, match="L.*Toggle viewer recording"):
        sim.user_interface.register_key(
            "L", owner="unit", description="Conflicting recording key"
        )

    markers = {"target": VisualizationMarkerConfig()}
    sim._initialize_with_markers(markers)

    assert sim.created is True
    assert sim._initialized is True
    assert sim._original_marker_configs == markers
    assert torch.equal(sim.data_conversion.dof_convert_to_common, torch.tensor([1, 0]))
    assert sim.created_projectile_config.num_projectiles == 2
    assert len(sim.projectile_sets) == 1
    with pytest.raises(RuntimeError, match="already initialized"):
        sim._initialize_with_markers({})

    default_state = sim.get_default_robot_reset_state()
    assert torch.equal(default_state.dof_pos[0], torch.tensor([0.25, -0.5]))
    assert torch.equal(default_state.root_pos[:, 2], torch.full((2,), 0.9))

    sim.user_interface.handle_key_event("L", pressed=True)
    assert sim._user_is_recording is True
    sim.user_interface.handle_key_event("Q", pressed=True)
    assert sim.is_simulation_running() is False
    assert sim.get_object_root_state() is None
    assert torch.equal(sim.get_previous_actions(), torch.zeros(2, 2))
    sim._previous_actions[:] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sim._common_actions[:] = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    assert torch.equal(sim.get_previous_actions(torch.tensor([1])), torch.tensor([[3.0, 4.0]]))
    assert torch.equal(sim.get_current_actions(), torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    assert torch.equal(sim.get_current_actions(torch.tensor([1])), torch.tensor([[7.0, 8.0]]))

    root = sim.get_root_state(torch.tensor([0]))
    bodies = sim.get_bodies_state()
    dof = sim.get_dof_state()
    forces = sim.get_dof_forces()
    contacts = sim.get_binary_body_contacts(threshold=0.1)
    robot = sim.get_robot_state()
    obj_contacts = sim.get_object_contact_buf(torch.tensor([1]))

    assert root.root_pos.shape == (1, 3)
    assert bodies.rigid_body_pos.shape == (2, 2, 3)
    assert torch.equal(dof.dof_pos[0], torch.tensor([0.5, 0.0]))
    assert torch.equal(forces.dof_forces[0], torch.tensor([2.0, 1.0]))
    assert torch.equal(contacts.rigid_body_contacts[0], torch.tensor([1.0, 0.0]))
    assert robot.dof_pos.shape == (2, 2)
    assert obj_contacts.contact_forces.shape == (1, 1, 3)

    sim.close()
    assert sim.is_simulation_running() is False


def test_simulator_step_applies_control_markers_push_and_projectile_timers():
    sim = _sim(scene_objects=0)
    sim._initialize_with_markers(None)
    sim._steps_since_reset[:] = torch.tensor([2, 1])
    sim._prev_prev_actions[:] = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    sim._previous_actions[:] = torch.tensor([[0.4, 0.0], [0.0, 0.0]])
    sim._common_actions[:] = torch.tensor([[0.8, 0.0], [0.0, 0.0]])
    sim._push_next_time[:] = 0.0

    marker_state = {
        "target": MarkerState(
            translation=torch.ones(2, 3),
            orientation=_identity_quat(2),
        )
    }
    sim.step(torch.tensor([[1.5, -1.0], [0.2, 0.3]]), lambda: marker_state)

    assert sim.physics_steps == 1
    assert sim._common_actions[0, 0] < 1.5
    assert len(sim.pd_targets) == 1
    assert len(sim.root_impulses) == 1
    assert sim.marker_updates[-1] == marker_state
    assert sim._last_markers_state == marker_state

    sim._throw_projectile()
    assert torch.equal(sim._proj_next_idx, torch.tensor([1, 1]))
    sim._proj_sim_time[:] = sim._proj_config.hide_delay + 1.0
    sim._update_projectiles()
    assert (sim._proj_throw_time == float("-inf")).all()

    sim._reset_projectiles(torch.tensor([1]))
    assert sim._proj_next_idx[1] == 0
    assert sim._proj_sim_time[1] == 0.0


def test_simulator_reset_and_park_env_paths_with_and_without_objects():
    sim = _sim(scene_objects=1)
    sim._initialize_with_markers(None)
    env_ids = torch.tensor([0])
    reset_state = ResetState(
        root_pos=torch.ones(1, 3),
        root_rot=_identity_quat(1),
        root_vel=torch.zeros(1, 3),
        root_ang_vel=torch.zeros(1, 3),
        dof_pos=torch.ones(1, 2),
        dof_vel=torch.zeros(1, 2),
        state_conversion=StateConversion.COMMON,
    )
    object_state = ObjectState(
        root_pos=torch.ones(1, 1, 3),
        root_rot=_identity_quat(1, 1),
        state_conversion=StateConversion.COMMON,
    )

    sim.reset_envs(reset_state, object_state, env_ids)

    state_arg, object_arg, ids_arg = sim.reset_calls[-1]
    assert state_arg.state_conversion is StateConversion.SIMULATOR
    assert object_arg.state_conversion is StateConversion.SIMULATOR
    assert torch.equal(ids_arg, env_ids)
    assert torch.equal(sim._previous_actions[env_ids], torch.zeros(1, 2))

    sim.park_envs(torch.tensor([], dtype=torch.long))
    sim.park_envs(torch.tensor([0]))
    parked_state, parked_objects, _ = sim.reset_calls[-1]
    assert torch.equal(parked_state.root_pos[:, 2], torch.tensor([-50.0]))
    assert torch.equal(parked_objects.root_pos[..., 2], torch.tensor([[-51.0]]))

    no_object_sim = _sim(scene_objects=0)
    no_object_sim._initialize_with_markers(None)
    no_object_sim.reset_envs(reset_state, object_state, env_ids)
    assert no_object_sim.reset_calls[-1][1] is None

    full_reset_state = ResetState(
        root_pos=torch.ones(2, 3),
        root_rot=_identity_quat(2),
        root_vel=torch.zeros(2, 3),
        root_ang_vel=torch.zeros(2, 3),
        dof_pos=torch.ones(2, 2),
        dof_vel=torch.zeros(2, 2),
        state_conversion=StateConversion.COMMON,
    )
    no_object_sim.reset_envs(full_reset_state)
    assert torch.equal(no_object_sim.reset_calls[-1][2], torch.tensor([0, 1]))


def test_simulator_control_modes_and_domain_randomization_helpers():
    proportional = _sim(scene_objects=0, control_type=ControlType.PROPORTIONAL)
    proportional._initialize_with_markers(None)
    proportional._common_actions[:] = torch.tensor([[2.0, -2.0], [0.5, 0.25]])
    proportional._apply_control()
    assert len(proportional.torques) == 1
    assert proportional.torques[-1].shape == (2, 2)
    assert torch.all(proportional.torques[-1].abs() <= torch.tensor([6.0, 5.0]))

    torque = _sim(scene_objects=0, control_type=ControlType.TORQUE)
    torque._initialize_with_markers(None)
    torque._common_actions[:] = torch.tensor([[100.0, -100.0], [0.5, -0.5]])
    torque._apply_control()
    assert torch.equal(torque.torques[-1][0], torch.tensor([-6.0, 5.0]))

    invalid = _sim(scene_objects=0, control_type=ControlType.TORQUE)
    invalid._initialize_with_markers(None)
    invalid.control_type = "bad"
    with pytest.raises(NameError, match="Unknown controller"):
        invalid._apply_control()

    no_randomization = _sim(domain_randomization=None)
    assert no_randomization._process_domain_randomization() is None
    no_push = _sim(
        domain_randomization=DomainRandomizationConfig(
            push=PushDomainRandomizationConfig(),
        )
    )
    no_push._init_push_randomization()
    assert no_push._push_enabled is False
    no_push._schedule_push(torch.tensor([], dtype=torch.long))
    no_push._apply_push_if_due()

    no_due = _sim()
    no_due._initialize_with_markers(None)
    no_due._simulation_time[:] = 0.0
    no_due._push_next_time[:] = 100.0
    no_due._apply_push_if_due()
    assert no_due.root_impulses == []

    no_due._steps_since_reset[:] = 0
    original_actions = no_due._common_actions.clone()
    no_due._apply_accel_clamp()
    assert torch.equal(no_due._common_actions, original_actions)


def test_simulator_joint_limit_mismatch_raises():
    class _BadLimitSimulator(_FakeSimulator):
        def _get_simulator_dof_limits_for_verification(self):
            return torch.tensor([-9.0, -1.0]), torch.tensor([2.0, 1.0])

    sim = _BadLimitSimulator(
        config=_sim_config(domain_randomization=None),
        robot_config=_robot_config(),
        terrain=None,
        device=torch.device("cpu"),
        scene_lib=SimpleNamespace(num_objects_per_scene=0),
    )

    with pytest.raises(ValueError, match="Joint limit mismatch"):
        sim._initialize_with_markers(None)

    class _BadUpperLimitSimulator(_FakeSimulator):
        def _get_simulator_dof_limits_for_verification(self):
            return torch.tensor([-2.0, -1.0]), torch.tensor([9.0, 1.0])

    upper_mismatch = _BadUpperLimitSimulator(
        config=_sim_config(domain_randomization=None),
        robot_config=_robot_config(),
        terrain=None,
        device=torch.device("cpu"),
        scene_lib=SimpleNamespace(num_objects_per_scene=0),
    )
    with pytest.raises(ValueError, match="upper"):
        upper_mismatch._initialize_with_markers(None)

    class _NoLimitSimulator(_FakeSimulator):
        def _get_simulator_dof_limits_for_verification(self):
            raise NotImplementedError

    no_limit = _NoLimitSimulator(
        config=_sim_config(domain_randomization=None),
        robot_config=_robot_config(),
        terrain=None,
        device=torch.device("cpu"),
        scene_lib=SimpleNamespace(num_objects_per_scene=0),
    )
    with pytest.raises(NotImplementedError):
        no_limit._initialize_with_markers(None)


def test_simulator_base_abstract_methods_raise_when_called_directly():
    sim = _sim()
    zeros_3 = torch.zeros(1, 3)
    zeros_4 = torch.zeros(1, 4)
    ids = torch.tensor([0])

    abstract_calls = [
        lambda: Simulator._create_simulation(sim),
        lambda: Simulator._apply_root_velocity_impulse(sim, zeros_3, zeros_3, ids),
        lambda: Simulator._create_projectiles(sim, ProjectileConfig()),
        lambda: Simulator._set_projectile_root_states(
            sim,
            ids,
            zeros_3,
            zeros_4,
            zeros_3,
            zeros_3,
            ids,
        ),
        lambda: Simulator._get_projectile_positions_rotations(sim),
        lambda: Simulator._set_simulator_env_state(sim, None, None, ids),
        lambda: Simulator._physics_step(sim),
        lambda: Simulator._get_sim_body_ordering(sim),
        lambda: Simulator._get_simulator_root_state(sim),
        lambda: Simulator._get_simulator_bodies_state(sim),
        lambda: Simulator._get_simulator_dof_forces(sim),
        lambda: Simulator._get_simulator_dof_state(sim),
        lambda: Simulator._get_simulator_dof_limits_for_verification(sim),
        lambda: Simulator._get_simulator_bodies_contact_buf(sim),
        lambda: Simulator._get_simulator_object_root_state(sim),
        lambda: Simulator._get_simulator_object_contact_buf(sim),
        lambda: Simulator._apply_simulator_pd_targets(sim, torch.zeros(1, 2)),
        lambda: Simulator._apply_simulator_torques(sim, torch.zeros(1, 2)),
        lambda: Simulator._write_viewport_to_file(sim, "frame.png"),
        lambda: Simulator._init_camera(sim),
        lambda: Simulator._update_simulator_markers(sim, None),
    ]

    for call in abstract_calls:
        with pytest.raises(NotImplementedError):
            call()
