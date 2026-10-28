# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU simulator boundary for sensor contracts, not a replacement environment.

MuJoCo supplies independent kinematic truth for a mesh-free articulated chain.
Actions prescribe joint velocities; this is deliberately NOT a dynamics/PD test.
There is no noise, observation derivation, or history implementation here. Tests
must use production BaseEnv, ComponentManager, views, FK and history for those.
Full-batch getters expose reused buffers like GPU backends; callers that retain
a frame must own a copy. Subset reads preserve the requested row order.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import torch

from protomotions.components.pose_lib import KinematicInfo, extract_kinematic_info
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.utils.baked_fk import BakedAnchorFK
from protomotions.envs.sensors.config import (
    IMUSensor,
    JointEncoderSensor,
    LocalizationSensor,
    NoNoise,
    SensorConfig,
)
from protomotions.envs.obs.target import compute_target_obs
from protomotions.envs.context_sources import sensed
from protomotions.envs.obs.humanoid import compute_humanoid_reduced_coords_observations
from protomotions.envs.obs.humanoid_historical import (
    compute_historical_reduced_coords_from_state,
)
from protomotions.simulator.base_simulator.config import (
    DomainRandomizationConfig,
    RobotNoiseConfig,
    SimulatorConfig,
    SimParams,
)
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
    RobotState,
    StateConversion,
)
from protomotions.simulator.base_simulator.user_interface import UserInterface
from protomotions.simulator.base_simulator.simulator import Simulator


CHAIN_XML = Path(__file__).parents[1] / "fixtures" / "sensor_chain.xml"


@dataclass
class SensorTestRobot:
    kinematic_info: KinematicInfo
    imu_body_name: str = "pelvis"
    imu_mount_quat: tuple = (0.0, 0.0, 0.0, 1.0)
    contact_bodies: tuple = ("hand",)
    non_termination_contact_bodies: tuple = ("hand",)
    anchor_body_index: int = 1
    number_of_actions: int = 2
    control_hz: int = 50
    reset_noise: Optional[RobotNoiseConfig] = None

    @property
    def resolved_anchor_body_name(self) -> str:
        return self.kinematic_info.body_names[self.anchor_body_index]

    @property
    def control_dt(self) -> float:
        """Seconds per control step, as RobotConfig.control_dt computes it."""
        return 1.0 / self.control_hz


@dataclass
class SimulatorSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    actions: torch.Tensor
    ticks: np.ndarray


class SensorTestSimulator:
    """Batched kinematic test double with real common-format state objects.

    ``contact_model=True`` makes ``get_contact_forces`` a deterministic function
    of the pose (see there) instead of zeros, so contact memory on the env side
    becomes observable. The default keeps today's impulse-free behaviour.
    ``contacts=False`` is a robot without contact bodies: the simulator reports
    no contacts (``contacts_enabled`` False, no contact labels), the robot
    config declares none and the state carries no ``rigid_body_contacts``.
    """

    save_backend_state = Simulator.save_backend_state
    restore_backend_state = Simulator.restore_backend_state

    # Bodies below this height press on the terrain in the synthetic contact
    # model; the force grows linearly with the penetration depth.
    CONTACT_HEIGHT = 0.9
    CONTACT_STIFFNESS = 100.0

    def __init__(self, num_envs=3, *, contact_model=False, contacts=True):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if contact_model and not contacts:
            raise ValueError("the contact model needs contacts enabled")
        self.num_envs = num_envs
        self.contact_model = contact_model
        self.num_act = 2
        self.dt = 0.02
        self.headless = True
        self.user_interface = UserInterface()
        # This fixture is copied onto the parity base export too, where
        # SimParams still carries `decimation` and the simulator derives
        # dt = decimation / fps. Pass decimation=1 only there; on this head the
        # robot's control_hz=50 gives the same 1/50 s step (decimation removed).
        sim_kwargs = {"fps": 50}
        if any(f.name == "decimation" for f in fields(SimParams)):
            sim_kwargs["decimation"] = 1
        self.config = SimulatorConfig(
            # This fixture implements MuJoCo's acquisition/reset contract with
            # actual mj_forward and COM-to-origin velocity conversion below.
            _target_="protomotions.simulator.mujoco.simulator.MujocoSimulator",
            w_last=True,
            headless=True,
            num_envs=num_envs,
            sim=SimParams(**sim_kwargs),
            experiment_name="sensor-contracts",
            domain_randomization=DomainRandomizationConfig(),
        )
        self.robot_config = SensorTestRobot(extract_kinematic_info(str(CHAIN_XML)))
        if not contacts:
            self.robot_config.contact_bodies = None
        self.model = mujoco.MjModel.from_xml_path(str(CHAIN_XML))
        self.data = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.actions = torch.zeros(num_envs, self.num_act)
        self.ticks = np.zeros(num_envs, dtype=np.int64)
        self.closed = False
        self.markers = None
        self.reset_calls = []
        # No push randomization here. ``push_enabled`` mirrors the real
        # Simulator's property (BaseEnv's perturbation switch calls
        # set_push_enabled, which refuses to turn pushes on); the private flag
        # stays because the parity harness runs older evaluators, which still
        # read it, against this double.
        self._push_enabled = False
        self.state = RobotState(
            state_conversion=StateConversion.COMMON,
            rigid_body_pos=torch.zeros(num_envs, 4, 3),
            rigid_body_rot=torch.zeros(num_envs, 4, 4),
            rigid_body_vel=torch.zeros(num_envs, 4, 3),
            rigid_body_ang_vel=torch.zeros(num_envs, 4, 3),
            dof_pos=torch.zeros(num_envs, 2),
            dof_vel=torch.zeros(num_envs, 2),
            dof_forces=torch.zeros(num_envs, 2),
            rigid_body_contacts=(
                torch.zeros(num_envs, 4, dtype=torch.bool) if contacts else None
            ),
        )
        # Deliberately asymmetric, nonzero, and not yaw-only. Identity poses hide
        # local/world mistakes and coincident root/anchor bugs.
        for i, data in enumerate(self.data):
            data.qpos[:3] = [0.4 + i, -0.2 * i, 0.9]
            quat = np.array([0.91, 0.16, -0.23, 0.27])
            data.qpos[3:7] = quat / np.linalg.norm(quat)
            data.qpos[7:] = [0.25 + 0.07 * i, -0.4 + 0.03 * i]
            data.qvel[:] = [0.1, -0.03, 0.02, 0.2, -0.1, 0.3, 0.4, -0.2]
        self._refresh(range(num_envs))
        self.default = ResetState.from_robot_state(self.state).clone()
        from protomotions.simulator.base_simulator.contacts import ContactLayout

        self.contacts_enabled = contacts
        self.has_complete_contact_labels = contacts
        self.contact_layout = ContactLayout.build(
            self.robot_config.kinematic_info.body_names, 0, 0
        )

    def get_contact_forces(self, env_ids=None):
        """Privileged ``forces[env, body, partner, xyz]`` for the current pose.

        Without ``contact_model`` this kinematic fixture has no solver impulses
        and returns zeros. With it, every body below ``CONTACT_HEIGHT`` pushes
        on the terrain partner column with a vertical force proportional to
        its depth below that height; the robot-link partner columns stay zero.
        """
        forces = torch.zeros(self.num_envs, 4, 5, 3)
        if self.contact_model:
            depth = (self.CONTACT_HEIGHT - self.state.rigid_body_pos[..., 2]).clamp_min(
                0.0
            )
            terrain = self.contact_layout.partner_index["terrain"]
            forces[:, :, terrain, 2] = depth * self.CONTACT_STIFFNESS
        return forces if env_ids is None else forces[env_ids]

    def _refresh(self, env_ids):
        for i in env_ids:
            data = self.data[int(i)]
            mujoco.mj_forward(self.model, data)
            self.state.dof_pos[i].copy_(torch.from_numpy(data.qpos[7:]))
            self.state.dof_vel[i].copy_(torch.from_numpy(data.qvel[6:]))
            for b in range(4):
                mj_id = b + 1  # world body is not part of common robot state
                self.state.rigid_body_pos[i, b].copy_(
                    torch.from_numpy(data.xpos[mj_id])
                )
                self.state.rigid_body_rot[i, b].copy_(
                    torch.from_numpy(data.xquat[mj_id][[1, 2, 3, 0]])
                )
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(
                    self.model, data, mujoco.mjtObj.mjOBJ_BODY, mj_id, velocity, 0
                )
                self.state.rigid_body_ang_vel[i, b].copy_(
                    torch.from_numpy(velocity[:3])
                )
                # mj_objectVelocity linear velocity is at the COM. Common state
                # is at the body origin; retain the lever-arm correction here.
                at_origin = velocity[3:] + np.cross(
                    velocity[:3], data.xpos[mj_id] - data.xipos[mj_id]
                )
                self.state.rigid_body_vel[i, b].copy_(torch.from_numpy(at_origin))

    def _initialize_with_markers(self, markers):
        self.markers = markers

    @property
    def push_enabled(self):
        return self._push_enabled

    @property
    def has_push_randomization(self):
        return False

    # The parity script copies this double into the export of the merge base,
    # whose env still reads the property under its old name; both names
    # answer here so one double serves both sides of the comparison.
    push_configured = has_push_randomization

    def set_push_enabled(self, enabled):
        if enabled:
            raise ValueError("the kinematic test double has no push randomization")
        self._push_enabled = False

    def is_simulation_running(self):
        return not self.closed

    def get_robot_state(self, env_ids=None):
        return self.state if env_ids is None else self.state[env_ids]

    def get_root_state(self, env_ids=None):
        return self.get_robot_state(env_ids)

    def get_bodies_state(self, env_ids=None):
        return self.get_robot_state(env_ids)

    def get_default_robot_reset_state(self):
        return self.default.clone()

    def get_current_actions(self):
        # Match BaseSimulator's live-buffer boundary. Snapshot callers must own
        # their copy; a clone here would conceal production aliasing defects.
        return self.actions

    def save_state(self):
        return {"actions": self.actions.clone(), "ticks": self.ticks.copy()}

    def restore_state(self, state):
        self.actions.copy_(state["actions"])
        self.ticks[:] = state["ticks"]

    def step(self, actions, markers_callback=None):
        if actions.shape != self.actions.shape:
            raise ValueError("Actions must match the simulator batch and DOFs")
        self.actions.copy_(actions)
        for i, data in enumerate(self.data):
            data.qvel[6:] = actions[i].detach().numpy()
            mujoco.mj_integratePos(self.model, data.qpos, data.qvel, self.dt)
        self.ticks += 1
        self._refresh(range(self.num_envs))
        if markers_callback is not None:
            markers_callback()

    def reset_envs(self, new_states, new_object_states=None, env_ids=None):
        if not isinstance(new_states, ResetState):
            raise TypeError("Reset accepts ResetState, not derived RobotState")
        if new_states.state_conversion != StateConversion.COMMON:
            raise ValueError("Expected common xyzw state")
        ids = torch.arange(self.num_envs) if env_ids is None else env_ids
        if len(ids) != new_states.root_pos.shape[0] or len(ids.unique()) != len(ids):
            raise ValueError("Reset rows must correspond one-to-one to unique env_ids")
        for row, i in enumerate(ids.tolist()):
            data = self.data[i]
            data.qpos[:3] = new_states.root_pos[row].numpy()
            data.qpos[3:7] = new_states.root_rot[row][[3, 0, 1, 2]].numpy()
            data.qpos[7:] = new_states.dof_pos[row].numpy()
            data.qvel[:3] = new_states.root_vel[row].numpy()
            # MuJoCo free-joint angular qvel is local; ResetState is world.
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            data.qvel[3:6] = (
                rotation.reshape(3, 3).T @ new_states.root_ang_vel[row].numpy()
            )
            data.qvel[6:] = new_states.dof_vel[row].numpy()
            self.actions[i].zero_()
            self.ticks[i] = 0
        self._refresh(ids.tolist())
        self.reset_calls.append(ids.clone())

    def snapshot(self):
        return SimulatorSnapshot(
            np.stack([d.qpos.copy() for d in self.data]),
            np.stack([d.qvel.copy() for d in self.data]),
            self.actions.clone(),
            self.ticks.copy(),
        )

    def restore(self, snapshot):
        for i, data in enumerate(self.data):
            data.qpos[:] = snapshot.qpos[i]
            data.qvel[:] = snapshot.qvel[i]
        self.actions.copy_(snapshot.actions)
        self.ticks[:] = snapshot.ticks
        self._refresh(range(self.num_envs))

    def close(self):
        self.closed = True


class FlatSensorTestTerrain:
    """Flat ground. ``spawn_spread=True`` hands out a distinct xy per sampled
    row (``[i, -i]``) instead of the origin, so respawn offsets are nonzero
    and a reset that forgets to apply them is visible."""

    def __init__(self, spawn_spread=False):
        self.config = TerrainConfig()
        self.num_height_points = 1
        self.spawn_spread = spawn_spread

    def is_flat(self):
        return True

    def get_ground_heights(self, positions):
        return torch.zeros(*positions.shape[:-1], 1)

    def get_height_maps(self, root_state, env_ids):
        return torch.zeros(len(env_ids), 1)

    def sample_valid_locations(self, num_envs, sample_flat=False):
        if not self.spawn_spread:
            return torch.zeros(num_envs, 2)
        rows = torch.arange(num_envs, dtype=torch.float)
        return torch.stack([rows, -rows], dim=-1)


class EmptySensorTestScenes:
    num_objects_per_scene = 0
    has_pointclouds = False  # an object-less library computes none

    def num_scenes(self):
        return 0

    def get_default_object_state(self, device):
        return ObjectState(state_conversion=StateConversion.COMMON)

    def get_scene_pose(self, scene_indices, time, respawn_offset=0.0):
        # Same shapes as the real empty SceneLib: zero objects per scene.
        num_scenes = scene_indices.shape[0] if scene_indices.numel() > 0 else 0
        return ObjectState(
            root_pos=torch.zeros(num_scenes, 0, 3),
            root_rot=torch.zeros(num_scenes, 0, 4),
            root_vel=torch.zeros(num_scenes, 0, 3),
            root_ang_vel=torch.zeros(num_scenes, 0, 3),
            state_conversion=StateConversion.COMMON,
        )


class EmptySensorTestMotions:
    def num_motions(self):
        return 0


def make_sensor_test_env(
    sensors=None,
    history_steps=3,
    num_envs=3,
    recovery_reset=None,
    *,
    observations=None,
    rewards=None,
    terminations=None,
    spawn_spread=False,
    contact_model=False,
    contacts=True,
    reset_noise=None,
    motion_lib=None,
    motion_manager=None,
    control_components=None,
    target_observations=True,
):
    """Construct the REAL environment and actor observation factories.

    ``rewards`` / ``terminations`` are component dicts installed before
    construction so the env validates their bindings. ``spawn_spread`` /
    ``contact_model`` turn on the informative variants of the terrain and
    simulator doubles; ``contacts=False`` is a robot without contact bodies
    (see ``SensorTestSimulator``); ``reset_noise`` (a ``RobotNoiseConfig``) is
    installed on the robot config. ``motion_lib`` (a real ``MotionLib``),
    ``motion_manager`` (its config) and ``control_components`` give the env a
    reference motion to track. ``target_observations=False`` leaves out the
    two target observations, which bind the anchor position and so need a
    sensor config with a localization channel; the other defaults bind only
    names every config measures. ``sensors`` (a ``SensorConfig``) overrides the
    default exact config with a localization sensor on body 0. The defaults keep
    today's motion-free, reward-free env with contacts (the parity scenarios
    build on them).
    """
    sim = SensorTestSimulator(num_envs, contact_model=contact_model, contacts=contacts)
    if reset_noise is not None:
        sim.robot_config.reset_noise = reset_noise
    config = EnvConfig(max_episode_length=1000)
    config.recovery_reset = recovery_reset
    if rewards is not None:
        config.reward_components = rewards
    if terminations is not None:
        config.termination_components = terminations
    if motion_manager is not None:
        config.motion_manager = motion_manager
    if control_components is not None:
        config.control_components = control_components
    if sensors is not None:
        config.sensors = sensors
    else:
        exact = NoNoise("fixture parity")
        root = sim.robot_config.kinematic_info.body_names[0]
        config.sensors = SensorConfig(
            joint_encoders=JointEncoderSensor(exact, exact),
            imu=IMUSensor(exact, exact),
            localization=LocalizationSensor(root, exact),
            seed=734,
        )
    reduced = MdpComponent.from_kernel(compute_humanoid_reduced_coords_observations)
    config.observation_components = {
        **sensed({"actor": reduced}, "noisy"),
        **sensed({"clean": reduced}, "clean"),
        "actor_fk": MdpComponent(
            compute_func=BakedAnchorFK(
                sim.robot_config.kinematic_info, 1, torch.tensor([3])
            ),
            dynamic_vars={"dof_pos": EnvContext.noisy.dof_pos},
            # Pin the body-fixed frame: BakedAnchorFK.current_anchor_rot now
            # carries a FieldPath default, so without this the direct constructor
            # would auto-bind it (heading frame). This fixture tests FK from joint
            # encoders alone, so it stays None.
            static_params={"current_anchor_rot": None},
        ),
    }
    targets = (
        (("actor_target", EnvContext.noisy), ("clean_target", EnvContext.clean))
        if target_observations
        else ()
    )
    for name, view in targets:
        config.observation_components[name] = MdpComponent(
            compute_func=compute_target_obs,
            dynamic_vars={"root_pos": view.anchor_pos, "root_rot": view.anchor_rot},
            static_params={"tar_pos": torch.tensor([[3.0, 1.0, 0.0]])},
        )
    if history_steps:
        history = MdpComponent.from_kernel(
            compute_historical_reduced_coords_from_state, history_steps=history_steps
        )
        config.observation_components.update(
            sensed({"actor_history": history}, "noisy")
        )
    if observations is not None:
        config.observation_components.update(observations)
    return BaseEnv(
        config,
        sim.robot_config,
        torch.device("cpu"),
        FlatSensorTestTerrain(spawn_spread=spawn_spread),
        sim,
        EmptySensorTestScenes(),
        EmptySensorTestMotions() if motion_lib is None else motion_lib,
    )


def tensor_state(state):
    """Owned snapshots for assertions; never let expected values alias buffers."""
    return {
        f.name: getattr(state, f.name).clone()
        for f in fields(state)
        if isinstance(getattr(state, f.name), torch.Tensor)
    }
