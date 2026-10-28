# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real GPU backend reset/restore check; run in the backend's environment.

python scripts/check_backend_reset.py --simulator isaaclab
python scripts/check_backend_reset.py --simulator isaacgym

This boots three G1s, checks immediate FK and velocity round trips, then checks
that a partial reset cannot discard a pending restore of the other robots.
It does not train a policy or require a checkpoint/motion dataset.
"""

# Simulator selection and startup must precede torch and its transitive imports.
# ruff: noqa: E402
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--simulator", choices=("isaacgym", "isaaclab"), required=True)
args = parser.parse_args()

from protomotions.utils.simulator_imports import import_simulator_before_torch

AppLauncher = import_simulator_before_torch(args.simulator)
app = AppLauncher({"headless": True, "device": "cuda:0"}).app if AppLauncher else None

import torch

from protomotions.envs.sensors.config import (
    BaseVelocitySensor,
    IMUSensor,
    JointEncoderSensor,
    LocalizationSensor,
    NoNoise,
    SensorConfig,
)
from protomotions.envs.sensors.runtime import SensorRuntime
from protomotions.robot_configs.factory import robot_config
from protomotions.simulator.factory import simulator_config
from protomotions.simulator.base_simulator.simulator_state import ResetState
from protomotions.components.scene_lib import SceneLib
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.terrains.terrain import Terrain
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
from protomotions.utils.hydra_replacement import get_class
from protomotions.utils.baked_fk import BakedAnchorFK
from protomotions.utils import rotations

device = torch.device("cuda:0")
robot = robot_config("g1")
# Exercise a mounted, moving IMU rather than the special case of a root IMU.
robot.imu_body_name = "torso_link"
robot.imu_mount_quat = (0.0, 0.0, 0.6, 0.8)
config = simulator_config(
    args.simulator, robot, headless=True, num_envs=3, experiment_name="reset-contract"
)
config.domain_randomization = None
terrain_config, config = convert_friction_for_simulator(TerrainConfig(), config)
terrain = Terrain(terrain_config, num_envs=3, device=device)
extra = {} if app is None else {"simulation_app": app}
sim = get_class(config._target_)(
    config=config,
    robot_config=robot,
    terrain=terrain,
    scene_lib=SceneLib.empty(num_envs=3, device=device),
    device=device,
    **extra,
)
try:
    sim._initialize_with_markers({})
    ids = torch.arange(3, device=device)
    state = sim.get_default_robot_reset_state()
    state.root_pos[:] = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 3.0], [7.0, 8.0, 3.0]], device=device
    )
    state.root_rot[:] = torch.tensor([0.2, -0.1, 0.3, 0.9], device=device)
    state.root_rot /= state.root_rot.norm(dim=-1, keepdim=True)
    state.root_vel[:] = torch.tensor([0.3, -0.2, 0.1], device=device)
    state.root_ang_vel[:] = torch.tensor([1.0, 2.0, 3.0], device=device)
    state.dof_vel[:] = 0.13
    expected = state.clone()
    sim.reset_envs(state, env_ids=ids)
    actual = sim.get_robot_state()
    torch.testing.assert_close(actual.root_pos, expected.root_pos, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual.root_vel, expected.root_vel, atol=2e-5, rtol=2e-5)
    fk = BakedAnchorFK(
        robot.kinematic_info, 0, torch.arange(robot.kinematic_info.num_bodies)
    )
    pos, _, angular, linear = fk.body_kinematics(expected.dof_pos, expected.dof_vel)
    root_rot = expected.root_rot[:, None].expand(-1, pos.shape[1], -1).reshape(-1, 4)
    offset = rotations.quat_rotate(root_rot, pos.reshape(-1, 3), True).reshape_as(pos)
    world = offset + expected.root_pos[:, None]
    torch.testing.assert_close(actual.rigid_body_pos, world, atol=2e-5, rtol=2e-5)
    # Rotate joint-induced velocities, then add the root's rigid motion. The
    # cross product is essential: velocities refer to link origins, not COMs.
    world_angular = expected.root_ang_vel[:, None] + rotations.quat_rotate(
        root_rot, angular.reshape(-1, 3), True
    ).reshape_as(angular)
    world_linear = (
        expected.root_vel[:, None]
        + rotations.quat_rotate(root_rot, linear.reshape(-1, 3), True).reshape_as(
            linear
        )
        + torch.cross(expected.root_ang_vel[:, None].expand_as(offset), offset, dim=-1)
    )
    torch.testing.assert_close(
        actual.rigid_body_vel, world_linear, atol=3e-5, rtol=3e-5
    )
    torch.testing.assert_close(
        actual.rigid_body_ang_vel, world_angular, atol=3e-5, rtol=3e-5
    )
    exact = NoNoise("Verify reconstruction against physical link state")
    sensors = SensorRuntime(
        SensorConfig(
            joint_encoders=JointEncoderSensor(exact, exact),
            imu=IMUSensor(exact, exact),
            localization=LocalizationSensor(body_name="pelvis", noise=exact),
            base_velocity=BaseVelocitySensor(exact),
        ),
        robot,
        3,
        device,
    )
    estimates = sensors.reset(ids, actual)
    for estimate in (estimates.clean, estimates.noisy):
        for name in ("rigid_body_pos", "rigid_body_vel", "rigid_body_ang_vel"):
            torch.testing.assert_close(
                getattr(estimate, name), getattr(actual, name), atol=3e-5, rtol=3e-5
            )
        # Quaternion sign is arbitrary; compare the represented rotations.
        alignment = (estimate.rigid_body_rot * actual.rigid_body_rot).sum(-1).abs()
        torch.testing.assert_close(
            alignment, torch.ones_like(alignment), atol=3e-5, rtol=3e-5
        )
    snapshot = actual.clone()
    displaced = expected.clone()
    displaced.root_pos[:, 0] += 50.0
    sim.reset_envs(displaced, env_ids=ids)
    sim.step(torch.zeros(3, robot.number_of_actions, device=device))
    sim.reset_envs(ResetState.from_robot_state(snapshot.clone()), env_ids=ids)
    restored = sim.get_robot_state()
    for name in (
        "rigid_body_pos",
        "rigid_body_rot",
        "rigid_body_vel",
        "rigid_body_ang_vel",
    ):
        torch.testing.assert_close(
            getattr(restored, name), getattr(snapshot, name), atol=3e-5, rtol=3e-5
        )
    partial = ResetState.from_robot_state(snapshot[[1]].clone())
    partial.root_pos[:, 0] += 25.0
    sim.reset_envs(partial, env_ids=ids[1:2])
    sim.step(torch.zeros(3, robot.number_of_actions, device=device))
    after = sim.get_robot_state()
    assert (after.root_pos[[0, 2]] - snapshot.root_pos[[0, 2]]).norm(dim=-1).max() < 0.3
    assert (after.root_pos[1] - partial.root_pos[0]).norm() < 0.3
    print(
        "PASS: immediate FK, origin velocity, full restore, restore + partial reset",
        flush=True,
    )
finally:
    sim.close()
    if app is not None:
        app.close()
