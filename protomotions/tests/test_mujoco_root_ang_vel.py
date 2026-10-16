# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MuJoCo backend: root angular velocity is world-frame on reset, read and push.

A free joint's qvel[3:6] is angular velocity in the BODY frame, while ProtoMotions'
root_ang_vel is world-frame. Writing the world vector straight into qvel[3:6] spins a
robot that was reset into a turned or tilted motion frame about the wrong axis. These
tests build the real simulator from the shipped SMPL assets, which use primitive geoms
and so load without Git LFS payloads, and check the pelvis spin with MuJoCo's own
world-oriented velocity.
"""

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")

from protomotions.components.scene_lib import SceneLib  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402
from protomotions.simulator.base_simulator.simulator_state import (  # noqa: E402
    ResetState,
    StateConversion,
)
from protomotions.simulator.mujoco.config import MujocoSimulatorConfig  # noqa: E402
from protomotions.simulator.mujoco.simulator import MujocoSimulator  # noqa: E402

# Tilted and turned well away from upright (xyzw), so any frame mix-up shows.
ROOT_ROT = torch.nn.functional.normalize(
    torch.tensor([[0.3, -0.25, 0.8, 0.45]]), dim=-1
)
ROOT_ANG_VEL = torch.tensor([[0.9, -0.4, 1.3]])  # world frame, rad/s


def _make_sim():
    rc = robot_config("smpl")
    config = MujocoSimulatorConfig(
        num_envs=1,
        headless=True,
        experiment_name="test_mujoco_root_ang_vel",
        sim=rc.simulation_params.mujoco,
    )
    sim = MujocoSimulator(
        config=config,
        robot_config=rc,
        terrain=None,
        device=torch.device("cpu"),
        scene_lib=SceneLib.empty(num_envs=1, device="cpu", terrain=None),
    )
    sim._initialize_with_markers({})
    return sim


def _reset(sim, root_ang_vel):
    n = sim._num_actuated_dofs
    sim.reset_envs(
        ResetState(
            root_pos=torch.tensor([[0.0, 0.0, 1.0]]),
            root_rot=ROOT_ROT.clone(),
            root_vel=torch.zeros(1, 3),
            root_ang_vel=root_ang_vel.clone(),
            dof_pos=torch.zeros(1, n),
            dof_vel=torch.zeros(1, n),
            state_conversion=StateConversion.COMMON,
        )
    )


def _pelvis_world_ang_vel(sim):
    """Pelvis angular velocity in world orientation, as MuJoCo itself computes it."""
    mujoco.mj_forward(sim.model, sim.data)
    assert sim.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    body = sim.model.jnt_bodyid[0]  # the root body, which the free joint moves
    velocity = np.zeros(6)  # [angular, linear]
    mujoco.mj_objectVelocity(
        sim.model, sim.data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0
    )
    return velocity[:3]


def test_reset_spins_the_pelvis_at_the_requested_world_angular_velocity():
    sim = _make_sim()

    _reset(sim, ROOT_ANG_VEL)

    np.testing.assert_allclose(
        _pelvis_world_ang_vel(sim), ROOT_ANG_VEL[0].numpy(), atol=1e-5
    )


def test_root_angular_velocity_reads_back_in_the_world_frame():
    sim = _make_sim()

    _reset(sim, ROOT_ANG_VEL)

    read_back = sim.get_root_state().root_ang_vel[0]
    np.testing.assert_allclose(read_back.numpy(), ROOT_ANG_VEL[0].numpy(), atol=1e-5)


def test_angular_push_is_applied_in_the_world_frame():
    sim = _make_sim()
    _reset(sim, torch.zeros(1, 3))

    sim._apply_root_velocity_impulse(
        torch.zeros(1, 3), ROOT_ANG_VEL.clone(), torch.tensor([0])
    )

    np.testing.assert_allclose(
        _pelvis_world_ang_vel(sim), ROOT_ANG_VEL[0].numpy(), atol=1e-5
    )
