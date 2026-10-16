# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Body velocities must match derivatives of the returned world-frame poses."""

import mujoco
import numpy as np
import pytest
import torch

from protomotions.simulator.base_simulator.simulator_state import StateConversion
from protomotions.simulator.mujoco.simulator import MujocoSimulator


XML = """
<mujoco>
  <worldbody>
    <body name="root" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .2 .1" pos=".07 0 .03" mass="5"/>
      <body name="left_foot" pos="0 .2 -.4">
        <joint name="left_hip" axis="0 1 0" pos="0 0 .3"/>
        <geom type="box" size=".1 .05 .05" pos=".04 0 -.02" mass="1"/>
      </body>
      <body name="right_foot" pos="0 -.2 -.4">
        <joint name="right_hip" axis="0 1 0" pos="0 0 .3"/>
        <geom type="box" size=".1 .05 .05" pos=".04 0 -.02" mass="1"/>
      </body>
    </body>
    <body name="scene_object" pos="2 0 1">
      <freejoint/>
      <geom type="sphere" size=".1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize("motion", ["still", "translation", "rotation", "left", "right", "combined"])
def test_body_velocities_match_pose_derivatives(motion):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    # A rotated base distinguishes world-frame from local-frame velocities.
    data.qpos[3:7] = [np.cos(.4), 0, 0, np.sin(.4)]
    data.qpos[7:9] = [.3, -.2]
    if motion in {"translation", "combined"}:
        data.qvel[:3] = [.7, -.2, .1]
    if motion in {"rotation", "combined"}:
        data.qvel[3:6] = [.4, -.3, .8]
    if motion in {"left", "combined"}:
        data.qvel[6] = 1.2
    if motion in {"right", "combined"}:
        data.qvel[7] = -.9
    mujoco.mj_forward(model, data)

    simulator = object.__new__(MujocoSimulator)
    simulator.model = model
    simulator.data = data
    simulator._num_robot_bodies = 3
    state = simulator._get_simulator_bodies_state()

    epsilon = 1e-7
    future = mujoco.MjData(model)
    future.qpos[:] = data.qpos
    mujoco.mj_integratePos(model, future.qpos, data.qvel, epsilon)
    mujoco.mj_forward(model, future)
    expected_linear = (future.xpos[1:4] - data.xpos[1:4]) / epsilon
    rotation = data.xmat[1:4].reshape(-1, 3, 3)
    next_rotation = future.xmat[1:4].reshape(-1, 3, 3)
    delta = next_rotation @ rotation.transpose(0, 2, 1)
    expected_angular = np.stack(
        [delta[:, 2, 1] - delta[:, 1, 2],
         delta[:, 0, 2] - delta[:, 2, 0],
         delta[:, 1, 0] - delta[:, 0, 1]], axis=-1
    ) / (2 * epsilon)

    assert state.rigid_body_vel.shape == (1, 3, 3)  # No world/scene bodies.
    assert state.rigid_body_vel.dtype == torch.float32
    assert state.state_conversion == StateConversion.SIMULATOR
    np.testing.assert_allclose(state.rigid_body_pos[0], data.xpos[1:4], atol=1e-7)
    np.testing.assert_allclose(state.rigid_body_rot[0], data.xquat[1:4], atol=1e-7)
    np.testing.assert_allclose(state.rigid_body_vel[0], expected_linear, atol=2e-6)
    np.testing.assert_allclose(state.rigid_body_ang_vel[0], expected_angular, atol=2e-6)
    if motion == "left":
        assert np.linalg.norm(state.rigid_body_vel[0, 1]) > .1
        np.testing.assert_array_equal(state.rigid_body_vel[0, 2], 0)
    if motion == "right":
        assert np.linalg.norm(state.rigid_body_vel[0, 2]) > .1
        np.testing.assert_array_equal(state.rigid_body_vel[0, 1], 0)
