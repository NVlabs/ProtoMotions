# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""World body state reconstructed from sensor readings and the kinematic tree.

Joint encoders give every body's pose relative to the root; the IMU supplies
the root's world orientation and angular velocity; localization and the
optional base-velocity estimate supply the translation and its rate. The same
function serves ideal and perturbed readings, and anything the readings do not
carry is left ``None`` rather than read from the simulator.
"""

from __future__ import annotations

import torch
from torch import Tensor

from protomotions.components.pose_lib import matrix_to_quaternion
from protomotions.envs.sensors.config import SensorMounts
from protomotions.envs.sensors.measurement import SensorReadings
from protomotions.envs.state import BodyState
from protomotions.utils import rotations as rot
from protomotions.utils.baked_fk import BakedAnchorFK


def estimate_body_state(
    readings: SensorReadings, fk: BakedAnchorFK, mounts: SensorMounts
) -> BodyState:
    """Reconstruct world poses and velocities of every body from the readings.

    With root-frame FK giving body positions p_i, rotations R_i, angular
    velocities w_i and origin velocities v_i (root fixed), sensor orientation
    Q, mount M (sensor axes to IMU-body axes) and gyro g:

        R       = Q M^T R_imu^T              world-from-root rotation
        w_root  = R_imu M g - w_imu          root angular velocity, root axes
        rot_i   = R R_i
        omega_i = R (w_root + w_i)
        pos_i   = L + R [p_i - (p_l + R_l o)]       localization L at body l, offset o
        vel_i   = V + R [v_i + w_root x p_i]         base velocity V of the root origin
    """
    lead = readings.joint_pos.shape[:-1]
    flat = readings.map(lambda t: t.reshape(-1, t.shape[-1]))
    p, R_i, w_i, v_i = fk.body_kinematics(flat.joint_pos, flat.joint_vel)

    mount = rot.quaternion_to_matrix(
        flat.imu_quat.new_tensor(mounts.imu_mount_quat), w_last=True
    )
    Q = rot.quaternion_to_matrix(flat.imu_quat, w_last=True)
    R = Q @ mount.T @ R_i[:, mounts.imu_body].transpose(-1, -2)
    w_root = (R_i[:, mounts.imu_body] @ (mount @ flat.imu_gyro[..., None])).squeeze(
        -1
    ) - w_i[:, mounts.imu_body]

    body_rot = R[:, None] @ R_i
    body_omega = (R[:, None] @ (w_root[:, None] + w_i)[..., None]).squeeze(-1)

    body_pos = None
    if flat.localization is not None:
        o = p.new_tensor(mounts.localization_offset)
        sensor_point = p[:, mounts.localization_body] + (
            R_i[:, mounts.localization_body] @ o[:, None]
        ).squeeze(-1)
        body_pos = flat.localization[:, None] + (
            R[:, None] @ (p - sensor_point[:, None])[..., None]
        ).squeeze(-1)

    body_vel = None
    if flat.base_velocity is not None:
        relative = v_i + torch.cross(w_root[:, None].expand_as(p), p, dim=-1)
        body_vel = flat.base_velocity[:, None] + (
            R[:, None] @ relative[..., None]
        ).squeeze(-1)

    ground = None
    if body_pos is not None and flat.height is not None:
        ground = body_pos[:, 0, 2] - flat.height[:, 0]

    num_bodies = p.shape[1]

    def restore_shape(t, *trailing):
        return None if t is None else t.reshape(*lead, *trailing)

    return BodyState(
        rigid_body_rot=matrix_to_quaternion(
            body_rot.reshape(-1, 3, 3), w_last=True
        ).reshape(*lead, num_bodies, 4),
        rigid_body_ang_vel=restore_shape(body_omega, num_bodies, 3),
        dof_pos=readings.joint_pos,
        dof_vel=readings.joint_vel,
        rigid_body_pos=restore_shape(body_pos, num_bodies, 3),
        rigid_body_vel=restore_shape(body_vel, num_bodies, 3),
        ground_heights=restore_shape(ground),
    )


def with_episode_odometry(
    state: BodyState, anchor_idx: int, start_xy: Tensor, start_heading_inv: Tensor
) -> BodyState:
    """Attach this source's odometer reading: anchor XY displacement from its own episode start.

    The displacement is expressed in the start heading frame, as a real odometer
    reports it (zero at the start, x along the robot's initial facing). The
    estimate is not modified; a new record carries the three odometry fields.
    """
    delta = state.rigid_body_pos[:, anchor_idx].clone()
    delta[:, :2] -= start_xy
    delta[:, 2] = 0.0
    displacement = rot.quat_rotate(start_heading_inv, delta, w_last=True)[:, :2]
    return BodyState(
        **{name: value for name, value in state.tensors()},
        odom_start_xy=start_xy,
        odom_start_heading_inv=start_heading_inv,
        odom_disp_start=displacement,
    )
