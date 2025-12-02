# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Reward utility functions for environments.

Provides common reward computation functions for the RewardComponentConfig system.

Function naming convention:
- Functions ending with `_exp` apply exponential transformation (need coefficient)
- Functions without `_exp` are linear (scaling done via weight in config)
- All functions accept optional `indices` for body/joint subsetting
"""

import torch
from torch import Tensor
from typing import Optional

from protomotions.utils.rotations import quat_angle_diff_norm


# =============================================================================
# Exponential Reward Functions (require coefficient parameter)
# =============================================================================


def mean_squared_error_exp(
    x: Tensor,
    ref_x: Tensor,
    coefficient: float,
    indices: Optional[Tensor] = None,
    mean_before_exp: bool = True,
) -> Tensor:
    """Mean squared error with exponential transformation.

    Computes exp(coefficient * mean(||x - ref_x||^2)).
    Use for position/velocity tracking rewards.

    Args:
        x: Current state [num_envs, num_bodies, dim] or [num_envs, dim] or [num_envs]
        ref_x: Reference state (same shape as x)
        coefficient: Exponential coefficient (typically negative)
        indices: Optional body indices to subset
        mean_before_exp: If True, mean before exp (more stable)

    Returns:
        Reward [num_envs] in range (0, 1] for negative coefficient
    """
    if indices is not None:
        x = x[:, indices]
        ref_x = ref_x[:, indices]

    diff_sq = (x - ref_x).pow(2)

    if diff_sq.dim() == 3:
        per_body = diff_sq.mean(dim=-1)
        if mean_before_exp:
            return per_body.mean(dim=-1).mul(coefficient).exp()
        else:
            return per_body.mul(coefficient).exp().mean(dim=-1)
    elif diff_sq.dim() == 2:
        return diff_sq.mean(dim=-1).mul(coefficient).exp()
    else:
        return diff_sq.mul(coefficient).exp()


def rotation_error_exp(
    q: Tensor,
    ref_q: Tensor,
    coefficient: float,
    indices: Optional[Tensor] = None,
    mean_before_exp: bool = True,
) -> Tensor:
    """Quaternion rotation error with exponential transformation.

    Computes exp(coefficient * mean(angle_diff^2)).
    Use for rotation tracking rewards.

    Args:
        q: Current quaternions [num_envs, num_bodies, 4]
        ref_q: Reference quaternions (same shape)
        coefficient: Exponential coefficient (typically negative)
        indices: Optional body indices to subset
        mean_before_exp: If True, mean before exp (more stable)

    Returns:
        Reward [num_envs] in range (0, 1] for negative coefficient
    """
    if indices is not None:
        q = q[:, indices]
        ref_q = ref_q[:, indices]

    angle_diff_sq = quat_angle_diff_norm(q, ref_q, w_last=True)

    if mean_before_exp:
        return angle_diff_sq.mean(dim=-1).mul(coefficient).exp()
    else:
        return angle_diff_sq.mul(coefficient).exp().mean(dim=-1)


def power_consumption_exp(
    dof_forces: Tensor,
    dof_vel: Tensor,
    coefficient: float,
    use_torque_squared: bool = False,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Power/energy consumption with exponential transformation.

    Computes exp(coefficient * sum(|force * vel|)).
    Use for energy efficiency rewards.

    Args:
        dof_forces: Joint torques [num_envs, num_dofs]
        dof_vel: Joint velocities [num_envs, num_dofs]
        coefficient: Exponential coefficient (typically negative)
        use_torque_squared: If True, use torque^2 instead of torque*vel
        indices: Optional DOF indices to subset

    Returns:
        Reward [num_envs]
    """
    if indices is not None:
        dof_forces = dof_forces[:, indices]
        dof_vel = dof_vel[:, indices]

    if use_torque_squared:
        power_val = torch.abs(dof_forces * dof_forces).sum(dim=-1)
    else:
        power_val = torch.abs(dof_forces * dof_vel).sum(dim=-1)

    return power_val.mul(coefficient).exp()


# =============================================================================
# Linear Reward/Penalty Functions (no coefficient, scaling via weight)
# =============================================================================


def mean_squared_error(
    x: Tensor,
    ref_x: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Mean squared error between tensors (linear).

    Computes mean(||x - ref_x||^2).

    Args:
        x: Current state [num_envs, num_bodies, dim] or [num_envs, dim]
        ref_x: Reference state (same shape)
        indices: Optional body indices to subset

    Returns:
        MSE error [num_envs]
    """
    if indices is not None:
        x = x[:, indices]
        ref_x = ref_x[:, indices]

    diff_sq = (x - ref_x).pow(2)

    if diff_sq.dim() == 3:
        return diff_sq.mean(dim=[1, 2])
    elif diff_sq.dim() == 2:
        return diff_sq.mean(dim=-1)
    else:
        return diff_sq


def norm(x: Tensor, indices: Optional[Tensor] = None) -> Tensor:
    """L2 norm of a tensor.

    Computes ||x||_2.

    Args:
        x: Tensor [num_envs, num_bodies, dim] or [num_envs, dim]
        indices: Optional body indices to subset
    """
    if indices is not None:
        x = x[:, indices]

    return torch.norm(x, dim=-1)


def rotation_error(
    q: Tensor,
    ref_q: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Quaternion rotation error (linear).

    Computes mean(angle_diff^2).

    Args:
        q: Current quaternions [num_envs, num_bodies, 4]
        ref_q: Reference quaternions (same shape)
        indices: Optional body indices to subset

    Returns:
        Rotation error [num_envs]
    """
    if indices is not None:
        q = q[:, indices]
        ref_q = ref_q[:, indices]

    angle_diff_sq = quat_angle_diff_norm(q, ref_q, w_last=True)
    return angle_diff_sq.mean(dim=-1)


def absolute_difference_sum(
    x: Tensor,
    ref_x: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum of absolute differences (L1 distance).

    Computes sum(|x - ref_x|).

    Args:
        x: Current state [num_envs, num_bodies, dim] or [num_envs, dim]
        ref_x: Reference state (same shape)
        indices: Optional body indices to subset

    Returns:
        L1 distance [num_envs]
    """
    if indices is not None:
        x = x[:, indices]
        ref_x = ref_x[:, indices]

    diff_abs = torch.abs(x - ref_x)

    if diff_abs.dim() == 3:
        return diff_abs.sum(dim=[1, 2])
    elif diff_abs.dim() == 2:
        return diff_abs.sum(dim=1)
    else:
        return diff_abs


def power_consumption_sum(
    dof_forces: Tensor,
    dof_vel: Tensor,
    use_torque_squared: bool = False,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum of power/energy consumption (linear).

    Computes sum(|force * vel|) or sum(force^2).

    Args:
        dof_forces: Joint torques [num_envs, num_dofs]
        dof_vel: Joint velocities [num_envs, num_dofs]
        use_torque_squared: If True, use torque^2 instead of torque*vel
        indices: Optional DOF indices to subset

    Returns:
        Power consumption [num_envs]
    """
    if indices is not None:
        dof_forces = dof_forces[:, indices]
        dof_vel = dof_vel[:, indices]

    if use_torque_squared:
        return torch.abs(dof_forces * dof_forces).sum(dim=-1)
    else:
        return torch.abs(dof_forces * dof_vel).sum(dim=-1)


def joint_limit_violation(
    dof_pos: Tensor,
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum of joint position limit violations.

    Penalizes positions outside [lower, upper] limits.

    Args:
        dof_pos: Joint positions [num_envs, num_dofs]
        dof_limits_lower: Lower limits [num_dofs]
        dof_limits_upper: Upper limits [num_dofs]
        indices: Optional DOF indices to subset

    Returns:
        Total violation [num_envs]
    """
    if indices is not None:
        dof_pos = dof_pos[:, indices]
        dof_limits_lower = dof_limits_lower[indices]
        dof_limits_upper = dof_limits_upper[indices]

    below_lower = -(dof_pos - dof_limits_lower).clip(max=0.0)
    above_upper = (dof_pos - dof_limits_upper).clip(min=0.0)
    return torch.sum(below_lower + above_upper, dim=1)


def contact_mismatch_sum(
    sim_contacts: Tensor,
    ref_contacts: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum of contact state mismatches.

    Computes sum(|sim_contacts - ref_contacts|).

    Args:
        sim_contacts: Simulated contacts [num_envs, num_bodies]
        ref_contacts: Reference contacts [num_envs, num_bodies]
        indices: Optional body indices to subset

    Returns:
        Total mismatch [num_envs]
    """
    if indices is not None:
        sim_contacts = sim_contacts[:, indices]
        ref_contacts = ref_contacts[:, indices]

    return torch.abs(sim_contacts.float() - ref_contacts.float()).sum(dim=1)


def impact_force_penalty(
    current_forces: Tensor,
    previous_forces: Tensor,
    indices: Optional[Tensor] = None,
    threshold: float = 30.0,
) -> Tensor:
    """Sum of sudden contact force changes above a threshold (impact penalty).

    Penalizes abrupt force changes (both increases and decreases) that exceed
    the threshold. Small force changes below the threshold are ignored.

    Args:
        current_forces: Current contact forces [num_envs, num_bodies]
        previous_forces: Previous contact forces [num_envs, num_bodies]
        indices: Optional body indices to subset
        threshold: Force change threshold below which changes are ignored (default: 30.0)

    Returns:
        Total force change above threshold [num_envs]
    """
    if indices is not None:
        current_forces = current_forces[:, indices]
        previous_forces = previous_forces[:, indices]

    force_changes = torch.abs(current_forces - previous_forces)
    force_changes = torch.clamp(force_changes - threshold, min=0)
    return force_changes.sum(dim=-1)


def velocity_squared_sum(
    velocity: Tensor,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Sum of squared velocities.

    Computes sum(velocity^2).

    Args:
        velocity: Velocity [num_envs, num_bodies, 3] or [num_envs, dim]
        indices: Optional indices to subset

    Returns:
        Velocity magnitude penalty [num_envs]
    """
    if indices is not None:
        velocity = velocity[:, indices]

    if velocity.dim() == 3:
        return velocity.pow(2).sum(dim=[1, 2])
    else:
        return velocity.pow(2).sum(dim=-1)


def soft_pos_limit_reward(dof_pos, dof_limits_lower, dof_limits_upper):
    out_of_limits = -(dof_pos - dof_limits_lower).clip(max=0.0)
    out_of_limits += (dof_pos - dof_limits_upper).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# =============================================================================
# Task-Specific Reward Functions
# =============================================================================


def heading_velocity_reward(
    root_pos: Tensor,
    prev_root_pos: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
    dt: float,
) -> Tensor:
    """Reward for moving in target direction at target speed.

    Computes exponential reward based on:
    - Error between target and actual velocity in target direction
    - Penalty for velocity tangent to target direction

    Args:
        root_pos: Current root position [num_envs, 3]
        prev_root_pos: Previous root position [num_envs, 3]
        tar_dir: Target direction [num_envs, 2]
        tar_speed: Target speed [num_envs]
        dt: Simulation timestep

    Returns:
        Reward [num_envs] in range [0, 1]
    """
    vel_err_scale = 0.25
    tangent_err_w = 0.1

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.sum(tangent_vel, dim=-1)

    tar_vel_err = tar_speed - tar_dir_speed
    tangent_vel_err = tangent_speed
    dir_reward = torch.exp(
        -vel_err_scale
        * (
            tar_vel_err * tar_vel_err
            + tangent_err_w * tangent_vel_err * tangent_vel_err
        )
    )

    speed_mask = tar_dir_speed < -0.5
    dir_reward[speed_mask] = 0

    return dir_reward


def path_following_reward(
    head_pos: Tensor,
    tar_pos: Tensor,
    height_conditioned: bool,
    pos_err_scale: float = 2.0,
    height_err_scale: float = 10.0,
) -> Tensor:
    """Reward for following a path (staying close to target position).

    Computes exponential reward based on:
    - Horizontal distance to target position
    - Optionally: vertical distance to target position

    Args:
        head_pos: Current head position [num_envs, 3] (ground-relative)
        tar_pos: Target position from path [num_envs, 3] (ground-relative)
        height_conditioned: Whether to include height in reward
        pos_err_scale: Coefficient for position error
        height_err_scale: Coefficient for height error

    Returns:
        Reward [num_envs] in range [0, 1]
    """
    pos_diff = tar_pos[..., 0:2] - head_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    height_diff = tar_pos[..., 2] - head_pos[..., 2]
    height_err = height_diff * height_diff

    pos_reward = torch.exp(-pos_err_scale * pos_err)
    height_reward = torch.exp(-height_err_scale * height_err)

    if height_conditioned:
        # Multiplicative reward ensures both terms are properly met.
        reward = pos_reward * height_reward
    else:
        reward = pos_reward

    return reward
