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
"""Base reward primitive functions.

Provides fundamental reward computation functions that can be composed
to create more complex rewards. These are the building blocks used by
tracking, task, and regularization rewards.

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

