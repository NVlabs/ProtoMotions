# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


# =============================================================================
# Linear Reward/Penalty Functions (no coefficient, scaling via weight)
# =============================================================================


def delta_norm(
    a: Tensor, b: Tensor, indices: Optional[Tensor] = None
) -> Tensor:
    """L2 norm of difference between two tensors.

    Computes ||a - b||_2.

    Args:
        a: First tensor [num_envs, dim] or [num_envs, num_bodies, dim]
        b: Second tensor (same shape as a)
        indices: Optional body indices to subset
    """
    diff = a - b
    if indices is not None:
        diff = diff[:, indices]
    return torch.norm(diff, dim=-1)


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

