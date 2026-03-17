# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
"""Tracking reward compute kernels for motion imitation.

Pure tensor functions (kernels) for computing tracking rewards.
Use MdpComponent in experiment configs to bind kernels to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.tracking import compute_gt_rew
    
    reward_components = {
        "gt_rew": MdpComponent(
            compute_func=compute_gt_rew,
            dynamic_vars={
                "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
                "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            },
            static_params={"coefficient": -100.0},
        ),
    }

Includes:
- Standard AMP/DeepMimic-style tracking rewards (gt, gr, gv, gav, rh)
- BeyondMimic-style rewards (global/relative position, orientation, velocity)
"""

import torch
from torch import Tensor
from typing import Optional

from protomotions.utils.rotations import (
    quat_angle_diff_norm,
    calc_heading_quat_inv,
    quat_rotate,
    quat_mul,
)
from protomotions.envs.rewards.base import mean_squared_error_exp, rotation_error_exp


# =============================================================================
# Standard Tracking Reward Kernels
# =============================================================================

def compute_gt_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    coefficient: float = -100.0,
) -> Tensor:
    """Position tracking reward (exponential MSE).
    
    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return mean_squared_error_exp(
        current_rigid_body_pos,
        ref_rigid_body_pos,
        coefficient,
    )


def compute_gr_rew(
    current_rigid_body_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    coefficient: float = -5.0,
) -> Tensor:
    """Rotation tracking reward (exponential quaternion error).
    
    Args:
        current_rigid_body_rot: Current body rotations [num_envs, num_bodies, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return rotation_error_exp(
        current_rigid_body_rot,
        ref_rigid_body_rot,
        coefficient,
    )


def compute_gv_rew(
    current_rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    coefficient: float = -0.5,
) -> Tensor:
    """Velocity tracking reward (exponential MSE).
    
    Args:
        current_rigid_body_vel: Current body velocities [num_envs, num_bodies, 3].
        ref_rigid_body_vel: Reference body velocities [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return mean_squared_error_exp(
        current_rigid_body_vel,
        ref_rigid_body_vel,
        coefficient,
    )


def compute_gav_rew(
    current_rigid_body_ang_vel: Tensor,
    ref_rigid_body_ang_vel: Tensor,
    coefficient: float = -0.1,
) -> Tensor:
    """Angular velocity tracking reward (exponential MSE).
    
    Args:
        current_rigid_body_ang_vel: Current angular velocities [num_envs, num_bodies, 3].
        ref_rigid_body_ang_vel: Reference angular velocities [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    return mean_squared_error_exp(
        current_rigid_body_ang_vel,
        ref_rigid_body_ang_vel,
        coefficient,
    )


def compute_rh_rew(
    current_root_height: Tensor,
    ref_rigid_body_pos: Tensor,
    coefficient: float = -100.0,
) -> Tensor:
    """Root height tracking reward (exponential MSE).
    
    Args:
        current_root_height: Current root height [num_envs] or [num_envs, 1].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        coefficient: Exponential coefficient for error.
    
    Returns:
        Reward tensor [num_envs].
    """
    # Extract reference root height (z-coordinate of root body)
    ref_root_height = ref_rigid_body_pos[:, 0, 2]
    
    return mean_squared_error_exp(
        current_root_height,
        ref_root_height,
        coefficient,
    )


# =============================================================================
# BeyondMimic-style Reward Kernels
# =============================================================================

def compute_global_position_error_exp(
    x: Tensor,
    ref_x: Tensor,
    sigma: float,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Position error: exp(-||x - ref_x||^2 / sigma^2).
    
    Args:
        x: Current positions [num_envs, num_bodies, 3] or [num_envs, 3].
        ref_x: Reference positions (same shape as x).
        sigma: Gaussian kernel width.
        indices: Optional body indices to select [num_bodies_subset].
    
    Returns:
        Reward tensor [num_envs].
    """
    if indices is not None and x.dim() == 3:
        x = x[:, indices]
        ref_x = ref_x[:, indices]

    error = (x - ref_x).pow(2).sum(dim=-1)
    if error.dim() == 2:
        error = error.mean(dim=-1)
    return torch.exp(-error / (sigma ** 2))


def compute_global_anchor_pos_rew(
    current_anchor_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    anchor_idx: int,
    sigma: float = 0.3,
) -> Tensor:
    """Global anchor position reward (BeyondMimic style).
    
    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
    
    Returns:
        Reward: exp(-||anchor_pos - ref_anchor_pos||^2 / sigma^2).
    """
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    return compute_global_position_error_exp(current_anchor_pos, ref_anchor_pos, sigma)


def compute_global_orientation_error_exp(
    q: Tensor,
    ref_q: Tensor,
    sigma: float,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Orientation error: exp(-angle_diff^2 / sigma^2).
    
    Args:
        q: Current orientations [num_envs, num_bodies, 4] or [num_envs, 4] (w-last).
        ref_q: Reference orientations (same shape as q).
        sigma: Gaussian kernel width.
        indices: Optional body indices to select [num_bodies_subset].
    
    Returns:
        Reward tensor [num_envs].
    """
    if indices is not None and q.dim() == 3:
        q = q[:, indices]
        ref_q = ref_q[:, indices]

    error = quat_angle_diff_norm(q, ref_q, w_last=True)
    if error.dim() == 2:
        error = error.mean(dim=-1)
    return torch.exp(-error / (sigma ** 2))


def compute_global_anchor_ori_rew(
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    anchor_idx: int,
    sigma: float = 0.4,
) -> Tensor:
    """Global anchor orientation reward (BeyondMimic style).
    
    Args:
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
    
    Returns:
        Reward: exp(-angle_diff^2 / sigma^2).
    """
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    return compute_global_orientation_error_exp(current_anchor_rot, ref_anchor_rot, sigma)


def compute_relative_body_pos_rew(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    current_anchor_pos: Tensor,
    anchor_idx: int,
    sigma: float = 0.3,
    body_indices: Optional[Tensor] = None,
) -> Tensor:
    """Relative body position reward (BeyondMimic style).
    
    Computes reward based on body positions relative to anchor in anchor's local frame.
    
    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        current_anchor_pos: Current anchor position [num_envs, 3].
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
        body_indices: Optional body indices to select [num_bodies_subset].
    
    Returns:
        Reward: exp(-||rel_pos - ref_rel_pos||^2 / sigma^2).
    """
    # Extract reference anchor pos and rot
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    
    # Compute heading rotations (yaw-only)
    current_heading_rot_inv = calc_heading_quat_inv(current_anchor_rot, w_last=True)
    ref_heading_rot_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)
    
    # Compute relative positions in world frame
    current_rel_pos = current_rigid_body_pos - current_anchor_pos.unsqueeze(1)
    ref_rel_pos = ref_rigid_body_pos - ref_anchor_pos.unsqueeze(1)
    
    # Rotate to anchor's local frame
    current_rel_pos_flat = current_rel_pos.reshape(-1, 3)
    current_heading_rot_inv_exp = current_heading_rot_inv.unsqueeze(1).expand(
        -1, current_rigid_body_pos.shape[1], -1
    ).reshape(-1, 4)
    current_rel_pos_local = quat_rotate(
        current_heading_rot_inv_exp, current_rel_pos_flat, w_last=True
    ).reshape(current_rigid_body_pos.shape)
    
    ref_rel_pos_flat = ref_rel_pos.reshape(-1, 3)
    ref_heading_rot_inv_exp = ref_heading_rot_inv.unsqueeze(1).expand(
        -1, ref_rigid_body_pos.shape[1], -1
    ).reshape(-1, 4)
    ref_rel_pos_local = quat_rotate(
        ref_heading_rot_inv_exp, ref_rel_pos_flat, w_last=True
    ).reshape(ref_rigid_body_pos.shape)
    
    return compute_global_position_error_exp(
        current_rel_pos_local, ref_rel_pos_local, sigma, body_indices
    )


def compute_relative_body_ori_rew(
    current_rigid_body_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    current_anchor_rot: Tensor,
    anchor_idx: int,
    sigma: float = 0.4,
    body_indices: Optional[Tensor] = None,
) -> Tensor:
    """Relative body orientation reward (BeyondMimic style).
    
    Computes reward based on body orientations relative to anchor.
    
    Args:
        current_rigid_body_rot: Current body rotations [num_envs, num_bodies, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        anchor_idx: Index of anchor body.
        sigma: Gaussian kernel width.
        body_indices: Optional body indices to select [num_bodies_subset].
    
    Returns:
        Reward: exp(-angle_diff^2 / sigma^2).
    """
    # Extract reference anchor rotation
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    
    # Compute heading rotations (yaw-only)
    current_heading_rot_inv = calc_heading_quat_inv(current_anchor_rot, w_last=True)
    ref_heading_rot_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)
    
    # Compute relative rotations
    current_heading_rot_inv_exp = current_heading_rot_inv.unsqueeze(1).expand(
        -1, current_rigid_body_rot.shape[1], -1
    )
    current_rel_rot = quat_mul(current_heading_rot_inv_exp, current_rigid_body_rot, w_last=True)
    
    ref_heading_rot_inv_exp = ref_heading_rot_inv.unsqueeze(1).expand(
        -1, ref_rigid_body_rot.shape[1], -1
    )
    ref_rel_rot = quat_mul(ref_heading_rot_inv_exp, ref_rigid_body_rot, w_last=True)
    
    return compute_global_orientation_error_exp(
        current_rel_rot, ref_rel_rot, sigma, body_indices
    )


def compute_global_body_lin_vel_rew(
    current_rigid_body_vel: Tensor,
    ref_rigid_body_vel: Tensor,
    sigma: float = 1.0,
) -> Tensor:
    """Global body linear velocity reward (BeyondMimic style).
    
    Args:
        current_rigid_body_vel: Current body velocities [num_envs, num_bodies, 3].
        ref_rigid_body_vel: Reference body velocities [num_envs, num_bodies, 3].
        sigma: Gaussian kernel width.
    
    Returns:
        Reward: exp(-||vel - ref_vel||^2 / sigma^2).
    """
    return compute_global_position_error_exp(current_rigid_body_vel, ref_rigid_body_vel, sigma)


def compute_global_body_ang_vel_rew(
    current_rigid_body_ang_vel: Tensor,
    ref_rigid_body_ang_vel: Tensor,
    sigma: float = 3.14,
) -> Tensor:
    """Global body angular velocity reward (BeyondMimic style).
    
    Args:
        current_rigid_body_ang_vel: Current angular velocities [num_envs, num_bodies, 3].
        ref_rigid_body_ang_vel: Reference angular velocities [num_envs, num_bodies, 3].
        sigma: Gaussian kernel width.
    
    Returns:
        Reward: exp(-||ang_vel - ref_ang_vel||^2 / sigma^2).
    """
    return compute_global_position_error_exp(
        current_rigid_body_ang_vel, ref_rigid_body_ang_vel, sigma
    )


__all__ = [
    # Standard tracking rewards
    "compute_gt_rew",
    "compute_gr_rew",
    "compute_gv_rew",
    "compute_gav_rew",
    "compute_rh_rew",
    # BeyondMimic-style rewards
    "compute_global_position_error_exp",
    "compute_global_anchor_pos_rew",
    "compute_global_orientation_error_exp",
    "compute_global_anchor_ori_rew",
    "compute_relative_body_pos_rew",
    "compute_relative_body_ori_rew",
    "compute_global_body_lin_vel_rew",
    "compute_global_body_ang_vel_rew",
]
