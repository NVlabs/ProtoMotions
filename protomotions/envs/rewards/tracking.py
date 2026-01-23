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
"""Tracking reward functions for motion imitation.

Provides reward functions for tracking reference motions, including:
- Standard AMP/DeepMimic-style tracking rewards (gt, gr, gv, gav, rh)
- BeyondMimic-style rewards (global/relative position, orientation, velocity)
"""

import torch
from torch import Tensor
from typing import Optional, TYPE_CHECKING

from protomotions.utils.rotations import (
    quat_angle_diff_norm,
    calc_heading_quat_inv,
    quat_rotate,
    quat_mul,
)
from protomotions.envs.rewards.base import mean_squared_error_exp, rotation_error_exp

if TYPE_CHECKING:
    from protomotions.envs.base_env.config import RewardComponentConfig


# =============================================================================
# Standard Tracking Reward Factories (AMP/DeepMimic style)
# =============================================================================


def gt_rew_factory(weight: float = 0.5, coefficient: float = -100.0) -> "RewardComponentConfig":
    """Factory for position tracking reward.
    
    Args:
        weight: Weight for the reward component.
        coefficient: Exponential coefficient for the error.
    
    Returns:
        Pre-configured RewardComponentConfig.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=mean_squared_error_exp,
        variables={
            "x": "current_state_rigid_body_pos",
            "ref_x": "ref_state_rigid_body_pos",
            "coefficient": coefficient,
        },
        weight=weight,
    )


def gr_rew_factory(weight: float = 0.3, coefficient: float = -5.0) -> "RewardComponentConfig":
    """Factory for rotation tracking reward.
    
    Args:
        weight: Weight for the reward component.
        coefficient: Exponential coefficient for the error.
    
    Returns:
        Pre-configured RewardComponentConfig.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=rotation_error_exp,
        variables={
            "q": "current_state_rigid_body_rot",
            "ref_q": "ref_state_rigid_body_rot",
            "coefficient": coefficient,
        },
        weight=weight,
    )


def gv_rew_factory(weight: float = 0.1, coefficient: float = -0.5) -> "RewardComponentConfig":
    """Factory for velocity tracking reward.
    
    Args:
        weight: Weight for the reward component.
        coefficient: Exponential coefficient for the error.
    
    Returns:
        Pre-configured RewardComponentConfig.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=mean_squared_error_exp,
        variables={
            "x": "current_state_rigid_body_vel",
            "ref_x": "ref_state_rigid_body_vel",
            "coefficient": coefficient,
        },
        weight=weight,
    )


def gav_rew_factory(weight: float = 0.1, coefficient: float = -0.1) -> "RewardComponentConfig":
    """Factory for angular velocity tracking reward.
    
    Args:
        weight: Weight for the reward component.
        coefficient: Exponential coefficient for the error.
    
    Returns:
        Pre-configured RewardComponentConfig.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=mean_squared_error_exp,
        variables={
            "x": "current_state_rigid_body_ang_vel",
            "ref_x": "ref_state_rigid_body_ang_vel",
            "coefficient": coefficient,
        },
        weight=weight,
    )


def rh_rew_factory(weight: float = 0.2, coefficient: float = -100.0) -> "RewardComponentConfig":
    """Factory for root height tracking reward.
    
    Args:
        weight: Weight for the reward component.
        coefficient: Exponential coefficient for the error.
    
    Returns:
        Pre-configured RewardComponentConfig.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=mean_squared_error_exp,
        variables={
            "x": "current_state_root_height",
            "ref_x": "ref_state_root_height",
            "coefficient": coefficient,
        },
        weight=weight,
    )


# =============================================================================
# BeyondMimic-style Reward Functions
# =============================================================================


def global_position_error_exp(
    x: Tensor,
    ref_x: Tensor,
    sigma: float,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Position error: exp(-||x - ref_x||^2 / sigma^2)."""
    if indices is not None and x.dim() == 3:
        x = x[:, indices]
        ref_x = ref_x[:, indices]

    error = (x - ref_x).pow(2).sum(dim=-1)
    if error.dim() == 2:
        error = error.mean(dim=-1)
    return torch.exp(-error / (sigma ** 2))


def global_anchor_pos_rew_factory(weight: float = 0.5, sigma: float = 0.3) -> "RewardComponentConfig":
    """Factory for global anchor position reward."""
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=global_position_error_exp,
        variables={
            "x": "current_state_anchor_pos",
            "ref_x": "ref_state_anchor_pos",
            "sigma": sigma,
        },
        weight=weight,
    )


def global_orientation_error_exp(
    q: Tensor,
    ref_q: Tensor,
    sigma: float,
    indices: Optional[Tensor] = None,
) -> Tensor:
    """Orientation error: exp(-angle_diff^2 / sigma^2)."""
    if indices is not None and q.dim() == 3:
        q = q[:, indices]
        ref_q = ref_q[:, indices]

    error = quat_angle_diff_norm(q, ref_q, w_last=True)
    if error.dim() == 2:
        error = error.mean(dim=-1)
    return torch.exp(-error / (sigma ** 2))


def global_anchor_ori_rew_factory(weight: float = 0.5, sigma: float = 0.4) -> "RewardComponentConfig":
    """Factory for global anchor orientation reward."""
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=global_orientation_error_exp,
        variables={
            "q": "current_state_anchor_rot",
            "ref_q": "ref_state_anchor_rot",
            "sigma": sigma,
        },
        weight=weight,
    )


def relative_body_position_error_exp(
    body_pos: Tensor,
    ref_body_pos: Tensor,
    anchor_pos: Tensor,
    ref_anchor_pos: Tensor,
    anchor_rot: Tensor,
    ref_anchor_rot: Tensor,
    sigma: float,
    indices: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Relative body position error in anchor's yaw-aligned local frame."""
    if indices is not None:
        body_pos = body_pos[:, indices]
        ref_body_pos = ref_body_pos[:, indices]
        if weights is not None:
            weights = weights[indices]
    
    # Get inverse yaw quaternions for both current and reference anchors
    heading_inv = calc_heading_quat_inv(anchor_rot, w_last=True)
    ref_heading_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)
    
    # Compute offsets in world frame
    offset = body_pos - anchor_pos.unsqueeze(1)
    ref_offset = ref_body_pos - ref_anchor_pos.unsqueeze(1)
    
    # Rotate offsets into anchor's local frame (yaw-aligned)
    num_envs, num_bodies, _ = offset.shape
    heading_inv_expanded = heading_inv.unsqueeze(1).expand(-1, num_bodies, -1)
    ref_heading_inv_expanded = ref_heading_inv.unsqueeze(1).expand(-1, num_bodies, -1)
    
    rel_pos = quat_rotate(heading_inv_expanded.reshape(-1, 4), offset.reshape(-1, 3), w_last=True).reshape(num_envs, num_bodies, 3)
    ref_rel_pos = quat_rotate(ref_heading_inv_expanded.reshape(-1, 4), ref_offset.reshape(-1, 3), w_last=True).reshape(num_envs, num_bodies, 3)
    
    per_body_error = (rel_pos - ref_rel_pos).pow(2).sum(dim=-1)
    if weights is not None:
        error = (per_body_error * weights).sum(dim=-1) / weights.sum()
    else:
        error = per_body_error.mean(dim=-1)
    return torch.exp(-error / (sigma ** 2))


def relative_body_pos_rew_factory(
    weight: float = 1.0,
    sigma: float = 0.3,
    indices_subset=None,
    use_density_weights: bool = False,
) -> "RewardComponentConfig":
    """Factory for relative body position reward."""
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=relative_body_position_error_exp,
        variables={
            "body_pos": "current_state_rigid_body_pos",
            "ref_body_pos": "ref_state_rigid_body_pos",
            "anchor_pos": "current_state_anchor_pos",
            "ref_anchor_pos": "ref_state_anchor_pos",
            "anchor_rot": "current_state_anchor_rot",
            "ref_anchor_rot": "ref_state_anchor_rot",
            "sigma": sigma,
        },
        indices_subset=indices_subset,
        weight=weight,
        use_density_weights=use_density_weights,
    )


def relative_body_orientation_error_exp(
    body_rot: Tensor,
    ref_body_rot: Tensor,
    anchor_rot: Tensor,
    ref_anchor_rot: Tensor,
    sigma: float,
    indices: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Relative body orientation error in anchor's yaw-aligned frame."""
    if indices is not None:
        body_rot = body_rot[:, indices]
        ref_body_rot = ref_body_rot[:, indices]
        if weights is not None:
            weights = weights[indices]
    
    # Get inverse yaw quaternions
    heading_inv = calc_heading_quat_inv(anchor_rot, w_last=True)
    ref_heading_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)
    
    num_envs, num_bodies, _ = body_rot.shape
    heading_inv_expanded = heading_inv.unsqueeze(1).expand(-1, num_bodies, -1)
    ref_heading_inv_expanded = ref_heading_inv.unsqueeze(1).expand(-1, num_bodies, -1)
    
    # Compute body rotation relative to anchor's yaw
    rel_rot = quat_mul(heading_inv_expanded.reshape(-1, 4), body_rot.reshape(-1, 4), w_last=True).reshape(body_rot.shape)
    ref_rel_rot = quat_mul(ref_heading_inv_expanded.reshape(-1, 4), ref_body_rot.reshape(-1, 4), w_last=True).reshape(ref_body_rot.shape)
    
    per_body_error = quat_angle_diff_norm(rel_rot, ref_rel_rot, w_last=True)
    if weights is not None:
        error = (per_body_error * weights).sum(dim=-1) / weights.sum()
    else:
        error = per_body_error.mean(dim=-1)
    return torch.exp(-error / (sigma ** 2))


def relative_body_ori_rew_factory(
    weight: float = 1.0,
    sigma: float = 0.4,
    indices_subset=None,
    use_density_weights: bool = False,
) -> "RewardComponentConfig":
    """Factory for relative body orientation reward."""
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=relative_body_orientation_error_exp,
        variables={
            "body_rot": "current_state_rigid_body_rot",
            "ref_body_rot": "ref_state_rigid_body_rot",
            "anchor_rot": "current_state_anchor_rot",
            "ref_anchor_rot": "ref_state_anchor_rot",
            "sigma": sigma,
        },
        indices_subset=indices_subset,
        weight=weight,
        use_density_weights=use_density_weights,
    )


def global_velocity_error_exp(
    x: Tensor,
    ref_x: Tensor,
    sigma: float,
    indices: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Velocity error: exp(-||x - ref_x||^2 / sigma^2)."""
    if indices is not None:
        x = x[:, indices]
        ref_x = ref_x[:, indices]
        if weights is not None:
            weights = weights[indices]

    per_body_error = (x - ref_x).pow(2).sum(dim=-1)
    if weights is not None:
        error = (per_body_error * weights).sum(dim=-1) / weights.sum()
    else:
        error = per_body_error.mean(dim=-1)
    return torch.exp(-error / (sigma ** 2))


def global_body_lin_vel_rew_factory(
    weight: float = 1.0,
    sigma: float = 1.0,
    indices_subset=None,
    use_density_weights: bool = False,
) -> "RewardComponentConfig":
    """Factory for global body linear velocity reward."""
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=global_velocity_error_exp,
        variables={
            "x": "current_state_rigid_body_vel",
            "ref_x": "ref_state_rigid_body_vel",
            "sigma": sigma,
        },
        indices_subset=indices_subset,
        weight=weight,
        use_density_weights=use_density_weights,
    )


def global_body_ang_vel_rew_factory(
    weight: float = 1.0,
    sigma: float = 3.14,
    indices_subset=None,
    use_density_weights: bool = False,
) -> "RewardComponentConfig":
    """Factory for global body angular velocity reward."""
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=global_velocity_error_exp,
        variables={
            "x": "current_state_rigid_body_ang_vel",
            "ref_x": "ref_state_rigid_body_ang_vel",
            "sigma": sigma,
        },
        indices_subset=indices_subset,
        weight=weight,
        use_density_weights=use_density_weights,
    )

