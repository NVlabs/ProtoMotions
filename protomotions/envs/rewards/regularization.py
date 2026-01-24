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
"""Regularization reward functions.

Provides reward/penalty functions for regularization:
- Action smoothness (normalized and physical units)
- Power consumption
- Joint limit violations
- Contact matching
"""

import torch
from torch import Tensor
from typing import Optional, TYPE_CHECKING

from protomotions.envs.rewards.base import norm, power_consumption_sum

if TYPE_CHECKING:
    from protomotions.envs.base_env.config import RewardComponentConfig


# =============================================================================
# Action Smoothness Rewards
# =============================================================================


def action_smoothness_factory(weight: float = -0.02) -> "RewardComponentConfig":
    """Factory for action smoothness reward (normalized action space).
    
    Computes L2 norm of action changes in normalized [-1, 1] space.
    Note: This does NOT account for different joint ranges.
    
    Args:
        weight: Weight for the reward component.
    
    Returns:
        Pre-configured RewardComponentConfig.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=norm,
        variables={"x": "current_actions - previous_actions"},
        weight=weight,
    )


def physical_action_rate(
    action_delta: Tensor,
    pd_action_scale: Tensor,
) -> Tensor:
    """Compute action rate in physical units (radians per step).
    
    Converts normalized action delta to physical radians by multiplying
    by the per-joint action scale, then computes L2 norm.
    
    Args:
        action_delta: Action change in normalized space [num_envs, num_dofs].
        pd_action_scale: Per-joint scale factor [num_dofs].
    
    Returns:
        L2 norm of physical action change in radians [num_envs].
    """
    # Convert normalized action delta to radians
    physical_delta = action_delta * pd_action_scale
    return torch.norm(physical_delta, dim=-1)


def action_smoothness_physical_factory(weight: float = -0.02) -> "RewardComponentConfig":
    """Factory for action smoothness reward in physical units (radians/step).
    
    Converts normalized action changes to physical radians using per-joint
    action scale before computing L2 norm. This ensures all joints are
    penalized equally for the same physical movement.
    
    Args:
        weight: Weight for the reward component.
    
    Returns:
        Pre-configured RewardComponentConfig.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=physical_action_rate,
        variables={
            "action_delta": "current_actions - previous_actions",
            "pd_action_scale": "pd_action_scale",
        },
        weight=weight,
    )


# =============================================================================
# Power Consumption Rewards
# =============================================================================


def pow_rew_factory(
    weight: float = -1e-5,
    min_value: float = -0.5,
    use_torque_squared: bool = False,
    zero_during_grace_period: bool = True,
) -> "RewardComponentConfig":
    """Factory for power consumption reward.
    
    Args:
        weight: Weight for the reward component.
        min_value: Minimum value to clamp the reward.
        use_torque_squared: Whether to use torque squared.
        zero_during_grace_period: Whether to zero this reward during grace period.
    
    Returns:
        Pre-configured RewardComponentConfig.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=power_consumption_sum,
        variables={
            "dof_forces": "current_state_dof_forces",
            "dof_vel": "current_state_dof_vel",
            "use_torque_squared": use_torque_squared,
        },
        weight=weight,
        min_value=min_value,
        zero_during_grace_period=zero_during_grace_period,
    )


# =============================================================================
# Joint Limit Rewards
# =============================================================================


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


def soft_pos_limit_reward(dof_pos, dof_limits_lower, dof_limits_upper):
    out_of_limits = -(dof_pos - dof_limits_lower).clip(max=0.0)
    out_of_limits += (dof_pos - dof_limits_upper).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def soft_pos_limit_rew_factory(
    weight: float = -0.1,
    min_value: float = None,
    zero_during_grace_period: bool = False,
) -> "RewardComponentConfig":
    """Factory for soft position limit reward.
    
    Penalizes joint positions outside soft limits (scaled version of hard limits).
    The soft limits are computed as dof_limits * soft_pos_limit factor from robot config.
    
    Args:
        weight: Weight for the reward component (negative to penalize violations).
        min_value: Minimum value to clamp the reward.
        zero_during_grace_period: Whether to zero this reward during grace period.
    
    Returns:
        Pre-configured RewardComponentConfig.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    return RewardComponentConfig(
        function=soft_pos_limit_reward,
        variables={
            "dof_pos": "current_state_dof_pos",
            "dof_limits_lower": "soft_dof_limits_lower",
            "dof_limits_upper": "soft_dof_limits_upper",
        },
        weight=weight,
        min_value=min_value,
        zero_during_grace_period=zero_during_grace_period,
    )


# =============================================================================
# Contact Rewards
# =============================================================================


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


def contact_match_rew_factory(
    weight: float = -0.1,
    indices_subset=None,
    zero_during_grace_period: bool = True,
) -> "RewardComponentConfig":
    """Factory for contact matching reward.
    
    Args:
        weight: Weight for the reward component.
        indices_subset: Optional body names or indices to subset (defaults to all foot bodies).
        zero_during_grace_period: Whether to zero this reward during grace period.
    
    Returns:
        Pre-configured RewardComponentConfig.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    if indices_subset is None:
        indices_subset = ["all_left_foot_bodies", "all_right_foot_bodies"]
    
    return RewardComponentConfig(
        function=contact_mismatch_sum,
        variables={
            "sim_contacts": "current_state_rigid_body_contacts",
            "ref_contacts": "ref_state_rigid_body_contacts",
        },
        indices_subset=indices_subset,
        weight=weight,
        zero_during_grace_period=zero_during_grace_period,
    )


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


def contact_force_change_rew_factory(
    weight: float = -1e-5,
    min_value: float = -0.5,
    indices_subset=None,
    zero_during_grace_period: bool = True,
) -> "RewardComponentConfig":
    """Factory for contact force change penalty.
    
    Args:
        weight: Weight for the reward component.
        min_value: Minimum value to clamp the reward.
        indices_subset: Optional body names or indices to subset (defaults to all foot bodies).
        zero_during_grace_period: Whether to zero this reward during grace period.
    
    Returns:
        Pre-configured RewardComponentConfig.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig
    
    if indices_subset is None:
        indices_subset = ["all_left_foot_bodies", "all_right_foot_bodies"]
    
    return RewardComponentConfig(
        function=impact_force_penalty,
        variables={
            "current_forces": "current_contact_force_magnitudes",
            "previous_forces": "prev_contact_force_magnitudes",
        },
        indices_subset=indices_subset,
        weight=weight,
        min_value=min_value,
        zero_during_grace_period=zero_during_grace_period,
    )

