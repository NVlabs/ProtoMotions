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
"""Regularization reward compute kernels.

Pure tensor functions (kernels) for computing regularization rewards.
Use MdpComponent in experiment configs to bind kernels to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.regularization import compute_action_smoothness
    
    reward_components = {
        "action_smoothness": MdpComponent(
            compute_func=compute_action_smoothness,
            dynamic_vars={
                "current_processed_action": EnvContext.current_processed_action,
                "previous_processed_action": EnvContext.previous_processed_action,
            },
        ),
    }

Includes:
- Action smoothness (L2 and Log-Mean-Exp variants)
- Power consumption
- Joint limit violations
- Contact matching
- Contact force change penalties
"""

import torch
from torch import Tensor
from typing import Optional

from protomotions.envs.rewards.base import power_consumption_sum, delta_norm, delta_logmeanexp


# =============================================================================
# Regularization Reward Kernels
# =============================================================================

def compute_action_smoothness(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
) -> Tensor:
    """Action smoothness reward (L2 norm of processed action changes).
    
    Requires num_state_history_steps >= 1 in env config.
    
    Args:
        current_processed_action: Current processed action [num_envs, action_dim].
        previous_processed_action: Previous processed action [num_envs, action_dim].
    
    Returns:
        Smoothness penalty tensor [num_envs].
    """
    return delta_norm(current_processed_action, previous_processed_action)


def compute_action_smoothness_logmeanexp(
    current_processed_action: Tensor,
    previous_processed_action: Tensor,
    beta: float = 3.0,
) -> Tensor:
    """Action smoothness using Log-Mean-Exp (soft L_infinity).
    
    Requires num_state_history_steps >= 1 in env config.
    
    Args:
        current_processed_action: Current processed action [num_envs, action_dim].
        previous_processed_action: Previous processed action [num_envs, action_dim].
        beta: Temperature parameter. Lower = more like mean, higher = more like max.
    
    Returns:
        Smoothness penalty tensor [num_envs].
    """
    return delta_logmeanexp(current_processed_action, previous_processed_action, beta)


def compute_pow_rew(
    dof_forces: Tensor,
    dof_vel: Tensor,
    use_torque_squared: bool = False,
) -> Tensor:
    """Power consumption reward.
    
    Args:
        dof_forces: Joint forces/torques [num_envs, num_dofs].
        dof_vel: Joint velocities [num_envs, num_dofs].
        use_torque_squared: Whether to use torque squared instead of absolute.
    
    Returns:
        Power consumption tensor [num_envs].
    """
    return power_consumption_sum(dof_forces, dof_vel, use_torque_squared)


def compute_soft_pos_limit_rew(
    dof_pos: Tensor,
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
) -> Tensor:
    """Soft joint position limit penalty.
    
    Penalizes when joints approach or exceed limits.
    
    Args:
        dof_pos: Joint positions [num_envs, num_dofs].
        dof_limits_lower: Lower joint limits [num_dofs].
        dof_limits_upper: Upper joint limits [num_dofs].
    
    Returns:
        Penalty tensor [num_envs].
    """
    out_of_limits = -(dof_pos - dof_limits_lower).clip(max=0.0)
    out_of_limits += (dof_pos - dof_limits_upper).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def compute_contact_match_rew(
    sim_contacts: Tensor,
    ref_contacts: Tensor,
    contact_body_ids: Tensor,
) -> Tensor:
    """Contact matching reward using foot contact bodies.
    
    Penalizes mismatch between simulated and reference foot contacts.
    Uses contact_body_ids (typically foot bodies).
    
    Args:
        sim_contacts: Simulated contact flags [num_envs, num_bodies].
        ref_contacts: Reference contact flags [num_envs, num_bodies].
        contact_body_ids: Indices of bodies to track contacts for [num_contact_bodies].
    
    Returns:
        Contact mismatch penalty tensor [num_envs].
    """
    sim_contacts_subset = sim_contacts[:, contact_body_ids]
    ref_contacts_subset = ref_contacts[:, contact_body_ids]
    return torch.abs(sim_contacts_subset.float() - ref_contacts_subset.float()).sum(dim=1)


def compute_contact_force_change_rew(
    current_contact_force_magnitudes: Tensor,
    prev_contact_force_magnitudes: Tensor,
    threshold: float = 30.0,
) -> Tensor:
    """Contact force change penalty.
    
    Penalizes sudden contact force changes above a threshold (impact penalty).
    
    Args:
        current_contact_force_magnitudes: Current contact forces [num_envs, num_bodies].
        prev_contact_force_magnitudes: Previous contact forces [num_envs, num_bodies].
        threshold: Force change threshold below which changes are ignored (default: 30.0).
    
    Returns:
        Total force change above threshold [num_envs].
    """
    force_changes = torch.abs(current_contact_force_magnitudes - prev_contact_force_magnitudes)
    force_changes = torch.clamp(force_changes - threshold, min=0)
    return force_changes.sum(dim=-1)


# =============================================================================
# Helper Functions (used by kernels or for advanced use cases)
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
        dof_pos: Joint positions [num_envs, num_dofs].
        dof_limits_lower: Lower limits [num_dofs].
        dof_limits_upper: Upper limits [num_dofs].
        indices: Optional DOF indices to subset.

    Returns:
        Total violation [num_envs].
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
        sim_contacts: Simulated contacts [num_envs, num_bodies].
        ref_contacts: Reference contacts [num_envs, num_bodies].
        indices: Optional body indices to subset.

    Returns:
        Total mismatch [num_envs].
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
        current_forces: Current contact forces [num_envs, num_bodies].
        previous_forces: Previous contact forces [num_envs, num_bodies].
        indices: Optional body indices to subset.
        threshold: Force change threshold below which changes are ignored (default: 30.0).

    Returns:
        Total force change above threshold [num_envs].
    """
    if indices is not None:
        current_forces = current_forces[:, indices]
        previous_forces = previous_forces[:, indices]

    force_changes = torch.abs(current_forces - previous_forces)
    force_changes = torch.clamp(force_changes - threshold, min=0)
    return force_changes.sum(dim=-1)


__all__ = [
    # Main reward kernels
    "compute_action_smoothness",
    "compute_action_smoothness_logmeanexp",
    "compute_pow_rew",
    "compute_soft_pos_limit_rew",
    "compute_contact_match_rew",
    "compute_contact_force_change_rew",
    # Helper functions
    "joint_limit_violation",
    "contact_mismatch_sum",
    "impact_force_penalty",
]
