# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    result = delta_norm(current_processed_action, previous_processed_action)
    if not torch.isfinite(result).all():
        # NaN forensics (2026-07-08 v2 resume crash): pin down whether the
        # non-finite values enter via the current processed action (action
        # processing chain), the previous one (state history), or both.
        for label, t in (
            ("current_processed_action", current_processed_action),
            ("previous_processed_action", previous_processed_action),
        ):
            nb = (~torch.isfinite(t)).sum().item()
            print(
                f"[nan-forensics] action_smoothness {label}: {nb}/{t.numel()} "
                f"non-finite, shape={tuple(t.shape)}, "
                f"finite_minmax=({t[torch.isfinite(t)].min().item() if torch.isfinite(t).any() else 'NA'}, "
                f"{t[torch.isfinite(t)].max().item() if torch.isfinite(t).any() else 'NA'})"
            )
    return result


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
    return delta_logmeanexp(
        current_processed_action,
        previous_processed_action,
        beta=beta,
    )


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


def compute_reference_contact_liftoff_penalty(
    sim_contacts: Tensor,
    ref_contacts: Tensor,
    contact_body_ids: Tensor,
    historical_body_contacts: Tensor,
    ref_contact_threshold: float = 0.5,
) -> Tensor:
    """Reference-gated penalty for unnecessary foot lift-offs.

    The raw unit is weighted foot-count per control step.  A foot contributes
    only on the simulated contact transition ``prev_contact -> no_contact`` while
    the reference contact schedule says that same foot should remain in stance.
    This gates the penalty on locomotion necessity: reference swing phases are
    not treated as nervous stepping.

    Smoothed reference contacts in [0, 1] are supported. Contacts above
    ``ref_contact_threshold`` scale linearly toward a full stance penalty.

    Args:
        sim_contacts: Simulated contact flags [num_envs, num_bodies].
        ref_contacts: Reference contact labels [num_envs, num_bodies].
        contact_body_ids: Indices of foot bodies to evaluate [num_feet].
        historical_body_contacts: Historical simulated contacts for configured
            contact bodies [num_envs, history_steps, num_feet]. The first history
            slot is the previous control step.
        ref_contact_threshold: Reference contact value where stance starts.

    Returns:
        Weighted foot-count penalty [num_envs].
    """
    if ref_contacts is None:
        raise ValueError(
            "reference_contact_liftoff_penalty requires reference motion contacts; "
            "re-run motion conversion with contacts enabled or remove this reward."
        )
    if historical_body_contacts is None:
        raise ValueError(
            "reference_contact_liftoff_penalty requires num_state_history_steps >= 1 "
            "so previous simulated foot contacts are available."
        )
    if not (0.0 <= ref_contact_threshold < 1.0):
        raise ValueError("ref_contact_threshold must be in [0, 1).")

    sim_contacts_subset = sim_contacts[:, contact_body_ids].float()
    if historical_body_contacts.dim() == 3:
        prev_sim_contacts_subset = historical_body_contacts[:, 0, :].float()
    else:
        prev_sim_contacts_subset = historical_body_contacts.float()
    ref_contacts_subset = ref_contacts[:, contact_body_ids].float()

    stance_denom = max(1.0 - ref_contact_threshold, 1e-6)
    ref_stance_weight = torch.clamp(
        (ref_contacts_subset - ref_contact_threshold) / stance_denom,
        min=0.0,
        max=1.0,
    )

    liftoffs = prev_sim_contacts_subset * (1.0 - sim_contacts_subset)
    return (ref_stance_weight * liftoffs).sum(dim=1)


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


def compute_foot_contact_force_penalty(
    current_contact_force_magnitudes: Tensor,
    contact_body_ids: Tensor,
    force_threshold: float = 400.0,
) -> Tensor:
    """Penalty on instantaneous foot contact-force magnitude above a threshold.

    Encourages gentle foot placement (anti-stomp). Only force ABOVE force_threshold
    is penalized, so normal weight-bearing stance (~250 N/foot for a ~50 kg robot) is
    free and only hard stomps/impacts are penalized. Summed over the foot bodies
    (``contact_body_ids``). Returns a per-env non-negative penalty [num_envs]; apply a
    small NEGATIVE weight (and a min_value floor) in the factory so push-recovery
    stomps stay affordable.

    NOTE: ``force_threshold`` is deliberately NOT named ``threshold`` because
    ``threshold`` is a reserved reward-metadata key stripped before the kernel call.

    Args:
        current_contact_force_magnitudes: Per-body contact force magnitudes
            [num_envs, num_bodies].
        contact_body_ids: Indices of the foot bodies to penalize [num_feet].
        force_threshold: Force (N) below which foot forces are not penalized.

    Returns:
        Sum over feet of force in excess of force_threshold [num_envs].
    """
    foot_forces = current_contact_force_magnitudes[:, contact_body_ids]
    excess = torch.clamp(foot_forces - force_threshold, min=0.0)
    return excess.sum(dim=-1)


def compute_fall_penalty(
    current_anchor_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    anchor_idx: int,
    height_threshold: float = 0.25,
) -> Tensor:
    """Explicit fall penalty.

    Returns 1.0 for envs whose anchor (root) height error exceeds ``height_threshold``
    -- the SAME condition used by the anchor-height fall termination
    (``compute_anchor_height_error_term``) -- else 0.0. Apply a NEGATIVE weight in the
    factory. Makes falling explicitly costly (previously it was only implicitly
    penalized via termination + zeroed bootstrap).

    NOTE: ``height_threshold`` is deliberately NOT named ``threshold`` (reserved
    reward-metadata key that would be stripped before the kernel call).

    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        anchor_idx: Index of the anchor body.
        height_threshold: Max allowed anchor height error (m) before it counts as a fall.

    Returns:
        Float fall indicator [num_envs] (1.0 fallen, 0.0 otherwise).
    """
    from protomotions.envs.terminations import anchor_height_error_value

    height_error = anchor_height_error_value(
        current_anchor_pos, ref_rigid_body_pos, anchor_idx
    )
    return (height_error > height_threshold).float()


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
    "compute_reference_contact_liftoff_penalty",
    "compute_contact_force_change_rew",
    "compute_foot_contact_force_penalty",
    "compute_fall_penalty",
    # Helper functions
    "joint_limit_violation",
    "contact_mismatch_sum",
    "impact_force_penalty",
]
