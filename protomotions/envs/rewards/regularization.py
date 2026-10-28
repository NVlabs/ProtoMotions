# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regularization reward compute kernels.

Pure tensor functions (kernels) for computing regularization rewards. Each
kernel's state inputs default to the context names it reads, so a recipe binds
it with ``MdpComponent.from_kernel`` and supplies only the constants that have
no default (the contact readers' name lists) or that it wants to change::

    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.envs.rewards.regularization import compute_action_smoothness

    reward_components = {
        "action_smoothness": MdpComponent.from_kernel(
            compute_action_smoothness, meta=RewardMeta(weight=-0.02)
        ),
    }

The three contact readers (``compute_self_contacts``,
``compute_contact_match_rew``, ``compute_terrain_foot_contact_match_rew``) take
the names of the rows and columns they read; the recipe also lists them in
``required_keys=`` so the build refuses a selection the simulator does not
measure.

Includes:
- Action smoothness (L2 and Log-Mean-Exp variants)
- Power consumption
- Joint limit violations
- Contact matching
- Contact force change penalties
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional
from protomotions.envs.contact_features import (
    check_force_threshold,
    contact_flags,
    contact_magnitudes,
    contact_rows,
)
from protomotions.envs.context_views import EnvContext
from protomotions.simulator.base_simulator.contacts import CONTACT_FORCE_THRESHOLD

from protomotions.envs.rewards.base import (
    power_consumption_sum,
    delta_norm,
)


# =============================================================================
# Regularization Reward Kernels
# =============================================================================


def compute_action_smoothness(
    current_processed_action: Tensor = EnvContext.current_processed_action,
    previous_processed_action: Tensor = EnvContext.previous_processed_action,
) -> Tensor:
    """Action smoothness reward (L2 norm of processed action changes).

    Reads the action history: ``previous_processed_action`` is its most recent
    slot (the default binding).

    Args:
        current_processed_action: Current processed action [num_envs, action_dim].
        previous_processed_action: Previous processed action [num_envs, action_dim].

    Returns:
        Smoothness penalty tensor [num_envs].
    """
    return delta_norm(current_processed_action, previous_processed_action)


def compute_action_rate_l2(
    raw_action: Tensor = EnvContext.current_action,
    previous_raw_action: Tensor = EnvContext.previous_action,
) -> Tensor:
    """Sum squared raw policy-action changes over joints, per environment.

    Inputs are dimensionless policy outputs [num_envs, action_dim], before
    clipping or PD scaling. No mean, square root, or timestep scaling is applied.
    """
    return (raw_action - previous_raw_action).square().sum(dim=-1)


def compute_pow_rew(
    dof_forces: Tensor = EnvContext.current.dof_forces,
    dof_vel: Tensor = EnvContext.current.dof_vel,
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
    max_violation: float = None,
) -> Tensor:
    """Soft joint position limit penalty.

    Penalizes when joints approach or exceed limits.

    The penalty is linear in the violation and summed over every DOF, so it has
    no floor. That is harmless while the simulator is healthy and unbounded when
    it is not: joint positions that blow up produce an arbitrarily large negative
    reward, which then sets the reward-normalizer scale for many epochs
    afterwards. A raw task reward of -7.9e7 was traced to this term.

    Args:
        dof_pos: Joint positions [num_envs, num_dofs].
        dof_limits_lower: Lower joint limits [num_dofs].
        dof_limits_upper: Upper joint limits [num_dofs].
        max_violation: If set, clamp the summed violation (in radians) before it
            is weighted, which puts a floor on the penalty. Past the clamp the
            state is broken rather than merely bad, and a larger number carries
            no useful gradient. ``None`` (default) keeps the unbounded behaviour.

    Returns:
        Penalty tensor [num_envs].
    """
    out_of_limits = -(dof_pos - dof_limits_lower).clip(max=0.0)
    out_of_limits += (dof_pos - dof_limits_upper).clip(min=0.0)
    total = torch.sum(out_of_limits, dim=1)
    if max_violation is not None:
        total = total.clip(max=max_violation)
    return total


def compute_self_contacts(
    body_names: List[str],
    partner_names: List[str],
    contact_forces: Tensor = EnvContext.contact_forces,
    contact_body_index: Dict[str, int] = EnvContext.contact_body_index,
    contact_partner_index: Dict[str, int] = EnvContext.contact_partner_index,
    force_threshold: float = 1.0,
) -> Tensor:
    """Count bodies whose largest named-partner normal force exceeds threshold.

    Forces are [env, body, partner, xyz], world frame, N. ``body_names`` are
    robot link names and ``partner_names`` exact ``robot/<link>`` names, so the
    count is a self-contact count; both selections must be measured by the
    simulator (list them in ``required_keys=``). Count each selected body at
    most once, even when several partners exceed the threshold. Equality does
    not count. Return an unweighted float count per environment.

    A partner outside the robot (``terrain``, an object) is refused: it would
    count as a self contact. Exact names and a finite, nonnegative threshold
    are checked as in :func:`contact_rows` / :func:`check_force_threshold`.
    """
    check_force_threshold(force_threshold)
    for name in partner_names:
        if not name.startswith("robot/") or name == "robot/":
            raise ValueError(
                "compute_self_contacts: partner_names are robot/<link> names, "
                f"got {name!r}"
            )
    rows = contact_rows(contact_body_index, body_names, "body_names")
    partners = contact_rows(contact_partner_index, partner_names, "partner_names")
    forces = contact_forces[:, rows][:, :, partners]
    return (forces.norm(dim=-1).amax(dim=-1) > force_threshold).float().sum(dim=-1)


def compute_contact_match_rew(
    contact_forces: Tensor = EnvContext.contact_forces,
    ref_contacts: Tensor = EnvContext.mimic.ref_state.rigid_body_contacts,
    contact_body_ids: Tensor = EnvContext.contact_body_ids,
    partner_ids: Optional[Tensor] = None,
    force_threshold: float = CONTACT_FORCE_THRESHOLD,
) -> Tensor:
    """Contact matching reward using foot contact bodies.

    Penalizes mismatch between simulated and reference foot contacts.
    The matrix already contains selected rows. ``contact_body_ids`` maps them
    back to full common-order kinematic contact annotations.

    Args:
        contact_forces: Normal forces [env, body, partner, xyz], world axes, N.
        ref_contacts: Reference contact flags [num_envs, num_bodies].
        contact_body_ids: Indices of bodies to track contacts for [num_contact_bodies].

    Returns:
        Contact mismatch penalty tensor [num_envs].
    """
    sim_contacts_subset = contact_flags(
        contact_forces, partner_ids=partner_ids, force_threshold=force_threshold
    )
    ref_contacts_subset = ref_contacts[:, contact_body_ids]
    return torch.abs(sim_contacts_subset.float() - ref_contacts_subset.float()).sum(
        dim=1
    )


def compute_terrain_foot_contact_match_rew(
    contact_body_ids: Tensor,
    body_names: List[str],
    contact_forces: Tensor = EnvContext.contact_forces,
    ref_contacts: Tensor = EnvContext.mimic.ref_state.rigid_body_contacts,
    contact_body_index: Dict[str, int] = EnvContext.contact_body_index,
    contact_partner_index: Dict[str, int] = EnvContext.contact_partner_index,
    force_threshold: float = CONTACT_FORCE_THRESHOLD,
) -> Tensor:
    """Compare foot-to-terrain contacts with reference labels (including soft labels).

    Select named foot rows and the terrain column from the privileged matrix
    [envs, selected robot bodies, selected partners, xyz], in world-frame N. Robot,
    object, and projectile columns never contribute to the summed mismatch.

    ``body_names`` selects the rows to match and ``contact_body_ids`` maps them
    to the reference labels' common body order: the recipe builds it once with
    ``build_body_ids_tensor(robot_cfg.kinematic_info.body_names, body_names,
    device=torch.device("cpu"))`` and lists the names (and ``"terrain"``) in
    ``required_keys=`` so the build refuses an unmeasured selection.
    ``force_threshold`` is in newtons (finite and nonnegative); the reference
    may contain soft labels.
    """
    check_force_threshold(force_threshold)
    rows = contact_rows(contact_body_index, body_names, "body_names")
    terrain_forces = contact_forces[:, rows, contact_partner_index["terrain"]]
    contacts = terrain_forces.norm(dim=-1) > force_threshold
    return contact_mismatch_sum(contacts, ref_contacts[:, contact_body_ids])


def compute_contact_force_change_rew(
    contact_forces: Tensor = EnvContext.contact_forces,
    prev_contact_force_magnitudes: Tensor = EnvContext.prev_contact_force_magnitudes,
    threshold: float = 30.0,
) -> Tensor:
    """Contact force change penalty.

    Penalizes sudden contact force changes above a threshold (impact penalty).

    Args:
        contact_forces: Current normal-force matrix [env, body, partner, xyz], N.
        prev_contact_force_magnitudes: Previous maximum pair magnitude [env, body], N.
        threshold: Force change threshold below which changes are ignored (default: 30.0).

    Returns:
        Total force change above threshold [num_envs].
    """
    return impact_force_penalty(
        contact_magnitudes(contact_forces),
        prev_contact_force_magnitudes,
        threshold=threshold,
    )


# =============================================================================
# Helper Functions (used by kernels or for advanced use cases)
# =============================================================================


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
    "compute_pow_rew",
    "compute_soft_pos_limit_rew",
    "compute_contact_match_rew",
    "compute_self_contacts",
    "compute_contact_force_change_rew",
    # Helper functions
    "contact_mismatch_sum",
    "impact_force_penalty",
]
