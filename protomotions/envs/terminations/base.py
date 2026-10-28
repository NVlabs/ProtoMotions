# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base termination checking utilities.

Provides core functions for checking common termination conditions:
- Fall detection (contact + height), joint-limit blow-up
- Episode length limits

The two boolean kernels (``named_contact_fall_termination``,
``joint_limit_termination``) carry their state inputs as context defaults; a
recipe binds one with ``MdpComponent.from_kernel`` (supplying the body
selection or the joint limits) and wraps it in a thresholdless
``TerminationRule``.
"""

from torch import Tensor
from typing import Dict, List
from protomotions.envs.contact_features import contact_flags, contact_rows
from protomotions.envs.context_views import EnvContext


def check_max_length_term(
    progress_buf: Tensor,
    max_episode_length: float,
) -> Tensor:
    """Check if episode has exceeded maximum length.

    Args:
        progress_buf: Episode progress counter [num_envs]
        max_episode_length: Maximum allowed episode length

    Returns:
        Boolean tensor [num_envs] indicating which episodes reached max length
    """
    return progress_buf >= max_episode_length - 1


# ==============================================================================
# Boolean termination kernels
# ==============================================================================


def named_contact_fall_termination(
    body_ids: Tensor,
    body_names: List[str],
    partner_names: List[str],
    rigid_body_pos: Tensor = EnvContext.current.rigid_body_pos,
    contact_forces: Tensor = EnvContext.contact_forces,
    contact_body_index: Dict[str, int] = EnvContext.contact_body_index,
    contact_partner_index: Dict[str, int] = EnvContext.contact_partner_index,
    ground_heights: Tensor = EnvContext.ground_heights,
    progress_buf: Tensor = EnvContext.progress_buf,
    termination_height: float = 0.15,
) -> Tensor:
    """Contact AND low height for named forbidden bodies, after reset grace.

    Positions/heights are in world metres. Forces are world normal forces in N.
    ``body_ids`` address the full kinematics (the recipe builds them once with
    ``build_body_ids_tensor(robot_cfg.kinematic_info.body_names, body_names,
    device=torch.device("cpu"))``); ``body_names`` and ``partner_names`` address
    the compact contact matrix and go in ``required_keys=`` too, so a
    foot-only matrix cannot silently stand in for whole-body fall detection.
    Both are exact names (:func:`contact_rows` refuses a pattern).
    """
    rows = contact_rows(contact_body_index, body_names, "body_names")
    columns = contact_rows(contact_partner_index, partner_names, "partner_names")
    forces = contact_forces[:, rows][:, :, columns]
    has_contact = contact_flags(forces).any(dim=-1)
    heights = rigid_body_pos[:, body_ids, 2] - ground_heights.unsqueeze(-1)
    low_body = (heights < termination_height).any(dim=-1)
    return has_contact & low_body & (progress_buf > 1)


def joint_limit_termination(
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
    dof_pos: Tensor = EnvContext.current.dof_pos,
    max_violation: float = 1.0,
) -> Tensor:
    """End the episode once joints are driven far past their limits.

    A humanoid whose joints are a radian outside their range is not in a state
    worth continuing to simulate or to learn from -- it is usually the first
    visible sign of the simulator diverging. Left alone the episode keeps
    running, the soft-limit penalty grows without bound, and the resulting
    reward sets the reward-normalizer scale for many epochs afterwards.

    Terminating instead keeps the damage inside one episode: the environment
    resets, and the penalty for the steps that did happen stays small.

    Args:
        dof_limits_lower: Lower joint limits [num_dofs], from
            ``robot_cfg.kinematic_info.dof_limits_lower``.
        dof_limits_upper: Upper joint limits [num_dofs].
        dof_pos: Joint positions [num_envs, num_dofs].
        max_violation: Radians past a limit, on any single joint, that ends the
            episode. Per-joint rather than summed so one badly diverged joint is
            caught without needing the whole body to be broken.

    Returns:
        Boolean tensor [num_envs].
    """
    out_of_limits = -(dof_pos - dof_limits_lower).clip(max=0.0)
    out_of_limits += (dof_pos - dof_limits_upper).clip(min=0.0)
    return out_of_limits.max(dim=-1).values > max_violation
