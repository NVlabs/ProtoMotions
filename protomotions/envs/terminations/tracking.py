# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tracking-related termination compute kernels.

Scalar tracking-error kernels whose inputs default to the current state and
the mimic control's reference. A recipe binds one with
``MdpComponent.from_kernel`` and cuts it with a ``TerminationRule``; the same
bound component serves as an evaluation metric, so training and evaluation
read one measurement::

    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.envs.terminations.tracking import max_body_pos_error

    termination_components = {
        "tracking_error": TerminationRule(
            MdpComponent.from_kernel(max_body_pos_error), 0.5
        ),
    }

Includes:
- Standard tracking error (mean and max body position, mean body rotation)
- BeyondMimic-style errors (anchor position, orientation, height, relative body)
"""

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext

from protomotions.utils.rotations import calc_heading_quat_inv, quat_diff_norm, quat_rotate, quat_rotate_inverse


# =============================================================================
# Value Functions (return numeric values, used by evaluation metrics)
# =============================================================================

def mean_body_pos_error(
    current_rigid_body_pos: Tensor = EnvContext.current.rigid_body_pos,
    ref_rigid_body_pos: Tensor = EnvContext.mimic.ref_state.rigid_body_pos,
) -> Tensor:
    """Mean body position error across all bodies.

    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].

    Returns:
        Mean position error per env [num_envs] in meters.
    """
    per_body_err = (ref_rigid_body_pos - current_rigid_body_pos).pow(2).sum(-1).sqrt()
    return per_body_err.mean(-1)


def max_body_pos_error(
    current_rigid_body_pos: Tensor = EnvContext.current.rigid_body_pos,
    ref_rigid_body_pos: Tensor = EnvContext.mimic.ref_state.rigid_body_pos,
) -> Tensor:
    """Maximum body position error across all bodies.

    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].

    Returns:
        Max position error per env [num_envs] in meters.
    """
    per_body_err = (ref_rigid_body_pos - current_rigid_body_pos).pow(2).sum(-1).sqrt()
    return per_body_err.max(-1)[0]


def mean_body_rot_error(
    current_rigid_body_rot: Tensor = EnvContext.current.rigid_body_rot,
    ref_rigid_body_rot: Tensor = EnvContext.mimic.ref_state.rigid_body_rot,
) -> Tensor:
    """Mean body rotation error across all bodies.

    Args:
        current_rigid_body_rot: Current body rotations [num_envs, num_bodies, 4].
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4].

    Returns:
        Mean rotation error per env [num_envs] in radians.
    """
    per_body_err = quat_diff_norm(current_rigid_body_rot, ref_rigid_body_rot, True)
    return per_body_err.mean(-1)


def anchor_pos_error_value(
    current_anchor_pos: Tensor = EnvContext.current.anchor_pos,
    ref_anchor_pos: Tensor = EnvContext.mimic.ref_anchor_pos,
) -> Tensor:
    """Anchor (root) position error in global coordinates.

    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        ref_anchor_pos: Reference anchor position [num_envs, 3].

    Returns:
        Position error per env [num_envs] in meters.
    """
    return (current_anchor_pos - ref_anchor_pos).pow(2).sum(-1).sqrt()


def anchor_ori_error_value(
    current_anchor_rot: Tensor = EnvContext.current.anchor_rot,
    ref_anchor_rot: Tensor = EnvContext.mimic.ref_anchor_rot,
) -> Tensor:
    """Anchor orientation error via projected gravity z-component difference.

    Args:
        current_anchor_rot: Current anchor rotation quaternion [num_envs, 4] (w-last).
        ref_anchor_rot: Reference anchor rotation quaternion [num_envs, 4] (w-last).

    Returns:
        Projected gravity z-component difference per env [num_envs].
    """
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=current_anchor_rot.device)
    gravity_vec = gravity_vec.unsqueeze(0).expand(current_anchor_rot.shape[0], -1)

    proj_grav = quat_rotate_inverse(current_anchor_rot, gravity_vec, w_last=True)
    ref_proj_grav = quat_rotate_inverse(ref_anchor_rot, gravity_vec, w_last=True)

    return torch.abs(proj_grav[:, 2] - ref_proj_grav[:, 2])


def relative_body_pos_max_error(
    current_rigid_body_pos: Tensor = EnvContext.current.rigid_body_pos,
    ref_rigid_body_pos: Tensor = EnvContext.mimic.ref_state.rigid_body_pos,
    current_anchor_pos: Tensor = EnvContext.current.anchor_pos,
    current_anchor_rot: Tensor = EnvContext.current.anchor_rot,
    ref_anchor_pos: Tensor = EnvContext.mimic.ref_anchor_pos,
    ref_anchor_rot: Tensor = EnvContext.mimic.ref_anchor_rot,
) -> Tensor:
    """Maximum relative body position error in heading-aligned frame.

    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        current_anchor_pos: Current anchor position [num_envs, 3].
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_anchor_pos: Reference anchor position [num_envs, 3].
        ref_anchor_rot: Reference anchor rotation [num_envs, 4] (w-last).

    Returns:
        Max body error per env [num_envs] in meters.
    """
    num_envs, num_bodies, _ = current_rigid_body_pos.shape

    heading_inv = calc_heading_quat_inv(current_anchor_rot, w_last=True)
    ref_heading_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)

    offset = current_rigid_body_pos - current_anchor_pos.unsqueeze(1)
    ref_offset = ref_rigid_body_pos - ref_anchor_pos.unsqueeze(1)

    heading_inv_expanded = heading_inv.unsqueeze(1).expand(-1, num_bodies, -1)
    ref_heading_inv_expanded = ref_heading_inv.unsqueeze(1).expand(-1, num_bodies, -1)

    rel_pos = quat_rotate(
        heading_inv_expanded.reshape(-1, 4),
        offset.reshape(-1, 3),
        w_last=True
    ).reshape(num_envs, num_bodies, 3)

    ref_rel_pos = quat_rotate(
        ref_heading_inv_expanded.reshape(-1, 4),
        ref_offset.reshape(-1, 3),
        w_last=True
    ).reshape(num_envs, num_bodies, 3)

    per_body_error = (rel_pos - ref_rel_pos).pow(2).sum(dim=-1).sqrt()
    return per_body_error.max(dim=-1)[0]


def anchor_height_error_value(
    current_anchor_pos: Tensor = EnvContext.current.anchor_pos,
    ref_anchor_pos: Tensor = EnvContext.mimic.ref_anchor_pos,
) -> Tensor:
    """Anchor (root) height error (Z-axis only).

    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        ref_anchor_pos: Reference anchor position [num_envs, 3].

    Returns:
        Absolute height error per env [num_envs] in meters.
    """
    ref_anchor_height = ref_anchor_pos[:, 2]
    current_height = current_anchor_pos[:, 2]
    return torch.abs(current_height - ref_anchor_height)


__all__ = [
    # Value functions (for evaluation metrics)
    "mean_body_pos_error",
    "max_body_pos_error",
    "mean_body_rot_error",
    "anchor_pos_error_value",
    "anchor_ori_error_value",
    "anchor_height_error_value",
    "relative_body_pos_max_error",
]
