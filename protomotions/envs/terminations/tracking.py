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
"""Tracking-related termination compute kernels.

Pure tensor functions (kernels) for computing tracking-related terminations.
Use MdpComponent in experiment configs to bind kernels to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.terminations.tracking import compute_tracking_error
    
    termination_components = {
        "tracking_error": MdpComponent(
            compute_func=compute_tracking_error,
            dynamic_vars={
                "current_rigid_body_pos": EnvContext.current.rigid_body_pos,
                "ref_rigid_body_pos": EnvContext.mimic.ref_state.rigid_body_pos,
            },
            static_params={"threshold": 0.5},
        ),
    }

Includes:
- Standard tracking error termination
- BeyondMimic-style terminations (anchor position, orientation, relative body)
- Motion clip completion
- Value functions for evaluation metrics
"""

import torch
from torch import Tensor

from protomotions.utils.rotations import calc_heading_quat_inv, quat_diff_norm, quat_rotate, quat_rotate_inverse


# =============================================================================
# Value Functions (return numeric values, used by evaluation metrics)
# =============================================================================

def mean_body_pos_error(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
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
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
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
    current_rigid_body_rot: Tensor,
    ref_rigid_body_rot: Tensor,
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
    current_anchor_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    anchor_idx: int,
) -> Tensor:
    """Anchor (root) position error in global coordinates.

    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        anchor_idx: Index of anchor body.

    Returns:
        Position error per env [num_envs] in meters.
    """
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    return (current_anchor_pos - ref_anchor_pos).pow(2).sum(-1).sqrt()


def anchor_ori_error_value(
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    anchor_idx: int,
) -> Tensor:
    """Anchor orientation error via projected gravity z-component difference.

    Args:
        current_anchor_rot: Current anchor rotation quaternion [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        anchor_idx: Index of anchor body.

    Returns:
        Projected gravity z-component difference per env [num_envs].
    """
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=current_anchor_rot.device)
    gravity_vec = gravity_vec.unsqueeze(0).expand(current_anchor_rot.shape[0], -1)

    proj_grav = quat_rotate_inverse(current_anchor_rot, gravity_vec, w_last=True)
    ref_proj_grav = quat_rotate_inverse(ref_anchor_rot, gravity_vec, w_last=True)

    return torch.abs(proj_grav[:, 2] - ref_proj_grav[:, 2])


def relative_body_pos_max_error(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    current_anchor_pos: Tensor,
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    anchor_idx: int,
) -> Tensor:
    """Maximum relative body position error in heading-aligned frame.

    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        current_anchor_pos: Current anchor position [num_envs, 3].
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        anchor_idx: Index of anchor body.

    Returns:
        Max body error per env [num_envs] in meters.
    """
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
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


# =============================================================================
# Termination Compute Kernels
# =============================================================================

def compute_tracking_error(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    threshold: float = 0.5,
) -> Tensor:
    """Tracking error termination based on max joint position error.
    
    Terminates episode when max joint position error exceeds threshold.
    
    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        threshold: Maximum joint error threshold in meters.
    
    Returns:
        Boolean tensor [num_envs] indicating which envs should terminate.
    """
    gt_per_joint_err = (ref_rigid_body_pos - current_rigid_body_pos).pow(2).sum(-1).sqrt()
    max_joint_err = gt_per_joint_err.max(-1)[0]
    terminate = max_joint_err > threshold
    return terminate


def anchor_height_error_value(
    current_anchor_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    anchor_idx: int,
) -> Tensor:
    """Anchor (root) height error (Z-axis only).

    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        anchor_idx: Index of anchor body.

    Returns:
        Absolute height error per env [num_envs] in meters.
    """
    ref_anchor_height = ref_rigid_body_pos[:, anchor_idx, 2]
    current_height = current_anchor_pos[:, 2]
    return torch.abs(current_height - ref_anchor_height)


def compute_anchor_pos_error_term(
    current_anchor_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    anchor_idx: int,
    threshold: float = 0.5,
) -> Tensor:
    """Anchor position error termination.
    
    Implements BeyondMimic's bad_ref_pos termination condition.
    Terminates when anchor (root) position error exceeds threshold in global coordinates.
    
    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        anchor_idx: Index of anchor body.
        threshold: Maximum allowed distance in meters.
    
    Returns:
        Boolean tensor [num_envs] indicating which envs should terminate.
    """
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    distance = (current_anchor_pos - ref_anchor_pos).pow(2).sum(-1).sqrt()
    return distance > threshold


def compute_anchor_ori_error_term(
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    anchor_idx: int,
    threshold: float = 0.8,
) -> Tensor:
    """Anchor orientation error termination.
    
    Implements BeyondMimic's bad_ref_ori termination condition.
    Compares the z-component of the projected gravity vectors to detect
    if the robot is tilted too much compared to the reference.
    
    Args:
        current_anchor_rot: Current anchor rotation quaternion [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        anchor_idx: Index of anchor body.
        threshold: Maximum allowed difference in projected gravity z-component.
    
    Returns:
        Boolean tensor [num_envs] indicating which envs should terminate.
    """
    # Extract reference anchor rotation
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    
    # Gravity vector in world frame (pointing down)
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=current_anchor_rot.device)
    gravity_vec = gravity_vec.unsqueeze(0).expand(current_anchor_rot.shape[0], -1)
    
    # Project gravity into anchor's local frame
    proj_grav = quat_rotate_inverse(current_anchor_rot, gravity_vec, w_last=True)
    ref_proj_grav = quat_rotate_inverse(ref_anchor_rot, gravity_vec, w_last=True)
    
    # Compare z-components (how "upright" each is)
    z_diff = torch.abs(proj_grav[:, 2] - ref_proj_grav[:, 2])
    return z_diff > threshold


def compute_relative_body_pos_error_term(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    current_anchor_pos: Tensor,
    current_anchor_rot: Tensor,
    ref_rigid_body_rot: Tensor,
    anchor_idx: int,
    threshold: float = 0.25,
) -> Tensor:
    """Relative body position error termination.
    
    Implements BeyondMimic's bad_motion_body_pos termination condition.
    Computes body positions relative to anchor in yaw-aligned local frame,
    making the check invariant to global heading/yaw.
    
    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        current_anchor_pos: Current anchor position [num_envs, 3].
        current_anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_rigid_body_rot: Reference body rotations [num_envs, num_bodies, 4] (w-last).
        anchor_idx: Index of anchor body.
        threshold: Maximum allowed error for any body in meters.
    
    Returns:
        Boolean tensor [num_envs] indicating which envs have any body exceeding threshold.
    """
    # Extract reference anchor pos and rot
    ref_anchor_pos = ref_rigid_body_pos[:, anchor_idx, :]
    ref_anchor_rot = ref_rigid_body_rot[:, anchor_idx, :]
    
    num_envs, num_bodies, _ = current_rigid_body_pos.shape
    
    # Get inverse yaw quaternions for both current and reference anchors
    heading_inv = calc_heading_quat_inv(current_anchor_rot, w_last=True)
    ref_heading_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)
    
    # Compute offsets in world frame (body position relative to anchor)
    offset = current_rigid_body_pos - current_anchor_pos.unsqueeze(1)
    ref_offset = ref_rigid_body_pos - ref_anchor_pos.unsqueeze(1)
    
    # Expand heading quaternions for all bodies
    heading_inv_expanded = heading_inv.unsqueeze(1).expand(-1, num_bodies, -1)
    ref_heading_inv_expanded = ref_heading_inv.unsqueeze(1).expand(-1, num_bodies, -1)
    
    # Rotate offsets into anchor's local frame (yaw-aligned)
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
    
    # Compute per-body position error
    per_body_error = (rel_pos - ref_rel_pos).pow(2).sum(dim=-1).sqrt()  # [num_envs, num_bodies]
    
    # Terminate if ANY body exceeds threshold
    return torch.any(per_body_error > threshold, dim=-1)


def compute_anchor_height_error_term(
    current_anchor_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    anchor_idx: int,
    threshold: float = 0.25,
) -> Tensor:
    """Anchor height (Z-axis) error termination.

    Terminates when the root height deviates from the reference root height
    by more than the threshold. Only checks the vertical (Z) component.

    Args:
        current_anchor_pos: Current anchor position [num_envs, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        anchor_idx: Index of anchor body.
        threshold: Maximum allowed height error in meters.

    Returns:
        Boolean termination mask [num_envs].
    """
    height_error = anchor_height_error_value(
        current_anchor_pos, ref_rigid_body_pos, anchor_idx
    )
    return height_error > threshold


# =============================================================================
# Helper Functions
# =============================================================================

def motion_clip_done(motion_times: Tensor, motion_ids: Tensor, motion_lib) -> Tensor:
    """Terminate when motion clip has finished playing.
    
    Args:
        motion_times: Current motion times [num_envs].
        motion_ids: Motion IDs [num_envs].
        motion_lib: Motion library instance.
    
    Returns:
        Boolean tensor [num_envs] indicating which motion clips are done.
    """
    motion_lengths = motion_lib.get_motion_length(motion_ids)
    return motion_times >= motion_lengths


__all__ = [
    # Value functions (for evaluation metrics)
    "mean_body_pos_error",
    "max_body_pos_error",
    "mean_body_rot_error",
    "anchor_pos_error_value",
    "anchor_ori_error_value",
    "anchor_height_error_value",
    "relative_body_pos_max_error",
    # Termination compute kernels
    "compute_tracking_error",
    "compute_anchor_pos_error_term",
    "compute_anchor_ori_error_term",
    "compute_relative_body_pos_error_term",
    "compute_anchor_height_error_term",
    "motion_clip_done",
]
