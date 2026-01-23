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
"""Tracking-related termination functions.

Provides termination conditions for motion tracking tasks:
- Standard tracking error termination
- BeyondMimic-style terminations (anchor position, orientation, relative body)
- Motion clip completion
"""

import torch
from torch import Tensor

from protomotions.utils.rotations import calc_heading_quat_inv, quat_rotate, quat_rotate_inverse


# ==============================================================================
# Standard Tracking Terminations
# ==============================================================================


def max_joint_err(
    current_rigid_body_pos: Tensor,
    ref_rigid_body_pos: Tensor,
    threshold: float,
) -> Tensor:
    """Check if maximum joint position error exceeds threshold.
    
    Computes the maximum absolute difference across all DOFs and checks if it
    exceeds the threshold for early termination.
    
    Args:
        current_rigid_body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_rigid_body_pos: Reference body positions [num_envs, num_bodies, 3].
        threshold: Maximum allowed joint error in radians.
    
    Returns:
        Boolean tensor indicating which envs exceed the threshold [num_envs].
    """
    gt_per_joint_err = (ref_rigid_body_pos - current_rigid_body_pos).pow(2).sum(-1).sqrt()
    max_joint_err = gt_per_joint_err.max(-1)[0]
    terminate = max_joint_err > threshold
    return terminate


def tracking_error_factory(threshold: float = 0.5):
    """Factory for tracking error termination.
    
    Terminates episode when max joint position error exceeds threshold.
    
    Args:
        threshold: Maximum joint error threshold in meters.
    
    Returns:
        Pre-configured TerminationComponentConfig.
    """
    from protomotions.envs.base_env.config import TerminationComponentConfig
    
    return TerminationComponentConfig(
        function=max_joint_err,
        variables={
            "current_rigid_body_pos": "current_state_rigid_body_pos",
            "ref_rigid_body_pos": "ref_state_rigid_body_pos",
            "threshold": threshold,
        },
    )


def motion_clip_done(motion_times: Tensor, motion_ids: Tensor, motion_lib) -> Tensor:
    """Terminate when motion clip has finished playing.
    
    For use with TerminationComponentConfig in mimic tasks.
    
    Args:
        motion_times: Current motion times [num_envs].
        motion_ids: Motion IDs [num_envs].
        motion_lib: Motion library instance.
    
    Returns:
        Boolean tensor [num_envs] indicating which motion clips are done.
    """
    motion_lengths = motion_lib.get_motion_length(motion_ids)
    return motion_times >= motion_lengths


# ==============================================================================
# BeyondMimic-style Terminations
# ==============================================================================


def anchor_pos_error(
    anchor_pos: Tensor,
    ref_anchor_pos: Tensor,
    threshold: float,
) -> Tensor:
    """Check if anchor (root) position error exceeds threshold in global coordinates.
    
    Implements BeyondMimic's bad_ref_pos termination condition.
    
    Args:
        anchor_pos: Current anchor position [num_envs, 3].
        ref_anchor_pos: Reference anchor position [num_envs, 3].
        threshold: Maximum allowed distance in meters.
    
    Returns:
        Boolean tensor [num_envs] indicating which envs exceed the threshold.
    """
    distance = (anchor_pos - ref_anchor_pos).pow(2).sum(-1).sqrt()
    return distance > threshold


def anchor_pos_error_factory(threshold: float = 0.5):
    """Factory for anchor position error termination (BeyondMimic bad_ref_pos).
    
    Terminates episode when anchor position error exceeds threshold in global coords.
    
    Args:
        threshold: Maximum anchor position error threshold in meters (default: 0.5).
    
    Returns:
        Pre-configured TerminationComponentConfig.
    """
    from protomotions.envs.base_env.config import TerminationComponentConfig
    
    return TerminationComponentConfig(
        function=anchor_pos_error,
        variables={
            "anchor_pos": "current_state_anchor_pos",
            "ref_anchor_pos": "ref_state_anchor_pos",
            "threshold": threshold,
        },
    )


def anchor_ori_error(
    anchor_rot: Tensor,
    ref_anchor_rot: Tensor,
    threshold: float,
) -> Tensor:
    """Check if anchor orientation error exceeds threshold via projected gravity.
    
    Implements BeyondMimic's bad_ref_ori termination condition.
    Compares the z-component of the projected gravity vectors to detect
    if the robot is tilted too much compared to the reference.
    
    Args:
        anchor_rot: Current anchor rotation quaternion [num_envs, 4] (w-last).
        ref_anchor_rot: Reference anchor rotation quaternion [num_envs, 4] (w-last).
        threshold: Maximum allowed difference in projected gravity z-component.
    
    Returns:
        Boolean tensor [num_envs] indicating which envs exceed the threshold.
    """
    # Gravity vector in world frame (pointing down)
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=anchor_rot.device)
    gravity_vec = gravity_vec.unsqueeze(0).expand(anchor_rot.shape[0], -1)
    
    # Project gravity into anchor's local frame
    proj_grav = quat_rotate_inverse(anchor_rot, gravity_vec, w_last=True)
    ref_proj_grav = quat_rotate_inverse(ref_anchor_rot, gravity_vec, w_last=True)
    
    # Compare z-components (how "upright" each is)
    z_diff = torch.abs(proj_grav[:, 2] - ref_proj_grav[:, 2])
    return z_diff > threshold


def anchor_ori_error_factory(threshold: float = 0.8):
    """Factory for anchor orientation error termination (BeyondMimic bad_ref_ori).
    
    Terminates episode when anchor orientation error (via projected gravity)
    exceeds threshold.
    
    Args:
        threshold: Maximum projected gravity z-component difference (default: 0.8).
    
    Returns:
        Pre-configured TerminationComponentConfig.
    """
    from protomotions.envs.base_env.config import TerminationComponentConfig
    
    return TerminationComponentConfig(
        function=anchor_ori_error,
        variables={
            "anchor_rot": "current_state_anchor_rot",
            "ref_anchor_rot": "ref_state_anchor_rot",
            "threshold": threshold,
        },
    )


def relative_body_pos_error(
    body_pos: Tensor,
    ref_body_pos: Tensor,
    anchor_pos: Tensor,
    ref_anchor_pos: Tensor,
    anchor_rot: Tensor,
    ref_anchor_rot: Tensor,
    threshold: float,
) -> Tensor:
    """Check if any body position error exceeds threshold in heading-aligned frame.
    
    Implements BeyondMimic's bad_motion_body_pos termination condition.
    Computes body positions relative to anchor in yaw-aligned local frame,
    making the check invariant to global heading/yaw.
    
    Args:
        body_pos: Current body positions [num_envs, num_bodies, 3].
        ref_body_pos: Reference body positions [num_envs, num_bodies, 3].
        anchor_pos: Current anchor position [num_envs, 3].
        ref_anchor_pos: Reference anchor position [num_envs, 3].
        anchor_rot: Current anchor rotation [num_envs, 4] (w-last).
        ref_anchor_rot: Reference anchor rotation [num_envs, 4] (w-last).
        threshold: Maximum allowed error for any body in meters.
    
    Returns:
        Boolean tensor [num_envs] indicating which envs have any body exceeding threshold.
    """
    num_envs, num_bodies, _ = body_pos.shape
    
    # Get inverse yaw quaternions for both current and reference anchors
    heading_inv = calc_heading_quat_inv(anchor_rot, w_last=True)
    ref_heading_inv = calc_heading_quat_inv(ref_anchor_rot, w_last=True)
    
    # Compute offsets in world frame (body position relative to anchor)
    offset = body_pos - anchor_pos.unsqueeze(1)
    ref_offset = ref_body_pos - ref_anchor_pos.unsqueeze(1)
    
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


def relative_body_pos_error_factory(threshold: float = 0.25):
    """Factory for relative body position error termination (BeyondMimic bad_motion_body_pos).
    
    Terminates episode when any body position error exceeds threshold in
    heading-aligned relative coordinates.
    
    Args:
        threshold: Maximum body position error threshold in meters (default: 0.25).
    
    Returns:
        Pre-configured TerminationComponentConfig.
    """
    from protomotions.envs.base_env.config import TerminationComponentConfig
    
    return TerminationComponentConfig(
        function=relative_body_pos_error,
        variables={
            "body_pos": "current_state_rigid_body_pos",
            "ref_body_pos": "ref_state_rigid_body_pos",
            "anchor_pos": "current_state_anchor_pos",
            "ref_anchor_pos": "ref_state_anchor_pos",
            "anchor_rot": "current_state_anchor_rot",
            "ref_anchor_rot": "ref_state_anchor_rot",
            "threshold": threshold,
        },
    )

