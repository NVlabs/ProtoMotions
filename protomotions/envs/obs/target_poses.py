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
"""Target pose building utilities for mimic environments.

Provides functions for building target pose observations from reference motions,
used for motion tracking and imitation learning.
"""

from typing import List, Union

import torch
from torch import Tensor

from protomotions.utils import rotations
from protomotions.envs.obs.utils import select_step_indices


def build_max_coords_target_poses_future_rel(
    current_state_body_pos: Tensor,
    current_state_body_rot: Tensor,
    mimic_ref_pos: Tensor,
    mimic_ref_rot: Tensor,
    w_last: bool,
    future_steps: Union[int, List[int]] = None,
):
    """Build target pose observations with relative deltas between consecutive future frames.

    Computes future target poses where each frame is expressed relative to the previous frame,
    providing incremental motion information for tracking.

    Args:
        current_state_body_pos: Current body positions [envs, bodies, 3]
        current_state_body_rot: Current body rotations [envs, bodies, 4]
        mimic_ref_pos: Reference body positions [envs, future_steps, bodies, 3]
        mimic_ref_rot: Reference body rotations [envs, future_steps, bodies, 4]
        w_last: If True, quaternions are in XYZW format, else WXYZ
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.

    Returns:
        Target pose observations [envs, features] in root-relative coordinates
    """

    num_envs = current_state_body_pos.shape[0]
    num_bodies = mimic_ref_pos.shape[2]

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        mimic_ref_pos = select_step_indices(mimic_ref_pos, future_steps)
        mimic_ref_rot = select_step_indices(mimic_ref_rot, future_steps)

    future_steps = mimic_ref_pos.shape[1]

    # Flatten reference tensors: [envs, future_steps, bodies, dim] -> [envs*future_steps, bodies, dim]
    ref_state_body_pos = mimic_ref_pos.reshape(-1, num_bodies, 3)
    ref_state_body_rot = mimic_ref_rot.reshape(-1, num_bodies, 4)

    reference_pos = mimic_ref_pos.clone().roll(shifts=1, dims=1)
    reference_pos[:, 0] = current_state_body_pos
    flat_reference_pos = reference_pos.reshape(ref_state_body_pos.shape)

    reference_rot = mimic_ref_rot.clone().roll(shifts=1, dims=1)
    reference_rot[:, 0] = current_state_body_rot
    flat_reference_rot = reference_rot.reshape(ref_state_body_rot.shape)

    reference_root_pos = flat_reference_pos[:, 0, :]
    reference_root_rot = flat_reference_rot[:, 0, :]

    heading_inv_rot = rotations.calc_heading_quat_inv(reference_root_rot, w_last)

    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    pos_heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_reference_pos.shape[1], 1)
    )
    rot_heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_reference_rot.shape[1], 1)
    )
    pos_flat_heading_inv_rot = pos_heading_inv_rot_expand.reshape(
        pos_heading_inv_rot_expand.shape[0] * pos_heading_inv_rot_expand.shape[1],
        pos_heading_inv_rot_expand.shape[2],
    )

    reference_root_pos_expand = reference_root_pos.unsqueeze(-2)

    """target"""
    # target body pos   [N, 3xB]
    target_rel_body_pos = ref_state_body_pos - flat_reference_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(
        target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
        target_rel_body_pos.shape[2],
    )
    flat_target_rel_body_pos = rotations.quat_rotate(
        pos_flat_heading_inv_rot, flat_target_rel_body_pos, w_last
    )

    # target body pos   [N, 3xB]
    flat_target_body_pos = (ref_state_body_pos - reference_root_pos_expand).reshape(
        ref_state_body_pos.shape[0] * ref_state_body_pos.shape[1],
        ref_state_body_pos.shape[2],
    )
    flat_target_body_pos = rotations.quat_rotate(
        pos_flat_heading_inv_rot, flat_target_body_pos, w_last
    )

    # target body rot   [N, 6xB]
    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_reference_rot, w_last), ref_state_body_rot, w_last
    )
    target_rel_body_rot_obs = (
        rotations.quat_to_tan_norm(target_rel_body_rot.view(-1, 4), w_last)
        .reshape(num_envs, future_steps, -1, 6)
        .reshape(target_rel_body_rot.shape[0], -1)
    )

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(
        rot_heading_inv_rot_expand, ref_state_body_rot, w_last
    )
    target_body_rot_obs = (
        rotations.quat_to_tan_norm(target_body_rot.view(-1, 4), w_last)
        .reshape(num_envs, future_steps, -1, 6)
        .reshape(target_rel_body_rot.shape[0], -1)
    )

    target_rel_body_pos = flat_target_rel_body_pos.reshape(
        num_envs, future_steps, -1, 3
    ).reshape(target_rel_body_pos.shape[0], -1)
    target_body_pos = flat_target_body_pos.reshape(
        num_envs, future_steps, -1, 3
    ).reshape(ref_state_body_pos.shape[0], -1)

    obs = torch.cat(
        (
            target_rel_body_pos,
            target_body_pos,
            target_rel_body_rot_obs,
            target_body_rot_obs,
        ),
        dim=-1,
    ).view(num_envs, -1)

    return obs


def build_max_coords_target_poses(
    current_state_body_pos: Tensor,
    current_state_body_rot: Tensor,
    current_state_body_vel: Tensor,
    current_state_body_ang_vel: Tensor,
    mimic_ref_pos: Tensor,
    mimic_ref_rot: Tensor,
    mimic_ref_vel: Tensor,
    mimic_ref_ang_vel: Tensor,
    with_velocities: bool,
    w_last: bool,
    future_steps: Union[int, List[int]] = None,
    with_relative: bool = True,
):
    """Build target pose observations in root-relative coordinates.

    Computes future target poses represented as both absolute (from root) and optionally
    relative (from current pose) transformations, in the root's heading-aligned frame.

    Args:
        current_state_body_pos: Current body positions [envs, bodies, 3]
        current_state_body_rot: Current body rotations [envs, bodies, 4]
        current_state_body_vel: Current body velocities [envs, bodies, 3]
        current_state_body_ang_vel: Current body angular velocities [envs, bodies, 3]
        mimic_ref_pos: Reference body positions [envs, future_steps, bodies, 3]
        mimic_ref_rot: Reference body rotations [envs, future_steps, bodies, 4]
        mimic_ref_vel: Reference body velocities [envs, future_steps, bodies, 3]
        mimic_ref_ang_vel: Reference body angular velocities [envs, future_steps, bodies, 3]
        with_velocities: If True, include velocity information
        w_last: If True, quaternions are in XYZW format, else WXYZ
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
        with_relative: If True, include relative pose observations (pos_rel, rot_rel)

    Returns:
        Target pose observations [envs, features] with absolute and optionally relative pose info
    """
    num_envs = current_state_body_pos.shape[0]
    num_bodies = mimic_ref_pos.shape[2]

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        mimic_ref_pos = select_step_indices(mimic_ref_pos, future_steps)
        mimic_ref_rot = select_step_indices(mimic_ref_rot, future_steps)
        mimic_ref_vel = select_step_indices(mimic_ref_vel, future_steps)
        mimic_ref_ang_vel = select_step_indices(mimic_ref_ang_vel, future_steps)

    future_steps = mimic_ref_pos.shape[1]

    # Flatten reference tensors: [envs, future_steps, bodies, dim] -> [envs*future_steps, bodies, dim]
    ref_state_body_pos = mimic_ref_pos.reshape(-1, num_bodies, 3)
    ref_state_body_rot = mimic_ref_rot.reshape(-1, num_bodies, 4)
    ref_state_body_vel = mimic_ref_vel.reshape(-1, num_bodies, 3)
    ref_state_body_ang_vel = mimic_ref_ang_vel.reshape(-1, num_bodies, 3)

    expanded_body_pos = current_state_body_pos.unsqueeze(1).expand(
        num_envs, future_steps, *current_state_body_pos.shape[1:]
    )
    expanded_body_rot = current_state_body_rot.unsqueeze(1).expand(
        num_envs, future_steps, *current_state_body_rot.shape[1:]
    )

    flat_current_state_body_pos = expanded_body_pos.reshape(ref_state_body_pos.shape)
    flat_current_state_body_rot = expanded_body_rot.reshape(ref_state_body_rot.shape)

    root_pos = flat_current_state_body_pos[:, 0, :]
    root_rot = flat_current_state_body_rot[:, 0, :]

    heading_inv_rot = rotations.calc_heading_quat_inv(root_rot, w_last)

    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    translation_heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_current_state_body_pos.shape[1], 1)
    )
    rotation_heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_current_state_body_rot.shape[1], 1)
    )
    flat_translation_heading_inv_rot = translation_heading_inv_rot_expand.reshape(
        translation_heading_inv_rot_expand.shape[0]
        * translation_heading_inv_rot_expand.shape[1],
        translation_heading_inv_rot_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)

    """target"""
    # target body pos   [N, 3xB]
    flat_target_body_pos = (ref_state_body_pos - root_pos_expand).reshape(
        ref_state_body_pos.shape[0] * ref_state_body_pos.shape[1],
        ref_state_body_pos.shape[2],
    )
    flat_target_body_pos = rotations.quat_rotate(
        flat_translation_heading_inv_rot, flat_target_body_pos, w_last
    )
    target_body_pos = flat_target_body_pos.reshape(num_envs, future_steps, -1)

    flat_target_body_pos_rel = (
        ref_state_body_pos - flat_current_state_body_pos
    ).reshape(
        ref_state_body_pos.shape[0] * ref_state_body_pos.shape[1],
        ref_state_body_pos.shape[2],
    )
    flat_target_body_pos_rel = rotations.quat_rotate(
        flat_translation_heading_inv_rot, flat_target_body_pos_rel, w_last
    )
    target_body_pos_rel = flat_target_body_pos_rel.reshape(num_envs, future_steps, -1)

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(
        rotation_heading_inv_rot_expand, ref_state_body_rot, w_last
    )

    target_body_rot_obs = rotations.quat_to_tan_norm(
        target_body_rot.view(-1, 4), w_last
    ).reshape(num_envs, future_steps, -1)

    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_current_state_body_rot, w_last),
        ref_state_body_rot,
        w_last,
    )
    target_rel_body_rot_obs = rotations.quat_to_tan_norm(
        target_rel_body_rot.view(-1, 4), w_last
    ).reshape(num_envs, future_steps, -1)

    if with_relative:
        obs = torch.cat(
            (
                target_body_pos,
                target_body_pos_rel,
                target_body_rot_obs,
                target_rel_body_rot_obs,
            ),
            dim=-1,
        )
    else:
        obs = torch.cat(
            (
                target_body_pos,
                target_body_rot_obs,
            ),
            dim=-1,
        )

    if with_velocities:
        expanded_body_vel = current_state_body_vel.unsqueeze(1).expand(
            num_envs, future_steps, *current_state_body_vel.shape[1:]
        )
        flat_current_state_body_vel = expanded_body_vel.reshape(
            ref_state_body_vel.shape
        )

        flat_target_vel = (ref_state_body_vel - flat_current_state_body_vel).reshape(
            ref_state_body_vel.shape[0] * ref_state_body_vel.shape[1],
            ref_state_body_vel.shape[2],
        )
        flat_local_target_vel = rotations.quat_rotate(
            translation_heading_inv_rot_expand, flat_target_vel, w_last
        )
        local_target_vel = flat_local_target_vel.reshape(num_envs, future_steps, -1)

        expanded_body_ang_vel = current_state_body_ang_vel.unsqueeze(1).expand(
            num_envs, future_steps, *current_state_body_ang_vel.shape[1:]
        )
        flat_current_state_body_ang_vel = expanded_body_ang_vel.reshape(
            ref_state_body_ang_vel.shape
        )

        flat_target_body_ang_vel = (
            ref_state_body_ang_vel - flat_current_state_body_ang_vel
        ).reshape(
            ref_state_body_ang_vel.shape[0] * ref_state_body_ang_vel.shape[1],
            ref_state_body_ang_vel.shape[2],
        )
        flat_local_target_ang_vel = rotations.quat_rotate(
            rotation_heading_inv_rot_expand, flat_target_body_ang_vel, w_last
        )
        local_target_ang_vel = flat_local_target_ang_vel.reshape(
            num_envs, future_steps, -1
        )

        obs = torch.cat(
            (
                obs,
                local_target_vel,
                local_target_ang_vel,
            ),
            dim=-1,
        )

    return obs.view(num_envs, -1)


# Context mapping for ONNX export


def build_reduced_coords_target_poses(
    current_state_anchor_rot: Tensor,
    mimic_ref_anchor_rot: Tensor,
    mimic_ref_dof_vel: Tensor,
    mimic_ref_dof_pos: Tensor,
    w_last: bool = True,
    current_state_anchor_pos: Tensor = None,
    mimic_ref_anchor_pos: Tensor = None,
    mimic_ref_anchor_vel: Tensor = None,
    mimic_ref_anchor_ang_vel: Tensor = None,
    include_xy_offset: bool = False,
    include_height: bool = False,
    include_dof_vel: bool = True,
    include_anchor_vel: bool = False,
    include_anchor_ang_vel: bool = False,
    future_steps: Union[int, List[int]] = None,
    current_ref_anchor_pos: Tensor = None,
    zero_xy_offset: bool = False,
):
    """Build target pose observations in reduced coordinates.

    Args:
        current_state_anchor_rot: Current anchor rotation [envs, 4]
        mimic_ref_anchor_rot: Reference body rotations [envs, future_steps, 4]
        mimic_ref_dof_vel: Reference DOF velocities [envs, future_steps, num_dofs]
        mimic_ref_dof_pos: Reference DOF positions [envs, future_steps, num_dofs]
        w_last: If True, quaternions are in XYZW format
        current_state_anchor_pos: Current anchor position [envs, 3]
        mimic_ref_anchor_pos: Reference body positions [envs, future_steps, 3]
        mimic_ref_anchor_vel: Reference anchor linear velocity [envs, future_steps, 3]
        mimic_ref_anchor_ang_vel: Reference anchor angular velocity [envs, future_steps, 3]
        include_xy_offset: If True, includes XY offset [2]
        include_height: If True, includes absolute height [1]
        include_dof_vel: If True, includes DOF velocities [num_dofs]
        include_anchor_vel: If True, includes anchor linear velocity in local frame [3]
        include_anchor_ang_vel: If True, includes anchor angular velocity in local frame [3]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
        current_ref_anchor_pos: Current-frame reference anchor position [envs, 3].
            Used with include_xy_offset to compute drift from current reference.
        zero_xy_offset: If True, emit zeros for the XY offset (for inference).

    Returns:
        Target pose observations [envs, features]
    """
    num_envs = current_state_anchor_rot.shape[0]

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        mimic_ref_anchor_rot = select_step_indices(mimic_ref_anchor_rot, future_steps)
        mimic_ref_dof_vel = select_step_indices(mimic_ref_dof_vel, future_steps)
        mimic_ref_dof_pos = select_step_indices(mimic_ref_dof_pos, future_steps)
        if mimic_ref_anchor_pos is not None:
            mimic_ref_anchor_pos = select_step_indices(
                mimic_ref_anchor_pos, future_steps
            )
        if mimic_ref_anchor_vel is not None:
            mimic_ref_anchor_vel = select_step_indices(
                mimic_ref_anchor_vel, future_steps
            )
        if mimic_ref_anchor_ang_vel is not None:
            mimic_ref_anchor_ang_vel = select_step_indices(
                mimic_ref_anchor_ang_vel, future_steps
            )

    future_steps = mimic_ref_anchor_rot.shape[1]

    # Flatten: [envs, future_steps, dim] -> [envs*future_steps, dim]
    ref_state_anchor_rot = mimic_ref_anchor_rot.reshape(-1, 4)
    ref_state_dof_vel = mimic_ref_dof_vel.reshape(-1, mimic_ref_dof_vel.shape[-1])
    ref_state_dof_pos = mimic_ref_dof_pos.reshape(-1, mimic_ref_dof_pos.shape[-1])

    heading_inv_rot = rotations.calc_heading_quat_inv(current_state_anchor_rot, w_last)

    current_state_anchor_rot_expanded = (
        current_state_anchor_rot.unsqueeze(1)
        .expand(num_envs, future_steps, 4)
        .contiguous()
        .view(-1, 4)
    )

    heading_inv_rot_expanded = (
        heading_inv_rot.unsqueeze(1)
        .expand(num_envs, future_steps, 4)
        .contiguous()
        .view(-1, 4)
    )

    # Target root rot relative to current root rot
    rel_target_anchor_rot = rotations.quat_mul(
        rotations.quat_conjugate(current_state_anchor_rot_expanded, w_last),
        ref_state_anchor_rot,
        w_last,
    )
    target_anchor_rot_obs = rotations.quat_to_tan_norm(rel_target_anchor_rot, w_last)

    # Build observation components
    obs_components = [target_anchor_rot_obs]  # [N, 6]

    if include_dof_vel:
        obs_components.append(ref_state_dof_vel)  # [N, num_dofs]

    obs_components.append(ref_state_dof_pos)  # [N, num_dofs]

    if include_xy_offset or include_height:
        # Compute position components
        # Extract root position: [envs, future_steps, 3] -> [envs*future_steps, 3]
        ref_state_anchor_pos = mimic_ref_anchor_pos.reshape(-1, 3)

        if include_xy_offset:
            if zero_xy_offset:
                # Inference: zero out XY offset ("you're tracking perfectly")
                xy_offset_local = torch.zeros(
                    num_envs * future_steps,
                    2,
                    device=current_state_anchor_rot.device,
                    dtype=current_state_anchor_rot.dtype,
                )
            else:
                # Training: drift from current reference frame (state_t - hat_state_t)
                drift_origin = (
                    current_ref_anchor_pos
                    if current_ref_anchor_pos is not None
                    else current_state_anchor_pos
                )
                drift_origin_expanded = (
                    drift_origin.unsqueeze(1)
                    .expand(num_envs, future_steps, 3)
                    .contiguous()
                    .view(-1, 3)
                )

                # XY drift in world frame: agent pos - reference pos
                xy_drift_world = (
                    current_state_anchor_pos.unsqueeze(1)
                    .expand(num_envs, future_steps, 3)
                    .contiguous()
                    .view(-1, 3)[:, :2]
                    - drift_origin_expanded[:, :2]
                )

                # Rotate to heading-aligned frame
                xy_drift_3d = torch.cat(
                    [xy_drift_world, torch.zeros_like(xy_drift_world[:, :1])], dim=-1
                )
                xy_drift_local_3d = rotations.quat_rotate(
                    heading_inv_rot_expanded, xy_drift_3d, w_last
                )
                xy_offset_local = xy_drift_local_3d[:, :2]  # [num_envs*future_steps, 2]
            obs_components.append(xy_offset_local)

        if include_height:
            # Absolute height (not offset from current)
            height = ref_state_anchor_pos[:, 2:3]  # [num_envs*future_steps, 1]
            obs_components.append(height)

    if include_anchor_vel:
        if mimic_ref_anchor_vel is None:
            raise ValueError(
                "mimic_ref_anchor_vel is required when include_anchor_vel=True"
            )
        ref_state_anchor_vel = mimic_ref_anchor_vel.reshape(-1, 3)
        # Transform to local frame (heading-aligned)
        local_anchor_vel = rotations.quat_rotate(
            heading_inv_rot_expanded, ref_state_anchor_vel, w_last
        )
        obs_components.append(local_anchor_vel)

    if include_anchor_ang_vel:
        if mimic_ref_anchor_ang_vel is None:
            raise ValueError(
                "mimic_ref_anchor_ang_vel is required when include_anchor_ang_vel=True"
            )
        ref_state_anchor_ang_vel = mimic_ref_anchor_ang_vel.reshape(-1, 3)
        # Transform to local frame
        local_anchor_ang_vel = rotations.quat_rotate(
            heading_inv_rot_expanded, ref_state_anchor_ang_vel, w_last
        )
        obs_components.append(local_anchor_ang_vel)

    # Concatenate all observations
    obs = torch.cat(obs_components, dim=-1)

    # Reshape to [num_envs, future_steps * features]
    return obs.view(num_envs, -1)


# Context mapping for reduced coords target poses


def build_sparse_target_poses(
    current_state_body_pos: Tensor,
    current_state_body_rot: Tensor,
    masked_mimic_ref_pos: Tensor,
    masked_mimic_ref_rot: Tensor,
    conditionable_body_ids: Tensor,
    w_last: bool,
    future_steps: Union[int, List[int]] = None,
    include_root_relative: bool = True,
):
    """Build target pose observations for sparse body tracking (MaskedMimic).

    Similar to max_coords but only includes conditionable bodies (e.g., head, hands for VR).
    Provides both absolute and relative pose encodings for partial body control.

    Args:
        current_state_body_pos: Current body positions [envs, bodies, 3]
        current_state_body_rot: Current body rotations [envs, bodies, 4]
        masked_mimic_ref_pos: Target positions [envs, future_steps, bodies, 3]
        masked_mimic_ref_rot: Target rotations [envs, future_steps, bodies, 4]
        conditionable_body_ids: Indices of trackable bodies
        w_last: If True, quaternions are in XYZW format, else WXYZ
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
        include_root_relative: If True (default), include both body-relative and root-relative
            poses (24 features per body). If False, only include body-relative poses
            (12 features per body: pos delta + rot delta from current to target).

    Returns:
        Sparse target pose observations [envs, features] for conditionable bodies only
    """
    num_envs = current_state_body_pos.shape[0]
    num_bodies = masked_mimic_ref_pos.shape[2]

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        masked_mimic_ref_pos = select_step_indices(masked_mimic_ref_pos, future_steps)
        masked_mimic_ref_rot = select_step_indices(masked_mimic_ref_rot, future_steps)

    future_steps = masked_mimic_ref_pos.shape[1]

    # Flatten reference tensors: [envs, future_steps, bodies, dim] -> [envs*future_steps, bodies, dim]
    flat_target_body_pos = masked_mimic_ref_pos.reshape(-1, num_bodies, 3)
    flat_target_body_rot = masked_mimic_ref_rot.reshape(-1, num_bodies, 4)

    expanded_body_pos = current_state_body_pos.unsqueeze(1).expand(
        num_envs, future_steps, *current_state_body_pos.shape[1:]
    )
    expanded_body_rot = current_state_body_rot.unsqueeze(1).expand(
        num_envs, future_steps, *current_state_body_rot.shape[1:]
    )

    flat_cur_pos = expanded_body_pos.reshape(flat_target_body_pos.shape)
    flat_cur_rot = expanded_body_rot.reshape(flat_target_body_rot.shape)

    current_state_root_pos = flat_cur_pos[:, 0, :]
    current_state_root_rot = flat_cur_rot[:, 0, :]

    heading_rot = rotations.calc_heading_quat_inv(current_state_root_rot, w_last)

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, flat_cur_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )

    current_state_root_pos_expand = current_state_root_pos.unsqueeze(-2)

    """target"""
    # target body pos   [N, 3xB]
    target_rel_body_pos = flat_target_body_pos - flat_cur_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(
        target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
        target_rel_body_pos.shape[2],
    )
    flat_target_rel_body_pos = rotations.quat_rotate(
        flat_heading_rot, flat_target_rel_body_pos, w_last
    )

    # target body pos   [N, 3xB]
    flat_target_body_pos = (
        flat_target_body_pos - current_state_root_pos_expand
    ).reshape(
        flat_target_body_pos.shape[0] * flat_target_body_pos.shape[1],
        flat_target_body_pos.shape[2],
    )
    flat_target_body_pos = rotations.quat_rotate(
        flat_heading_rot, flat_target_body_pos, w_last
    )

    # target body rot   [N, 6xB]
    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_cur_rot, w_last), flat_target_body_rot, w_last
    )
    target_rel_body_rot_obs = rotations.quat_to_tan_norm(
        target_rel_body_rot.view(-1, 4), w_last
    ).view(target_rel_body_rot.shape[0], -1)

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(
        heading_rot_expand, flat_target_body_rot, w_last
    )
    target_body_rot_obs = rotations.quat_to_tan_norm(
        target_body_rot.view(-1, 4), w_last
    ).view(target_rel_body_rot.shape[0], -1)

    padded_flat_target_rel_body_pos = torch.nn.functional.pad(
        flat_target_rel_body_pos, [0, 3], "constant", 0
    )
    sub_sampled_target_rel_body_pos = padded_flat_target_rel_body_pos.reshape(
        num_envs, future_steps, -1, 6
    )[:, :, conditionable_body_ids]

    padded_flat_target_body_pos = torch.nn.functional.pad(
        flat_target_body_pos, [0, 3], "constant", 0
    )
    sub_sampled_target_body_pos = padded_flat_target_body_pos.reshape(
        num_envs, future_steps, -1, 6
    )[:, :, conditionable_body_ids]

    sub_sampled_target_rel_body_rot_obs = target_rel_body_rot_obs.reshape(
        num_envs, future_steps, -1, 6
    )[:, :, conditionable_body_ids]
    sub_sampled_target_body_rot_obs = target_body_rot_obs.reshape(
        num_envs, future_steps, -1, 6
    )[:, :, conditionable_body_ids]

    # In masked_mimic allow easy re-shape to [batch, time, joint, type (transform/rotate), features]
    if include_root_relative:
        # Full output: body-relative + root-relative (24 features per body)
        obs = torch.cat(
            (
                sub_sampled_target_rel_body_pos,
                sub_sampled_target_body_pos,
                sub_sampled_target_rel_body_rot_obs,
                sub_sampled_target_body_rot_obs,
            ),
            dim=-1,  # [batch, timesteps, joints, 24]
        ).view(num_envs, -1)
    else:
        # Reduced output: only body-relative (12 features per body)
        # pos delta (current body -> target body) + rot delta (current rot -> target rot)
        obs = torch.cat(
            (
                sub_sampled_target_rel_body_pos,
                sub_sampled_target_rel_body_rot_obs,
            ),
            dim=-1,  # [batch, timesteps, joints, 12]
        ).view(num_envs, -1)

    return obs


# =============================================================================
# Individual Component Build Functions (for modular factories)
# =============================================================================
# All functions support future_steps for multi-frame targets.
# Output is flattened: [envs, future_steps * feature_dim]


def build_target_root_rot(
    current_state_root_rot: Tensor,
    mimic_ref_anchor_rot: Tensor,
    future_steps: Union[int, List[int]] = 1,
    w_last: bool = True,
) -> Tensor:
    """Build target root rotation observation (6D tan-norm).

    Args:
        current_state_root_rot: Current root rotation [envs, 4]
        mimic_ref_anchor_rot: Reference anchor rotation [envs, future_steps, 4]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]).
        w_last: If True, quaternions are in XYZW format

    Returns:
        Relative root rotation [envs, future_steps * 6]
    """
    num_envs = current_state_root_rot.shape[0]

    # Slice to requested steps
    ref_anchor_rot = select_step_indices(mimic_ref_anchor_rot, future_steps)
    actual_steps = ref_anchor_rot.shape[1]

    # Expand current rotation to match
    current_expanded = current_state_root_rot.unsqueeze(1).expand(-1, actual_steps, -1)
    current_flat = current_expanded.reshape(-1, 4)  # [envs*steps, 4]
    ref_flat = ref_anchor_rot.reshape(-1, 4)  # [envs*steps, 4]

    # Relative rotation from current to target
    rel_target_root_rot = rotations.quat_mul(
        rotations.quat_conjugate(current_flat, w_last), ref_flat, w_last
    )
    rot_6d = rotations.quat_to_tan_norm(rel_target_root_rot, w_last)  # [envs*steps, 6]
    return rot_6d.reshape(num_envs, -1)


def build_target_xy_offset(
    current_state_anchor_pos: Tensor,
    current_state_anchor_rot: Tensor,
    mimic_ref_anchor_pos: Tensor,
    future_steps: Union[int, List[int]] = 1,
    w_last: bool = True,
    current_ref_anchor_pos: Tensor = None,
    zero_xy_offset: bool = False,
) -> Tensor:
    """Build target XY offset in heading frame.

    Computes drift from current reference frame (state_t - hat_state_t).

    Args:
        current_state_anchor_pos: Current anchor position [envs, 3]
        current_state_anchor_rot: Current anchor rotation [envs, 4]
        mimic_ref_anchor_pos: Reference anchor position [envs, future_steps, 3]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]).
        w_last: If True, quaternions are in XYZW format
        current_ref_anchor_pos: Current-frame reference anchor position [envs, 3].
            Used to compute drift from current reference.
        zero_xy_offset: If True, emit zeros (for inference).

    Returns:
        XY offset in heading frame [envs, future_steps * 2]
    """
    num_envs = current_state_anchor_pos.shape[0]

    # Slice to requested steps
    ref_anchor_pos = select_step_indices(mimic_ref_anchor_pos, future_steps)
    actual_steps = ref_anchor_pos.shape[1]

    if zero_xy_offset:
        return torch.zeros(
            num_envs,
            actual_steps * 2,
            device=current_state_anchor_pos.device,
            dtype=current_state_anchor_pos.dtype,
        )

    heading_inv_rot = rotations.calc_heading_quat_inv(current_state_anchor_rot, w_last)
    heading_inv_expanded = heading_inv_rot.unsqueeze(1).expand(-1, actual_steps, -1)
    heading_inv_flat = heading_inv_expanded.reshape(-1, 4)  # [envs*steps, 4]

    # Drift from current reference frame
    drift_origin = (
        current_ref_anchor_pos
        if current_ref_anchor_pos is not None
        else current_state_anchor_pos
    )
    drift_origin_expanded = drift_origin.unsqueeze(1).expand(-1, actual_steps, -1)

    # XY drift in world frame: agent pos - reference pos
    xy_drift_world = (
        current_state_anchor_pos.unsqueeze(1).expand(-1, actual_steps, -1)[:, :, :2]
        - drift_origin_expanded[:, :, :2]
    )

    # Rotate to heading-aligned frame
    xy_drift_flat = xy_drift_world.reshape(-1, 2)  # [envs*steps, 2]
    xy_drift_3d = torch.cat(
        [xy_drift_flat, torch.zeros_like(xy_drift_flat[:, :1])], dim=-1
    )
    xy_drift_local_3d = rotations.quat_rotate(heading_inv_flat, xy_drift_3d, w_last)
    xy_drift_local = xy_drift_local_3d[:, :2]  # [envs*steps, 2]

    return xy_drift_local.reshape(num_envs, -1)


def build_target_height(
    mimic_ref_anchor_pos: Tensor,
    future_steps: Union[int, List[int]] = 1,
) -> Tensor:
    """Build target absolute height observation.

    Args:
        mimic_ref_anchor_pos: Reference anchor position [envs, future_steps, 3]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]).

    Returns:
        Absolute height [envs, future_steps * 1]
    """
    num_envs = mimic_ref_anchor_pos.shape[0]
    ref_pos = select_step_indices(mimic_ref_anchor_pos, future_steps)
    heights = ref_pos[:, :, 2:3]  # [envs, steps, 1]
    return heights.reshape(num_envs, -1)


def build_target_root_vel(
    current_state_anchor_rot: Tensor,
    mimic_ref_root_vel: Tensor,
    future_steps: Union[int, List[int]] = 1,
    w_last: bool = True,
) -> Tensor:
    """Build target root linear velocity in local frame.

    Args:
        current_state_anchor_rot: Current anchor rotation [envs, 4]
        mimic_ref_root_vel: Reference root velocity [envs, future_steps, 3]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]).
        w_last: If True, quaternions are in XYZW format

    Returns:
        Root velocity in heading frame [envs, future_steps * 3]
    """
    num_envs = current_state_anchor_rot.shape[0]

    ref_root_vel = select_step_indices(mimic_ref_root_vel, future_steps)
    actual_steps = ref_root_vel.shape[1]

    heading_inv_rot = rotations.calc_heading_quat_inv(current_state_anchor_rot, w_last)
    heading_inv_expanded = heading_inv_rot.unsqueeze(1).expand(-1, actual_steps, -1)
    heading_inv_flat = heading_inv_expanded.reshape(-1, 4)  # [envs*steps, 4]

    ref_vel_flat = ref_root_vel.reshape(-1, 3)  # [envs*steps, 3]
    local_vel = rotations.quat_rotate(
        heading_inv_flat, ref_vel_flat, w_last
    )  # [envs*steps, 3]

    return local_vel.reshape(num_envs, -1)


def build_target_root_ang_vel(
    current_state_anchor_rot: Tensor,
    mimic_ref_root_ang_vel: Tensor,
    future_steps: Union[int, List[int]] = 1,
    w_last: bool = True,
) -> Tensor:
    """Build target root angular velocity in local frame.

    Args:
        current_state_anchor_rot: Current anchor rotation [envs, 4]
        mimic_ref_root_ang_vel: Reference root angular velocity [envs, future_steps, 3]
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]).
        w_last: If True, quaternions are in XYZW format

    Returns:
        Root angular velocity in local frame [envs, future_steps * 3]
    """
    num_envs = current_state_anchor_rot.shape[0]

    ref_root_ang_vel = select_step_indices(mimic_ref_root_ang_vel, future_steps)
    actual_steps = ref_root_ang_vel.shape[1]

    heading_inv_rot = rotations.calc_heading_quat_inv(current_state_anchor_rot, w_last)
    heading_inv_expanded = heading_inv_rot.unsqueeze(1).expand(-1, actual_steps, -1)
    heading_inv_flat = heading_inv_expanded.reshape(-1, 4)  # [envs*steps, 4]

    ref_ang_vel_flat = ref_root_ang_vel.reshape(-1, 3)  # [envs*steps, 3]
    local_ang_vel = rotations.quat_rotate(
        heading_inv_flat, ref_ang_vel_flat, w_last
    )  # [envs*steps, 3]

    return local_ang_vel.reshape(num_envs, -1)


def build_deploy_target_poses(
    current_anchor_rot: Tensor,
    mimic_ref_rot: Tensor,
    mimic_ref_dof_pos: Tensor,
    mimic_ref_dof_vel: Tensor,
    w_last: bool = True,
    include_dof_vel: bool = True,
    future_steps: Union[int, List[int]] = None,
):
    """Build deployment-ready target pose observations.

    Only requires the robot's anchor/root orientation (from IMU during deployment)
    and reference motion data.  No position information is needed, making this
    suitable for deployment without external position tracking.

    The observation encodes:
    - Reference DOF positions (joint targets, frame-invariant)
    - Reference DOF velocities (joint velocity targets, frame-invariant)
    - Reference body rotations relative to current anchor orientation (6D per body)

    During deployment:
    1. At start, compute heading offset: heading(IMU) - heading(motion_root) at t=0
    2. Each step, apply heading offset to raw reference body rotations before feeding
       to this function.  In training, realign_motion_with_humanoid_on_each_step
       handles XY position alignment; rotation alignment at episode start is implicit
       from spawning at the reference pose.

    Args:
        current_anchor_rot: Current anchor body rotation [envs, 4] (IMU during deploy)
        mimic_ref_rot: Reference body rotations [envs, future_steps, num_bodies, 4]
        mimic_ref_dof_pos: Reference DOF positions [envs, future_steps, num_dofs]
        mimic_ref_dof_vel: Reference DOF velocities [envs, future_steps, num_dofs]
        w_last: If True, quaternions are in XYZW format
        include_dof_vel: If True, include DOF velocities in observation
        future_steps: Steps to select.  Int N for first N consecutive steps,
            list for specific step indices.  None = use all.

    Returns:
        Target pose observations [envs, features] containing:
        [ref_dof_pos, (ref_dof_vel), local_ref_body_rot_6d] per future step
    """
    num_envs = current_anchor_rot.shape[0]

    if future_steps is not None:
        mimic_ref_rot = select_step_indices(mimic_ref_rot, future_steps)
        mimic_ref_dof_pos = select_step_indices(mimic_ref_dof_pos, future_steps)
        mimic_ref_dof_vel = select_step_indices(mimic_ref_dof_vel, future_steps)

    n_future = mimic_ref_rot.shape[1]
    num_bodies = mimic_ref_rot.shape[2]

    ref_body_rot = mimic_ref_rot.reshape(-1, num_bodies, 4)
    ref_dof_pos = mimic_ref_dof_pos.reshape(-1, mimic_ref_dof_pos.shape[-1])
    ref_dof_vel = mimic_ref_dof_vel.reshape(-1, mimic_ref_dof_vel.shape[-1])

    current_rot_expanded = (
        current_anchor_rot.unsqueeze(1)
        .expand(num_envs, n_future, 4)
        .contiguous()
        .view(-1, 4)
    )
    current_rot_inv = rotations.quat_conjugate(current_rot_expanded, w_last)

    # ref body rotations in current anchor frame
    current_rot_inv_expand = current_rot_inv.unsqueeze(1).repeat(1, num_bodies, 1)
    local_ref_body_rot = rotations.quat_mul(
        current_rot_inv_expand, ref_body_rot, w_last
    )

    local_ref_body_rot_6d = rotations.quat_to_tan_norm(
        local_ref_body_rot.view(-1, 4), w_last
    ).reshape(num_envs * n_future, num_bodies * 6)

    obs_components = [ref_dof_pos]
    if include_dof_vel:
        obs_components.append(ref_dof_vel)
    obs_components.append(local_ref_body_rot_6d)

    obs = torch.cat(obs_components, dim=-1)
    return obs.view(num_envs, -1)
