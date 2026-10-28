# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target pose building utilities for mimic environments.

Provides functions for building target pose observations from reference motions,
used for motion tracking and imitation learning.
"""

from typing import List, Optional

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.steps import StepSelection, select_steps
from protomotions.utils import rotations


def build_max_coords_target_poses_future_rel(
    current_state_body_pos: Tensor = EnvContext.current.rigid_body_pos,
    current_state_body_rot: Tensor = EnvContext.current.rigid_body_rot,
    mimic_ref_pos: Tensor = EnvContext.mimic.future_pos,
    mimic_ref_rot: Tensor = EnvContext.mimic.future_rot,
    w_last: bool = True,
    future_steps: Optional[StepSelection] = None,
    anchor_body_index: int = EnvContext.mimic.anchor_idx,
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
        future_steps: The 1-based steps to read, in output order (a recipe may
            write an int N for 1..N). None = every published step.
        anchor_body_index: Body index used as the current/reference anchor frame.

    Returns:
        Target pose observations [envs, features] in anchor-relative coordinates
    """

    num_envs = current_state_body_pos.shape[0]
    num_bodies = mimic_ref_pos.shape[2]
    if not 0 <= anchor_body_index < num_bodies:
        raise ValueError(
            f"anchor_body_index {anchor_body_index} is outside [0, {num_bodies})"
        )

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        mimic_ref_pos = select_steps(mimic_ref_pos, future_steps)
        mimic_ref_rot = select_steps(mimic_ref_rot, future_steps)

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

    reference_root_pos = flat_reference_pos[:, anchor_body_index, :]
    reference_root_rot = flat_reference_rot[:, anchor_body_index, :]

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
    current_state_body_pos: Tensor = EnvContext.current.rigid_body_pos,
    current_state_body_rot: Tensor = EnvContext.current.rigid_body_rot,
    current_state_body_vel: Tensor = EnvContext.current.rigid_body_vel,
    current_state_body_ang_vel: Tensor = EnvContext.current.rigid_body_ang_vel,
    mimic_ref_pos: Tensor = EnvContext.mimic.future_pos,
    mimic_ref_rot: Tensor = EnvContext.mimic.future_rot,
    mimic_ref_vel: Tensor = EnvContext.mimic.future_vel,
    mimic_ref_ang_vel: Tensor = EnvContext.mimic.future_ang_vel,
    with_velocities: bool = True,
    w_last: bool = True,
    future_steps: Optional[StepSelection] = None,
    with_relative: bool = True,
    anchor_body_index: int = EnvContext.mimic.anchor_idx,
    body_indices: Optional[List[int]] = None,
):
    """Build target pose observations in anchor-relative coordinates.

    Computes future target poses represented as both absolute (from the anchor) and optionally
    relative (from current pose) transformations, in the anchor's heading-aligned frame.

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
        future_steps: The 1-based steps to read, in output order (a recipe may
            write an int N for 1..N). None = every published step.
        with_relative: If True, include relative pose observations (pos_rel, rot_rel)
        anchor_body_index: Body index used as the current/reference anchor frame.
        body_indices: Observed bodies in output order. Anchor indexing always
            refers to the original body array, even if the anchor is not selected.

    Returns:
        Target pose observations [envs, features] with absolute and optionally relative pose info
    """
    num_envs = current_state_body_pos.shape[0]
    num_bodies = mimic_ref_pos.shape[2]
    if not 0 <= anchor_body_index < num_bodies:
        raise ValueError(
            f"anchor_body_index {anchor_body_index} is outside [0, {num_bodies})"
        )

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        mimic_ref_pos = select_steps(mimic_ref_pos, future_steps)
        mimic_ref_rot = select_steps(mimic_ref_rot, future_steps)
        mimic_ref_vel = select_steps(mimic_ref_vel, future_steps)
        mimic_ref_ang_vel = select_steps(mimic_ref_ang_vel, future_steps)

    future_steps = mimic_ref_pos.shape[1]

    anchor_pos = current_state_body_pos[:, anchor_body_index].unsqueeze(1).expand(
        -1, future_steps, -1
    ).reshape(-1, 3)
    anchor_rot = current_state_body_rot[:, anchor_body_index].unsqueeze(1).expand(
        -1, future_steps, -1
    ).reshape(-1, 4)
    if body_indices is not None:
        if not body_indices or any(i < 0 or i >= num_bodies for i in body_indices):
            raise ValueError("body_indices must select at least one valid body")
        current_state_body_pos = current_state_body_pos[:, body_indices]
        current_state_body_rot = current_state_body_rot[:, body_indices]
        current_state_body_vel = current_state_body_vel[:, body_indices]
        current_state_body_ang_vel = current_state_body_ang_vel[:, body_indices]
        mimic_ref_pos = mimic_ref_pos[:, :, body_indices]
        mimic_ref_rot = mimic_ref_rot[:, :, body_indices]
        mimic_ref_vel = mimic_ref_vel[:, :, body_indices]
        mimic_ref_ang_vel = mimic_ref_ang_vel[:, :, body_indices]
        num_bodies = len(body_indices)

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

    heading_inv_rot = rotations.calc_heading_quat_inv(anchor_rot, w_last)

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

    root_pos_expand = anchor_pos.unsqueeze(-2)

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


def build_target_dof_pos(
    mimic_ref_dof_pos: Tensor,
    future_steps: Optional[StepSelection] = None,
) -> Tensor:
    """Flatten target joint angles in time-major, then DOF order."""
    if future_steps is not None:
        mimic_ref_dof_pos = select_steps(mimic_ref_dof_pos, future_steps)
    return mimic_ref_dof_pos.flatten(start_dim=1)


def build_reduced_coords_target_poses(
    current_state_anchor_rot: Tensor = EnvContext.current.anchor_rot,
    mimic_ref_anchor_rot: Tensor = EnvContext.mimic.future_anchor_rot,
    mimic_ref_dof_vel: Tensor = EnvContext.mimic.future_dof_vel,
    mimic_ref_dof_pos: Tensor = EnvContext.mimic.future_dof_pos,
    w_last: bool = True,
    current_state_anchor_pos: Tensor = None,
    mimic_ref_anchor_pos: Tensor = EnvContext.mimic.future_anchor_pos,
    mimic_ref_anchor_vel: Tensor = EnvContext.mimic.future_anchor_vel,
    mimic_ref_anchor_ang_vel: Tensor = EnvContext.mimic.future_anchor_ang_vel,
    include_xy_offset: bool = False,
    include_height: bool = False,
    include_dof_vel: bool = True,
    include_anchor_vel: bool = False,
    include_anchor_ang_vel: bool = False,
    future_steps: Optional[StepSelection] = None,
    current_ref_anchor_pos: Tensor = EnvContext.mimic.ref_anchor_pos,
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
        future_steps: The 1-based steps to read, in output order (a recipe may
            write an int N for 1..N). None = every published step.
        current_ref_anchor_pos: Current-frame reference anchor position [envs, 3].
            Used with include_xy_offset to compute drift from current reference.
        zero_xy_offset: If True, emit zeros for the XY offset (for inference).

    Returns:
        Target pose observations [envs, features]
    """
    num_envs = current_state_anchor_rot.shape[0]

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        mimic_ref_anchor_rot = select_steps(mimic_ref_anchor_rot, future_steps)
        mimic_ref_dof_vel = select_steps(mimic_ref_dof_vel, future_steps)
        mimic_ref_dof_pos = select_steps(mimic_ref_dof_pos, future_steps)
        if mimic_ref_anchor_pos is not None:
            mimic_ref_anchor_pos = select_steps(
                mimic_ref_anchor_pos, future_steps
            )
        if mimic_ref_anchor_vel is not None:
            mimic_ref_anchor_vel = select_steps(
                mimic_ref_anchor_vel, future_steps
            )
        if mimic_ref_anchor_ang_vel is not None:
            mimic_ref_anchor_ang_vel = select_steps(
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
                # Training: drift from current reference frame (state_t - hat_state_t).
                # The measured drift needs the robot's own anchor position; the
                # kernel default for it is None (localization is not universal),
                # so a recipe with include_xy_offset must bind it explicitly.
                if current_state_anchor_pos is None:
                    raise ValueError(
                        "current_state_anchor_pos is required when "
                        "include_xy_offset=True and zero_xy_offset=False"
                    )
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
    future_steps: Optional[StepSelection] = None,
    include_root_relative: bool = True,
    anchor_body_index: int = EnvContext.mimic.anchor_idx,
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
        future_steps: The 1-based steps to read, in output order (a recipe may
            write an int N for 1..N). None = every published step.
        include_root_relative: If True (default), include both body-relative and root-relative
            poses (24 features per body). If False, only include body-relative poses
            (12 features per body: pos delta + rot delta from current to target).
            The root-relative stream uses anchor_body_index (root by default).
        anchor_body_index: Body used as the heading and position reference.

    Returns:
        Sparse target pose observations [envs, features] for conditionable bodies only
    """
    num_envs = current_state_body_pos.shape[0]
    num_bodies = masked_mimic_ref_pos.shape[2]

    # Slice to requested number of future steps if specified
    if future_steps is not None:
        masked_mimic_ref_pos = select_steps(masked_mimic_ref_pos, future_steps)
        masked_mimic_ref_rot = select_steps(masked_mimic_ref_rot, future_steps)

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

    current_state_root_pos = flat_cur_pos[:, anchor_body_index, :]
    current_state_root_rot = flat_cur_rot[:, anchor_body_index, :]

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


def build_deploy_target_poses(
    current_anchor_rot: Tensor = EnvContext.current.anchor_rot,
    mimic_ref_rot: Tensor = EnvContext.mimic.future_rot,
    mimic_ref_dof_pos: Tensor = EnvContext.mimic.future_dof_pos,
    mimic_ref_dof_vel: Tensor = EnvContext.mimic.future_dof_vel,
    w_last: bool = True,
    include_dof_vel: bool = True,
    future_steps: Optional[StepSelection] = None,
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
        future_steps: The 1-based steps to read, in output order (a recipe may
            write an int N for 1..N). None = every published step.

    Returns:
        Target pose observations [envs, features] containing:
        [ref_dof_pos, (ref_dof_vel), local_ref_body_rot_6d] per future step
    """
    num_envs = current_anchor_rot.shape[0]

    if future_steps is not None:
        mimic_ref_rot = select_steps(mimic_ref_rot, future_steps)
        mimic_ref_dof_pos = select_steps(mimic_ref_dof_pos, future_steps)
        mimic_ref_dof_vel = select_steps(mimic_ref_dof_vel, future_steps)

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


def compute_odom_offset_local(
    ref_anchor_pos: Tensor = EnvContext.mimic.ref_anchor_pos,
    current_state_anchor_rot: Tensor = EnvContext.current.anchor_rot,
    w_last: bool = True,
    *,
    odom_disp_start: Tensor,  # [B, 2] believed displacement, start-heading frame
    odom_start_xy: Tensor,  # [B, 2] anchor XY at episode start
    odom_start_heading_inv: Tensor,  # [B, 4] inverse start-heading quat
) -> Tensor:  # [B, 2] heading-local offset to reference anchor
    """Derive the heading-local odometer offset from raw sensor ingredients.

    Reconstructs the believed odometer position from the raw ``odom_disp_start``
    reading (the stochastic corruption boundary), then expresses the vector from
    that believed position to the current reference anchor XY in the live anchor's
    heading-aligned (yaw-only, z-dropped) frame.  Reproduces exactly the
    corrupted heading-local offset that the env's context used to store
    (feed ``odom_disp_start_corrupt``); feeding ``odom_disp_start_clean`` yields the
    clean twin, since the clean displacement reconstructs the true position.

    Deterministic (only quat ops) → ONNX/torch.compile-safe.

    Takes the reference ANCHOR position only, not the full reference body array:
    the anchor XY is all this needs, so deployment does not have to ship every
    reference body pose.

    Args:
        ref_anchor_pos: Reference anchor position [B, 3] at the current step
            (``mimic.ref_anchor_pos``).
        current_state_anchor_rot: Robot current anchor rotation [B, 4] (IMU heading).
        odom_disp_start: Believed displacement from start [B, 2], start-heading frame.
        odom_start_xy: Anchor XY at episode start [B, 2].
        odom_start_heading_inv: Inverse start-heading quaternion [B, 4].
        w_last: If True, quaternions are XYZW; else WXYZ.

    Returns:
        Heading-local offset to reference anchor [B, 2].
    """
    # believed current position = start_xy + rotate(start_heading, disp)
    start_heading = rotations.quat_conjugate(odom_start_heading_inv, w_last)
    disp_3d = torch.cat(
        [odom_disp_start, torch.zeros_like(odom_disp_start[:, :1])], dim=-1
    )
    disp_world = rotations.quat_rotate(start_heading, disp_3d, w_last)[:, :2]
    believed_pos = odom_start_xy + disp_world
    xy_world = ref_anchor_pos[:, :2] - believed_pos
    heading_inv = rotations.calc_heading_quat_inv(current_state_anchor_rot, w_last)
    xy_3d = torch.cat([xy_world, torch.zeros_like(xy_world[:, :1])], dim=-1)
    return rotations.quat_rotate(heading_inv, xy_3d, w_last)[:, :2]
