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
"""Target pose building utilities for mimic environments.

Provides functions for building target pose observations from reference motions,
used for motion tracking and imitation learning.
"""

import torch
from torch import Tensor
from protomotions.utils import rotations


@torch.jit.script_if_tracing
def build_max_coords_target_poses_future_rel(
    cur_gt: Tensor,
    cur_gr: Tensor,
    flat_target_pos: Tensor,
    flat_target_rot: Tensor,
    cur_vel: Tensor,
    cur_ang_vel: Tensor,
    flat_target_vel: Tensor,
    flat_target_ang_vel: Tensor,
    num_future_steps: int,
    num_envs: int,
    with_velocities: bool,
    w_last: bool,
):
    """Build target pose observations with relative deltas between consecutive future frames.

    Computes future target poses where each frame is expressed relative to the previous frame,
    providing incremental motion information for tracking.

    Args:
        cur_gt: Current body positions [batch, bodies, 3]
        cur_gr: Current body rotations [batch, bodies, 4]
        flat_target_pos: Flattened target positions [batch*num_future_steps, bodies, 3]
        flat_target_rot: Flattened target rotations [batch*num_future_steps, bodies, 4]
        cur_vel: Current body velocities [batch, bodies, 3]
        cur_ang_vel: Current body angular velocities [batch, bodies, 3]
        flat_target_vel: Flattened target velocities [batch*num_future_steps, bodies, 3]
        flat_target_ang_vel: Flattened target angular velocities [batch*num_future_steps, bodies, 3]
        num_future_steps: Number of future frames to encode
        num_envs: Number of parallel environments
        with_velocities: If True, include velocity information
        w_last: If True, quaternions are in XYZW format, else WXYZ

    Returns:
        Target pose observations [batch, features] in root-relative coordinates
    """
    if with_velocities:
        raise NotImplementedError(
            "Velocity not yet implemented for future rel target poses"
        )

    reference_pos = (
        flat_target_pos.reshape(num_envs, num_future_steps, *cur_gt.shape[1:])
        .clone()
        .roll(shifts=1, dims=1)
    )
    reference_pos[:, 0] = cur_gt
    flat_reference_pos = reference_pos.reshape(flat_target_pos.shape)

    reference_rot = (
        flat_target_rot.reshape(num_envs, num_future_steps, *flat_target_rot.shape[1:])
        .clone()
        .roll(shifts=1, dims=1)
    )
    reference_rot[:, 0] = cur_gr
    flat_reference_rot = reference_rot.reshape(flat_target_rot.shape)

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
    target_rel_body_pos = flat_target_pos - flat_reference_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(
        target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
        target_rel_body_pos.shape[2],
    )
    flat_target_rel_body_pos = rotations.quat_rotate(
        pos_flat_heading_inv_rot, flat_target_rel_body_pos, w_last
    )

    # target body pos   [N, 3xB]
    flat_target_body_pos = (flat_target_pos - reference_root_pos_expand).reshape(
        flat_target_pos.shape[0] * flat_target_pos.shape[1], flat_target_pos.shape[2]
    )
    flat_target_body_pos = rotations.quat_rotate(
        pos_flat_heading_inv_rot, flat_target_body_pos, w_last
    )

    # target body rot   [N, 6xB]
    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_reference_rot, w_last), flat_target_rot, w_last
    )
    target_rel_body_rot_obs = (
        rotations.quat_to_tan_norm(target_rel_body_rot.view(-1, 4), w_last)
        .reshape(num_envs, num_future_steps, -1, 6)
        .reshape(target_rel_body_rot.shape[0], -1)
    )

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(
        rot_heading_inv_rot_expand, flat_target_rot, w_last
    )
    target_body_rot_obs = (
        rotations.quat_to_tan_norm(target_body_rot.view(-1, 4), w_last)
        .reshape(num_envs, num_future_steps, -1, 6)
        .reshape(target_rel_body_rot.shape[0], -1)
    )

    target_rel_body_pos = flat_target_rel_body_pos.reshape(
        num_envs, num_future_steps, -1, 3
    ).reshape(target_rel_body_pos.shape[0], -1)
    target_body_pos = flat_target_body_pos.reshape(
        num_envs, num_future_steps, -1, 3
    ).reshape(flat_target_pos.shape[0], -1)

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


@torch.jit.script_if_tracing
def build_max_coords_target_poses(
    cur_gt: Tensor,
    cur_gr: Tensor,
    flat_target_pos: Tensor,
    flat_target_rot: Tensor,
    cur_vel: Tensor,
    cur_ang_vel: Tensor,
    flat_target_vel: Tensor,
    flat_target_ang_vel: Tensor,
    num_envs: int,
    num_future_steps: int,
    with_velocities: bool,
    w_last: bool,
):
    """Build target pose observations in root-relative coordinates.

    Computes future target poses represented as both absolute (from root) and relative
    (from current pose) transformations, in the root's heading-aligned frame.

    Args:
        cur_gt: Current body positions [batch, bodies, 3]
        cur_gr: Current body rotations [batch, bodies, 4]
        flat_target_pos: Flattened target positions [batch*num_future_steps, bodies, 3]
        flat_target_rot: Flattened target rotations [batch*num_future_steps, bodies, 4]
        cur_vel: Current body velocities [batch, bodies, 3]
        cur_ang_vel: Current body angular velocities [batch, bodies, 3]
        flat_target_vel: Flattened target velocities [batch*num_future_steps, bodies, 3]
        flat_target_ang_vel: Flattened target angular velocities [batch*num_future_steps, bodies, 3]
        num_envs: Number of parallel environments
        num_future_steps: Number of future frames to encode
        with_velocities: If True, include velocity deltas
        w_last: If True, quaternions are in XYZW format, else WXYZ

    Returns:
        Target pose observations [batch, features] with both absolute and relative pose info
    """
    expanded_body_pos = cur_gt.unsqueeze(1).expand(
        num_envs, num_future_steps, *cur_gt.shape[1:]
    )
    expanded_body_rot = cur_gr.unsqueeze(1).expand(
        num_envs, num_future_steps, *cur_gr.shape[1:]
    )

    flat_cur_pos = expanded_body_pos.reshape(flat_target_pos.shape)
    flat_cur_rot = expanded_body_rot.reshape(flat_target_rot.shape)

    root_pos = flat_cur_pos[:, 0, :]
    root_rot = flat_cur_rot[:, 0, :]

    heading_inv_rot = rotations.calc_heading_quat_inv(root_rot, w_last)

    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    translation_heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_cur_pos.shape[1], 1)
    )
    rotation_heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_cur_rot.shape[1], 1)
    )
    flat_translation_heading_inv_rot = translation_heading_inv_rot_expand.reshape(
        translation_heading_inv_rot_expand.shape[0]
        * translation_heading_inv_rot_expand.shape[1],
        translation_heading_inv_rot_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)

    """target"""
    # target body pos   [N, 3xB]
    flat_target_body_pos = (flat_target_pos - root_pos_expand).reshape(
        flat_target_pos.shape[0] * flat_target_pos.shape[1], flat_target_pos.shape[2]
    )
    flat_target_body_pos = rotations.quat_rotate(
        flat_translation_heading_inv_rot, flat_target_body_pos, w_last
    )
    target_body_pos = flat_target_body_pos.reshape(num_envs, num_future_steps, -1)

    flat_target_body_pos_rel = (flat_target_pos - flat_cur_pos).reshape(
        flat_target_pos.shape[0] * flat_target_pos.shape[1], flat_target_pos.shape[2]
    )
    flat_target_body_pos_rel = rotations.quat_rotate(
        flat_translation_heading_inv_rot, flat_target_body_pos_rel, w_last
    )
    target_body_pos_rel = flat_target_body_pos_rel.reshape(
        num_envs, num_future_steps, -1
    )

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(
        rotation_heading_inv_rot_expand, flat_target_rot, w_last
    )

    target_body_rot_obs = rotations.quat_to_tan_norm(
        target_body_rot.view(-1, 4), w_last
    ).reshape(num_envs, num_future_steps, -1)

    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_cur_rot, w_last), flat_target_rot, w_last
    )
    target_rel_body_rot_obs = rotations.quat_to_tan_norm(
        target_rel_body_rot.view(-1, 4), w_last
    ).reshape(num_envs, num_future_steps, -1)

    obs = torch.cat(
        (
            target_body_pos,
            target_body_pos_rel,
            target_body_rot_obs,
            target_rel_body_rot_obs,
        ),
        dim=-1,
    )

    if with_velocities:
        expanded_body_vel = cur_vel.unsqueeze(1).expand(
            num_envs, num_future_steps, *cur_vel.shape[1:]
        )
        flat_cur_vel = expanded_body_vel.reshape(flat_target_vel.shape)

        flat_target_vel = (flat_target_vel - flat_cur_vel).reshape(
            flat_target_vel.shape[0] * flat_target_vel.shape[1],
            flat_target_vel.shape[2],
        )
        flat_local_target_vel = rotations.quat_rotate(
            translation_heading_inv_rot_expand, flat_target_vel, w_last
        )
        local_target_vel = flat_local_target_vel.reshape(num_envs, num_future_steps, -1)

        expanded_body_ang_vel = cur_ang_vel.unsqueeze(1).expand(
            num_envs, num_future_steps, *cur_ang_vel.shape[1:]
        )
        flat_cur_ang_vel = expanded_body_ang_vel.reshape(flat_target_ang_vel.shape)

        flat_target_ang_vel = (flat_target_ang_vel - flat_cur_ang_vel).reshape(
            flat_target_ang_vel.shape[0] * flat_target_ang_vel.shape[1],
            flat_target_ang_vel.shape[2],
        )
        flat_local_target_ang_vel = rotations.quat_rotate(
            rotation_heading_inv_rot_expand, flat_target_ang_vel, w_last
        )
        local_target_ang_vel = flat_local_target_ang_vel.reshape(
            num_envs, num_future_steps, -1
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


@torch.jit.script_if_tracing
def build_max_coords_target_poses_simple(
    cur_gt: Tensor,
    cur_gr: Tensor,
    flat_target_pos: Tensor,
    flat_target_rot: Tensor,
    flat_target_vel: Tensor,
    flat_target_ang_vel: Tensor,
    num_envs: int,
    num_future_steps: int,
    with_velocities: bool,
    w_last: bool,
):
    """Build simplified target pose observations (absolute only, no relative deltas).

    Encodes future target poses in root-heading-aligned frame using only absolute
    positions and rotations (no current-to-target deltas).

    Args:
        cur_gt: Current body positions [batch, bodies, 3]
        cur_gr: Current body rotations [batch, bodies, 4]
        flat_target_pos: Flattened target positions [batch*num_future_steps, bodies, 3]
        flat_target_rot: Flattened target rotations [batch*num_future_steps, bodies, 4]
        flat_target_vel: Flattened target velocities [batch*num_future_steps, bodies, 3]
        flat_target_ang_vel: Flattened target angular velocities [batch*num_future_steps, bodies, 3]
        num_envs: Number of parallel environments
        num_future_steps: Number of future frames to encode
        with_velocities: If True, include velocity information
        w_last: If True, quaternions are in XYZW format, else WXYZ

    Returns:
        Target pose observations [batch, features]
    """
    pos_normalizer = cur_gt[:, 0, :]
    rot_normalizer = rotations.calc_heading_quat_inv(cur_gr[:, 0, :], w_last)

    pos_normalizer_expand = pos_normalizer.unsqueeze(1).expand(
        num_envs, num_future_steps, *pos_normalizer.shape[1:]
    )
    rot_normalizer_expand = rot_normalizer.unsqueeze(1).expand(
        num_envs, num_future_steps, *rot_normalizer.shape[1:]
    )

    pos_normalizer_expand_flat = pos_normalizer_expand.reshape(
        flat_target_pos.shape[0], -1
    )
    rot_normalizer_expand_flat = rot_normalizer_expand.reshape(
        flat_target_rot.shape[0], -1
    )

    rot_normalizer_expand_flat_expand = rot_normalizer_expand_flat.unsqueeze(-2)
    pos_rot_normalizer_expand_flat_expand = rot_normalizer_expand_flat_expand.repeat(
        (1, flat_target_pos.shape[1], 1)
    )
    rot_rot_normalizer_expand_flat_expand = rot_normalizer_expand_flat_expand.repeat(
        (1, flat_target_rot.shape[1], 1)
    )

    pos_normalizer_expand_flat_expand = pos_normalizer_expand_flat.unsqueeze(-2)

    """target"""
    # target body pos   [N, 3xB]
    flat_target_body_pos = (
        flat_target_pos - pos_normalizer_expand_flat_expand
    ).reshape(
        flat_target_pos.shape[0] * flat_target_pos.shape[1], flat_target_pos.shape[2]
    )
    flat_target_body_pos = rotations.quat_rotate(
        pos_rot_normalizer_expand_flat_expand.reshape(
            -1, pos_rot_normalizer_expand_flat_expand.shape[2]
        ),
        flat_target_body_pos,
        w_last,
    )
    target_body_pos = flat_target_body_pos.reshape(num_envs, num_future_steps, -1)

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(
        rot_rot_normalizer_expand_flat_expand, flat_target_rot, w_last
    )

    target_body_rot_obs = rotations.quat_to_tan_norm(
        target_body_rot.view(-1, 4), w_last
    ).reshape(num_envs, num_future_steps, -1)

    obs = torch.cat(
        (
            target_body_pos,
            target_body_rot_obs,
        ),
        dim=-1,
    )

    if with_velocities:
        flat_target_vel = flat_target_vel.reshape(
            flat_target_vel.shape[0] * flat_target_vel.shape[1],
            flat_target_vel.shape[2],
        )
        flat_local_target_vel = rotations.quat_rotate(
            pos_rot_normalizer_expand_flat_expand, flat_target_vel, w_last
        )
        local_target_vel = flat_local_target_vel.reshape(num_envs, num_future_steps, -1)

        flat_target_ang_vel = flat_target_ang_vel.reshape(
            flat_target_ang_vel.shape[0] * flat_target_ang_vel.shape[1],
            flat_target_ang_vel.shape[2],
        )
        flat_local_target_ang_vel = rotations.quat_rotate(
            rot_rot_normalizer_expand_flat_expand, flat_target_ang_vel, w_last
        )
        local_target_ang_vel = flat_local_target_ang_vel.reshape(
            num_envs, num_future_steps, -1
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


@torch.jit.script_if_tracing
def build_sparse_target_poses(
    cur_gt: Tensor,
    cur_gr: Tensor,
    flat_target_pos: Tensor,
    flat_target_rot: Tensor,
    masked_mimic_conditionable_bodies_ids: Tensor,
    num_future_steps: int,
    num_envs: int,
    w_last: bool,
):
    """Build target pose observations for sparse body tracking (MaskedMimic).

    Similar to max_coords but only includes conditionable bodies (e.g., head, hands for VR).
    Provides both absolute and relative pose encodings for partial body control.

    Args:
        cur_gt: Current body positions [batch, bodies, 3]
        cur_gr: Current body rotations [batch, bodies, 4]
        flat_target_pos: Flattened target positions [batch*num_future_steps, bodies, 3]
        flat_target_rot: Flattened target rotations [batch*num_future_steps, bodies, 4]
        masked_mimic_conditionable_bodies_ids: Indices of trackable bodies
        num_future_steps: Number of future frames to encode
        num_envs: Number of parallel environments
        w_last: If True, quaternions are in XYZW format, else WXYZ

    Returns:
        Sparse target pose observations [batch, features] for conditionable bodies only
    """

    expanded_body_pos = cur_gt.unsqueeze(1).expand(
        num_envs, num_future_steps, *cur_gt.shape[1:]
    )
    expanded_body_rot = cur_gr.unsqueeze(1).expand(
        num_envs, num_future_steps, *cur_gr.shape[1:]
    )

    flat_cur_pos = expanded_body_pos.reshape(flat_target_pos.shape)
    flat_cur_rot = expanded_body_rot.reshape(flat_target_rot.shape)

    root_pos = flat_cur_pos[:, 0, :]
    root_rot = flat_cur_rot[:, 0, :]

    heading_rot = rotations.calc_heading_quat_inv(root_rot, w_last)

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, flat_cur_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)

    """target"""
    # target body pos   [N, 3xB]
    target_rel_body_pos = flat_target_pos - flat_cur_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(
        target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
        target_rel_body_pos.shape[2],
    )
    flat_target_rel_body_pos = rotations.quat_rotate(
        flat_heading_rot, flat_target_rel_body_pos, w_last
    )

    # target body pos   [N, 3xB]
    flat_target_body_pos = (flat_target_pos - root_pos_expand).reshape(
        flat_target_pos.shape[0] * flat_target_pos.shape[1], flat_target_pos.shape[2]
    )
    flat_target_body_pos = rotations.quat_rotate(
        flat_heading_rot, flat_target_body_pos, w_last
    )

    # target body rot   [N, 6xB]
    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_cur_rot, w_last), flat_target_rot, w_last
    )
    target_rel_body_rot_obs = rotations.quat_to_tan_norm(
        target_rel_body_rot.view(-1, 4), w_last
    ).view(target_rel_body_rot.shape[0], -1)

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(heading_rot_expand, flat_target_rot, w_last)
    target_body_rot_obs = rotations.quat_to_tan_norm(
        target_body_rot.view(-1, 4), w_last
    ).view(target_rel_body_rot.shape[0], -1)

    padded_flat_target_rel_body_pos = torch.nn.functional.pad(
        flat_target_rel_body_pos, [0, 3], "constant", 0
    )
    sub_sampled_target_rel_body_pos = padded_flat_target_rel_body_pos.reshape(
        num_envs, num_future_steps, -1, 6
    )[:, :, masked_mimic_conditionable_bodies_ids]

    padded_flat_target_body_pos = torch.nn.functional.pad(
        flat_target_body_pos, [0, 3], "constant", 0
    )
    sub_sampled_target_body_pos = padded_flat_target_body_pos.reshape(
        num_envs, num_future_steps, -1, 6
    )[:, :, masked_mimic_conditionable_bodies_ids]

    sub_sampled_target_rel_body_rot_obs = target_rel_body_rot_obs.reshape(
        num_envs, num_future_steps, -1, 6
    )[:, :, masked_mimic_conditionable_bodies_ids]
    sub_sampled_target_body_rot_obs = target_body_rot_obs.reshape(
        num_envs, num_future_steps, -1, 6
    )[:, :, masked_mimic_conditionable_bodies_ids]

    # In masked_mimic allow easy re-shape to [batch, time, joint, type (transform/rotate), features]
    obs = torch.cat(
        (
            sub_sampled_target_rel_body_pos,
            sub_sampled_target_body_pos,
            sub_sampled_target_rel_body_rot_obs,
            sub_sampled_target_body_rot_obs,
        ),
        dim=-1,  # [batch, timesteps, joints, 24]
    ).view(num_envs, -1)

    return obs
