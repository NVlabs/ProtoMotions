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
"""Humanoid-specific utility functions.

Provides functions for computing humanoid observations and transformations.
"""

from typing import Dict
import torch
from torch import Tensor

from protomotions.utils import rotations
from protomotions.components.pose_lib import (
    extract_transforms_from_qpos_non_root_ignore_fixed_helper,
    matrix_to_quaternion,
)


@torch.jit.script
def dof_to_local(
    pose: Tensor, hinge_axes_map: Dict[int, torch.Tensor], w_last: bool
) -> Tensor:
    """Convert degrees of freedom (DoF) representation (articulated joint angles)
    to local joint rotation quaternions.
    Assumes pose is [N, total_articulated_dofs] and output is [N, num_articulated_joints, 4],

    Args:
        pose: Input pose tensor with shape [N, total_articulated_dofs].
              These are the angles for the articulated joints only.
        hinge_axes_map: Dictionary mapping body indices to hinge axes.
        w_last: Whether output quaternion w component is last (XYZW) or first (WXYZ).

    Returns:
        Local rotation quaternions with shape [N, num_articulated_joints, 4].
    """
    if pose.ndim != 2:
        raise ValueError(f"Input pose must be 2D [N, num_dofs], got shape {pose.shape}")

    joint_rot_mats = extract_transforms_from_qpos_non_root_ignore_fixed_helper(
        hinge_axes_map, pose, qpos_is_exp_map_on_3dof_joints=True
    )

    local_rot_quats = matrix_to_quaternion(
        joint_rot_mats, w_last=w_last
    )  # [N, num_articulated_joints, 4]

    return local_rot_quats


@torch.jit.script
def dof_to_obs(
    pose: Tensor, hinge_axes_map: Dict[int, torch.Tensor], w_last: bool
) -> Tensor:
    local_joint_quats = dof_to_local(
        pose, hinge_axes_map, w_last
    )  # [B, num_articulated_joints, 4]

    dof_obs = rotations.quat_to_tan_norm(
        local_joint_quats, w_last
    )  # [B, num_articulated_joints, 6]
    dof_obs = dof_obs.view(pose.shape[0], -1)

    return dof_obs


@torch.jit.script
def compute_humanoid_reduced_coords_observations(
    dof_pos: Tensor,
    dof_vel: Tensor,
    root_ang_vel: Tensor,
    root_projected_gravity: Tensor,
    hinge_axes_map: Dict[int, torch.Tensor],
    w_last: bool,
) -> Tensor:
    num_envs = dof_pos.shape[0]
    dof_obs = dof_to_obs(dof_pos, hinge_axes_map, w_last)

    obs = [
        dof_obs.view(num_envs, -1),
        dof_vel.view(num_envs, -1),
        root_ang_vel.view(num_envs, -1),
        root_projected_gravity.view(num_envs, -1),
    ]

    obs = torch.cat(obs, dim=-1)
    return obs


@torch.jit.script
def compute_humanoid_max_coords_observations(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
    body_ang_vel: Tensor,
    ground_height: Tensor,
    body_contacts: Tensor,
    local_obs: bool,
    root_height_obs: bool,
    observe_contacts: bool,
    w_last: bool,
) -> Tensor:
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h - ground_height

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )
    flat_body_vel = body_vel.reshape(
        body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
    )
    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )

    if local_obs:
        pos_normalizer = root_pos
        rot_normalizer = rotations.calc_heading_quat_inv(root_rot, w_last)

        rot_normalizer_expand = rot_normalizer.unsqueeze(-2)
        rot_normalizer_expand = rot_normalizer_expand.repeat((1, body_pos.shape[1], 1))
        flat_rot_normalizer = rot_normalizer_expand.reshape(
            rot_normalizer_expand.shape[0] * rot_normalizer_expand.shape[1],
            rot_normalizer_expand.shape[2],
        )

        pos_normalizer_expand = pos_normalizer.unsqueeze(-2)
        normalized_body_pos = body_pos - pos_normalizer_expand

        flat_normalized_body_pos = normalized_body_pos.reshape(
            normalized_body_pos.shape[0] * normalized_body_pos.shape[1],
            normalized_body_pos.shape[2],
        )
        flat_normalized_body_pos = rotations.quat_rotate(
            flat_rot_normalizer, flat_normalized_body_pos, w_last
        )
        normalized_body_pos = flat_normalized_body_pos.reshape(
            normalized_body_pos.shape[0],
            normalized_body_pos.shape[1] * normalized_body_pos.shape[2],
        )
        full_body_pos_obs = normalized_body_pos

        flat_normalized_body_rot = rotations.quat_mul(
            flat_rot_normalizer, flat_body_rot, w_last
        )
        full_body_rot_obs = flat_normalized_body_rot

        flat_body_vel = rotations.quat_rotate(
            flat_rot_normalizer, flat_body_vel, w_last
        )

        flat_body_ang_vel = rotations.quat_rotate(
            flat_rot_normalizer, flat_body_ang_vel, w_last
        )
    else:
        full_body_pos_obs = body_pos.reshape(body_pos.shape[0], -1)
        full_body_rot_obs = flat_body_rot

    body_pos_obs = full_body_pos_obs[..., 3:]  # remove root pos
    body_rot_obs = rotations.quat_to_tan_norm(full_body_rot_obs, w_last)
    body_rot_obs = body_rot_obs.reshape(body_rot.shape[0], -1)
    body_vel_obs = flat_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
    )
    body_ang_vel_obs = flat_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    obs = [
        root_h_obs,
        body_pos_obs,
        body_rot_obs,
        body_vel_obs,
        body_ang_vel_obs,
    ]

    if observe_contacts:
        # body_contacts is binary flags: [num_envs, num_contact_bodies]
        # Convert to float for observation
        contact_obs = body_contacts.float()
        obs.append(contact_obs)

    obs = torch.cat(obs, dim=-1)
    return obs


@torch.jit.script
def root_projected_gravity(root_rot: Tensor, w_last: bool = True) -> torch.Tensor:
    GRAVITY_VEC_W = torch.tensor([0.0, 0.0, -1.0]).reshape(1, 3)
    if root_rot is not None:
        if GRAVITY_VEC_W.device != root_rot.device:
            GRAVITY_VEC_W = GRAVITY_VEC_W.to(root_rot.device)
        return rotations.quat_rotate_inverse(
            root_rot, GRAVITY_VEC_W.repeat(root_rot.shape[0], 1), w_last=w_last
        )
    return None
