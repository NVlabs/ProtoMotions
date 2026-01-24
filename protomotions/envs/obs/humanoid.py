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
from protomotions.utils.rotations import angle_from_matrix_axis
from protomotions.components.pose_lib import (
    extract_transforms_from_qpos_non_root_ignore_fixed_helper,
    matrix_to_quaternion,
    quaternion_to_matrix,
)


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


def obs_to_dof(
    obs_6d: Tensor, hinge_axes_map: Dict[int, torch.Tensor], w_last: bool
) -> Tensor:
    """Inverse of dof_to_obs. Converts 6D tan-norm back to DOF angles."""
    device = obs_6d.device
    dtype = obs_6d.dtype
    B = obs_6d.shape[0]
    num_joints = len(hinge_axes_map)
    
    obs_6d = obs_6d.view(B, num_joints, 6)
    flat_6d = obs_6d.view(B * num_joints, 6)
    flat_quat = rotations.tan_norm_to_quat(flat_6d, w_last)
    joint_quats = flat_quat.view(B, num_joints, 4)
    joint_rot_mats = quaternion_to_matrix(joint_quats, w_last)
    
    hinge_axes_map = {k: v.to(device).to(dtype) for k, v in hinge_axes_map.items()}
    
    dof_list = []
    for i, (body_idx, axes) in enumerate(hinge_axes_map.items()):
        num_body_dofs = len(axes)
        rot_mat = joint_rot_mats[:, i, :, :]
        
        if num_body_dofs == 1:
            angle = angle_from_matrix_axis(rot_mat, axes[0])
            dof_list.append(angle.unsqueeze(-1))
        elif num_body_dofs == 3:
            quat = matrix_to_quaternion(rot_mat, w_last=False)
            exp_map = rotations.quat_to_exp_map(quat, w_last=False)
            dof_list.append(exp_map)
        else:
            raise ValueError(f"Unsupported number of DOFs: {num_body_dofs}")
    
    return torch.cat(dof_list, dim=-1)


def root_projected_gravity(root_rot: Tensor, w_last: bool = True) -> torch.Tensor:
    GRAVITY_VEC_W = torch.tensor([0.0, 0.0, -1.0]).reshape(1, 3)
    if root_rot is not None:
        if GRAVITY_VEC_W.device != root_rot.device:
            GRAVITY_VEC_W = GRAVITY_VEC_W.to(root_rot.device)
        return rotations.quat_rotate_inverse(
            root_rot, GRAVITY_VEC_W.repeat(root_rot.shape[0], 1), w_last=w_last
        )
    return None


def compute_local_ang_vel(
    root_rot: Tensor,
    root_ang_vel: Tensor,
    w_last: bool = True,
) -> Tensor:
    """Transform angular velocity from world frame to local (body) frame.
    
    Args:
        root_rot: Root orientation quaternion [num_envs, 4] or [num_envs, num_steps, 4].
        root_ang_vel: Root angular velocity in world frame, same shape as root_rot but last dim is 3.
        w_last: Whether quaternion w component is last.
    
    Returns:
        Angular velocity in local frame, same shape as input root_ang_vel.
    """
    original_shape = root_ang_vel.shape
    
    # Handle both 2D [envs, 3] and 3D [envs, steps, 3] tensors
    if root_ang_vel.dim() == 3:
        flat_rot = root_rot.reshape(-1, 4)
        flat_ang_vel = root_ang_vel.reshape(-1, 3)
        flat_local_ang_vel = rotations.quat_rotate_inverse(flat_rot, flat_ang_vel, w_last)
        return flat_local_ang_vel.reshape(original_shape)
    else:
        return rotations.quat_rotate_inverse(root_rot, root_ang_vel, w_last)


def compute_humanoid_reduced_coords_observations(
    dof_pos: Tensor,
    dof_vel: Tensor,
    anchor_rot: Tensor,
    root_local_ang_vel: Tensor,
    w_last: bool,
    root_rot: Tensor = None,
    root_pos: Tensor = None,
    root_vel: Tensor = None,
    ground_height: Tensor = None,
    root_height_obs: bool = False,
    root_vel_obs: bool = False,
) -> Tensor:
    """Compute reduced coordinates observations for humanoid.
    
    Output format (in order):
        - dof_pos [num_dofs]
        - dof_vel [num_dofs]
        - root_ang_vel [3]: root angular velocity in local frame
        - proj_gravity [3]: projected gravity vector
        - root_vel [3]: root linear velocity in local frame (if root_vel_obs=True)
        - root_height [1]: root height above ground (if root_height_obs=True)
    
    Args:
        dof_pos: Joint positions [num_envs, num_dofs].
        dof_vel: Joint velocities [num_envs, num_dofs].
        anchor_rot: Anchor body orientation for gravity projection [num_envs, 4].
        root_local_ang_vel: Root angular velocity pre-normalized to local frame [num_envs, 3].
        w_last: Whether quaternion w component is last.
        root_rot: Root orientation quaternion [num_envs, 4]. Required if root_vel_obs=True.
        root_pos: Root position [num_envs, 3]. Required if root_height_obs=True.
        root_vel: Root linear velocity in world frame [num_envs, 3]. Required if root_vel_obs=True.
        ground_height: Ground height beneath root [num_envs] or [num_envs, 1].
        root_height_obs: Whether to include root height observation.
        root_vel_obs: Whether to include root velocity observation.
    
    Returns:
        Concatenated observation tensor [num_envs, obs_dim].
    """
    num_envs = dof_pos.shape[0]
    proj_gravity = root_projected_gravity(anchor_rot, w_last)

    obs = [
        dof_pos.view(num_envs, -1),
        dof_vel.view(num_envs, -1),
        root_local_ang_vel.view(num_envs, -1),
        proj_gravity.view(num_envs, -1),
    ]

    if root_vel_obs:
        if root_vel is None:
            raise ValueError("root_vel is required when root_vel_obs=True")
        if root_rot is None:
            raise ValueError("root_rot is required when root_vel_obs=True")
        normalized_root_vel = rotations.quat_rotate_inverse(root_rot, root_vel, w_last)
        obs.append(normalized_root_vel.view(num_envs, -1))

    if root_height_obs:
        if root_pos is None:
            raise ValueError("root_pos is required when root_height_obs=True")
        if ground_height is None:
            ground_height = torch.zeros(num_envs, 1, device=root_pos.device)
        elif ground_height.dim() == 1:
            ground_height = ground_height.unsqueeze(-1)
        root_h = root_pos[:, 2:3] - ground_height
        obs.append(root_h)

    obs = torch.cat(obs, dim=-1)
    return obs


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
    if ground_height.dim() == 1:
        ground_height = ground_height.unsqueeze(-1)

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
