# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import List, Union, Tuple, Dict

import torch
from torch import Tensor
import numpy as np
import math

from isaac_utils import torch_utils, rotations, maths


@torch.jit.script
def dof_to_obs(
    pose: Tensor, dof_obs_size: int, dof_offsets: List[int], joint_axis: List[str], w_last: bool
) -> Tensor:
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    assert pose.shape[-1] == dof_offsets[-1]
    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset : (dof_offset + dof_size)]

        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose, w_last)
        elif dof_size == 1:
            configured_joint_axis = joint_axis[j]
            axis = torch.tensor(
                [0.0, 0.0, 0.0], dtype=joint_pose.dtype, device=pose.device
            )
            if configured_joint_axis == "x":
                axis[0] = 1.0
            elif configured_joint_axis == "y":
                axis[1] = 1.0
            elif configured_joint_axis == "z":
                axis[2] = 1.0
            joint_pose_q = rotations.quat_from_angle_axis(
                joint_pose[..., 0], axis, w_last
            )
        else:
            joint_pose_q = None
            assert False, "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q, w_last)
        dof_obs[:, (j * joint_obs_size) : ((j + 1) * joint_obs_size)] = joint_dof_obs

    assert (num_joints * joint_obs_size) == dof_obs_size

    return dof_obs


def build_pd_action_offset_scale(
    dof_offsets, dof_limits_lower, dof_limits_upper, device
):
    num_joints = len(dof_offsets) - 1

    lim_low = dof_limits_lower.cpu().numpy()
    lim_high = dof_limits_upper.cpu().numpy()

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]

        if dof_size == 3:
            curr_low = lim_low[dof_offset : (dof_offset + dof_size)]
            curr_high = lim_high[dof_offset : (dof_offset + dof_size)]
            curr_low = np.max(np.abs(curr_low))
            curr_high = np.max(np.abs(curr_high))
            curr_scale = max([curr_low, curr_high])
            curr_scale = 1.2 * curr_scale
            curr_scale = min([curr_scale, np.pi])

            lim_low[dof_offset : (dof_offset + dof_size)] = -curr_scale
            lim_high[dof_offset : (dof_offset + dof_size)] = curr_scale

            # lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
            # lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

        elif dof_size == 1:
            curr_low = lim_low[dof_offset]
            curr_high = lim_high[dof_offset]
            curr_mid = 0.5 * (curr_high + curr_low)

            # extend the action range to be a bit beyond the joint limits so that the motors
            # don't lose their strength as they approach the joint limits
            curr_scale = 0.7 * (curr_high - curr_low)
            curr_low = curr_mid - curr_scale
            curr_high = curr_mid + curr_scale

            lim_low[dof_offset] = curr_low
            lim_high[dof_offset] = curr_high

    pd_action_offset = 0.5 * (lim_high + lim_low)
    pd_action_scale = 0.5 * (lim_high - lim_low)
    pd_action_offset = torch.tensor(pd_action_offset, device=device)
    pd_action_scale = torch.tensor(pd_action_scale, device=device)

    return pd_action_offset, pd_action_scale


@torch.jit.script
def compute_humanoid_observations(
    root_pos: Tensor,
    root_rot: Tensor,
    root_vel: Tensor,
    root_ang_vel: Tensor,
    dof_pos: Tensor,
    dof_vel: Tensor,
    key_body_pos: Tensor,
    ground_height: Tensor,
    local_root_obs: bool,
    dof_obs_size: int,
    dof_offsets: List[int],
    joint_axis: List[str],
    w_last: bool,
) -> Tensor:
    root_h = root_pos[:, 2:3] - ground_height
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    if local_root_obs:
        root_rot_obs = rotations.quat_mul(heading_rot, root_rot, w_last)
    else:
        root_rot_obs = root_rot

    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs, w_last)

    local_root_vel = rotations.quat_rotate(heading_rot, root_vel, w_last)
    local_root_ang_vel = rotations.quat_rotate(heading_rot, root_ang_vel, w_last)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = rotations.quat_rotate(flat_heading_rot, flat_end_pos, w_last)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets, joint_axis, w_last)

    obs = torch.cat(
        (
            root_h,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_humanoid_observations_max(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
    body_ang_vel: Tensor,
    ground_height: Tensor,
    local_root_obs: bool,
    root_height_obs: bool,
    w_last: bool,
) -> Tensor:
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h - ground_height

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = rotations.quat_rotate(
        flat_heading_rot, flat_local_body_pos, w_last
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )
    flat_local_body_rot = rotations.quat_mul(flat_heading_rot, flat_body_rot, w_last)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot, w_last)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    if not local_root_obs:
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot, w_last)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(
        body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
    )
    flat_local_body_vel = rotations.quat_rotate(flat_heading_rot, flat_body_vel, w_last)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
    )

    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )
    flat_local_body_ang_vel = rotations.quat_rotate(
        flat_heading_rot, flat_body_ang_vel, w_last
    )
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    obs = torch.cat(
        (
            root_h_obs,
            local_body_pos,
            local_body_rot_obs,
            local_body_vel,
            local_body_ang_vel,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script_if_tracing
def compute_humanoid_reset(
    reset_buf: Tensor,
    progress_buf: Tensor,
    contact_buf: Tensor,
    non_termination_contact_body_ids: Tensor,
    rigid_body_pos: Tensor,
    max_episode_length: float,
    enable_early_termination: bool,
    termination_heights: Tensor,
) -> Tuple[Tensor, Tensor]:
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, non_termination_contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, non_termination_contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated


@torch.jit.script
def build_disc_observations(
    root_pos: Tensor,
    root_rot: Tensor,
    root_vel: Tensor,
    root_ang_vel: Tensor,
    dof_pos: Tensor,
    dof_vel: Tensor,
    key_body_pos: Tensor,
    ground_height: Tensor,
    local_root_obs: bool,
    root_height_obs: bool,
    dof_obs_size: int,
    dof_offsets: List[int],
    joint_axis: List[str],
    w_last: bool,
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    if local_root_obs:
        root_rot_obs = rotations.quat_mul(heading_rot, root_rot, w_last)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs, w_last)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h - ground_height

    local_root_vel = torch_utils.quat_rotate(heading_rot, root_vel, w_last)
    local_root_ang_vel = torch_utils.quat_rotate(heading_rot, root_ang_vel, w_last)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = torch_utils.quat_rotate(flat_heading_rot, flat_end_pos, w_last)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets, joint_axis, w_last)

    obs = torch.cat(
        (
            root_h_obs,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )

    return obs


@torch.jit.script
def quat_diff_norm(quat1: Tensor, quat2: Tensor, w_last: bool):
    if w_last:
        w = 3
    else:
        w = 0
    quat1inv = rotations.quat_conjugate(quat1, w_last)
    mul = rotations.quat_mul(quat1inv, quat2, w_last)
    norm = mul[..., w].clip(-1, 1).arccos() * 2
    # Trying both rotation directions
    norm = torch.min(norm, math.pi * 2 - norm)
    return norm


@torch.jit.script
def quat_angle_diff_norm(quat1: Tensor, quat2: Tensor, w_last: bool):
    diff_quat = rotations.quat_mul(
        quat2, rotations.quat_conjugate(quat1, w_last), w_last
    )
    angle_axis = torch_utils.quat_to_angle_axis(diff_quat, w_last)[0]
    return angle_axis**2


@torch.jit.script
def remove_base_rot(quat: Tensor, w_last: bool):
    base_rot = rotations.quat_conjugate(
        torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat), w_last
    )  # SMPL
    shape = quat.shape[0]
    return rotations.quat_mul(quat, base_rot.repeat(shape, 1), w_last)


@torch.jit.script_if_tracing
def get_relative_object_pointclouds_jit(
    root_pos: Tensor, root_rot: Tensor, pointclouds: Tensor, w_last: bool
) -> Tensor:
    """
    Computes the relative point clouds of objects with respect to the root states.

    Args:
        root_pos (Tensor): Tensor containing the root positions of the humanoid.
        root_rot (Tensor): Tensor containing the root rotations of the humanoid.
        pointclouds (Tensor): Tensor containing the point clouds of the objects.
        object_root_states (Tensor): Tensor containing the root states of the objects.
        w_last (bool): Boolean indicating if the last dimension is the w component of the quaternion.

    Returns:
        Tensor: The relative point clouds of the objects.
    """
    # Expand root positions and rotations to match the shape of pointclouds
    expanded_root_pos = (
        root_pos.unsqueeze(1)
        .unsqueeze(1)
        .expand(pointclouds.shape[0], pointclouds.shape[1], pointclouds.shape[2], 3)
        .reshape(-1, 3)
    )
    expanded_root_rot = (
        root_rot.unsqueeze(1)
        .unsqueeze(1)
        .expand(pointclouds.shape[0], pointclouds.shape[1], pointclouds.shape[2], 4)
        .reshape(-1, 4)
    )

    # Calculate the inverse of the expanded root rotations
    expanded_root_rot_inv = torch_utils.calc_heading_quat_inv(expanded_root_rot, w_last)

    # Flatten pointclouds and object root states for processing
    flat_pointclouds = pointclouds.reshape(-1, 3)

    # Compute point clouds relative to the root
    flat_pointclouds_wrt_root = flat_pointclouds - expanded_root_pos
    flat_pointclouds_wrt_root = torch_utils.quat_rotate(
        expanded_root_rot_inv, flat_pointclouds_wrt_root, w_last
    )

    # Reshape the result to match the original pointclouds shape
    return flat_pointclouds_wrt_root.view(
        pointclouds.shape[0], pointclouds.shape[1], pointclouds.shape[2], 3
    )


@torch.jit.script_if_tracing
def compute_relative_to_object_pointcloud_contact_bodies_jit(
    ego_object_pointclouds: Tensor, ego_contact_bodies: Tensor, w_last: bool
) -> Tensor:
    num_envs = ego_contact_bodies.shape[0]
    num_contact_bodies = ego_contact_bodies.shape[1]
    num_objects_per_env = ego_object_pointclouds.shape[1]
    num_points_per_object = ego_object_pointclouds.shape[2]

    # Handle single point case differently
    if num_points_per_object == 1:
        # Reshape points to [num_envs * num_objects_per_env, 3]
        points = ego_object_pointclouds.reshape(num_envs * num_objects_per_env, 3)

        # Expand bodies
        bodies = ego_contact_bodies.unsqueeze(1).expand(
            num_envs, num_objects_per_env, num_contact_bodies, 3
        )
        bodies = bodies.reshape(num_envs * num_objects_per_env, num_contact_bodies, 3)

        # Compute vectors directly without distance calculation
        points_expanded = points.unsqueeze(1).expand(-1, num_contact_bodies, 3)
        global_vectors = points_expanded - bodies
    else:
        # Implementation for multiple points
        points = ego_object_pointclouds.reshape(
            num_envs * num_objects_per_env, -1, 3
        ).contiguous()
        bodies = ego_contact_bodies.unsqueeze(1).expand(
            num_envs, num_objects_per_env, num_contact_bodies, 3
        )
        bodies = bodies.reshape(
            num_envs * num_objects_per_env, num_contact_bodies, 3
        ).contiguous()

        points_expanded = points.unsqueeze(1)
        bodies_expanded = bodies.unsqueeze(2)

        squared_diff = (points_expanded - bodies_expanded) ** 2
        distances = squared_diff.sum(dim=-1)

        # Find minimum distances and corresponding indices
        min_distances, min_indices = distances.min(dim=-1)

        # Gather closest points efficiently
        batch_indices = torch.arange(points.shape[0], device=points.device).unsqueeze(1)
        batch_indices = batch_indices.expand(-1, num_contact_bodies)

        closest_points = points[batch_indices, min_indices]
        global_vectors = closest_points - bodies

    return global_vectors.reshape(num_envs, num_objects_per_env, num_contact_bodies, 3)


@torch.jit.script_if_tracing
def compute_relative_to_object_contacts_contact_bodies_jit(
    target_contact_positions: Tensor,  # [num_envs, num_objects, num_contact_bodies, 3]
    contact_bodies: Tensor,  # [num_envs, num_contact_bodies, 3]
    expected_contacts: Tensor,  # [num_envs, num_objects, num_contact_bodies]
    w_last: bool,
) -> Tensor:
    """
    Computes vectors from current contact body positions to their target positions on the object.

    Args:
        target_contact_positions: Target positions for each contact body on the object
        contact_bodies: Current positions of the contact bodies
        w_last: Flag for quaternion w-last format

    Returns:
        Tensor: Vectors from current contact body positions to target positions
        Shape: [num_envs, num_objects, num_contact_bodies, 3]
    """
    # Compute vectors from current positions to target positions
    # Shape: [num_envs, num_objects, num_contact_bodies, 3]
    contact_vectors = target_contact_positions - contact_bodies.unsqueeze(1)

    return torch.cat(
        [
            contact_vectors * expected_contacts.unsqueeze(-1),
            expected_contacts.unsqueeze(-1),
        ],
        dim=-1,
    )


@torch.jit.script_if_tracing
def get_object_bounding_box_obs_jit(
    object_ids: Tensor,
    root_pos: Tensor,
    root_quat: Tensor,
    object_root_states: Tensor,
    object_bounding_box: Tensor,
    w_last: bool,
):
    num_objects = object_ids.shape[0]
    expanded_root_pos = root_pos.unsqueeze(1).expand(num_objects, 8, 3).reshape(-1, 3)
    expanded_root_rot = root_quat.unsqueeze(1).expand(num_objects, 8, 4).reshape(-1, 4)

    root_rot_inv = torch_utils.calc_heading_quat_inv(root_quat, w_last)
    expanded_root_rot_inv = torch_utils.calc_heading_quat_inv(expanded_root_rot, w_last)

    obj_root_pos = object_root_states[object_ids, :3]
    obj_root_rot = object_root_states[object_ids, 3:7]

    # Ensure the quaternion is normalized
    obj_root_rot = maths.normalize(obj_root_rot)

    object_bbs = object_bounding_box.view(-1, 3)
    expanded_obj_root_pos = (
        obj_root_pos.unsqueeze(1).expand(num_objects, 8, 3).reshape(-1, 3)
    )

    expanded_obj_root_pos[..., -1] = 0

    obj_relative_to_env = object_bbs - expanded_root_pos

    object_rotated_relative_to_env = torch_utils.quat_rotate(
        expanded_root_rot_inv, obj_relative_to_env, w_last
    ).view(-1, 3)

    object_root_rot_relative_to_env = rotations.quat_mul(
        root_rot_inv, obj_root_rot, w_last
    ).view(-1, 4)

    object_root_rot_relative_to_env = torch_utils.quat_to_tan_norm(
        object_root_rot_relative_to_env, w_last
    )

    return torch.cat(
        (
            object_rotated_relative_to_env.view(num_objects, -1),
            object_root_rot_relative_to_env.view(num_objects, -1),
        ),
        dim=-1,
    )
