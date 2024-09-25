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

from typing import List

import torch
from torch import Tensor

from isaac_utils import rotations, torch_utils, maths


@torch.jit.script_if_tracing
def build_sparse_target_poses(
    cur_gt: Tensor,
    cur_gr: Tensor,
    flat_target_pos: Tensor,
    flat_target_rot: Tensor,
    flat_target_vel: Tensor,
    masked_mimic_conditionable_bodies_ids: Tensor,
    num_future_steps: int,
    num_envs: int,
    w_last: bool,
):
    """
    This is identical to the max_coords humanoid observation, only in relative to the current pose.
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

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

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
    flat_target_rel_body_pos = torch_utils.quat_rotate(
        flat_heading_rot, flat_target_rel_body_pos, w_last
    )

    # target body pos   [N, 3xB]
    flat_target_body_pos = (flat_target_pos - root_pos_expand).reshape(
        flat_target_pos.shape[0] * flat_target_pos.shape[1], flat_target_pos.shape[2]
    )
    flat_target_body_pos = torch_utils.quat_rotate(
        flat_heading_rot, flat_target_body_pos, w_last
    )

    # target body rot   [N, 6xB]
    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_cur_rot, w_last), flat_target_rot, w_last
    )
    target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(
        target_rel_body_rot.view(-1, 4), w_last
    ).view(target_rel_body_rot.shape[0], -1)

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(heading_rot_expand, flat_target_rot, w_last)
    target_body_rot_obs = torch_utils.quat_to_tan_norm(
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

    # Heading
    target_heading_rot = torch_utils.calc_heading_quat(flat_target_rot[:, 0, :], w_last)
    target_rel_heading_rot = torch_utils.quat_to_tan_norm(
        rotations.quat_mul(
            heading_rot_expand[:, 0, :], target_heading_rot, w_last
        ).view(-1, 4),
        w_last,
    ).reshape(num_envs, num_future_steps, 1, 6)

    # Velocity
    target_root_vel = flat_target_vel[:, 0, :]
    target_root_vel[..., -1] = 0  # ignore vertical speed
    target_rel_vel = rotations.quat_rotate(
        heading_rot, target_root_vel, w_last
    ).reshape(-1, 3)
    padded_target_rel_vel = torch.nn.functional.pad(
        target_rel_vel, [0, 3], "constant", 0
    )
    padded_target_rel_vel = padded_target_rel_vel.reshape(
        num_envs, num_future_steps, 1, 6
    )

    heading_and_velocity = torch.cat(
        [
            target_rel_heading_rot,
            target_rel_heading_rot,
            padded_target_rel_vel,
            padded_target_rel_vel,
        ],
        dim=-1,
    )

    # In masked_mimic allow easy re-shape to [batch, time, joint, type (transform/rotate), features]
    obs = torch.cat(
        (
            sub_sampled_target_rel_body_pos,
            sub_sampled_target_body_pos,
            sub_sampled_target_rel_body_rot_obs,
            sub_sampled_target_body_rot_obs,
        ),
        dim=-1,
    )  # [batch, timesteps, joints, 24]
    obs = torch.cat((obs, heading_and_velocity), dim=-2).view(num_envs, -1)

    return obs


@torch.jit.script_if_tracing
def get_object_bounding_box_obs(
    object_ids: Tensor,
    root_pos: Tensor,
    root_quat: Tensor,
    num_object_envs: Tensor,
    object_root_states: Tensor,
    object_root_states_offsets: Tensor,
    object_bounding_box: Tensor,
    num_object_types: int,
    w_last: bool,
):
    expanded_root_pos = (
        root_pos.unsqueeze(1).expand(num_object_envs, 8, 3).reshape(-1, 3)
    )
    expanded_root_rot = (
        root_quat.unsqueeze(1).expand(num_object_envs, 8, 4).reshape(-1, 4)
    )

    root_rot_inv = torch_utils.calc_heading_quat_inv(root_quat, w_last)
    expanded_root_rot_inv = torch_utils.calc_heading_quat_inv(expanded_root_rot, w_last)

    obj_root_pos = object_root_states[object_ids, :3]
    obj_root_rot = object_root_states[object_ids, 3:7]

    # Apply translation offset
    obj_root_pos += object_root_states_offsets[object_ids, :3]

    # Apply rotation offset
    obj_root_rot = rotations.quat_mul(
        object_root_states_offsets[object_ids, 3:7], obj_root_rot, w_last
    )

    # Ensure the quaternion is normalized
    obj_root_rot = maths.normalize(obj_root_rot)

    object_bbs = object_bounding_box.view(-1, 3)
    expanded_obj_root_pos = (
        obj_root_pos.unsqueeze(1).expand(num_object_envs, 8, 3).reshape(-1, 3)
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

    # TODO: move this to a better place with SIGGRAPH Asia 2024 specific object observation.
    object_type = object_root_states_offsets[object_ids, -1]
    object_type_one_hot = torch.nn.functional.one_hot(
        object_type.to(torch.int64).view(-1), num_object_types
    )

    return torch.cat(
        (
            object_rotated_relative_to_env.view(num_object_envs, -1),
            object_root_rot_relative_to_env.view(num_object_envs, -1),
            object_type_one_hot.view(num_object_envs, -1),
        ),
        dim=-1,
    )


@torch.jit.script_if_tracing
def build_historical_body_poses(
    cur_gt: Tensor,
    cur_gr: Tensor,
    hist_gt: Tensor,
    hist_gr: Tensor,
    num_historical_stored_steps: int,
    num_historical_conditioned_steps: int,
    dt: float,
    num_envs: int,
    w_last: bool,
):
    sub_sampling = num_historical_stored_steps // num_historical_conditioned_steps

    cur_root_gt = cur_gt[:, 0]

    # unsqueeze [num_envs, 3] into [num_envs, 1 (for historical steps), 1 (for joints), 3]
    relative_gt = hist_gt - cur_root_gt.unsqueeze(1).unsqueeze(1)

    rr = cur_gr[:, 0]
    heading_inv: Tensor = torch_utils.calc_heading_quat_inv(rr, w_last)

    # from [num_envs, 4] -> [num_envs, num_future_steps, num_joints, 4]
    heading_inv_expand = heading_inv.unsqueeze(1).unsqueeze(1).expand(hist_gr.shape)

    norm_gt = rotations.quat_rotate(
        heading_inv_expand.reshape(-1, 4), relative_gt.reshape(-1, 3), w_last
    ).view(relative_gt.shape)

    relative_gr = rotations.quat_mul(heading_inv_expand, hist_gr, w_last)
    flat_relative_gr = torch_utils.quat_to_tan_norm(
        relative_gr.view(-1, 4), w_last
    ).view(num_envs, -1)

    flat_norm_gt = norm_gt.view(num_envs, -1)

    obs = torch.cat([flat_norm_gt, flat_relative_gr], dim=1)

    # now compute valid masks
    time_offsets = (
        torch.arange(
            1, num_historical_stored_steps + 1, device=cur_gt.device, dtype=torch.long
        )
        * dt
    )

    obs_with_time = torch.cat(
        [
            obs.view(num_envs, num_historical_conditioned_steps, -1),
            time_offsets[::sub_sampling]
            .view(1, num_historical_conditioned_steps, 1)
            .expand(num_envs, -1, -1),
        ],
        dim=-1,
    ).view(num_envs, -1)

    return obs_with_time
