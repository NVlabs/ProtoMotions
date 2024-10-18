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

import torch
from torch import Tensor
from typing import List, Dict, Union, Tuple
from omegaconf.dictconfig import DictConfig

from phys_anim.envs.humanoid.humanoid_utils import quat_diff_norm
from isaac_utils import torch_utils, rotations


@torch.jit.script
def mul_exp_sum(x: Tensor, coef: float, sum_before_exp: bool):
    if sum_before_exp:
        return x.sum(-1).sqrt().mul(coef).exp()
    else:
        return x.sqrt().mul(coef).exp().sum(-1)


@torch.jit.script_if_tracing  # This is important to ensure it doesn't compile the omega config early.
def exp_tracking_reward(
    gt: Tensor,
    rt: Tensor,
    rv: Tensor,
    rav: Tensor,
    gv: Tensor,
    gav: Tensor,
    kb: Tensor,
    gr: Tensor,
    lr: Tensor,
    dv: Tensor,
    ref_gt: Tensor,
    ref_rt: Tensor,
    ref_rv: Tensor,
    ref_rav: Tensor,
    ref_gv: Tensor,
    ref_gav: Tensor,
    ref_kb: Tensor,
    ref_gr: Tensor,
    ref_lr: Tensor,
    ref_dv: Tensor,
    joint_reward_weights: Union[Tensor, float],
    config: DictConfig,
    w_last: bool,
) -> Dict[str, Tensor]:
    # First sum across xyz
    if isinstance(joint_reward_weights, Tensor):
        joint_reward_weights = joint_reward_weights.unsqueeze(0)
    else:
        joint_reward_weights = joint_reward_weights
    gt_rew = (
        (gt - ref_gt)
        .pow(2)
        .mean(-1)
        .mul(joint_reward_weights)
        .mean(-1)
        .mul(config.component_coefficients.gt_rew_c)
        .exp()
    )

    rh = gt[:, 0, 2]
    ref_rh = ref_gt[:, 0, 2]

    rh_rew = (rh - ref_rh).pow(2).mul(config.component_coefficients.rh_rew_c).exp()

    rt_rew = (
        (rt - ref_rt).pow(2).mean(-1).mul(config.component_coefficients.rt_rew_c).exp()
    )
    rv_rew = (
        (rv - ref_rv).pow(2).mean(-1).mul(config.component_coefficients.rv_rew_c).exp()
    )
    rav_rew = (
        (rav - ref_rav)
        .pow(2)
        .mean(-1)
        .mul(config.component_coefficients.rav_rew_c)
        .exp()
    )

    gv_rew = (
        (gv - ref_gv)
        .pow(2)
        .mean(-1)
        .mean(-1)
        .mul(config.component_coefficients.gv_rew_c)
        .exp()
    )
    gav_rew = (
        (gav - ref_gav)
        .pow(2)
        .mean(-1)
        .mean(-1)
        .mul(config.component_coefficients.gav_rew_c)
        .exp()
    )

    # First sum across xyz
    kb_rew = (
        (kb - ref_kb)
        .pow(2)
        .mean(-1)
        .mean(-1)
        .mul(config.component_coefficients.kb_rew_c)
        .exp()
    )

    if config.add_rr_to_lr:
        joint_reward_weights_for_lr = joint_reward_weights
    else:
        if isinstance(joint_reward_weights, Tensor):
            joint_reward_weights_for_lr = joint_reward_weights[:, 1:]
        else:
            joint_reward_weights_for_lr = joint_reward_weights

    if config.tan_norm_reward:
        gr = torch_utils.quat_to_tan_norm(gr.view(-1, 4), w_last).view(
            *gr.shape[:-1], 6
        )
        ref_gr = torch_utils.quat_to_tan_norm(ref_gr.view(-1, 4), w_last).view(
            *ref_gr.shape[:-1], 6
        )

        gr_rew = mul_exp_sum(
            (gr - ref_gr).pow(2).sum(-1).mul(joint_reward_weights),
            config.component_coefficients.gr_rew_c,
            config.sum_before_exp,
        )

        lr = torch_utils.quat_to_tan_norm(lr.view(-1, 4), w_last).view(
            *lr.shape[:-1], 6
        )
        ref_lr = torch_utils.quat_to_tan_norm(ref_lr.view(-1, 4), w_last).view(
            *ref_lr.shape[:-1], 6
        )

        lr_rew = mul_exp_sum(
            (lr - ref_lr).pow(2).sum(-1).mul(joint_reward_weights_for_lr),
            config.component_coefficients.lr_rew_c,
            config.sum_before_exp,
        )
    else:
        diff_global_body_rot = (
            quat_diff_norm(gr, ref_gr, w_last).mul(joint_reward_weights).mean(dim=-1)
        )
        gr_rew = diff_global_body_rot.mul(config.component_coefficients.gr_rew_c).exp()

        diff_local_body_rot = (
            quat_diff_norm(lr, ref_lr, w_last)
            .mul(joint_reward_weights_for_lr)
            .mean(dim=-1)
        )
        lr_rew = diff_local_body_rot.mul(config.component_coefficients.lr_rew_c).exp()

    dv_rew = (
        (dv - ref_dv).pow(2).mean(-1).mul(config.component_coefficients.dv_rew_c).exp()
    )

    rew_dict = {
        "gt_rew": gt_rew,
        "rt_rew": rt_rew,
        "rh_rew": rh_rew,
        "rv_rew": rv_rew,
        "rav_rew": rav_rew,
        "gv_rew": gv_rew,
        "gav_rew": gav_rew,
        "kb_rew": kb_rew,
        "gr_rew": gr_rew,
        "lr_rew": lr_rew,
        "dv_rew": dv_rew,
    }

    return rew_dict


@torch.jit.script
def dof_to_local(pose: Tensor, dof_offsets: List[int], w_last: bool) -> Tensor:
    num_joints = len(dof_offsets) - 1

    assert pose.shape[-1] == dof_offsets[-1]
    local_rot_shape = pose.shape[:-1] + (num_joints, 4)
    local_rot = torch.zeros(local_rot_shape, device=pose.device)

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset : (dof_offset + dof_size)]

        # jp hack
        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose, w_last)
        elif dof_size == 1:
            axis = torch.tensor(
                [0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device
            )
            joint_pose_q = rotations.quat_from_angle_axis(
                joint_pose[..., 0], axis, w_last
            )
        else:
            joint_pose_q = None
            assert False, "Unsupported joint type"

        local_rot[:, j] = joint_pose_q

    return local_rot


@torch.jit.script_if_tracing
def build_max_coords_target_poses_future_rel(
    cur_gt: Tensor,
    cur_gr: Tensor,
    flat_target_pos: Tensor,
    flat_target_rot: Tensor,
    num_future_steps: int,
    num_envs: int,
    mimic_conditionable_bodies_ids: Tensor,
    w_last: bool,
):
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

    heading_inv_rot = torch_utils.calc_heading_quat_inv(reference_root_rot, w_last)

    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_reference_pos.shape[1], 1)
    )
    flat_heading_inv_rot = heading_inv_rot_expand.reshape(
        heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1],
        heading_inv_rot_expand.shape[2],
    )

    reference_root_pos_expand = reference_root_pos.unsqueeze(-2)

    """target"""
    # target body pos   [N, 3xB]
    target_rel_body_pos = flat_target_pos - flat_reference_pos
    flat_target_rel_body_pos = target_rel_body_pos.reshape(
        target_rel_body_pos.shape[0] * target_rel_body_pos.shape[1],
        target_rel_body_pos.shape[2],
    )
    flat_target_rel_body_pos = torch_utils.quat_rotate(
        flat_heading_inv_rot, flat_target_rel_body_pos, w_last
    )

    # target body pos   [N, 3xB]
    flat_target_body_pos = (flat_target_pos - reference_root_pos_expand).reshape(
        flat_target_pos.shape[0] * flat_target_pos.shape[1], flat_target_pos.shape[2]
    )
    flat_target_body_pos = torch_utils.quat_rotate(
        flat_heading_inv_rot, flat_target_body_pos, w_last
    )

    # target body rot   [N, 6xB]
    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_reference_rot, w_last), flat_target_rot, w_last
    )
    target_rel_body_rot_obs = (
        torch_utils.quat_to_tan_norm(target_rel_body_rot.view(-1, 4), w_last)
        .reshape(num_envs, num_future_steps, -1, 6)[
            :, :, mimic_conditionable_bodies_ids
        ]
        .reshape(target_rel_body_rot.shape[0], -1)
    )

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(
        heading_inv_rot_expand, flat_target_rot, w_last
    )
    target_body_rot_obs = (
        torch_utils.quat_to_tan_norm(target_body_rot.view(-1, 4), w_last)
        .reshape(num_envs, num_future_steps, -1, 6)[
            :, :, mimic_conditionable_bodies_ids
        ]
        .reshape(target_rel_body_rot.shape[0], -1)
    )

    target_rel_body_pos = flat_target_rel_body_pos.reshape(
        num_envs, num_future_steps, -1, 3
    )[:, :, mimic_conditionable_bodies_ids].reshape(target_rel_body_pos.shape[0], -1)
    target_body_pos = flat_target_body_pos.reshape(num_envs, num_future_steps, -1, 3)[
        :, :, mimic_conditionable_bodies_ids
    ].reshape(flat_target_pos.shape[0], -1)

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
    num_envs: int,
    num_future_steps: int,
    mimic_conditionable_bodies_ids: Tensor,
    w_last: bool,
):
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

    heading_inv_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
    heading_inv_rot_expand = heading_inv_rot_expand.repeat(
        (1, flat_cur_pos.shape[1], 1)
    )
    flat_heading_inv_rot = heading_inv_rot_expand.reshape(
        heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1],
        heading_inv_rot_expand.shape[2],
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
        flat_heading_inv_rot, flat_target_rel_body_pos, w_last
    )
    target_rel_body_pos = flat_target_rel_body_pos.reshape(
        num_envs, num_future_steps, -1, 3
    )[:, :, mimic_conditionable_bodies_ids].reshape(target_rel_body_pos.shape[0], -1)

    # target body pos   [N, 3xB]
    flat_target_body_pos = (flat_target_pos - root_pos_expand).reshape(
        flat_target_pos.shape[0] * flat_target_pos.shape[1], flat_target_pos.shape[2]
    )
    flat_target_body_pos = torch_utils.quat_rotate(
        flat_heading_inv_rot, flat_target_body_pos, w_last
    )
    target_body_pos = flat_target_body_pos.reshape(num_envs, num_future_steps, -1, 3)[
        :, :, mimic_conditionable_bodies_ids
    ].reshape(flat_target_pos.shape[0], -1)

    # target body rot   [N, 6xB]
    target_rel_body_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_cur_rot, w_last), flat_target_rot, w_last
    )
    target_rel_body_rot_obs = (
        torch_utils.quat_to_tan_norm(target_rel_body_rot.view(-1, 4), w_last)
        .reshape(num_envs, num_future_steps, -1, 6)[
            :, :, mimic_conditionable_bodies_ids
        ]
        .reshape(target_rel_body_rot.shape[0], -1)
    )

    # target body rot   [N, 6xB]
    target_body_rot = rotations.quat_mul(
        heading_inv_rot_expand, flat_target_rot, w_last
    )
    target_body_rot_obs = (
        torch_utils.quat_to_tan_norm(target_body_rot.view(-1, 4), w_last)
        .reshape(num_envs, num_future_steps, -1, 6)[
            :, :, mimic_conditionable_bodies_ids
        ]
        .reshape(target_rel_body_rot.shape[0], -1)
    )

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
def build_max_coords_object_target_poses(
    cur_object_gt: Tensor,
    cur_object_gr: Tensor,
    cur_robot_gt: Tensor,
    cur_robot_gr: Tensor,
    flat_object_target_pos: Tensor,
    flat_object_target_rot: Tensor,
    num_scenes: int,
    max_objects_per_scene: int,
    num_future_steps: int,
    w_last: bool,
):
    # Reshape inputs
    target_pos = flat_object_target_pos.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 3
    )
    target_rot = flat_object_target_rot.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 4
    )
    cur_object_pos = cur_object_gt.reshape(num_scenes, max_objects_per_scene, 3)
    cur_object_rot = cur_object_gr.reshape(num_scenes, max_objects_per_scene, 4)
    cur_robot_pos = cur_robot_gt.reshape(num_scenes, 3)
    cur_robot_rot = cur_robot_gr.reshape(num_scenes, 4)

    # Calculate heading inverse rotation for objects and robot
    object_heading_inv_rot = torch_utils.calc_heading_quat_inv(cur_object_rot, w_last)
    robot_heading_inv_rot = torch_utils.calc_heading_quat_inv(cur_robot_rot, w_last)

    # Expand current positions and rotations
    cur_object_pos_expanded = cur_object_pos.unsqueeze(2).expand(
        -1, -1, num_future_steps, -1
    )
    cur_object_rot_expanded = cur_object_rot.unsqueeze(2).expand(
        -1, -1, num_future_steps, -1
    )
    cur_robot_pos_expanded = (
        cur_robot_pos.unsqueeze(1)
        .unsqueeze(1)
        .expand(-1, max_objects_per_scene, num_future_steps, -1)
    )
    cur_robot_rot_expanded = (
        cur_robot_rot.unsqueeze(1)
        .unsqueeze(1)
        .expand(-1, max_objects_per_scene, num_future_steps, -1)
    )

    # Flatten tensors
    flat_target_pos = target_pos.reshape(-1, 3)
    flat_target_rot = target_rot.reshape(-1, 4)
    flat_cur_object_pos = cur_object_pos_expanded.reshape(-1, 3)
    flat_cur_object_rot = cur_object_rot_expanded.reshape(-1, 4)
    flat_cur_robot_pos = cur_robot_pos_expanded.reshape(-1, 3)
    flat_cur_robot_rot = cur_robot_rot_expanded.reshape(-1, 4)
    flat_object_heading_inv_rot = (
        object_heading_inv_rot.unsqueeze(1)
        .expand(-1, num_future_steps, -1, -1)
        .reshape(-1, 4)
    )
    flat_robot_heading_inv_rot = (
        robot_heading_inv_rot.unsqueeze(1)
        .unsqueeze(2)
        .expand(-1, num_future_steps, max_objects_per_scene, -1)
        .reshape(-1, 4)
    )

    # 1. Relative change w.r.t. object's initial state
    object_rel_pos = flat_target_pos - flat_cur_object_pos
    object_rel_pos_heading_inv = torch_utils.quat_rotate(
        flat_object_heading_inv_rot, object_rel_pos, w_last
    )

    object_rel_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_cur_object_rot, w_last), flat_target_rot, w_last
    )
    object_rel_rot_obs = torch_utils.quat_to_tan_norm(object_rel_rot, w_last)

    # 2. Relative change w.r.t. humanoid root
    robot_rel_pos = flat_target_pos - flat_cur_robot_pos
    robot_rel_pos_heading_inv = torch_utils.quat_rotate(
        flat_robot_heading_inv_rot, robot_rel_pos, w_last
    )

    robot_rel_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_cur_robot_rot, w_last), flat_target_rot, w_last
    )
    robot_rel_rot_obs = torch_utils.quat_to_tan_norm(robot_rel_rot, w_last)

    # Reshape observations
    object_rel_pos_heading_inv = object_rel_pos_heading_inv.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 3
    )
    object_rel_rot_obs = object_rel_rot_obs.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 6
    )
    robot_rel_pos_heading_inv = robot_rel_pos_heading_inv.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 3
    )
    robot_rel_rot_obs = robot_rel_rot_obs.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 6
    )

    # Concatenate all observations
    obs = torch.cat(
        (
            object_rel_pos_heading_inv,
            object_rel_rot_obs,
            robot_rel_pos_heading_inv,
            robot_rel_rot_obs,
        ),
        dim=-1,
    )

    return obs.reshape(num_scenes, -1)


@torch.jit.script_if_tracing
def build_max_coords_object_target_poses_future_rel(
    cur_object_gt: Tensor,
    cur_object_gr: Tensor,
    cur_robot_gt: Tensor,
    cur_robot_gr: Tensor,
    flat_object_target_pos: Tensor,
    flat_object_target_rot: Tensor,
    num_scenes: int,
    max_objects_per_scene: int,
    num_future_steps: int,
    w_last: bool,
):
    # Reshape inputs
    target_pos = flat_object_target_pos.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 3
    )
    target_rot = flat_object_target_rot.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 4
    )

    # Calculate reference positions and rotations (shifted by one step)
    reference_pos = target_pos.clone().roll(shifts=1, dims=2)
    reference_pos[:, :, 0] = cur_object_gt.reshape(num_scenes, max_objects_per_scene, 3)
    reference_rot = target_rot.clone().roll(shifts=1, dims=2)
    reference_rot[:, :, 0] = cur_object_gr.reshape(num_scenes, max_objects_per_scene, 4)

    # Flatten tensors for easier computation
    flat_reference_pos = reference_pos.reshape(-1, 3)
    flat_reference_rot = reference_rot.reshape(-1, 4)
    flat_target_pos = target_pos.reshape(-1, 3)
    flat_target_rot = target_rot.reshape(-1, 4)

    # Calculate heading inverse rotation for objects
    heading_inv_rot = torch_utils.calc_heading_quat_inv(flat_reference_rot, w_last)

    # 1. Relative change w.r.t. object's previous state
    object_rel_pos = flat_target_pos - flat_reference_pos
    object_rel_pos_heading_inv = torch_utils.quat_rotate(
        heading_inv_rot, object_rel_pos, w_last
    )

    object_rel_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_reference_rot, w_last), flat_target_rot, w_last
    )
    object_rel_rot_obs = torch_utils.quat_to_tan_norm(object_rel_rot, w_last)

    # 2. Relative change w.r.t. humanoid root
    robot_root_pos = (
        cur_robot_gt.unsqueeze(1)
        .unsqueeze(1)
        .expand(-1, max_objects_per_scene, num_future_steps, -1)
    )
    robot_root_rot = (
        cur_robot_gr.unsqueeze(1)
        .unsqueeze(1)
        .expand(-1, max_objects_per_scene, num_future_steps, -1)
    )
    flat_robot_root_pos = robot_root_pos.reshape(-1, 3)
    flat_robot_root_rot = robot_root_rot.reshape(-1, 4)

    robot_heading_inv_rot = torch_utils.calc_heading_quat_inv(
        flat_robot_root_rot, w_last
    )

    robot_rel_pos = flat_target_pos - flat_robot_root_pos
    robot_rel_pos_heading_inv = torch_utils.quat_rotate(
        robot_heading_inv_rot, robot_rel_pos, w_last
    )

    robot_rel_rot = rotations.quat_mul(
        rotations.quat_conjugate(flat_robot_root_rot, w_last), flat_target_rot, w_last
    )
    robot_rel_rot_obs = torch_utils.quat_to_tan_norm(robot_rel_rot, w_last)

    # Reshape observations
    object_rel_pos_heading_inv = object_rel_pos_heading_inv.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 3
    )
    object_rel_rot_obs = object_rel_rot_obs.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 6
    )
    robot_rel_pos_heading_inv = robot_rel_pos_heading_inv.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 3
    )
    robot_rel_rot_obs = robot_rel_rot_obs.reshape(
        num_scenes, max_objects_per_scene, num_future_steps, 6
    )

    # Concatenate all observations
    obs = torch.cat(
        (
            object_rel_pos_heading_inv,
            object_rel_rot_obs,
            robot_rel_pos_heading_inv,
            robot_rel_rot_obs,
        ),
        dim=-1,
    )

    return obs.reshape(num_scenes, -1)
