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

from isaac_utils import torch_utils, rotations

import torch
import numpy as np

from phys_anim.envs.mimic.common import quat_diff_norm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phys_anim.envs.masked_mimic.tasks.speed.isaacgym import (
        MaskedMimicSpeedHumanoid,
    )
else:
    MaskedMimicSpeedHumanoid = object


class BaseMaskedMimicSpeed(MaskedMimicSpeedHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config=config, device=device)

        self._heading_change_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )
        self._tar_speed = (
            torch.ones([self.num_envs], device=self.device, dtype=torch.float) * 5
        )
        self._tar_dir = torch.zeros(
            [self.num_envs, 2], device=self.device, dtype=torch.float
        )
        self._tar_dir[..., 0] = 1.0

        self._tar_facing_dir = torch.zeros(
            [self.num_envs, 2], device=self.device, dtype=torch.float
        )
        self._tar_facing_dir[..., 0] = 1.0

        self._heading_turn_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )

        self._failures = []
        self._distances = []
        self._current_accumulated_errors = (
            torch.zeros([self.num_envs], device=self.device, dtype=torch.float) - 1
        )
        self._current_failures = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.float
        )
        self._last_length = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.long
        )

    ###############################################################
    # Handle resets
    ###############################################################
    def reset_task(self, env_ids):
        if len(env_ids) > 0:
            # Make sure the test has started + agent started from a valid position (if it failed, then it's not valid)
            active_envs = (self._current_accumulated_errors[env_ids] > 0) & (
                (self._last_length[env_ids] - self._heading_turn_steps[env_ids]) > 0
            )
            average_distances = self._current_accumulated_errors[env_ids][
                active_envs
            ] / (
                self._last_length[env_ids][active_envs]
                - self._heading_turn_steps[env_ids][active_envs]
            )
            self._distances.extend(average_distances.cpu().tolist())
            self._current_accumulated_errors[env_ids] = 0
            self._failures.extend(
                (self._current_failures[env_ids][active_envs] > 0).cpu().tolist()
            )
            self._current_failures[env_ids] = 0

        super().reset_task(env_ids)
        n = len(env_ids)
        if self.config.speed_params.enable_rand_heading:
            rand_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
        else:
            rand_theta = torch.zeros(n, device=self.device)

        tar_dir = torch.stack([torch.cos(rand_theta), torch.sin(rand_theta)], dim=-1)
        tar_speed = (
            self.config.speed_params.tar_speed_max
            - self.config.speed_params.tar_speed_min
        ) * torch.rand(n, device=self.device) + self.config.tar_speed_min
        change_steps = torch.randint(
            low=self.config.speed_params.change_steps_min,
            high=self.config.speed_params.change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )

        self._tar_speed[env_ids] = tar_speed
        self._tar_dir[env_ids] = tar_dir
        self._tar_facing_dir[env_ids] = tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        self._heading_turn_steps[env_ids] = 60 * 1 + self.progress_buf[env_ids]
        self._prev_root_pos[:, :2] = self.get_bodies_state().body_pos[:, 0, :2]

    def store_motion_data(self, skip=False):
        super().store_motion_data(skip=True)
        if skip:
            return

        if "target_poses" not in self.motion_recording:
            self.motion_recording["target_poses"] = []

        data = torch.stack([self._tar_dir, self._tar_facing_dir], dim=1)
        self.motion_recording["target_poses"].append(
            data.view(self.num_envs, 2, 2).cpu().numpy()
        )

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_reward(self, actions):
        super().compute_reward(actions)

        current_state = self.get_bodies_state()
        body_pos, body_rot = (
            current_state.body_pos,
            current_state.body_rot,
        )

        root_vel = (body_pos[:, 0, :2] - self._prev_root_pos[:, :2]) / self.dt
        tar_dir_vel = self._tar_dir[:] * self._tar_speed[:].unsqueeze(-1)

        tangent_vel = root_vel - tar_dir_vel
        tangent_vel_error = torch.norm(tangent_vel, dim=-1)

        turning_envs = self._heading_turn_steps > self.progress_buf
        turned_envs = ~turning_envs

        print(f"Velocity: {root_vel[0]}")
        print(f"Target: {tar_dir_vel[0]}")
        print(f"Error: {tangent_vel_error[0]}")

        # Turn 3d rotation to flat heading quaternion
        facing_quat = torch_utils.calc_heading_quat(body_rot[:, 0], w_last=self.w_last)

        # Turn 2 vector to quaternion
        angle = rotations.vec_to_heading(self._tar_facing_dir)
        neg = angle < 0
        angle[neg] += 2 * torch.pi
        tar_facing_quat = rotations.heading_to_quat(angle, w_last=self.w_last)

        # Compute angle error
        facing_err = quat_diff_norm(facing_quat, tar_facing_quat, self.w_last)
        facing_err_degrees = facing_err * 180 / torch.pi

        self._current_accumulated_errors[turned_envs] += tangent_vel_error[turned_envs]
        self._current_failures[turned_envs] += (
            45 < facing_err_degrees[turned_envs]
        ) | (facing_err_degrees[turned_envs] < -45)
        self._current_failures[turning_envs] = 0
        self._current_accumulated_errors[turning_envs] = 0
        self._last_length[:] = self.progress_buf[:]

        if len(self._failures) > 0:
            self.last_other_rewards["reach_success"] = 1.0 - sum(self._failures) / len(
                self._failures
            )
            self.last_other_rewards["reach_distance"] = sum(self._distances) / len(
                self._distances
            )

    def compute_observations(self, env_ids=None):
        self.mask_everything()
        super().compute_observations(env_ids)
        self.mask_everything()

        turning_envs = self._heading_turn_steps < 0  # > (self.progress_buf + 15)
        turned_envs = ~turning_envs

        time_left_to_turn = (self._heading_turn_steps - self.progress_buf - 15).clamp(5)

        pelvis_body_index = self.config.masked_mimic_conditionable_bodies.index(
            "Pelvis"
        )

        self.target_pose_time[turning_envs] = (
            self.motion_times[turning_envs] + self.dt * time_left_to_turn[turning_envs]
        )
        self.target_pose_time[turned_envs] = (
            self.motion_times[turned_envs] + 1.0
        )  # .5 second
        self.target_pose_obs_mask[:] = True
        self.target_pose_joints[:] = False
        self.target_pose_joints[turned_envs, pelvis_body_index * 2] = True
        # self.target_pose_joints[turning_envs, pelvis_body_index * 2 + 1] = True
        self.target_pose_joints[turning_envs, -1] = True

        single_step_mask_size = self.num_conditionable_bodies * 2
        new_mask = torch.zeros(
            self.num_envs,
            self.num_conditionable_bodies,
            2,
            dtype=torch.bool,
            device=self.device,
        )
        new_mask[:, pelvis_body_index, :] = True
        new_mask[:, -1, -1] = True
        new_mask = (
            new_mask.view(self.num_envs, 1, single_step_mask_size)
            .expand(-1, self.config.masked_mimic_obs.num_future_steps, -1)
            .reshape(self.num_envs, -1)
        )
        self.masked_mimic_target_bodies_masks[:] = False
        self.masked_mimic_target_bodies_masks[:] = new_mask

        sparse_target_poses = self.build_sparse_target_heading_poses_masked_with_time(
            self.config.masked_mimic_obs.num_future_steps
        )
        self.masked_mimic_target_poses[:] = sparse_target_poses

        self.masked_mimic_target_poses_masks[:] = False
        self.masked_mimic_target_poses_masks[:, -1] = True
        self.masked_mimic_target_poses_masks[turned_envs, -2] = True

    def update_task(self, actions):
        super().update_task(actions)

        reset_task_mask = self.progress_buf >= self._heading_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self.reset_task(rest_env_ids)

    ###############################################################
    # Helpers
    ###############################################################
    def build_sparse_target_heading_poses(self, raw_future_times):
        """
        This is identical to the max_coords humanoid observation, only in relative to the current pose.
        """
        num_future_steps = raw_future_times.shape[1]

        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        lengths = self.motion_lib.get_motion_length(flat_ids)

        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        ref_state = self.motion_lib.get_mimic_motion_state(flat_ids, flat_times)
        flat_target_pos, flat_target_rot, flat_target_vel = (
            ref_state.rb_pos,
            ref_state.rb_rot,
            ref_state.rb_vel,
        )

        current_state = self.get_bodies_state()
        cur_gt, cur_gr = current_state.body_pos, current_state.body_rot
        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        cur_gt[:, :, -1:] -= self.get_ground_heights(cur_gt[:, 0, :2]).view(
            self.num_envs, 1, 1
        )
        # cur_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[..., :2].view(self.num_envs, 1, 2)

        # override to set the target root parameters
        reshaped_target_pos = flat_target_pos.reshape(
            self.num_envs, num_future_steps, -1, 3
        )
        reshaped_target_rot = flat_target_rot.reshape(
            self.num_envs, num_future_steps, -1, 4
        )

        turning_envs = self._heading_turn_steps < 0  # > (self.progress_buf + 15)
        # turned_envs = ~turning_envs

        reshaped_target_pos[:, :, 0, :2] = cur_gt[:, 0, :2].unsqueeze(1).clone()
        for frame_idx in range(num_future_steps):
            reshaped_target_pos[:, frame_idx, 0, :2] += (
                self._tar_dir[:]
                * self._tar_speed[:].unsqueeze(-1)
                * (raw_future_times[:, frame_idx] - self.motion_times).unsqueeze(-1)
            )

        reshaped_target_pos[turning_envs, :, 0, :2] = (
            cur_gt[turning_envs, 0, :2].unsqueeze(1).clone()
        )

        reshaped_target_pos[:, :, 0, -1] = 0.88  # standing up

        angle = rotations.vec_to_heading(self._tar_facing_dir)
        neg = angle < 0
        angle[neg] += 2 * torch.pi
        quat = rotations.heading_to_quat(angle, w_last=self.w_last)
        reshaped_target_rot[:, :, 0] = quat.unsqueeze(1)

        flat_target_pos = reshaped_target_pos.reshape(flat_target_pos.shape)
        flat_target_rot = reshaped_target_rot.reshape(flat_target_rot.shape)

        non_flat_target_vel = flat_target_vel.reshape(
            self.num_envs, num_future_steps, -1, 3
        )

        non_flat_target_vel[:, :, 0, :2] = (
            self._tar_dir[:] * self._tar_speed[:].unsqueeze(-1)
        ).view(self._tar_dir.shape[0], 1, 2)
        flat_target_vel = non_flat_target_vel.reshape(flat_target_vel.shape)
        # override to set the target root parameters

        expanded_body_pos = cur_gt.unsqueeze(1).expand(
            self.num_envs, num_future_steps, *cur_gt.shape[1:]
        )
        expanded_body_rot = cur_gr.unsqueeze(1).expand(
            self.num_envs, num_future_steps, *cur_gr.shape[1:]
        )

        flat_cur_pos = expanded_body_pos.reshape(flat_target_pos.shape)
        flat_cur_rot = expanded_body_rot.reshape(flat_target_rot.shape)

        root_pos = flat_cur_pos[:, 0, :]
        root_rot = flat_cur_rot[:, 0, :]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot, self.w_last)

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
            flat_heading_rot, flat_target_rel_body_pos, self.w_last
        )

        # target body pos   [N, 3xB]
        flat_target_body_pos = (flat_target_pos - root_pos_expand).reshape(
            flat_target_pos.shape[0] * flat_target_pos.shape[1],
            flat_target_pos.shape[2],
        )
        flat_target_body_pos = torch_utils.quat_rotate(
            flat_heading_rot, flat_target_body_pos, self.w_last
        )

        # target body rot   [N, 6xB]
        target_rel_body_rot = rotations.quat_mul(
            rotations.quat_conjugate(flat_cur_rot, self.w_last),
            flat_target_rot,
            self.w_last,
        )
        target_rel_body_rot_obs = torch_utils.quat_to_tan_norm(
            target_rel_body_rot.view(-1, 4), self.w_last
        ).view(target_rel_body_rot.shape[0], -1)

        # target body rot   [N, 6xB]
        target_body_rot = rotations.quat_mul(
            heading_rot_expand, flat_target_rot, self.w_last
        )
        target_body_rot_obs = torch_utils.quat_to_tan_norm(
            target_body_rot.view(-1, 4), self.w_last
        ).view(target_rel_body_rot.shape[0], -1)

        padded_flat_target_rel_body_pos = torch.nn.functional.pad(
            flat_target_rel_body_pos, [0, 3], "constant", 0
        )
        sub_sampled_target_rel_body_pos = padded_flat_target_rel_body_pos.reshape(
            self.num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]

        padded_flat_target_body_pos = torch.nn.functional.pad(
            flat_target_body_pos, [0, 3], "constant", 0
        )
        sub_sampled_target_body_pos = padded_flat_target_body_pos.reshape(
            self.num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]

        sub_sampled_target_rel_body_rot_obs = target_rel_body_rot_obs.reshape(
            self.num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]
        sub_sampled_target_body_rot_obs = target_body_rot_obs.reshape(
            self.num_envs, num_future_steps, -1, 6
        )[:, :, self.masked_mimic_conditionable_bodies_ids]

        # Heading
        target_heading_rot = torch_utils.calc_heading_quat(
            flat_target_rot[:, 0, :], self.w_last
        )
        target_rel_heading_rot = torch_utils.quat_to_tan_norm(
            rotations.quat_mul(
                heading_rot_expand[:, 0, :], target_heading_rot, self.w_last
            ).view(-1, 4),
            self.w_last,
        ).reshape(self.num_envs, num_future_steps, 1, 6)

        # Velocity
        target_root_vel = flat_target_vel[:, 0, :]
        target_root_vel[..., -1] = 0  # ignore vertical speed
        target_rel_vel = rotations.quat_rotate(
            heading_rot, target_root_vel, self.w_last
        ).reshape(-1, 3)
        padded_target_rel_vel = torch.nn.functional.pad(
            target_rel_vel, [0, 3], "constant", 0
        )
        padded_target_rel_vel = padded_target_rel_vel.reshape(
            self.num_envs, num_future_steps, 1, 6
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
        obs = torch.cat((obs, heading_and_velocity), dim=-2).view(self.num_envs, -1)

        return obs

    def build_sparse_target_heading_poses_masked_with_time(self, num_future_steps):
        time_offsets = (
            torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
            * self.dt
        )

        near_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        all_future_times = torch.cat(
            [near_future_times, self.target_pose_time.view(-1, 1)], dim=1
        )

        obs = self.build_sparse_target_heading_poses(all_future_times).view(
            self.num_envs,
            num_future_steps + 1,
            self.masked_mimic_conditionable_bodies_ids.shape[0] + 1,
            2,
            12,
        )

        near_mask = self.masked_mimic_target_bodies_masks.view(
            self.num_envs, num_future_steps, self.num_conditionable_bodies, 2, 1
        )
        far_mask = self.target_pose_joints.view(self.num_envs, 1, -1, 2, 1)
        mask = torch.cat([near_mask, far_mask], dim=1)

        masked_obs = obs * mask

        masked_obs_with_joints = torch.cat((masked_obs, mask), dim=-1).view(
            self.num_envs, num_future_steps + 1, -1
        )

        times = all_future_times.view(-1).view(
            self.num_envs, num_future_steps + 1, 1
        ) - self.motion_times.view(self.num_envs, 1, 1)
        ones_vec = torch.ones(
            self.num_envs, num_future_steps + 1, 1, device=self.device
        )
        times_with_mask = torch.cat((times, ones_vec), dim=-1)
        combined_sparse_future_pose_obs = torch.cat(
            (masked_obs_with_joints, times_with_mask), dim=-1
        )

        return combined_sparse_future_pose_obs.view(self.num_envs, -1)
