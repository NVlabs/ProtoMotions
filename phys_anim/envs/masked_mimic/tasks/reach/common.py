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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phys_anim.envs.masked_mimic.tasks.reach.isaacgym import (
        MaskedMimicReachHumanoid,
    )
else:
    MaskedMimicReachHumanoid = object


class BaseMaskedMimicReach(MaskedMimicReachHumanoid):
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        self._tar_change_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._tar_reach_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._tar_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )
        self._failures = []
        self._distances = []
        self._current_accumulated_errors = (
            torch.zeros([self.num_envs], device=self.device, dtype=torch.float) - 1
        )
        self._current_failures = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.float
        )
        self._last_failures = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.bool
        )
        self._last_reset_time = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.long
        )
        self._last_length = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.long
        )

        self._use_text = True
        self._text_embedding = None
        if self._use_text:
            from transformers import AutoTokenizer, XCLIPTextModel

            model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

            # text_command = ["the person gets on their hand and knees and crawls around"]
            # text_command = ["a person raises both hands and walks forward"]
            text_command = ["a person reaches with its right hand"]
            with torch.inference_mode():
                inputs = tokenizer(
                    text_command, padding=True, truncation=True, return_tensors="pt"
                )
                outputs = model(**inputs)
                pooled_output = outputs.pooler_output  # pooled (EOS token) states
                self._text_embedding = pooled_output[0].to(self.device)

    ###############################################################
    # Handle resets
    ###############################################################
    def reset_task(self, env_ids):
        if len(env_ids) > 0:
            # Make sure the test has started + agent started from a valid position (if it failed, then it's not valid)
            active_envs = (self._current_accumulated_errors[env_ids] > 0) & (
                ~self._last_failures[env_ids]
            )
            average_distances = self._current_accumulated_errors[env_ids][
                active_envs
            ] / (
                self._last_length[env_ids][active_envs]
                - self._tar_reach_steps[env_ids][active_envs]
            )
            self._distances.extend(average_distances.cpu().tolist())
            self._current_accumulated_errors[env_ids] = 0
            self._failures.extend(
                (self._current_failures[env_ids][active_envs] > 0).cpu().tolist()
            )
            self._last_failures[env_ids] = self._current_failures[env_ids] > 0
            self._current_failures[env_ids] = 0

        super().reset_task(env_ids)
        n = len(env_ids)

        rand_pos = torch.rand([n, 3], device=self.device)
        rand_pos[..., 0:2] = self.config.reach_params.tar_dist_max * (
            1.5 * rand_pos[..., 0:2] - 0.75
        )
        rand_pos[..., 2] = (
            self.config.reach_params.tar_height_max
            - self.config.reach_params.tar_height_min
        ) * rand_pos[..., 2] + self.config.reach_params.tar_height_min

        change_steps = torch.randint(
            low=self.config.reach_params.change_steps_min,
            high=self.config.reach_params.change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )
        reach_steps = torch.randint(
            low=self.config.reach_params.reach_steps_min,
            high=self.config.reach_params.reach_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )
        # min with change_steps to avoid reaching AFTER changing
        reach_steps = torch.min(reach_steps, change_steps)

        bodies_positions = self.get_body_positions()
        root_pos = bodies_positions[env_ids, 0, :]
        root_pos[:, -1:] = 0

        marker_pos = root_pos + rand_pos
        marker_pos[:, -1:] += self.terrain_obs_cb.get_ground_heights(
            marker_pos[:, :2]
        ).view(-1, 1)

        # Marker position is represented relative to the character pos, without terrains.
        self._tar_pos[env_ids, :] = marker_pos
        self._tar_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        self._tar_reach_steps[env_ids] = self.progress_buf[env_ids] + reach_steps
        self._last_reset_time[env_ids] = self.progress_buf[env_ids]

    def store_motion_data(self, skip=False):
        super().store_motion_data(skip=True)
        if skip:
            return

        if "target_poses" not in self.motion_recording:
            self.motion_recording["target_poses"] = []

        self.motion_recording["target_poses"].append(
            self._tar_pos[:].view(self.num_envs, 1, 3).cpu().numpy()
        )

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_reward(self, actions):
        super().compute_reward(actions)

        body_pos = self.get_bodies_state().body_pos
        reach_actual_pos = body_pos[:, self.reach_body_id, :]

        goal_pos = self._tar_pos

        distance_to_target = torch.norm(reach_actual_pos - goal_pos, dim=-1).view(
            self.num_envs
        )

        measurement_started = self._tar_reach_steps < self.progress_buf
        measurement_not_started = ~measurement_started

        self._current_accumulated_errors[measurement_started] += distance_to_target[
            measurement_started
        ]
        self._current_failures[measurement_started] += (
            distance_to_target[measurement_started] > 0.5
        )
        self._current_failures[measurement_not_started] = 0
        self._current_accumulated_errors[measurement_not_started] = 0
        self._last_length[:] = self.progress_buf[:]

        if len(self._failures) > 0:
            self.mimic_info_dict["reach_success"] = 1.0 - sum(self._failures) / len(
                self._failures
            )
            self.mimic_info_dict["reach_distance"] = sum(self._distances) / len(
                self._distances
            )

    def compute_observations(self, env_ids=None):
        self.mask_everything()
        super().compute_observations(env_ids)
        self.mask_everything()

        reach_steps_left = (self._tar_reach_steps - self.progress_buf - 10).clamp(
            min=3
        )  # just incase keep above 0 to ensure it isn't hidden

        target_body_index = self.config.masked_mimic_conditionable_bodies.index(
            self.config.reach_params.reach_body_name
        )

        self.target_pose_time[:] = self.motion_times + self.dt * reach_steps_left
        self.target_pose_obs_mask[:] = True
        self.target_pose_joints[:] = False
        self.target_pose_joints[:, target_body_index * 2] = True

        sparse_target_poses = self.build_sparse_target_reach_poses_masked_with_time(
            self.config.masked_mimic_obs.num_future_steps
        )
        self.masked_mimic_target_poses[:] = sparse_target_poses

        self.masked_mimic_target_poses_masks[:] = False
        self.masked_mimic_target_poses_masks[:, -1] = True

        if self._use_text:
            self.motion_text_embeddings_mask[:] = True
            self.motion_text_embeddings[:] = self._text_embedding

    def update_task(self, actions):
        super().update_task(actions)

        reset_task_mask = self.progress_buf >= self._tar_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self.reset_task(rest_env_ids)

    ###############################################################
    # Helpers
    ###############################################################
    def build_sparse_target_reach_poses(self, raw_future_times):
        """
        This is identical to the max_coords humanoid observation, only in relative to the current pose.
        """
        num_future_steps = raw_future_times.shape[1]

        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        lengths = self.motion_lib.get_motion_length(flat_ids)

        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        ref_state = self.motion_lib.get_mimic_motion_state(flat_ids, flat_times)
        flat_target_pos = ref_state.rb_pos
        flat_target_rot = ref_state.rb_rot
        flat_target_vel = ref_state.rb_vel

        current_state = self.get_bodies_state()
        cur_gt, cur_gr = current_state.body_pos, current_state.body_rot

        # override to set the target root parameters
        reshaped_target_pos = flat_target_pos.reshape(
            self.num_envs, num_future_steps, -1, 3
        )
        reshaped_target_pos[:, :, self.reach_body_id, :] = self._tar_pos.unsqueeze(1)
        flat_target_pos = reshaped_target_pos.reshape(flat_target_pos.shape)
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

    def build_sparse_target_reach_poses_masked_with_time(self, num_future_steps):
        time_offsets = (
            torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
            * self.dt
        )

        near_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        all_future_times = torch.cat(
            [near_future_times, self.target_pose_time.view(-1, 1)], dim=1
        )

        obs = self.build_sparse_target_reach_poses(all_future_times).view(
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
