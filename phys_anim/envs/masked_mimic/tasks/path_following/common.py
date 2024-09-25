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
from torch import Tensor

from phys_anim.envs.env_utils.path_generator import PathGenerator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phys_anim.envs.masked_mimic.tasks.path_following.isaacgym import (
        MaskedMimicPathFollowingHumanoid,
    )
else:
    MaskedMimicPathFollowingHumanoid = object


class BaseMaskedMimicPathFollowing(MaskedMimicPathFollowingHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        self._num_traj_samples = config.path_follower_params.num_traj_samples
        self._traj_sample_timestep = config.path_follower_params.traj_sample_timestep

        super().__init__(config=config, device=device)

        self.build_path_generator()
        self.reset_path_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self.condition_body_part = "Head"

        self._failures = []
        self._distances = []
        self._current_accumulated_errors = (
            torch.zeros([self.num_envs], device=self.device, dtype=torch.float) - 1
        )
        self._current_failures = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.float
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
            # text_command = ["swinging arms up and down"]
            text_command = ["a person walks casually"]
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
            active_envs = self._current_accumulated_errors[env_ids] > 0
            average_distances = (
                self._current_accumulated_errors[env_ids][active_envs]
                / self._last_length[env_ids][active_envs]
            )
            self._distances.extend(average_distances.cpu().tolist())
            self._current_accumulated_errors[env_ids] = 0
            self._failures.extend(
                (self._current_failures[env_ids][active_envs] > 0).cpu().tolist()
            )
            self._current_failures[env_ids] = 0

        super().reset_task(env_ids)
        self.reset_path_ids = env_ids

    def store_motion_data(self, skip=False):
        super().store_motion_data(skip=True)
        if skip:
            return

        if "target_poses" not in self.motion_recording:
            self.motion_recording["target_poses"] = []

        traj_samples = self.fetch_path_samples(time_offset=0)[0].clone()
        self._marker_pos[:] = traj_samples
        if not self.config.path_follower_params.path_generator.height_conditioned:
            self._marker_pos[..., 2] = 0.92  # CT hack

        ground_below_marker = self.get_ground_heights(
            traj_samples[..., :2].view(-1, 2)
        ).view(traj_samples.shape[:-1])

        self._marker_pos[..., 2] += ground_below_marker

        self.motion_recording["target_poses"].append(
            self._marker_pos[:].view(self.num_envs, -1, 3).cpu().numpy()
        )

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_reward(self, actions):
        super().compute_reward(actions)

        body_part = self.gym.find_asset_rigid_body_index(
            self.humanoid_asset, self.condition_body_part
        )
        current_state = self.get_bodies_state()
        cur_gt = current_state.body_pos

        cur_gt[:, :, -1:] -= self.get_ground_heights(cur_gt[:, 0, :2]).view(
            self.num_envs, 1, 1
        )

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time).clone()

        distance_to_target = torch.norm(
            cur_gt[:, body_part, :3] - tar_pos[:, :3], dim=-1
        ).view(self.num_envs)

        warmup_passed = self.progress_buf > 10  # 10 frames

        self._current_accumulated_errors[warmup_passed] += distance_to_target[
            warmup_passed
        ]
        self._current_failures[warmup_passed] += distance_to_target[warmup_passed] > 2.0
        self._last_length[warmup_passed] = self.progress_buf[warmup_passed]

        self._current_accumulated_errors[~warmup_passed] = 0
        self._current_failures[~warmup_passed] = 0

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

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        bodies_positions = self.get_body_positions()

        body_part = self.gym.find_asset_rigid_body_index(
            self.humanoid_asset, self.condition_body_part
        )
        head_position = bodies_positions[:, body_part, :]
        ground_below_head = self.get_ground_heights(bodies_positions[:, 0, :2])
        head_position[..., 2] -= ground_below_head.view(-1)

        if self.reset_path_ids is not None and len(self.reset_path_ids) > 0:
            self.path_generator.reset(
                self.reset_path_ids, head_position[self.reset_path_ids]
            )

            assert (
                self.progress_buf[self.reset_path_ids] <= 1
            ).all(), (
                f"Progress should be reset {self.progress_buf[self.reset_path_ids]}"
            )

            self.reset_path_ids = None

        traj_samples, traj_samples_p1 = self.fetch_path_samples(env_ids)

        dir_p_to_p_1 = traj_samples_p1[..., :2] - traj_samples[..., :2]
        dir_p_to_p_1_flat = dir_p_to_p_1.view(-1, 2)
        angle = rotations.vec_to_heading(dir_p_to_p_1_flat).view(
            dir_p_to_p_1_flat.shape[0], -1
        )
        neg = angle < 0
        angle[neg] += 2 * torch.pi
        direction = rotations.heading_to_quat(angle, w_last=self.w_last).view(
            dir_p_to_p_1.shape[0], -1, 4
        )

        body_index = self.config.masked_mimic_conditionable_bodies.index(
            self.condition_body_part
        )
        single_step_mask_size = self.num_conditionable_bodies * 2
        new_mask = torch.zeros(
            self.num_envs,
            self.num_conditionable_bodies,
            2,
            dtype=torch.bool,
            device=self.device,
        )
        new_mask[:, body_index, 0] = True
        # new_mask[:, -1, :] = True  # heading & speed
        new_mask = (
            new_mask.view(self.num_envs, 1, single_step_mask_size)
            .expand(-1, self.config.masked_mimic_obs.num_future_steps, -1)
            .reshape(self.num_envs, -1)
        )
        # new_mask = new_mask.view(self.num_envs, 1, single_step_mask_size).expand(-1, self.config.num_future_steps, -1).reshape(self.num_envs, -1)
        self.masked_mimic_target_bodies_masks[:] = new_mask
        # self.masked_mimic_target_bodies_masks[:] = new_mask

        self.target_pose_joints[:] = False
        self.target_pose_joints[:, body_index * 2] = True
        self.target_pose_joints[:, body_index * 2 + 1] = True
        # self.target_pose_joints[:, -2:] = True  # heading & speed
        self.target_pose_time[:] = self.motion_times + self._traj_sample_timestep

        target_poses = self.build_sparse_target_path_poses_masked_with_time(
            traj_samples, direction
        )
        self.masked_mimic_target_poses[:] = target_poses

        self.masked_mimic_target_poses_masks[:] = True
        # self.masked_mimic_target_poses_masks[:, 5] = True
        # # self.masked_mimic_target_poses_masks[:, 1] = True
        # self.masked_mimic_target_poses_masks[:, 2] = True
        # self.masked_mimic_target_poses_masks[:, -1] = True

        too_far = (
            torch.norm(traj_samples[:, 0, :2] - bodies_positions[:, 0, :2], dim=-1)
            > 0.4
        )
        self.masked_mimic_target_poses_masks[too_far, :-1] = False

        if self._use_text:
            self.motion_text_embeddings_mask[:] = True
            self.motion_text_embeddings[:] = self._text_embedding

    ###############################################################
    # Helpers
    ###############################################################
    def build_path_generator(self):
        episode_dur = self.max_episode_length * self.dt
        self.path_generator = PathGenerator(
            self.config.path_follower_params.path_generator,
            self.device,
            self.num_envs,
            episode_dur,
            self.config.path_follower_params.path_generator.height_conditioned,
        )

    def fetch_path_samples(self, env_ids=None, time_offset=10):
        # 5 seconds with 0.5 second intervals, 10 samples.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        timestep_beg = self.progress_buf[env_ids] * self.dt
        timesteps = torch.arange(
            self._num_traj_samples, device=self.device, dtype=torch.float
        )
        if time_offset == 0:
            timesteps = timesteps * self._traj_sample_timestep
        else:
            timesteps = (timesteps + 1) * self.dt
            timesteps[-1] += self._traj_sample_timestep
        timesteps_p1 = timesteps + self.dt
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps
        traj_timesteps_p1 = timestep_beg.unsqueeze(-1) + timesteps_p1

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self.path_generator.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps.flatten()
        ).clone()
        traj_samples_p1_flat = self.path_generator.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps_p1.flatten()
        ).clone()
        traj_samples = torch.reshape(
            traj_samples_flat,
            shape=(
                env_ids.shape[0],
                self._num_traj_samples,
                traj_samples_flat.shape[-1],
            ),
        )
        traj_samples_p1 = torch.reshape(
            traj_samples_p1_flat,
            shape=(
                env_ids.shape[0],
                self._num_traj_samples,
                traj_samples_p1_flat.shape[-1],
            ),
        )

        return traj_samples, traj_samples_p1

    def build_sparse_target_path_poses(
        self, raw_future_times, target_root_pos, target_root_rot
    ):
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

        # override to set the target root parameters
        reshaped_target_pos = flat_target_pos.reshape(
            self.num_envs, num_future_steps, -1, 3
        )
        reshaped_target_rot = flat_target_rot.reshape(
            self.num_envs, num_future_steps, -1, 4
        )

        body_part = self.gym.find_asset_rigid_body_index(
            self.humanoid_asset, self.condition_body_part
        )

        reshaped_target_pos[:, :, body_part, :3] = target_root_pos[..., :3]
        reshaped_target_rot[:, :, body_part] = target_root_rot[:]

        # reshaped_target_pos[:, :, body_part, -1] = 0.92  # standing up

        flat_target_pos = reshaped_target_pos.reshape(flat_target_pos.shape)
        flat_target_rot = reshaped_target_rot.reshape(flat_target_rot.shape)
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

    def build_sparse_target_path_poses_masked_with_time(
        self, target_root_pos: Tensor, target_root_rot: Tensor
    ):
        num_future_steps = target_root_pos.shape[1] - 1
        time_offsets = (
            torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
            * self.dt
        )

        near_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        all_future_times = torch.cat(
            [near_future_times, self.target_pose_time.view(-1, 1)], dim=1
        )

        obs = self.build_sparse_target_path_poses(
            all_future_times, target_root_pos, target_root_rot
        ).view(
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
