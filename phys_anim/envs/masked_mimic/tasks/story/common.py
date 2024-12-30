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
    from phys_anim.envs.masked_mimic.tasks.story.isaacgym import (
        MaskedMimicStoryHumanoid,
    )
else:
    MaskedMimicStoryHumanoid = object


class BaseMaskedMimicStory(MaskedMimicStoryHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        self._fsm_state = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._last_fsm_switch_time = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.long
        )

        from transformers import AutoTokenizer, XCLIPTextModel

        model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

        self._story_markers = torch.zeros(
            [self.num_envs, 8, 3], device=self.device, dtype=torch.float32
        )  # 8 for the bounding box

        self._is_king = True

        if self._is_king:
            self.condition_body_part = "Head"
            # Start with hml3d_id: 741
            text_commands = [
                # "a figure crosses their arms at their chest",
                "a person sits on the floor then stands back up",
                "a person walks casually",
                # "a person waves hello, then come here, then hello again",
                "a person gesturing with their right hand similar to waving a greeting",
                "a person walks casually",
                "a person walks casually",
                "the person sits down in the chair",
            ]
        else:
            self.condition_body_part = "Head"  # "Pelvis"
            text_commands = [
                "a man raises his right hand to his head as in a salute",
                "a man raises his right hand to his head as in a salute",
            ]
        with torch.inference_mode():
            inputs = tokenizer(
                text_commands, padding=True, truncation=True, return_tensors="pt"
            )
            outputs = model(**inputs)
            pooled_output = outputs.pooler_output  # pooled (EOS token) states
            self._text_embedding = pooled_output.to(self.device)
            assert self._text_embedding.shape[0] == len(text_commands)

    def get_envs_respawn_position(
        self,
        env_ids,
        offset=0,
        rb_pos: torch.tensor = None,
        scene_ids: torch.tensor = None,
    ):
        xy_position = (
            torch.tensor([[40.0, 110.0]], device=self.device, dtype=torch.float32)
            .view(1, 2)
            .expand(len(env_ids), -1)
        )  # starting position
        # xy_position = torch.tensor([[150., 90.]], device=self.device, dtype=torch.float32).view(1, 2).expand(len(env_ids), -1)
        # xy_position = torch.tensor([[150., 60.]], device=self.device, dtype=torch.float32).view(1, 2).expand(len(env_ids), -1)

        if rb_pos is not None:
            normalized_dof_pos = rb_pos.clone()
            normalized_dof_pos[:, :, :2] -= rb_pos[:, :1, :2]  # remove root position
            normalized_dof_pos[:, :, :2] += xy_position.unsqueeze(
                1
            )  # add respawn offset
            flat_normalized_dof_pos = normalized_dof_pos.view(-1, 3)
            z_all_joints = self.terrain_obs_cb.get_heights_with_scene(
                flat_normalized_dof_pos
            )
            z_all_joints = z_all_joints.view(normalized_dof_pos.shape[:-1])

            z_diff = z_all_joints - normalized_dof_pos[:, :, 2]
            z_indices = torch.max(z_diff, dim=1).indices.view(-1, 1)

            # Extra offset is added to ensure the character is above the terrain.
            # This is the minimal required distance of any joint to avoid collisions.
            # If the character is above this height (e.g., jumping), do not add any offset.
            min_joint_height = rb_pos[:, :, 2].min(dim=1).values.view(-1, 1)
            extra_offset = (self.config.ref_respawn_offset - min_joint_height).clamp(
                min=0
            )

            z_offset = z_all_joints.gather(1, z_indices).view(-1, 1) + extra_offset
        else:
            z_root = self.terrain_obs_cb.get_heights_with_scene(xy_position)
            z_offset = z_root.view(-1, 1) + self.config.ref_respawn_offset

        respawn_position = torch.cat([xy_position, z_offset], dim=-1)

        return respawn_position

    def sample_scene_ids(self, motion_ids):
        sampled_scene_id_per_env = torch.randint(
            low=0,
            high=len(self.scene_position),
            size=(len(motion_ids),),
            device=self.device,
        ).view(motion_ids.shape)

        return (
            sampled_scene_id_per_env,
            torch.ones(
                sampled_scene_id_per_env.shape[0], dtype=torch.bool, device=self.device
            )
            * True,
        )

    def reset_task(self, env_ids):
        super().reset_task(env_ids)
        self._fsm_state[env_ids] = 0
        self._last_fsm_switch_time[env_ids] = self.progress_buf[env_ids]

    def store_motion_data(self, skip=False):
        super().store_motion_data(skip=True)
        if skip:
            return

        if "target_poses" not in self.motion_recording:
            self.motion_recording["target_poses"] = []

        self.motion_recording["target_poses"].append(self._marker_pos[:].cpu().numpy())

    def compute_observations(self, env_ids=None):
        self.mask_everything()
        super().compute_observations(env_ids)
        self.mask_everything()

        print(f"Currently FSM at stage: {self._fsm_state[0]}")

        # Compute goals based on FSM state
        # 0: stretch for 3 seconds
        # 1: walk to top of the hill [150., 90.]
        # 2: wave for 3 seconds
        # 3: walk to bottom of the hill [150., 60.]
        # 4: walk into the castle towards the chair
        # 5: sit on the chair

        root_pos = self.get_humanoid_root_states()[..., :3]

        root_pos[..., -1:] -= self.terrain_obs_cb.ground_heights.view(self.num_envs, 1)
        root_pos2d = root_pos[..., :2]

        self.motion_text_embeddings_mask[self._fsm_state < 5] = True
        self.motion_text_embeddings[:] = self._text_embedding[self._fsm_state]

        target_positions = self.object_target_position[self.scene_ids].clone()
        target_positions_for_rotation = self.object_target_position[
            self.scene_ids
        ].clone()

        if self._is_king:
            # If sufficient time passed at stage 0 -> move to stage 1
            time_passed = (self._last_fsm_switch_time[:] + 30 * 6) < self.progress_buf[
                :
            ]  # 3 seconds
            swap_to_stage_1 = time_passed & (self._fsm_state == 0)
            self._fsm_state[swap_to_stage_1] = 1
            self._last_fsm_switch_time[swap_to_stage_1] = self.progress_buf[
                swap_to_stage_1
            ]

            move_to_top = self._fsm_state == 1
            target_positions[move_to_top] = torch.tensor(
                [[50.0, 90.0, 1.5]], device=self.device, dtype=torch.float32
            ).view(-1, 3)
            target_positions_for_rotation[move_to_top] = torch.tensor(
                [[50.0, 90.0, 1.5]], device=self.device, dtype=torch.float32
            ).view(-1, 3)

            # If reached target location -> move to stage 2
            reached_target = (
                torch.norm(root_pos2d - target_positions[:, :2], dim=-1) < 0.5
            )
            swap_to_stage_2 = reached_target & (self._fsm_state == 1)
            self._fsm_state[swap_to_stage_2] = 2
            self._last_fsm_switch_time[swap_to_stage_2] = self.progress_buf[
                swap_to_stage_2
            ]

            target_positions[self._fsm_state == 2] = torch.tensor(
                [[50.0, 90.0, 1.5]], device=self.device, dtype=torch.float32
            ).view(-1, 3)

            # If sufficient time passed at stage 2 -> move to stage 3
            time_passed = (self._last_fsm_switch_time[:] + 30 * 6) < self.progress_buf[
                :
            ]  # 3 seconds
            swap_to_stage_3 = time_passed & (self._fsm_state == 2)
            self._fsm_state[swap_to_stage_3] = 3
            self._last_fsm_switch_time[swap_to_stage_3] = self.progress_buf[
                swap_to_stage_3
            ]

            move_to_bottom = self._fsm_state == 3
            target_positions[move_to_bottom] = torch.tensor(
                [[50.0, 60.0, 1.5]], device=self.device, dtype=torch.float32
            ).view(-1, 3)
            target_positions_for_rotation[move_to_bottom] = torch.tensor(
                [[50.0, 60.0, 1.5]], device=self.device, dtype=torch.float32
            ).view(-1, 3)

            # If reached target location -> move to stage 4
            reached_target = (
                torch.norm(root_pos2d - target_positions[:, :2], dim=-1) < 2
            )
            swap_to_stage_4 = reached_target & (self._fsm_state == 3)
            self._fsm_state[swap_to_stage_4] = 4
            self._last_fsm_switch_time[swap_to_stage_4] = self.progress_buf[
                swap_to_stage_4
            ]

            # compute 2d distance from object bounding box positions
            objects_bounding_box = self.object_id_to_object_bounding_box[self.scene_ids]

            # compute distance from each corner of the bounding box (only 2d distance)
            distance_to_object = torch.norm(
                root_pos2d - objects_bounding_box[:, 0, :2], dim=-1
            )
            distance_to_object = torch.minimum(
                torch.norm(root_pos2d - objects_bounding_box[:, 1, :2], dim=-1),
                distance_to_object,
            )
            distance_to_object = torch.minimum(
                torch.norm(root_pos2d - objects_bounding_box[:, 2, :2], dim=-1),
                distance_to_object,
            )
            distance_to_object = torch.minimum(
                torch.norm(root_pos2d - objects_bounding_box[:, 3, :2], dim=-1),
                distance_to_object,
            )
            # distance_to_object = torch.minimum(torch.norm(root_pos2d - target_positions[:, :2], dim=-1), distance_to_object)

            # Once within range, swap to "sit on"
            move_to_object = self._fsm_state == 4
            close_to_object = distance_to_object < 3
            sit_on_object = move_to_object & close_to_object

            self._fsm_state[sit_on_object] = 5

            self.object_bounding_box_obs_mask[self._fsm_state == 5] = True
            all_env_ids = torch.arange(self.num_envs, device=self.device)
            self.object_bounding_box_obs[:] = self.get_object_bounding_box_obs(
                all_env_ids
            )
        else:  # if not self._is_king:
            stop_condition_pose = (
                self._last_fsm_switch_time[:] + 30 * 6
            ) < self.progress_buf[:]
            self._fsm_state[:] = 1
            self._fsm_state[stop_condition_pose] = 0

            self.motion_text_embeddings_mask[self._fsm_state == 0] = True
            self.motion_text_embeddings_mask[self._fsm_state != 0] = False

            target_positions[:, :2] = torch.tensor(
                [[40.0, 110.0]], device=self.device, dtype=torch.float32
            ).view(1, 2)
            target_positions_for_rotation[:, :2] = torch.tensor(
                [[40.0, 120.0]], device=self.device, dtype=torch.float32
            ).view(1, 2)

        dir_to_target = target_positions_for_rotation - root_pos
        angle = rotations.vec_to_heading(dir_to_target).view(self.num_envs, -1)
        neg = angle < 0
        angle[neg] += 2 * torch.pi
        target_directions = rotations.heading_to_quat(angle, w_last=self.w_last).view(
            self.num_envs, 4
        )

        body_index = self.config.masked_mimic_conditionable_bodies.index(
            self.condition_body_part
        )
        single_step_mask_size = self.masked_mimic_conditionable_bodies_ids.shape[0] * 2
        new_mask = torch.zeros(
            self.num_envs,
            self.masked_mimic_conditionable_bodies_ids.shape[0],
            2,
            dtype=torch.bool,
            device=self.device,
        )

        new_mask[:, body_index, :] = True
        new_mask_pos = torch.zeros(
            self.num_envs,
            self.masked_mimic_conditionable_bodies_ids.shape[0],
            2,
            dtype=torch.bool,
            device=self.device,
        )
        new_mask_pos[:, body_index, 0] = True

        self.masked_mimic_target_bodies_masks[:] = False
        self.masked_mimic_target_bodies_masks[:, -single_step_mask_size:] = (
            new_mask.view(self.num_envs, -1)
        )
        # self.masked_mimic_target_bodies_masks[self._fsm_state == 2, -single_step_mask_size:] = new_mask_pos.view(self.num_envs, -1)[self._fsm_state == 2, :]

        self._story_markers[:, 0] = target_positions

        self.masked_mimic_target_poses[:] = (
            self.build_sparse_target_object_poses_masked_with_time(
                self.config.masked_mimic_obs.num_future_steps,
                target_positions,
                target_directions,
            )
        )
        self.masked_mimic_target_poses_masks[:] = False
        self.masked_mimic_target_poses_masks[self._fsm_state == 1, -2] = True
        self.masked_mimic_target_poses_masks[self._fsm_state == 2, -2] = True
        self.masked_mimic_target_poses_masks[self._fsm_state == 3, -2] = True
        self.masked_mimic_target_poses_masks[self._fsm_state == 4, -2] = True

    def compute_reward(self, actions):
        super().compute_reward(actions)

        root_pos = self.get_humanoid_root_states().clone()[..., :3]

        target_positions = self.object_target_position[self.scene_ids].clone()
        target_positions[
            ..., 2
        ] += 0.053  # Add half pelvis-size  # TODO: not hard coded

        # compute 2d distance from object bounding box positions
        objects_bounding_box = self.object_id_to_object_bounding_box[self.scene_ids]
        min_range = objects_bounding_box[:, 0, :2]
        max_range = objects_bounding_box[:, 6, :2]

        in_range = (
            (root_pos[..., 0] > (min_range[:, 0] - 0.1))
            & (root_pos[..., 1] > (min_range[:, 1] - 0.1))
            & (root_pos[..., 0] < (max_range[:, 0] + 0.1))
            & (root_pos[..., 1] < (max_range[:, 1] + 0.1))
            & (root_pos[..., 2] < (target_positions[..., 2] + 0.2))
            & (root_pos[..., 2] > (target_positions[..., 2] - 0.2 - 2 * 0.053))
        )

        target_positions[in_range, :2] = root_pos[in_range, :2]

        distance_to_target = torch.norm(root_pos - target_positions, dim=-1)
        self.mimic_info_dict["distance_to_object_position"] = distance_to_target
        self.mimic_info_dict["success_object_position"] = (
            in_range.float()
        )  # (distance_to_target < 0.4).float()

    def build_sparse_target_object_poses(
        self, raw_future_times, target_positions, target_directions
    ):
        """
        This is identical to the max_coords humanoid observation, only in relative to the current pose.
        """
        if self.condition_body_part == "Head":
            target_height = 1.5
        elif self.condition_body_part == "Pelvis":
            target_height = 0.9
        else:
            raise NotImplementedError

        num_future_steps = raw_future_times.shape[1]

        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        lengths = self.motion_lib.get_motion_length(flat_ids)

        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        ref_state = self.motion_lib.get_motion_state(flat_ids, flat_times)
        flat_target_pos, flat_target_rot, flat_target_vel = (
            ref_state.rb_pos,
            ref_state.rb_rot,
            ref_state.rb_vel,
        )

        current_state = self.get_bodies_state()
        cur_gt, cur_gr = current_state.body_pos, current_state.body_rot
        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        cur_gt[:, :, -1:] -= self.terrain_obs_cb.ground_heights.view(
            self.num_envs, 1, 1
        )
        # cur_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[..., :2].view(self.num_envs, 1, 2)

        # override to set the target root parameters
        body_part = self.gym.find_asset_rigid_body_index(
            self.humanoid_asset, self.condition_body_part
        )

        reshaped_target_pos = flat_target_pos.reshape(
            self.num_envs, num_future_steps, -1, 3
        )
        target_positions[..., 2] = target_height  # standing up  # cur_gt[:, 0, 2]
        vec_to_target = target_positions - cur_gt[:, body_part, :3]
        normalized_vec_to_target = vec_to_target / torch.norm(
            vec_to_target[..., :2], dim=-1, keepdim=True
        )

        proposed_target_positions_local = (
            cur_gt[:, body_part, :3] + 1.0 * normalized_vec_to_target * 10.0 / 30
        )  # 10 frames/fps = 1 m/sec
        proposed_target_positions_local[..., 2] = (
            target_height  # standing up  # cur_gt[:, 0, 2
        )

        dist_to_target = torch.norm(
            target_positions[..., :2] - cur_gt[:, body_part, :2], dim=-1
        ).view(-1)
        dist_from_proposed = torch.norm(
            target_positions[..., :2] - proposed_target_positions_local[..., :2],
            dim=-1,
        ).view(-1)

        closer_than_proposed = dist_to_target < dist_from_proposed

        if any(closer_than_proposed):
            reshaped_target_pos[closer_than_proposed, :, body_part, :] = (
                target_positions[closer_than_proposed].unsqueeze(1)
            )
        if any(~closer_than_proposed):
            reshaped_target_pos[~closer_than_proposed, :, body_part, :] = (
                proposed_target_positions_local[~closer_than_proposed].unsqueeze(1)
            )

        flat_target_pos = reshaped_target_pos.reshape(flat_target_pos.shape)

        reshaped_target_rot = flat_target_rot.reshape(
            self.num_envs, num_future_steps, -1, 4
        )
        reshaped_target_rot[:, :, body_part, :] = target_directions.unsqueeze(1)
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
                rotations.quat_conjugate(heading_rot_expand[:, 0, :], self.w_last),
                target_heading_rot,
                self.w_last,
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

    def build_sparse_target_object_poses_masked_with_time(
        self, num_future_steps, target_positions, target_directions
    ):
        time_offsets = (
            torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
            * self.dt
        )

        near_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        all_future_times = torch.cat(
            [near_future_times, self.target_pose_time.view(-1, 1)], dim=1
        )

        obs = self.build_sparse_target_object_poses(
            all_future_times, target_positions, target_directions
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
