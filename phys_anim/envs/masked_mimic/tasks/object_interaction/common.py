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
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phys_anim.envs.masked_mimic.tasks.object_interaction.isaacgym import (
        MaskedMimicObjectHumanoid,
    )
else:
    MaskedMimicObjectHumanoid = object


class BaseMaskedMimicObject(MaskedMimicObjectHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config=config, device=device)

        self._tar_reach_time = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.float
        )
        self._use_text = True
        self._close_to_object = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._text_embedding = None
        if self._use_text:
            from transformers import AutoTokenizer, XCLIPTextModel

            model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

            text_command = ["a person walks casually"]
            with torch.inference_mode():
                inputs = tokenizer(
                    text_command, padding=True, truncation=True, return_tensors="pt"
                )
                outputs = model(**inputs)
                pooled_output = outputs.pooler_output  # pooled (EOS token) states
                self._text_embedding = pooled_output[0].to(self.device)

    def get_envs_respawn_position(
        self,
        env_ids,
        offset=0,
        rb_pos: Tensor = None,
        scene_ids: Tensor = None,
    ):
        respawn_position = super().get_envs_respawn_position(
            env_ids, rb_pos=rb_pos, offset=offset, scene_ids=scene_ids
        )

        if scene_ids is not None:
            scene_interaction_envs_mask = (
                scene_ids != -1
            )  # Object id -1 corresponds to no object
            if torch.any(scene_interaction_envs_mask):
                xy_position = self.object_target_position[
                    scene_ids[scene_interaction_envs_mask], :2
                ].clone()

                random_distance = (
                    torch.rand(respawn_position.shape[0], device=self.device) * 8 + 2
                )
                random_angle = (
                    torch.rand(respawn_position.shape[0], device=self.device)
                    * 2
                    * np.pi
                    - np.pi
                )
                random_offset = torch.stack(
                    [
                        random_distance * torch.cos(random_angle),
                        random_distance * torch.sin(random_angle),
                    ],
                    dim=-1,
                )

                xy_position += random_offset[scene_interaction_envs_mask]

                if rb_pos is not None:
                    normalized_dof_pos = rb_pos[scene_interaction_envs_mask].clone()
                    normalized_dof_pos[:, :, :2] -= rb_pos[
                        :, :1, :2
                    ]  # remove root position
                    normalized_dof_pos[:, :, :2] += xy_position.unsqueeze(
                        1
                    )  # add respawn offset
                    flat_normalized_dof_pos = normalized_dof_pos.view(-1, 3)
                    z_all_joints = self.get_heights_with_scene(flat_normalized_dof_pos)
                    z_all_joints = z_all_joints.view(normalized_dof_pos.shape[:-1])

                    z_diff = z_all_joints - normalized_dof_pos[:, :, 2]
                    z_indices = torch.max(z_diff, dim=1).indices.view(-1, 1)

                    # Extra offset is added to ensure the character is above the terrain.
                    # This is the minimal required distance of any joint to avoid collisions.
                    # If the character is above this height (e.g., jumping), do not add any offset.
                    min_joint_height = rb_pos[:, :, 2].min(dim=1).values.view(-1, 1)
                    extra_offset = (
                        self.config.ref_respawn_offset - min_joint_height
                    ).clamp(min=0)

                    z_offset = (
                        z_all_joints.gather(1, z_indices).view(-1, 1) + extra_offset
                    )
                else:
                    z_root = self.get_heights_with_scene(xy_position)
                    z_offset = z_root.view(-1, 1) + self.config.ref_respawn_offset

                object_respawn_position = torch.cat([xy_position, z_offset], dim=-1)

                self.respawn_offset_relative_to_data[env_ids, :2][
                    scene_interaction_envs_mask
                ] -= respawn_position[scene_interaction_envs_mask, :2]
                self.respawn_offset_relative_to_data[env_ids, :2][
                    scene_interaction_envs_mask
                ] += object_respawn_position[:, :2]

                respawn_position[scene_interaction_envs_mask] = object_respawn_position

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

        self._tar_reach_time[env_ids] = 0
        self._close_to_object[env_ids] = False
        self.motion_times[env_ids] = 0

    def compute_observations(self, env_ids=None):
        self.mask_everything()
        super().compute_observations(env_ids)
        self.mask_everything()

        target_positions = self.object_target_position[self.scene_ids]
        # compute 2d distance from object bounding box positions
        object_ids = self.scene_lib.scene_to_object_ids[self.scene_ids]

        assert (
            len(object_ids.shape) == 1 or object_ids.shape[1] == 1
        ), "This observation does not yet support multiple objects per scene."

        object_ids = object_ids.view(-1)
        objects_bounding_box = self.object_id_to_object_bounding_box(object_ids)
        root_pos = self.get_humanoid_root_states()[..., :3]
        root_pos2d = root_pos[..., :2]

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
        distance_to_object = torch.minimum(
            torch.norm(root_pos2d - target_positions[:, :2], dim=-1), distance_to_object
        )

        # far_from_object = distance_to_object > 3
        # close_to_object = ~far_from_object

        was_far_from_object = ~self._close_to_object.clone()

        close_to_object = distance_to_object < 2
        self._close_to_object[close_to_object] = True
        far_from_object = ~self._close_to_object
        close_to_object = self._close_to_object

        just_reached_range = was_far_from_object & close_to_object
        just_reached_range_env_ids = torch.nonzero(just_reached_range, as_tuple=True)[0]
        self.valid_hist_buf.set_all(False, just_reached_range_env_ids)

        # self.object_bounding_box_obs_mask[:] = True
        self.object_bounding_box_obs_mask[close_to_object] = True
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.object_bounding_box_obs[:] = self.get_object_bounding_box_obs(all_env_ids)

        # time_until_sitting = 3  # (self._tar_reach_steps - self.progress_buf).clamp(min=10)  # just incase keep above 0 to ensure it isn't hidden
        self._tar_reach_time[just_reached_range] = (
            self.motion_times[just_reached_range] + 5
        )
        self._tar_reach_time[:] = self._tar_reach_time.clamp(
            min=self.motion_times[:] + 2
        )

        # time_until_sitting = 6
        # time_to_object = distance_to_object / 1.5  # 1.5 m/s
        # self.target_pose_time[:] = self.motion_times + time_until_sitting
        self.target_pose_time[:] = self._tar_reach_time[:]
        # self.target_pose_obs_mask[:] = True
        target_body_index = self.config.masked_mimic_conditionable_bodies.index(
            "Pelvis"
        )
        self.target_pose_joints[:] = False
        self.target_pose_joints[:, target_body_index * 2] = True

        # nudge from here
        pelvis_body_index = self.config.masked_mimic_conditionable_bodies.index("Head")
        single_step_mask_size = self.num_conditionable_bodies * 2
        new_mask = torch.zeros(
            self.num_envs,
            self.num_conditionable_bodies,
            2,
            dtype=torch.bool,
            device=self.device,
        )
        # new_mask[:, pelvis_body_index, 1] = True
        # new_mask[self.motion_times > 1, pelvis_body_index, 0] = True
        new_mask[:, -1, :] = True  # velocity and heading
        new_mask = (
            new_mask.view(self.num_envs, 1, single_step_mask_size)
            .expand(-1, self.config.masked_mimic_obs.num_future_steps, -1)
            .reshape(self.num_envs, -1)
        )

        self.masked_mimic_target_bodies_masks[:] = new_mask

        dir_to_object = target_positions - root_pos
        angle = rotations.vec_to_heading(dir_to_object).view(self.num_envs, -1)
        neg = angle < 0
        angle[neg] += 2 * torch.pi
        direction = rotations.heading_to_quat(angle, w_last=self.w_last).view(
            self.num_envs, 4
        )

        self.masked_mimic_target_poses[:] = (
            self.build_sparse_target_object_poses_masked_with_time(
                self.config.masked_mimic_obs.num_future_steps, direction
            )
        )
        self.masked_mimic_target_poses_masks[:] = False
        self.masked_mimic_target_poses_masks[far_from_object, -2] = True
        # self.masked_mimic_target_poses_masks[far_from_object, 6] = True
        # self.masked_mimic_target_poses_masks[close_to_object, -1] = True

        self.motion_text_embeddings_mask[:] = False
        if self._use_text:
            self.motion_text_embeddings_mask[far_from_object] = True
            self.motion_text_embeddings[far_from_object] = self._text_embedding

    def compute_reward(self, actions):
        super().compute_reward(actions)

        root_pos = self.get_humanoid_root_states().clone()[..., :3]

        target_positions = self.object_target_position[self.scene_ids].clone()
        target_positions[
            ..., 2
        ] += 0.053  # Add half pelvis-size  # TODO: not hard coded

        # compute 2d distance from object bounding box positions
        # compute 2d distance from object bounding box positions
        object_ids = self.scene_lib.scene_to_object_ids[self.scene_ids]

        assert (
            len(object_ids.shape) == 1 or object_ids.shape[1] == 1
        ), "This observation does not yet support multiple objects per scene."

        object_ids = object_ids.view(-1)
        objects_bounding_box = self.object_id_to_object_bounding_box(object_ids)
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
        self.last_other_rewards["distance_to_object_position"] = (
            distance_to_target.detach()
        )
        self.last_other_rewards["success_object_position"] = (
            in_range.float().detach()
        )  # (distance_to_target < 0.4).float()

    def build_sparse_target_object_poses(self, raw_future_times, direction):
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
        target_positions = self.object_target_position[self.scene_ids]

        # For pelvis  --- # HARD CODED FOR THE 10th frame!
        target_positions[..., 2] = 0.95  # standing up  # cur_gt[:, 0, 2]
        vec_to_target = target_positions - cur_gt[:, 0, :3]
        normalized_vec_to_target = vec_to_target / torch.norm(
            vec_to_target[..., :2], dim=-1, keepdim=True
        )
        reshaped_target_pos[:, :, 0, :] = (
            cur_gt[:, 0, :3] + 1 * normalized_vec_to_target * 10.0 / 30
        ).unsqueeze(1)
        # we condition on 10 frames into the future. At 30 fps this conditions on 1.5 m/sec

        reshaped_target_pos[:, -1, 0, :] = target_positions
        # Done pelvis

        # For head
        target_positions_local_head = target_positions.clone()
        target_positions_local_head[..., 2] = 1  # standing up  # cur_gt[:, 0, 2]
        vec_to_target = target_positions_local_head.unsqueeze(1) - cur_gt[:, :, :3]
        normalized_vec_to_target = vec_to_target / torch.norm(
            vec_to_target[..., :2], dim=-1, keepdim=True
        )
        reshaped_target_pos[:, :, 1:, :] = (
            cur_gt[:, 1:, :3] + 1 * normalized_vec_to_target[:, 1:, :3] * 10.0 / 30
        ).unsqueeze(
            1
        )  # 10 frames/fps = 1.5 m/sec]
        reshaped_target_pos[:, -1, :, :] = target_positions_local_head.unsqueeze(1)
        # Done head

        # Override speed
        non_flat_target_vel = flat_target_vel.reshape(
            self.num_envs, num_future_steps, -1, 3
        )

        non_flat_target_vel[:, :, 0, :2] = (
            normalized_vec_to_target[:, 0, :2] * 1
        ).unsqueeze(1)
        flat_target_vel = non_flat_target_vel.reshape(flat_target_vel.shape)
        # Done speed

        flat_target_pos = reshaped_target_pos.reshape(flat_target_pos.shape)

        reshaped_target_rot = flat_target_rot.reshape(
            self.num_envs, num_future_steps, -1, 4
        )
        reshaped_target_rot[:, :, :, :] = direction.unsqueeze(1).unsqueeze(1)
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

    def build_sparse_target_object_poses_masked_with_time(
        self, num_future_steps, direction
    ):
        time_offsets = (
            torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
            * self.dt
        )

        near_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        all_future_times = torch.cat(
            [near_future_times, self.target_pose_time.view(-1, 1)], dim=1
        )

        # +1 for "far future step"
        obs = self.build_sparse_target_object_poses(all_future_times, direction).view(
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
