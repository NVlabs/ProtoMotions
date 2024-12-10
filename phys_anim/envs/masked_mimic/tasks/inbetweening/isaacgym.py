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

from isaacgym import gymtorch

import torch

from phys_anim.envs.masked_mimic.tasks.inbetweening.common import (
    BaseMaskedMimicInbetweening,
)
from phys_anim.envs.masked_mimic.isaacgym import MaskedMimicHumanoid


class MaskedMimicInbetweeningHumanoid(BaseMaskedMimicInbetweening, MaskedMimicHumanoid):  # type: ignore[misc]
    def __init__(self, config, device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

    def set_env_state(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        rb_pos,
        rb_rot,
        rb_vel,
        rb_ang_vel,
    ):
        if self.start_pose is not None:
            global_rb_pos = rb_pos.clone()

            flat_global_rb_pos = global_rb_pos.view(-1, 3)
            z_all_joints = self.terrain_obs_cb.get_ground_heights(flat_global_rb_pos)
            z_all_joints = z_all_joints.view(global_rb_pos.shape[:-1])

            z_diff = z_all_joints - global_rb_pos[:, :, 2]
            z_indices = torch.max(z_diff, dim=1).indices.view(-1, 1)

            # We want to add the offset based on the ground terrain-height below the joint.
            # Unlike the diff. The reason is that while jumping, we want to ensure the character retains
            # the relative height above the terrain.
            z_offset = z_all_joints.gather(1, z_indices).view(-1, 1)

            z_all_joints_with_objects = self.terrain_obs_cb.get_heights_with_scene(
                flat_global_rb_pos
            )
            z_all_joints_with_objects = z_all_joints_with_objects.view(
                global_rb_pos.shape[:-1]
            )
            # Check if after added offset, if any joint is BELOW the object (+ respawn offset). If yes, shift up.
            # Otherwise, don't change the height.
            z_diff_with_objects = (
                z_all_joints_with_objects
                + self.config.ref_respawn_offset
                - (global_rb_pos[:, :, 2] + z_offset)
            )
            z_with_objects_offset = torch.max(z_diff_with_objects, dim=1).values.view(
                -1, 1
            )
            z_with_objects_offset = torch.clamp(z_with_objects_offset, min=0)

            z_offset = z_offset + z_with_objects_offset

            root_pos[:, 2] += z_offset.view(-1)
            rb_pos[:, :, 2] += z_offset.view(-1, 1)

            new_translation = self.start_pose["translation"].view(1, -1, 3).clone()
            new_rotation = self.start_pose["rotation"].view(1, -1, 4).clone()
            new_dof_pos = self.start_pose["dof_pos"].view(1, -1).clone()

            root_pos[:] = new_translation[:, 0]
            rb_pos[:] = new_translation
            root_rot[:] = new_rotation[:, 0]
            rb_rot[:] = new_rotation
            dof_pos[:] = new_dof_pos

            root_vel[:] = 0
            root_ang_vel[:] = 0
            rb_vel[:] = 0
            rb_ang_vel[:] = 0
            dof_vel[:] = 0

        self.humanoid_root_states[env_ids, 0:3] = root_pos
        self.humanoid_root_states[env_ids, 3:7] = root_rot
        self.humanoid_root_states[env_ids, 7:10] = root_vel
        self.humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self.rigid_body_pos[env_ids] = rb_pos
        self.rigid_body_rot[env_ids] = rb_rot
        self.rigid_body_vel[env_ids] = rb_vel
        self.rigid_body_ang_vel[env_ids] = rb_ang_vel

        self.reset_states = {
            "root_pos": root_pos.clone(),
            "root_rot": root_rot.clone(),
            "root_vel": root_vel.clone(),
            "root_ang_vel": root_ang_vel.clone(),
            "dof_pos": dof_pos.clone(),
            "dof_vel": dof_vel.clone(),
            "rb_pos": rb_pos.clone(),
            "rb_rot": rb_rot.clone(),
            "rb_vel": rb_vel.clone(),
            "rb_ang_vel": rb_ang_vel.clone(),
        }

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

    def _update_marker(self):
        # Standard future poses
        ref_state = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.motion_times
        )
        target_pos = ref_state.rb_pos
        target_pos += self.respawn_offset_relative_to_data.clone().view(
            self.num_envs, 1, 3
        )

        target_pos[..., -1:] += self.terrain_obs_cb.get_ground_heights(
            target_pos[:, 0, :2]
        ).view(self.num_envs, 1, 1)

        target_pos = target_pos[:, self.masked_mimic_conditionable_bodies_ids, :]

        inactive_markers = torch.ones(
            self.num_envs,
            len(self.config.masked_mimic_conditionable_bodies),
            dtype=torch.bool,
            device=self.device,
        )

        if self.config.masked_mimic_masking.joint_masking.masked_mimic_time_mask:
            mask_time_len = self.config.masked_mimic_obs.num_future_steps
        else:
            mask_time_len = 1

        translation_view = self.masked_mimic_target_bodies_masks.view(
            self.num_envs, mask_time_len, self.num_conditionable_bodies, 2
        )[
            :, 0, :-1, 0
        ]  # ignore the last entry, that is for speed/heading
        active_translations = translation_view == 1

        inactive_markers[active_translations] = False

        target_pos[inactive_markers] += 100

        self._marker_pos[:, : self.masked_mimic_conditionable_bodies_ids.shape[0]] = (
            target_pos
        )

        # Inbetweening target pose
        ref_state = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.target_pose_time
        )
        target_pos = ref_state.rb_pos

        first_pose = self.fsm_state == 1
        if torch.any(first_pose) and self.start_pose is not None:
            target_pos[first_pose] = (
                self.start_pose["translation"].view(1, 1, -1, 3).clone()
            )
        second_pose = self.fsm_state == 2
        if torch.any(second_pose) and self.end_pose is not None:
            target_pos[second_pose] = (
                self.end_pose["translation"].view(1, 1, -1, 3).clone()
            )

        target_pos[..., -1:] += self.terrain_obs_cb.get_ground_heights(
            target_pos[:, 0, :2]
        ).view(self.num_envs, 1, 1)

        target_pos = target_pos[:, self.masked_mimic_conditionable_bodies_ids, :]

        translation_view = self.target_pose_joints.view(
            self.num_envs, self.num_conditionable_bodies, 2
        )[
            :, :-1, 0
        ]  # ignore the last entry, that is for speed/heading
        active_translations = translation_view == 1

        inactive_markers[active_translations] = False

        target_pos[inactive_markers] += 100

        target_pos[torch.logical_not(self.target_pose_obs_mask.view(-1))] += 100
        self._marker_pos[
            :,
            self.masked_mimic_conditionable_bodies_ids.shape[
                0
            ] : self.masked_mimic_conditionable_bodies_ids.shape[0]
            * 2,
        ] = target_pos

        # Terrain
        if self.terrain is not None:
            height_maps = self.terrain_obs_cb.get_height_maps(
                None, None, return_all_dims=True
            )
            height_maps = height_maps.view(self.num_envs, -1, 3)
            self._marker_pos[
                :, self.masked_mimic_conditionable_bodies_ids.shape[0] * 2 :
            ] = height_maps

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )
