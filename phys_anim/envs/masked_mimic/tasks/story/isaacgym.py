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

from isaacgym import gymapi, gymtorch  # type: ignore[misc]
from isaac_utils import torch_utils

import torch

from phys_anim.envs.masked_mimic.tasks.base_task.isaacgym import (
    MaskedMimicTaskHumanoid,
)
from phys_anim.envs.masked_mimic.tasks.story.common import BaseMaskedMimicStory


class MaskedMimicStoryHumanoid(BaseMaskedMimicStory, MaskedMimicTaskHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config=config, device=device)

        if not self.headless:
            self._build_marker_state_tensors()

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless:
            self._marker_handles = [[] for _ in range(num_envs)]
            self._load_marker_asset()

        super().create_envs(num_envs, spacing, num_per_row)

    def _load_marker_asset(self):
        asset_root = "phys_anim/data/assets/urdf/"
        asset_file = "traj_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )

    def build_env(self, env_id, env_ptr, humanoid_asset):
        super().build_env(env_id, env_ptr, humanoid_asset)

        if not self.headless:
            self._build_marker(env_id, env_ptr)

    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()

        num_markers_per_env = 9

        for i in range(num_markers_per_env):
            marker_handle = self.gym.create_actor(
                env_ptr,
                self._marker_asset,
                default_pose,
                "marker",
                self.num_envs + 10,
                0,
                0,
            )
            color = gymapi.Vec3(0.0, 0.0, 0.8)
            self.gym.set_rigid_body_color(
                env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color
            )
            self._marker_handles[env_id].append(marker_handle)

    def _build_marker_state_tensors(self):
        num_markers_per_env = 9

        num_actors = self.get_num_actors_per_env()
        if self.total_num_objects > 0:
            self._marker_states = self.root_states[: -self.total_num_objects].view(
                self.num_envs, num_actors, self.root_states.shape[-1]
            )[..., 1 : (1 + num_markers_per_env), :]
        else:
            self._marker_states = self.root_states.view(
                self.num_envs, num_actors, self.root_states.shape[-1]
            )[..., 1 : (1 + num_markers_per_env), :]

        self._marker_pos = self._marker_states[..., :3]

        self._marker_actor_ids = self.humanoid_actor_ids.unsqueeze(
            -1
        ) + torch_utils.to_torch(
            self._marker_handles, dtype=torch.int32, device=self.device
        )
        self._marker_actor_ids = self._marker_actor_ids.flatten()

    ###############################################################
    # Helpers
    ###############################################################
    def _update_marker(self):
        current_root = self.get_humanoid_root_states().clone()[..., :3]
        current_root[..., -1:] -= self.get_ground_heights(current_root[..., :2]).view(
            self.num_envs, 1
        )

        vec_to_target = self._story_markers[:, 0] - current_root
        normalized_vec_to_target = vec_to_target[...] / torch.norm(
            vec_to_target[..., :2], dim=-1, keepdim=True
        )

        proposed_target_positions = (
            current_root[:, :3] + 1.0 * normalized_vec_to_target * 10.0 / 30
        )  # 10 frames/fps = 1 m/sec
        proposed_target_positions[..., 2] = 1.5

        dist_to_target = torch.norm(
            self._story_markers[:, 0, :2] - current_root[:, :2], dim=-1
        ).view(-1)
        dist_from_proposed = torch.norm(
            self._story_markers[:, 0, :2] - proposed_target_positions[..., :2], dim=-1
        ).view(-1)

        closer_than_proposed = dist_to_target < dist_from_proposed

        if any(closer_than_proposed):
            proposed_target_positions[closer_than_proposed, :2] = self._story_markers[
                :, 0, :2
            ][closer_than_proposed]
        if any(~closer_than_proposed):
            proposed_target_positions[~closer_than_proposed] = (
                proposed_target_positions[~closer_than_proposed]
            )

        ground_below_marker = self.get_ground_heights(proposed_target_positions).view(
            proposed_target_positions.shape[0]
        )

        proposed_target_positions[..., 2] += ground_below_marker

        self._marker_pos[:, 0] = proposed_target_positions.view(-1, 3)

        self._marker_pos[self._fsm_state == 0, :] = 1000
        self._marker_pos[self._fsm_state == 1, 1:] = 1000
        self._marker_pos[self._fsm_state == 2, 1:] = 1000
        self._marker_pos[self._fsm_state == 3, 1:] = 1000
        self._marker_pos[self._fsm_state == 4, 1:] = 1000
        self._marker_pos[self._fsm_state == 5, 0] = 1000

        object_ids = self.scene_lib.scene_to_object_ids[self.scene_ids]

        objects_bounding_box = self.object_id_to_object_bounding_box[object_ids].clone()

        object_root_states = self.object_root_states[object_ids].clone()
        height_below_object = self.get_ground_heights(object_root_states[..., :2]).view(
            -1, 1
        )
        objects_bounding_box[..., 2] += height_below_object

        self._marker_pos[self._fsm_state == 5, 1:] = objects_bounding_box[
            self._fsm_state == 5
        ]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )

    def draw_task(self):
        self._update_marker()
