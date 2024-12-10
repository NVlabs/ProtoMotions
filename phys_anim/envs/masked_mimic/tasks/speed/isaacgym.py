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
from isaac_utils import rotations

import torch

import numpy as np

from phys_anim.envs.masked_mimic.tasks.base_task.isaacgym import (
    MaskedMimicTaskHumanoid,
)
from phys_anim.envs.masked_mimic.tasks.speed.common import BaseMaskedMimicSpeed

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2


class MaskedMimicSpeedHumanoid(BaseMaskedMimicSpeed, MaskedMimicTaskHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device, *args, **kwargs):
        super().__init__(config=config, device=device, *args, **kwargs)

        if not self.headless:
            self._build_marker_state_tensors()

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless:
            self._marker_handles = []
            self._face_marker_handles = []
            self._load_marker_asset()

        super().create_envs(num_envs, spacing, num_per_row)

    def _load_marker_asset(self):
        asset_root = "phys_anim/data/assets/urdf/"

        asset_file = "heading_marker.urdf"

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

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self.humanoid_root_states[..., 0:3]

    def _build_marker(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 2
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0
        default_pose.p.z = 0.0

        marker_handle = self.gym.create_actor(
            env_ptr,
            self._marker_asset,
            default_pose,
            "marker",
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.set_rigid_body_color(
            env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0)
        )
        self._marker_handles.append(marker_handle)

        face_marker_handle = self.gym.create_actor(
            env_ptr,
            self._marker_asset,
            default_pose,
            "face_marker",
            col_group,
            col_filter,
            segmentation_id,
        )
        self.gym.set_rigid_body_color(
            env_ptr,
            face_marker_handle,
            0,
            gymapi.MESH_VISUAL,
            gymapi.Vec3(0.0, 0.0, 0.8),
        )
        self._face_marker_handles.append(face_marker_handle)

    def _build_marker_state_tensors(self):
        num_actors = self.get_num_actors_per_env()
        if self.total_num_objects > 0:
            self._marker_states = self.root_states[: -self.total_num_objects].view(
                self.num_envs, num_actors, self.root_states.shape[-1]
            )[..., TAR_ACTOR_ID, :]
            self._face_marker_states = self.root_states[: -self.total_num_objects].view(
                self.num_envs, num_actors, self.root_states.shape[-1]
            )[..., TAR_FACING_ACTOR_ID, :]
        else:
            self._marker_states = self.root_states.view(
                self.num_envs, num_actors, self.root_states.shape[-1]
            )[..., TAR_ACTOR_ID, :]
            self._face_marker_states = self.root_states.view(
                self.num_envs, num_actors, self.root_states.shape[-1]
            )[..., TAR_FACING_ACTOR_ID, :]

        self._marker_pos = self._marker_states[..., :3]
        self._marker_rot = self._marker_states[..., 3:7]
        self._marker_actor_ids = self.humanoid_actor_ids + TAR_ACTOR_ID

        self._face_marker_pos = self._face_marker_states[..., :3]
        self._face_marker_rot = self._face_marker_states[..., 3:7]
        self._face_marker_actor_ids = self.humanoid_actor_ids + TAR_FACING_ACTOR_ID

    ###############################################################
    # Helpers
    ###############################################################
    def _update_marker(self):
        humanoid_root_pos = self.humanoid_root_states[..., :3].clone()
        self._marker_pos[..., 0:2] = humanoid_root_pos[..., :2] + self._tar_dir[:]

        height_below_marker = humanoid_root_pos[..., -1:]
        self._marker_pos[..., -1:] = height_below_marker

        heading_theta = torch.atan2(self._tar_dir[..., 1], self._tar_dir[..., 0])
        heading_axis = torch.zeros_like(self._marker_pos)
        heading_axis[..., -1] = 1.0
        heading_q = rotations.quat_from_angle_axis(
            heading_theta, heading_axis, w_last=self.w_last
        )
        self._marker_rot[:] = heading_q

        self._face_marker_pos[..., :2] = (
            humanoid_root_pos[..., :2] + self._tar_facing_dir[:]
        )
        self._face_marker_pos[..., -1:] = height_below_marker

        face_theta = torch.atan2(
            self._tar_facing_dir[..., 1], self._tar_facing_dir[..., 0]
        )
        face_axis = torch.zeros_like(self._marker_pos)
        face_axis[..., -1] = 1.0
        face_q = rotations.quat_from_angle_axis(
            face_theta, heading_axis, w_last=self.w_last
        )
        self._face_marker_rot[:] = face_q

        marker_ids = torch.cat(
            [self._marker_actor_ids, self._face_marker_actor_ids], dim=0
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(marker_ids),
            len(marker_ids),
        )

    def draw_task(self):
        self._update_marker()

        vel_scale = 0.2
        heading_cols = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        root_pos = self.humanoid_root_states[..., 0:3]
        prev_root_pos = self._prev_root_pos[:]
        sim_vel = (root_pos - prev_root_pos) / self.dt
        sim_vel[..., -1] = 0

        starts = root_pos
        tar_ends = torch.clone(starts)
        tar_ends[..., 0:2] += (
            vel_scale * self._tar_speed[:].unsqueeze(-1) * self._tar_dir[:]
        )
        sim_ends = starts + vel_scale * sim_vel

        verts = torch.cat([starts, tar_ends, starts, sim_ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i : i + 1]
            curr_verts = curr_verts.reshape([2, 6])
            self.gym.add_lines(
                self.viewer, env_ptr, curr_verts.shape[0], curr_verts, heading_cols
            )
