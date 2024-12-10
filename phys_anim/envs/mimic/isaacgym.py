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
import torch

from isaac_utils import torch_utils, rotations
from phys_anim.envs.mimic.common import BaseMimic
from phys_anim.envs.humanoid.isaacgym import Humanoid


class MimicHumanoid(BaseMimic, Humanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)

        if not self.headless and self.config.visualize_markers:
            self._build_marker_state_tensors()

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def create_envs(self, num_envs, spacing, num_per_row):
        if not self.headless and self.config.visualize_markers:
            self._marker_handles = [[] for _ in range(num_envs)]
            self._load_marker_asset()

        super().create_envs(num_envs, spacing, num_per_row)

    def _load_marker_asset(self):
        asset_root = "phys_anim/data/assets/urdf/"
        asset_file = "traj_marker.urdf"
        small_asset_file = "traj_marker_small.urdf"
        tiny_asset_file = "traj_marker_tiny.urdf"

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
        self._marker_asset_small = self.gym.load_asset(
            self.sim, asset_root, small_asset_file, asset_options
        )
        self._marker_asset_tiny = self.gym.load_asset(
            self.sim, asset_root, tiny_asset_file, asset_options
        )

    def build_env(self, env_id, env_ptr, humanoid_asset):
        super().build_env(env_id, env_ptr, humanoid_asset)

        if not self.headless and self.config.visualize_markers:
            self._build_marker(env_id, env_ptr)

    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()

        num_mimic_tracking_points = self.num_bodies

        # Mimic markers
        for i in range(num_mimic_tracking_points):
            if (
                self.config.robot.mimic_small_marker_bodies is not None
                and self.body_names[i] in self.config.robot.mimic_small_marker_bodies
            ):
                marker_handle = self.gym.create_actor(
                    env_ptr,
                    self._marker_asset_small,
                    default_pose,
                    "marker",
                    self.num_envs + 10,
                    0,
                    0,
                )
            else:
                marker_handle = self.gym.create_actor(
                    env_ptr,
                    self._marker_asset,
                    default_pose,
                    "marker",
                    self.num_envs + 10,
                    0,
                    0,
                )
            color = gymapi.Vec3(0.8, 0.0, 0.0)
            self.gym.set_rigid_body_color(
                env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color
            )
            self._marker_handles[env_id].append(marker_handle)

        # Terrain markers
        for i in range(self.terrain_obs_cb.num_height_points):
            marker_handle = self.gym.create_actor(
                env_ptr,
                self._marker_asset_small,
                default_pose,
                "marker",
                self.num_envs + 10,
                0,
                0,
            )
            color = gymapi.Vec3(0.0, 0.8, 0.0)
            self.gym.set_rigid_body_color(
                env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color
            )
            self._marker_handles[env_id].append(marker_handle)

        # Point cloud markers
        if self.scene_lib is not None and self.config.point_cloud_obs.enabled:
            num_pointcloud_markers = (
                self.config.point_cloud_obs.num_pointcloud_samples
                * self.max_objects_per_scene
            )

            for i in range(num_pointcloud_markers):
                marker_handle = self.gym.create_actor(
                    env_ptr,
                    self._marker_asset_tiny,
                    default_pose,
                    "pointcloud_marker",
                    self.num_envs + 10,
                    0,
                    0,
                )
                object_number = (
                    i // self.config.point_cloud_obs.num_pointcloud_samples
                ) % self.max_objects_per_scene
                color_interpolation = (
                    object_number * 1.0 / max((self.max_objects_per_scene - 1), 1)
                )
                lightblue = [0.3, 0.7, 0.9]  # RGB values for a more vibrant light blue
                light_purple = [
                    0.6,
                    0.4,
                    0.7,
                ]  # RGB values for a more distinct light purple
                color = gymapi.Vec3(
                    lightblue[0] * (1 - color_interpolation)
                    + light_purple[0] * color_interpolation,
                    lightblue[1] * (1 - color_interpolation)
                    + light_purple[1] * color_interpolation,
                    lightblue[2] * (1 - color_interpolation)
                    + light_purple[2] * color_interpolation,
                )
                self.gym.set_rigid_body_color(
                    env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color
                )
                self._marker_handles[env_id].append(marker_handle)

            num_contact_markers = len(self.config.robot.contact_bodies)
            for i in range(num_contact_markers):
                marker_handle = self.gym.create_actor(
                    env_ptr,
                    self._marker_asset_tiny,
                    default_pose,
                    "contact_marker",
                    self.num_envs + 10,
                    0,
                    0,
                )
                color = gymapi.Vec3(1.0, 0.4, 0.7)  # Pink color
                self.gym.set_rigid_body_color(
                    env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, color
                )
                self._marker_handles[env_id].append(marker_handle)

    def _build_marker_state_tensors(self):
        num_markers_per_env = self.num_bodies
        if self.terrain is not None:
            num_markers_per_env += self.terrain_obs_cb.num_height_points
        if self.scene_lib is not None and self.config.point_cloud_obs.enabled:
            num_markers_per_env += (
                self.config.point_cloud_obs.num_pointcloud_samples
                * self.max_objects_per_scene
            )
            num_contact_markers = len(self.config.robot.contact_bodies)
            num_markers_per_env += num_contact_markers

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
        # Update mimic markers
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

        self._marker_pos[:, : self.num_bodies] = target_pos

        markers_offset = self.num_bodies

        # Update terrain markers
        num_terrain_markers = self.terrain_obs_cb.num_height_points
        height_maps = self.terrain_obs_cb.get_height_maps(
            None, None, return_all_dims=True
        )
        height_maps = height_maps.view(self.num_envs, -1, 3)
        self._marker_pos[:, markers_offset : markers_offset + num_terrain_markers] = (
            height_maps
        )
        markers_offset += num_terrain_markers

        # Update scene markers
        if self.scene_lib is not None and self.config.point_cloud_obs.enabled:
            num_pointcloud_markers = (
                self.config.point_cloud_obs.num_pointcloud_samples
                * self.max_objects_per_scene
            )
            self._marker_pos[
                :,
                markers_offset : markers_offset + num_pointcloud_markers,
            ] = self.object_obs_cb.object_pointclouds.reshape(
                self.num_envs,
                self.config.point_cloud_obs.num_pointcloud_samples
                * self.max_objects_per_scene,
                3,
            )

            markers_offset += num_pointcloud_markers

            # Update contact markers
            object_ids = self.env_id_to_object_ids
            flat_object_ids = object_ids.flatten()
            expanded_times = self.motion_times.unsqueeze(-1).expand(
                self.num_envs, self.max_objects_per_scene
            )
            object_root_states = self.get_object_root_states()[object_ids.flatten()]
            object_gt, object_gr = (
                object_root_states[..., 0:3],
                object_root_states[..., 3:7],
            )
            object_gt = object_gt.view(self.num_envs, self.max_objects_per_scene, -1)
            object_gr = object_gr.view(self.num_envs, self.max_objects_per_scene, -1)

            ref_object_state = self.scene_lib.get_object_pose(
                flat_object_ids, expanded_times.flatten()
            )

            non_static_object_mask = self.scene_target_poses_mask.view(
                self.num_envs, self.max_objects_per_scene, -1
            )[:, :, 0]

            bodies_in_contact_target_positions = (
                ref_object_state.bodies_in_contact_target_positions.view(
                    self.num_envs,
                    self.max_objects_per_scene,
                    self.num_bodies,
                    3,
                )
            )
            ref_contact_joint_positions = (
                bodies_in_contact_target_positions
                * non_static_object_mask.unsqueeze(-1).unsqueeze(-1)
            ).sum(dim=1)
            # Remove the reference object offset and rotation
            # TODO: For now only support single dynamic object
            object_ids = (self.env_id_to_object_ids * non_static_object_mask).sum(dim=1)

            cur_object_pos = (object_gt * non_static_object_mask.unsqueeze(-1)).sum(
                dim=1
            )

            # Apply translation and rotation from current object position
            cur_object_gr_expanded = (
                (object_gr * non_static_object_mask.unsqueeze(-1))
                .sum(dim=-2)
                .unsqueeze(1)
                .expand(self.num_envs, self.num_bodies, 4)
            )
            rotated_contact_joint_positions = rotations.quat_rotate(
                cur_object_gr_expanded, ref_contact_joint_positions, self.w_last
            )
            target_contact_joint_positions = (
                rotated_contact_joint_positions + cur_object_pos.unsqueeze(1)
            )

            expected_contacts = ref_object_state.bodies_in_contact.view(
                self.num_envs,
                self.max_objects_per_scene,
                self.num_bodies,
            )
            expected_contacts = (
                expected_contacts * non_static_object_mask.unsqueeze(-1)
            ).sum(dim=1)

            target_contact_joint_positions[expected_contacts == 0] = 100

            num_contact_markers = len(self.config.robot.contact_bodies)
            self._marker_pos[
                :, markers_offset : markers_offset + num_contact_markers
            ] = target_contact_joint_positions[:, self.contact_body_ids]

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )

    def draw_mimic_markers(self):
        self._update_marker()

    def render(self):
        super().render()

        if not self.headless and self.config.visualize_markers:
            self.draw_mimic_markers()
