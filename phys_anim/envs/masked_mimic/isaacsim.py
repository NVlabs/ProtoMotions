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

from phys_anim.envs.masked_mimic.common import BaseMaskedMimic
from phys_anim.envs.mimic.isaacsim import MimicHumanoid

from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom, UsdPhysics, Gf


class MaskedMimicHumanoid(BaseMaskedMimic, MimicHumanoid):
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)

    ###############################################################
    # Set up IsaacSim environment
    ###############################################################
    def _load_marker_asset(self):
        self.mimic_total_markers = len(self.config.masked_mimic_conditionable_bodies)

        base_mimic_marker_path = self.default_zero_env_path + "/TrackingMarker_"

        for i in range(len(self.config.masked_mimic_conditionable_bodies)):
            if (
                self.config.robot.mimic_small_marker_bodies is not None
                and self.config.masked_mimic_conditionable_bodies[i]
                in self.config.robot.mimic_small_marker_bodies
            ):
                scale = 0.01
            else:
                scale = 0.05

            sphere = UsdGeom.Sphere.Define(
                get_current_stage(), base_mimic_marker_path + str(i)
            )
            color_attribute = sphere.GetDisplayColorAttr()
            color_attribute.Set([(1.0, 0.0, 0.0)])
            UsdGeom.Xformable(sphere).AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
            UsdPhysics.RigidBodyAPI(sphere).CreateKinematicEnabledAttr(False)

        base_mimic_future_marker_path = (
            self.default_zero_env_path + "/FutureTrackingMarker_"
        )

        for i in range(len(self.config.masked_mimic_conditionable_bodies)):
            if (
                self.config.robot.mimic_small_marker_bodies is not None
                and self.config.masked_mimic_conditionable_bodies[i]
                in self.config.robot.mimic_small_marker_bodies
            ):
                scale = 0.01
            else:
                scale = 0.05

            sphere = UsdGeom.Sphere.Define(
                get_current_stage(), base_mimic_future_marker_path + str(i)
            )
            color_attribute = sphere.GetDisplayColorAttr()
            color_attribute.Set([(1.0, 1.0, 0.0)])
            UsdGeom.Xformable(sphere).AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
            UsdPhysics.RigidBodyAPI(sphere).CreateKinematicEnabledAttr(False)

        base_terrain_marker_path = self.default_zero_env_path + "/TerrainMarker_"

        for i in range(self.num_height_points):
            scale = 0.01

            sphere = UsdGeom.Sphere.Define(
                get_current_stage(), base_terrain_marker_path + str(i)
            )
            color_attribute = sphere.GetDisplayColorAttr()
            color_attribute.Set([(0.008, 0.345, 0.224)])
            UsdGeom.Xformable(sphere).AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))
            UsdPhysics.RigidBodyAPI(sphere).CreateKinematicEnabledAttr(False)

    def post_set_up_scene(self):
        self.mimic_markers = XFormPrimView(
            self.default_base_env_path + "/env_*/TrackingMarker_*"
        )
        self.future_mimic_markers = XFormPrimView(
            self.default_base_env_path + "/env_*/FutureTrackingMarker_*"
        )
        self.terrain_markers = XFormPrimView(
            self.default_base_env_path + "/env_*/TerrainMarker_*"
        )

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

        target_pos[..., -1:] += self.get_ground_heights(target_pos[:, 0, :2]).view(
            self.num_envs, 1, 1
        )
        target_pos = target_pos[:, self.masked_mimic_conditionable_bodies_ids, :]

        inactive_markers = torch.ones(
            self.num_envs,
            len(self.config.masked_mimic_conditionable_bodies),
            dtype=torch.bool,
            device=self.device,
        )

        if self.config.masked_mimic_masking.joint_masking.masked_mimic_time_mask:
            mask_time_len = self.config.mimic_target_pose.num_future_steps
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

        self.mimic_markers.set_world_poses(target_pos.view(-1, 3))

        # Inbetweening markers
        ref_state = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.target_pose_time
        )
        target_pos = ref_state.rb_pos
        target_pos += self.respawn_offset_relative_to_data.clone().view(
            self.num_envs, 1, 3
        )
        target_pos[..., -1:] += self.get_ground_heights(target_pos[:, 0, :2]).view(
            self.num_envs, 1, 1
        )

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

        self.future_mimic_markers.set_world_poses(target_pos.view(-1, 3))

        # Update terrain markers
        height_maps = self.get_height_maps(None, return_all_dims=True)
        self.terrain_markers.set_world_poses(height_maps.view(-1, 3))

    def draw_mimic_markers(self):
        self._update_marker()

    def render(self):
        super().render()

        if not self.headless and self.config.visualize_markers:
            self.draw_mimic_markers()
