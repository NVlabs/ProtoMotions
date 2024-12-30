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

from phys_anim.envs.mimic.common import BaseMimic
from phys_anim.envs.humanoid.isaaclab import Humanoid

from isaac_utils import rotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg


class MimicHumanoid(BaseMimic, Humanoid):
    def __init__(self, config, device: torch.device, simulation_app):
        super().__init__(config, device, simulation_app)

    ###############################################################
    # Set up IsaacSim environment
    ###############################################################
    def set_up_scene(self) -> None:
        if not self.headless and self.config.visualize_markers:
            self._load_marker_asset()
        super().set_up_scene()

    def _load_marker_asset(self):
        mimic_marker_obj_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TrackingMarker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=1,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)
                    ),
                ),
            },
        )
        tracking_marker_scale = []

        for i in range(self.config.robot.num_bodies):
            if (
                self.config.robot.mimic_small_marker_bodies is not None
                and self.config.robot.isaacgym_body_names[i]
                in self.config.robot.mimic_small_marker_bodies
            ):
                tracking_marker_scale.append([0.01, 0.01, 0.01])
            else:
                tracking_marker_scale.append([0.05, 0.05, 0.05])

        self.tracking_markers = VisualizationMarkers(mimic_marker_obj_cfg)
        self.tracking_marker_scale = torch.tensor(
            tracking_marker_scale, device=self.device
        )

        terrain_marker_obj_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TerrainMarker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=1,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.008, 0.345, 0.224)
                    ),
                ),
            },
        )
        terrain_marker_scale = []

        for i in range(self.terrain_obs_cb.num_height_points):
            terrain_marker_scale.append([0.01, 0.01, 0.01])

        self.terrain_markers = VisualizationMarkers(terrain_marker_obj_cfg)
        self.terrain_marker_scale = torch.tensor(
            terrain_marker_scale, device=self.device
        )

        if self.scene_lib is not None and self.config.point_cloud_obs.enabled:
            # Point cloud markers
            pointcloud_marker_obj_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/PointCloudMarker",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=1,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.3, 0.7, 0.9)
                        ),
                    ),
                },
            )
            pointcloud_marker_scale = []

            num_pointcloud_markers = (
                self.config.point_cloud_obs.num_pointcloud_samples
                * self.max_objects_per_scene
            )

            for i in range(num_pointcloud_markers):
                pointcloud_marker_scale.append([0.005, 0.005, 0.005])
            self.pointcloud_markers = VisualizationMarkers(pointcloud_marker_obj_cfg)
            self.pointcloud_marker_scale = torch.tensor(
                pointcloud_marker_scale, device=self.device
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

        target_pos[..., -1:] += self.terrain_obs_cb.get_ground_heights(
            target_pos[:, 0, :2]
        ).view(self.num_envs, 1, 1)
        self.tracking_markers.visualize(
            translations=target_pos.view(-1, 3), scales=self.tracking_marker_scale
        )

        # Update terrain markers
        height_maps = self.terrain_obs_cb.get_height_maps(
            None, None, return_all_dims=True
        )
        self.terrain_markers.visualize(
            translations=height_maps.view(-1, 3), scales=self.terrain_marker_scale
        )

        # Update scene markers
        if self.scene_lib is not None and self.config.point_cloud_obs.enabled:
            self.pointcloud_markers.visualize(
                translations=self.object_obs_cb.object_pointclouds.view(-1, 3),
                scales=self.pointcloud_marker_scale,
            )

    def draw_mimic_markers(self):
        self._update_marker()

    def render(self):
        super().render()

        if not self.headless and self.config.visualize_markers:
            self.draw_mimic_markers()
