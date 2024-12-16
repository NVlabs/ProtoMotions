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

from phys_anim.envs.path_following.common import BasePathFollowing
from phys_anim.envs.base_task.isaaclab import TaskHumanoid

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg


class PathFollowingHumanoid(BasePathFollowing, TaskHumanoid):
    def __init__(self, config, device: torch.device, simulation_app):
        super().__init__(config=config, device=device, simulation_app=simulation_app)

    ###############################################################
    # Set up IsaacSim environment
    ###############################################################
    def set_up_scene(self) -> None:
        if not self.headless:
            self._load_marker_asset()
        super().set_up_scene()

    def _load_marker_asset(self):
        traj_marker_obj_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TrajectoryMarker",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=1,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0)
                    ),
                ),
            },
        )
        trajectory_marker_scale = []
        for i in range(
            self.num_envs * self.config.path_follower_params.num_traj_samples
        ):
            trajectory_marker_scale.append([0.05, 0.05, 0.05])

        self.trajectory_markers = VisualizationMarkers(traj_marker_obj_cfg)
        self.trajectory_marker_scale = torch.tensor(
            trajectory_marker_scale, device=self.device
        )

    ###############################################################
    # Helpers
    ###############################################################
    def _update_marker(self):
        traj_samples = self.fetch_path_samples().clone()
        if not self.config.path_follower_params.height_conditioned:
            traj_samples[..., 2] = 0.8  # CT hack

        ground_below_marker = self.terrain_obs_cb.get_ground_heights(
            traj_samples[..., :2].view(-1, 2)
        ).view(traj_samples.shape[:-1])
        traj_samples[..., 2] += ground_below_marker

        self.trajectory_markers.visualize(
            translations=traj_samples.view(-1, 3),
            scales=self.trajectory_marker_scale,
        )

    def draw_task(self):
        self._update_marker()
        super().draw_task()
