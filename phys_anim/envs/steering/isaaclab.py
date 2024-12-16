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
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkersCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from isaac_utils import rotations
from phys_anim.envs.steering.common import BaseSteering
from phys_anim.envs.base_task.isaaclab import TaskHumanoid


class SteeringHumanoid(BaseSteering, TaskHumanoid):
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
        steering_marker_obj_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/SteeringMarker",
            markers={
                "arrow_x": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.1, 0.1, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 1.0), opacity=0.5
                    ),
                ),
            },
        )

        self.steering_markers = VisualizationMarkers(steering_marker_obj_cfg)

    ###############################################################
    # Helpers
    ###############################################################
    def _update_marker(self):
        marker_root_pos = self.get_humanoid_root_states()[..., 0:3].clone()
        marker_root_pos[..., 0:2] += self._tar_dir

        heading_axis = torch.zeros_like(marker_root_pos)
        heading_axis[..., -1] = 1.0
        marker_rot = rotations.quat_from_angle_axis(
            self._tar_dir_theta, heading_axis, self.w_last
        )

        self.steering_markers.visualize(
            translations=marker_root_pos,
            orientations=rotations.xyzw_to_wxyz(marker_rot),
        )

    def draw_task(self):
        self._update_marker()
        super().draw_task()
