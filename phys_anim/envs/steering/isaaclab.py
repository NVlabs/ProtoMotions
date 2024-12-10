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
from omni.isaac.cloner import GridCloner
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.stage import get_current_stage
from pxr import UsdGeom

from isaac_utils import rotations
from phys_anim.envs.steering.common import BaseSteering
from phys_anim.envs.base_task.isaaclab import TaskHumanoid

TAR_ACTOR_ID = 1
TAR_FACING_ACTOR_ID = 2


class SteeringHumanoid(BaseSteering, TaskHumanoid):
    def __init__(self, config, device: torch.device, simulation_app):
        super().__init__(config=config, device=device, simulation_app=simulation_app)

    ###############################################################
    # Set up IsaacSim environment
    ###############################################################
    def set_up_scene(self, scene) -> None:
        if not self.headless:
            self._load_marker_asset(scene)
        super().set_up_scene(scene)
        if not self.headless:
            self.post_set_up_scene()

    def post_set_up_scene(self):
        self.markers = ArticulationView(
            self.default_base_env_path + "/env_*/DirectionMarker_*"
        )
        self.markers.set_local_scales(0.1 * torch.ones((self.num_envs, 3)))

    def _load_marker_asset(self, scene):
        # Each marker will be a sphere
        base_env_path = self.default_zero_env_path + "/DirectionMarker_0"
        UsdGeom.Cone.Define(get_current_stage(), base_env_path)

        # Create a grid cloner instance
        cloner = GridCloner(spacing=3)

        # Create 10 clones, the num_env clones will be created by the main cloner
        target_paths = cloner.generate_paths(
            self.default_zero_env_path + "/DirectionMarker", 1
        )

        # Clone the marker at target paths
        cloner.clone(
            source_prim_path=self.default_zero_env_path + "/DirectionMarker_0",
            prim_paths=target_paths,
        )

        self._marker_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )

    ###############################################################
    # Helpers
    ###############################################################
    def _update_marker(self):
        humanoid_root_pos = self.get_humanoid_root_states()[..., 0:3]
        self._marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2] + self._tar_dir
        self._marker_pos[..., 2] = humanoid_root_pos[..., 2]

        heading_theta = (
            self._tar_dir_theta
        )  # torch.atan2(self._tar_dir[..., 1], self._tar_dir[..., 0])
        heading_axis = torch.zeros_like(self._marker_pos)
        heading_axis[..., -1] = 1.0
        marker_rot = rotations.quat_from_angle_axis(
            heading_theta, heading_axis, self.w_last
        )

        self.markers.set_world_poses(self._marker_pos, marker_rot)

        for env_id in range(self.num_envs):
            cone_prim = UsdGeom.Cone.Define(
                get_current_stage(),
                self.default_base_env_path + f"/env_{env_id}/DirectionMarker_0",
            )
            color_attr = cone_prim.GetDisplayColorAttr()

            weight = self._tar_speed[env_id].cpu().item() / self._tar_speed_max
            color = [(weight, 0, 1 - weight)]
            color_attr.Set(color)

    def draw_task(self):
        self._update_marker()
        super().draw_task()
