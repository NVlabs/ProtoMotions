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

from phys_anim.envs.path_following.common import BasePathFollowing
from phys_anim.envs.base_task.isaacsim import TaskHumanoid


class PathFollowingHumanoid(BasePathFollowing, TaskHumanoid):
    def __init__(self, config, device: torch.device):
        super().__init__(config=config, device=device)

        self.head_body_id = self.bodies_names.index("head")

    ###############################################################
    # Set up IsaacSim environment
    ###############################################################
    def set_up_scene(self, scene) -> None:
        if self._env._render:
            self._load_marker_asset(scene)
        super().set_up_scene(scene)
        if self._env._render:
            self.post_set_up_scene()

    def post_set_up_scene(self):
        self.markers = ArticulationView(
            self.default_base_env_path + "/env_*/TrajectoryMarker_*"
        )
        self.markers.set_local_scales(0.05 * torch.ones((self.num_envs * 10, 3)))

    def _load_marker_asset(self, scene):
        # Each marker will be a sphere
        base_env_path = self.default_zero_env_path + "/TrajectoryMarker_0"
        sphere = UsdGeom.Sphere.Define(get_current_stage(), base_env_path)
        color_attribute = sphere.GetDisplayColorAttr()
        color_attribute.Set([(0.21, 0.46, 0.53)])

        # Create a grid cloner instance
        cloner = GridCloner(spacing=3)

        # Create 10 clones, the num_env clones will be created by the main cloner
        target_paths = cloner.generate_paths(
            self.default_zero_env_path + "/TrajectoryMarker", 10
        )

        # Clone the marker at target paths
        cloner.clone(
            source_prim_path=self.default_zero_env_path + "/TrajectoryMarker_0",
            prim_paths=target_paths,
        )

    ###############################################################
    # Helpers
    ###############################################################
    def _update_marker(self):
        traj_samples = self.fetch_path_samples().clone()
        if not self.config.height_conditioned:
            traj_samples[..., 2] = 0.8  # CT hack

        ground_below_marker = self.get_ground_heights(
            traj_samples[..., :2].view(-1, 2)
        ).view(traj_samples.shape[:-1])
        traj_samples[..., 2] += ground_below_marker

        self.markers.set_world_poses(traj_samples.view(-1, 3))

    def draw_task(self):
        self._update_marker()
        super().draw_task()
