# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Scene observation component.

Provides object-based observations including bounding boxes, point clouds,
masks, and human-object relative observations.
"""

import torch
from typing import TYPE_CHECKING
from protomotions.envs.utils.scene import (
    get_object_pointcloud,
    get_contact_bodies_to_object_pointcloud,
    get_local_object_coordinates,
)
from protomotions.envs.obs.config import SceneObsConfig

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


class SceneObs:
    """Handles computation of scene and object observations.

    Computes point clouds and object states relative to the robot.

    Args:
        config: Configuration for scene observations.
        env: Parent environment instance.
    """

    def __init__(self, config: SceneObsConfig, env: "BaseEnv"):
        self.config = config
        self.env = env

        if self.env.scene_lib.num_scenes() == 0 or not self.config.enabled:
            self.scene_pointclouds = None
            self.contact_bodies_to_pointcloud = None
            return

        if self.config.obs_object_index is not None:
            self.obs_object_index = self.config.obs_object_index
        else:
            self.obs_object_index = list(
                range(self.env.scene_lib.num_objects_per_scene)
            )  # default to all objects
        num_obs_objects_per_scene = len(self.obs_object_index)

        self.scene_pointclouds = torch.zeros(
            self.env.num_envs,
            num_obs_objects_per_scene
            * self.env.scene_lib.pointcloud_samples_per_object
            * 3,
            device=self.env.device,
        )

        self.contact_bodies_to_pointcloud = torch.zeros(
            self.env.num_envs,
            num_obs_objects_per_scene * len(self.env.robot_config.contact_bodies) * 3,
            device=self.env.device,
        )

        self.scene_object_coordinates = torch.zeros(
            self.env.num_envs,
            num_obs_objects_per_scene * (3 + 6),  # Translation 3D + rotation 6D
            device=self.env.device,
        )

    def compute_observations(self, env_ids: torch.Tensor) -> None:
        if self.env.scene_lib.num_scenes() == 0 or not self.config.enabled:
            return

        scene_neutral_pointclouds = self.env.scene_lib.get_scene_neutral_pointcloud(
            env_ids
        )  # [num_envs, num_objects, num_points, 3]
        current_object_state = self.env.simulator.get_object_root_state(env_ids)
        current_object_pos = current_object_state.root_pos  # [num_envs, num_objects, 3]
        current_object_rot = current_object_state.root_rot  # [num_envs, num_objects, 4]

        scene_neutral_pointclouds = scene_neutral_pointclouds[
            :, self.obs_object_index, :, :
        ]
        current_object_pos = current_object_pos[:, self.obs_object_index, :]
        current_object_rot = current_object_rot[:, self.obs_object_index, :]

        robot_state = self.env.simulator.get_bodies_state(env_ids)
        robot_root_pos = robot_state.rigid_body_pos[:, 0, :]
        robot_root_rot = robot_state.rigid_body_rot[:, 0, :]
        contact_bodies_pos = robot_state.rigid_body_pos[:, self.env.contact_body_ids, :]
        contact_bodies_rot = robot_state.rigid_body_rot[:, self.env.contact_body_ids, :]

        contact_bodies_to_pointcloud = get_contact_bodies_to_object_pointcloud(
            current_object_rot=current_object_rot,
            current_object_pos=current_object_pos,
            scene_neutral_pointclouds=scene_neutral_pointclouds,
            contact_bodies_pos=contact_bodies_pos,
            contact_bodies_rot=contact_bodies_rot,
        )

        self.contact_bodies_to_pointcloud[env_ids] = contact_bodies_to_pointcloud

        scene_pointclouds = get_object_pointcloud(
            current_object_rot=current_object_rot,
            current_object_pos=current_object_pos,
            scene_neutral_pointclouds=scene_neutral_pointclouds,
            robot_root_pos=robot_root_pos,
            robot_root_rot=robot_root_rot,
        )

        self.scene_pointclouds[env_ids] = scene_pointclouds

        object_coordinates = get_local_object_coordinates(
            current_object_rot=current_object_rot,
            current_object_pos=current_object_pos,
            robot_root_pos=robot_root_pos,
            robot_root_rot=robot_root_rot,
        )
        self.scene_object_coordinates[env_ids] = object_coordinates

    def get_obs(self):
        assert self.scene_pointclouds is not None
        assert self.contact_bodies_to_pointcloud is not None
        return {
            "scene_pointclouds": self.scene_pointclouds.clone(),
            "contact_bodies_to_pointcloud": self.contact_bodies_to_pointcloud.clone(),
            "scene_object_coordinates": self.scene_object_coordinates.clone(),
        }
