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
"""Stateless observation functions for steering tasks.

These functions compute observations from steering control context and can be used
with the ObservationComponentConfig system.
"""

import torch
from torch import Tensor

from protomotions.utils import rotations
from protomotions.envs.obs.observation_component import ObservationComponentConfig


def steering_obs_factory() -> ObservationComponentConfig:
    """Factory for steering observation component.

    Returns an ObservationComponentConfig that computes steering observations
    (target direction + target speed + target facing direction, all in local frame).

    Returns:
        ObservationComponentConfig for steering observations.
    """
    return ObservationComponentConfig(
        function=compute_steering_obs,
        variables={
            "root_rot": "current_state_root_rot",
            "tar_dir": "tar_dir",  # From SteeringControl.get_context()
            "tar_speed": "tar_speed",  # From SteeringControl.get_context()
            "tar_face_dir": "tar_face_dir",  # From SteeringControl.get_context()
        },
    )


def compute_steering_obs(
    root_rot: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
    tar_face_dir: Tensor,
) -> Tensor:
    """Compute steering observations in the robot's local frame.

    Transforms the target direction and facing direction from world frame to the
    robot's local frame and concatenates with the target speed.

    Args:
        root_rot: Root orientation quaternions [num_envs, 4] (w-last format).
        tar_dir: Target movement direction vectors [num_envs, 2] in world frame.
        tar_speed: Target speeds [num_envs].
        tar_face_dir: Target facing direction vectors [num_envs, 2] in world frame.

    Returns:
        Steering observations [num_envs, 5]: [local_dir(2), tar_speed(1), local_face_dir(2)].
    """
    # Extend 2D directions to 3D (z=0)
    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    tar_face_dir3d = torch.cat([tar_face_dir, torch.zeros_like(tar_face_dir[..., 0:1])], dim=-1)

    # Get inverse heading rotation (to transform world -> local)
    heading_rot = rotations.calc_heading_quat_inv(root_rot, True)

    # Transform target direction to local frame
    local_tar_dir = rotations.quat_rotate(heading_rot, tar_dir3d, True)
    local_tar_dir = local_tar_dir[..., 0:2]

    # Transform target facing direction to local frame
    local_tar_face_dir = rotations.quat_rotate(heading_rot, tar_face_dir3d, True)
    local_tar_face_dir = local_tar_face_dir[..., 0:2]

    # Concatenate: [local_tar_dir(2), tar_speed(1), local_tar_face_dir(2)]
    tar_speed_expanded = tar_speed.unsqueeze(-1)
    obs = torch.cat([local_tar_dir, tar_speed_expanded, local_tar_face_dir], dim=-1)

    return obs
