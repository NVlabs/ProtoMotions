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
"""Scene and object utilities for environment observations.

Provides functions for computing object representations including point clouds,
bounding boxes, and local coordinates in robot-centric frames.
"""

import torch
from torch import Tensor
from protomotions.utils import rotations
from typing import Tuple


@torch.jit.script_if_tracing
def get_contact_bodies_to_object_pointcloud(
    current_object_rot: Tensor,
    current_object_pos: Tensor,
    scene_neutral_pointclouds: Tensor,
    contact_bodies_pos: Tensor,
    contact_bodies_rot: Tensor,
):
    """Compute vectors from contact bodies to nearest object surface points.

    For each contact body (e.g., hands, feet), finds the nearest point on each
    object's surface and returns the vector in the contact body's local frame.

    Args:
        current_object_rot: Object rotations [num_envs, num_objects, 4]
        current_object_pos: Object positions [num_envs, num_objects, 3]
        scene_neutral_pointclouds: Neutral pose point clouds [num_envs, num_objects, num_points, 3]
        contact_bodies_pos: Contact body positions [num_envs, num_contact_bodies, 3]
        contact_bodies_rot: Contact body rotations [num_envs, num_contact_bodies, 4]

    Returns:
        Vectors from contact bodies to nearest points [num_envs, num_objects * num_contact_bodies * 3]
    """
    num_envs = contact_bodies_pos.shape[0]
    contact_bodies_quat_inv = rotations.quat_conjugate(contact_bodies_rot, True)

    # Get the nearest points: [envs, objects, contact_bodies, 3]
    nearest_points, _ = closest_points_on_object_surface(
        current_object_rot=current_object_rot,
        current_object_pos=current_object_pos,
        scene_neutral_pointclouds=scene_neutral_pointclouds,
        contact_bodies_pos=contact_bodies_pos,
    )

    # Calculate vectors from contact bodies to nearest points in global frame
    # [envs, objects, contact_bodies, 3]
    contact_bodies_to_pointcloud = nearest_points - contact_bodies_pos.unsqueeze(1)

    # Transform vectors to contact-body-egocentric frame using the same heading rotation
    # [envs, objects, contact_bodies, 3]

    egocentric_contact_bodies_to_pointcloud = rotations.quat_rotate(
        contact_bodies_quat_inv.unsqueeze(1).expand(
            -1, contact_bodies_to_pointcloud.shape[1], -1, -1
        ),
        contact_bodies_to_pointcloud,
        True,
    )

    return egocentric_contact_bodies_to_pointcloud.reshape(num_envs, -1)


@torch.jit.script_if_tracing
def closest_points_on_object_surface(
    current_object_rot: Tensor,
    current_object_pos: Tensor,
    scene_neutral_pointclouds: Tensor,
    contact_bodies_pos: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Find nearest surface points on objects for each contact body.

    Args:
        current_object_rot: Object rotations [num_envs, num_objects, 4]
        current_object_pos: Object positions [num_envs, num_objects, 3]
        scene_neutral_pointclouds: Neutral pose point clouds [num_envs, num_objects, num_points, 3]
        contact_bodies_pos: Contact body positions [num_envs, num_contact_bodies, 3]

    Returns:
        Tuple of (nearest_points [num_envs, num_objects, num_contact_bodies, 3],
                  min_indices [num_envs, num_objects, num_contact_bodies])
    """
    num_envs = current_object_pos.shape[0]
    num_objects = scene_neutral_pointclouds.shape[1]
    num_points = scene_neutral_pointclouds.shape[2]

    rotated_neutral_object_pointcloud = rotations.quat_rotate(
        current_object_rot.unsqueeze(2).expand(-1, -1, num_points, -1),
        scene_neutral_pointclouds,
        True,
    )
    global_scene_pointcloud = (
        rotated_neutral_object_pointcloud + current_object_pos.unsqueeze(2)
    )

    # Reshape tensors for broadcasting
    # [envs, objects, points, 3] -> [envs, objects, 1, points, 3]
    global_scene_pointcloud = global_scene_pointcloud.unsqueeze(2)
    # [envs, contact_bodies, 3] -> [envs, 1, contact_bodies, 1, 3]
    contact_bodies_pos = contact_bodies_pos.unsqueeze(1).unsqueeze(3)

    # Calculate distances between each contact body and all points in each object's pointcloud
    # This will be [envs, objects, contact_bodies, points]
    distances = torch.norm(global_scene_pointcloud - contact_bodies_pos, dim=-1)

    # Find indices of minimum distances for each contact body and object
    # [envs, objects, contact_bodies]
    min_indices = torch.argmin(distances, dim=-1)

    # Create index grid for proper gathering
    batch_idx = torch.arange(num_envs, device=current_object_rot.device)[:, None, None]
    obj_idx = torch.arange(num_objects, device=current_object_rot.device)[None, :, None]

    # Get the nearest points: [envs, objects, contact_bodies, 3]
    nearest_points = global_scene_pointcloud[batch_idx, obj_idx, 0, min_indices]

    return nearest_points, min_indices


@torch.jit.script_if_tracing
def get_object_pointcloud(
    current_object_rot: Tensor,
    current_object_pos: Tensor,
    scene_neutral_pointclouds: Tensor,
    robot_root_pos: Tensor,
    robot_root_rot: Tensor,
) -> Tensor:
    """
    Calculates the egocentric point cloud of scene objects relative to the robot's root frame.

    Args:
        current_object_rot (Tensor): Current rotation of objects. Shape: [num_envs, num_objects, 4].
        current_object_pos (Tensor): Current position of objects. Shape: [num_envs, num_objects, 3].
        scene_neutral_pointclouds (Tensor): Neutral pose point clouds for objects.
                                            Shape: [num_envs, num_objects, num_points, 3].
        robot_root_pos (Tensor): Position of the robot root. Shape: [num_envs, 3].
        robot_root_rot (Tensor): Rotation of the robot root. Shape: [num_envs, 4].

    Returns:
        Tensor: Egocentric point cloud flattened per environment.
                Shape: [num_envs, num_objects * num_points * 3].
    """
    num_envs = robot_root_pos.shape[0]
    num_objects = scene_neutral_pointclouds.shape[1]
    num_points = scene_neutral_pointclouds.shape[2]

    # Calculate inverse heading rotation of the robot
    heading_rot_inv = rotations.calc_heading_quat_inv(
        robot_root_rot, True
    )  # [num_envs, 4]

    # Rotate neutral point clouds by current object rotations
    # Expand current_object_rot: [num_envs, num_objects, 1, 4] -> [num_envs, num_objects, num_points, 4]
    expanded_object_rot = current_object_rot.unsqueeze(2).expand(-1, -1, num_points, -1)
    rotated_neutral_object_pointcloud = rotations.quat_rotate(
        expanded_object_rot, scene_neutral_pointclouds, True
    )  # [num_envs, num_objects, num_points, 3]

    # Translate point clouds to global positions
    # Expand current_object_pos: [num_envs, num_objects, 1, 3]
    global_scene_pointcloud = (
        rotated_neutral_object_pointcloud + current_object_pos.unsqueeze(2)
    )  # [num_envs, num_objects, num_points, 3]

    # Calculate point clouds relative to robot root position (local frame)
    # Expand robot_root_pos: [num_envs, 1, 1, 3]
    local_scene_pointcloud = global_scene_pointcloud - robot_root_pos.unsqueeze(
        1
    ).unsqueeze(2)  # [num_envs, num_objects, num_points, 3]

    # Rotate local point clouds by inverse heading rotation to get egocentric view
    # Expand heading_rot_inv: [num_envs, 1, 1, 4] -> [num_envs, num_objects, num_points, 4]
    expanded_heading_rot_inv = (
        heading_rot_inv.unsqueeze(1)
        .unsqueeze(2)
        .expand(-1, num_objects, num_points, -1)
    )
    egocentric_scene_pointcloud = rotations.quat_rotate(
        expanded_heading_rot_inv, local_scene_pointcloud, True
    )  # [num_envs, num_objects, num_points, 3]

    # Reshape to flatten points per environment
    return egocentric_scene_pointcloud.reshape(
        num_envs, -1
    )  # [num_envs, num_objects * num_points * 3]


@torch.jit.script_if_tracing
def get_local_object_coordinates(
    current_object_rot: Tensor,
    current_object_pos: Tensor,
    robot_root_pos: Tensor,
    robot_root_rot: Tensor,
) -> Tensor:
    """
    Transforms object positions and rotations to the robot's egocentric frame.

    Args:
        current_object_rot (Tensor): Current rotation of objects. Shape: [num_envs, num_objects, 4].
        current_object_pos (Tensor): Current position of objects. Shape: [num_envs, num_objects, 3].
        robot_root_pos (Tensor): Position of the robot root. Shape: [num_envs, 3].
        robot_root_rot (Tensor): Rotation of the robot root. Shape: [num_envs, 4].

    Returns:
        Tensor: Object coordinates in the robot's egocentric frame.
                Shape: [num_envs, num_objects * (3 + 6)], where 3 is for position and 6 is for rotation.
    """
    num_envs = robot_root_pos.shape[0]
    num_objects = current_object_pos.shape[1]

    # Calculate inverse heading rotation of the robot
    heading_rot_inv = rotations.calc_heading_quat_inv(
        robot_root_rot, True
    )  # [num_envs, 4]

    # Calculate local positions (relative to robot position)
    # [num_envs, num_objects, 3] - [num_envs, 1, 3] -> [num_envs, num_objects, 3]
    local_object_pos = current_object_pos - robot_root_pos.unsqueeze(1)

    # Rotate local positions to get egocentric view
    # Expand heading_rot_inv: [num_envs, 1, 4] -> [num_envs, num_objects, 4]
    expanded_heading_rot_inv = heading_rot_inv.unsqueeze(1).expand(-1, num_objects, -1)
    egocentric_object_pos = rotations.quat_rotate(
        expanded_heading_rot_inv, local_object_pos, True
    )  # [num_envs, num_objects, 3]

    # Calculate relative rotations and convert to 6D representation
    # [num_envs, num_objects, 4] relative to [num_envs, 1, 4] -> [num_envs, num_objects, 4]
    relative_object_rot = rotations.quat_mul(
        expanded_heading_rot_inv, current_object_rot, True
    )

    # Convert quaternions to 6D tangent-normal representation
    # [num_envs, num_objects, 4] -> [num_envs, num_objects, 6]
    rot_6d = rotations.quat_to_tan_norm(relative_object_rot, True)

    # Concatenate position and rotation
    # [num_envs, num_objects, 3] + [num_envs, num_objects, 6] -> [num_envs, num_objects, 9]
    object_coordinates = torch.cat([egocentric_object_pos, rot_6d], dim=-1)

    # Reshape to flatten objects per environment
    # [num_envs, num_objects, 9] -> [num_envs, num_objects * 9]
    return object_coordinates.reshape(num_envs, -1)
