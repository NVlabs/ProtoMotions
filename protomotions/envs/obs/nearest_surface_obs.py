# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nearest surface vector observation.

For each selected body, computes an ego-centric heading-relative vector to the
closest interaction surface. The surface can be sampled terrain points, scene
object pointclouds, or both.
"""

from typing import Optional

import torch
from torch import Tensor

from protomotions.components.terrains.terrain_utils import get_height_maps_jit
from protomotions.envs.utils.scene import closest_points_on_object_surface
from protomotions.utils.rotations import calc_heading_quat_inv, quat_rotate


def compute_nearest_surface_vectors(
    rigid_body_pos: Tensor,
    root_pos: Tensor,
    root_rot: Tensor,
    height_points: Optional[Tensor] = None,
    height_samples: Optional[Tensor] = None,
    object_pos: Optional[Tensor] = None,
    object_rot: Optional[Tensor] = None,
    neutral_pointclouds: Optional[Tensor] = None,
    object_valid_mask: Optional[Tensor] = None,
    terrain_horizontal_scale: float = 0.1,
    body_ids: Optional[list] = None,
) -> Tensor:
    """Return per-body vectors to the nearest terrain/object surface."""
    if body_ids is not None:
        rigid_body_pos = rigid_body_pos[:, body_ids]
    num_envs, num_bodies = rigid_body_pos.shape[:2]
    device = rigid_body_pos.device

    best_dist = torch.full((num_envs, num_bodies), float("inf"), device=device)
    best_vector = torch.zeros(num_envs, num_bodies, 3, device=device)

    if height_points is not None and height_samples is not None:
        num_terrain_points = height_points.shape[1]
        terrain_world = get_height_maps_jit(
            base_rot=root_rot,
            base_pos=root_pos,
            height_points=height_points.clone(),
            height_samples=height_samples,
            num_height_points=num_terrain_points,
            terrain_horizontal_scale=terrain_horizontal_scale,
            w_last=True,
            return_all_dims=True,
        )
        terrain_diff = terrain_world.unsqueeze(1) - rigid_body_pos.unsqueeze(2)
        terrain_dists = terrain_diff.norm(dim=-1)
        terrain_nearest_idx = terrain_dists.argmin(dim=-1)

        terrain_vector = terrain_diff.gather(
            2,
            terrain_nearest_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 3),
        ).squeeze(2)
        terrain_dist = terrain_dists.gather(
            2,
            terrain_nearest_idx.unsqueeze(-1),
        ).squeeze(-1)

        closer = terrain_dist < best_dist
        best_dist = torch.where(closer, terrain_dist, best_dist)
        best_vector = torch.where(closer.unsqueeze(-1), terrain_vector, best_vector)

    has_objects = (
        object_pos is not None
        and object_rot is not None
        and neutral_pointclouds is not None
        and object_valid_mask is not None
        and object_pos.shape[1] > 0
    )
    if has_objects:
        nearest_points, _ = closest_points_on_object_surface(
            current_object_rot=object_rot,
            current_object_pos=object_pos,
            scene_neutral_pointclouds=neutral_pointclouds,
            contact_bodies_pos=rigid_body_pos,
        )

        obj_vectors = nearest_points - rigid_body_pos.unsqueeze(1)
        obj_dists = obj_vectors.norm(dim=-1)
        obj_dists = obj_dists.masked_fill(
            (object_valid_mask < 0.5).unsqueeze(-1),
            float("inf"),
        )

        best_obj_idx = obj_dists.argmin(dim=1)
        obj_dist = obj_dists.gather(1, best_obj_idx.unsqueeze(1)).squeeze(1)
        obj_vector = obj_vectors.gather(
            1,
            best_obj_idx.unsqueeze(1).unsqueeze(-1).expand(-1, 1, num_bodies, 3),
        ).squeeze(1)

        closer = obj_dist < best_dist
        best_dist = torch.where(closer, obj_dist, best_dist)
        best_vector = torch.where(closer.unsqueeze(-1), obj_vector, best_vector)

    heading_rot_inv = calc_heading_quat_inv(root_rot, True)
    heading_expanded = heading_rot_inv.unsqueeze(1).expand(-1, num_bodies, -1)
    ego_vectors = quat_rotate(
        heading_expanded.reshape(-1, 4),
        best_vector.reshape(-1, 3),
        True,
    ).reshape(num_envs, num_bodies, 3)

    return ego_vectors.reshape(num_envs, -1)
