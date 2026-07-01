# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment utilities.

Contains:
- path_generator: Path generation for path following tasks
- scene: Scene/object point cloud and coordinate utilities
"""

from protomotions.envs.utils.path_generator import PathGenerator, PathGeneratorConfig
from protomotions.envs.utils.scene import (
    get_contact_bodies_to_object_pointcloud,
    closest_points_on_object_surface,
    get_object_pointcloud,
    get_local_object_coordinates,
)

__all__ = [
    # Path generator
    "PathGenerator",
    "PathGeneratorConfig",
    # Scene utilities
    "get_contact_bodies_to_object_pointcloud",
    "closest_points_on_object_surface",
    "get_object_pointcloud",
    "get_local_object_coordinates",
]
