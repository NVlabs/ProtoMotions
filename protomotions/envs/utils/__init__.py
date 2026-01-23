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
