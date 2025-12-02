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
"""Sub-terrain data structure.

Defines the SubTerrain class for representing individual terrain patches.
"""

import numpy as np


class SubTerrain:
    """Represents a single sub-terrain patch in the terrain grid.

    Stores height field data, dimensions, and scaling information for a
    specific terrain type (e.g., a patch of stairs).

    Args:
        config_terrain: Terrain configuration.
        terrain_name: Name of the terrain type.
        device: Device for data storage.
    """

    def __init__(self, config_terrain, terrain_name="terrain", device="cuda:0"):
        self.terrain_name = terrain_name
        self.config_terrain = config_terrain
        self.device = device
        self.vertical_scale = config_terrain.vertical_scale
        self.horizontal_scale = config_terrain.horizontal_scale
        self.width = int(config_terrain.map_width / self.horizontal_scale)
        self.length = int(config_terrain.map_length / self.horizontal_scale)
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
        self.ceiling_field_raw = np.zeros((self.width, self.length), dtype=np.int16) + (
            3 / self.vertical_scale
        )
        self.walkable_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
