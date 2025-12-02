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
from typing import Optional, List
from dataclasses import dataclass, field
from protomotions.utils.config_builder import ConfigBuilder
from enum import Enum


class CombineMode(Enum):
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    MULTIPLY = "multiply"

    @classmethod
    def from_str(cls, value: str) -> "CombineMode":
        """Create enum from string, case-insensitive."""
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. "
                f"Valid values are: {[e.value for e in cls]}"
            )
        return cls(value)


@dataclass
class TerrainSimConfig(ConfigBuilder):
    """Configuration for terrain simulation properties (friction, restitution, height offset).

    These properties affect the physical behavior of the terrain in simulation.
    Separate from TerrainConfig which defines terrain geometry.
    """

    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    height_offset: float = (
        0.0  # Height offset for the terrain (negative values move terrain down)
    )
    combine_mode: CombineMode = CombineMode.AVERAGE


@dataclass
class TerrainConfig(ConfigBuilder):
    """Configuration for terrain generation."""

    # Flat terrain is a class that inherits from Terrain. It creates a simple and minimal ground mesh.
    # We require some terrain class to manage spawning, observations and logic. The default terrain should act similarly
    #   to the default ground mesh.
    # The opt/terrain config overrides this config to use irregular terrains.

    _target_: str = "protomotions.components.terrains.terrain.Terrain"
    map_length: float = 20.0
    map_width: float = 20.0
    border_size: float = 40.0  # ensure sufficient space from the edges
    num_levels: int = 10
    num_terrains: int = 10

    # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping, poles, flat]
    terrain_proportions: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    )
    slope_threshold: float = 0.9
    num_samples_per_axis: int = 16
    sample_width: float = 1.0
    terrain_obs_num_samples: Optional[int] = None

    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005

    spacing_between_scenes: float = (
        10.0  # Scenes are created in a grid. This is the distance between scenes.
    )

    # For non-scene regions in the terrain, this is the minimal distance between humanoids if spaced out evenly in a grid fashion.
    # This replaces the "env-spacing" that is usually used in Isaac.
    minimal_humanoid_spacing: float = 1.0

    # We can save the terrain to a file. This is useful for debugging and for loading the terrain from a file.
    terrain_path: Optional[str] = None
    load_terrain: bool = False
    save_terrain: bool = False

    # Simulation properties (friction, restitution, height offset)
    sim_config: TerrainSimConfig = field(default_factory=TerrainSimConfig)

    def __post_init__(self):
        if self.terrain_obs_num_samples is None:
            # The observation model for the terrain will observe the terrain as a 2D grid of height values.
            # The samples are spaced out on a grid with spacing sample_width*horizontal_scale.
            # On each axis we have num_samples_per_axis samples.
            self.terrain_obs_num_samples = self.num_samples_per_axis**2


@dataclass
class ComplexTerrainConfig(TerrainConfig):
    num_terrains: int = 7
    num_levels: int = 7
    terrain_proportions: List[float] = field(
        default_factory=lambda: [0.2, 0.1, 0.1, 0.1, 0.05, 0.0, 0.0, 0.45]
    )
    minimal_humanoid_spacing: float = 0

    # minimal_humanoid_spacing: float = 1.0

    # terrain_proportions: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4])
    # slope_threshold: float = 0.4
