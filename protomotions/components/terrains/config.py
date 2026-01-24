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
"""Configuration classes for terrain generation and simulation properties."""

from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum


class CombineMode(Enum):
    """Physics material combine mode for friction/restitution."""
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
class TerrainSimConfig:
    """Configuration for terrain simulation properties (friction, restitution, height offset).

    These properties affect the physical behavior of the terrain in simulation.
    Separate from TerrainConfig which defines terrain geometry.
    """

    static_friction: float = field(
        default=1.0,
        metadata={"help": "Static friction coefficient.", "min": 0.0}
    )
    dynamic_friction: float = field(
        default=1.0,
        metadata={"help": "Dynamic friction coefficient.", "min": 0.0}
    )
    restitution: float = field(
        default=0.0,
        metadata={"help": "Restitution (bounciness) coefficient.", "min": 0.0, "max": 1.0}
    )
    height_offset: float = field(
        default=0.0,
        metadata={"help": "Height offset for terrain (negative = lower)."}
    )
    combine_mode: CombineMode = field(
        default=CombineMode.AVERAGE,
        metadata={"help": "How to combine friction values between objects."}
    )


@dataclass
class TerrainConfig:
    """Configuration for terrain generation.
    
    Defines terrain geometry, procedural generation parameters, and simulation properties.
    """

    _target_: str = "protomotions.components.terrains.terrain.Terrain"
    map_length: float = field(
        default=20.0,
        metadata={"help": "Length of terrain map in meters.", "min": 1.0}
    )
    map_width: float = field(
        default=20.0,
        metadata={"help": "Width of terrain map in meters.", "min": 1.0}
    )
    border_size: float = field(
        default=40.0,
        metadata={"help": "Border size to ensure space from edges.", "min": 0.0}
    )
    num_levels: int = field(
        default=10,
        metadata={"help": "Number of difficulty levels for curriculum.", "min": 1}
    )
    num_terrains: int = field(
        default=10,
        metadata={"help": "Number of terrain variations to generate.", "min": 1}
    )

    terrain_proportions: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        metadata={"help": "Proportions: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping, poles, flat]"}
    )
    slope_threshold: float = field(
        default=0.9,
        metadata={"help": "Maximum slope angle threshold.", "min": 0.0, "max": 1.0}
    )
    num_samples_per_axis: int = field(
        default=16,
        metadata={"help": "Samples per axis for height observation.", "min": 1}
    )
    sample_width: float = field(
        default=1.0,
        metadata={"help": "Width between sample points in meters.", "min": 0.01}
    )
    terrain_obs_num_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Total terrain observation samples. Auto-computed if None."}
    )

    horizontal_scale: float = field(
        default=0.1,
        metadata={"help": "Horizontal resolution scale.", "min": 0.001}
    )
    vertical_scale: float = field(
        default=0.005,
        metadata={"help": "Vertical resolution scale.", "min": 0.001}
    )

    spacing_between_scenes: float = field(
        default=10.0,
        metadata={"help": "Distance between scenes in grid layout.", "min": 0.0}
    )

    minimal_humanoid_spacing: float = field(
        default=1.0,
        metadata={"help": "Minimum spacing between humanoids in non-scene regions.", "min": 0.0}
    )

    terrain_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save/load terrain file."}
    )
    load_terrain: bool = field(
        default=False,
        metadata={"help": "Load terrain from file instead of generating."}
    )
    save_terrain: bool = field(
        default=False,
        metadata={"help": "Save generated terrain to file."}
    )

    sim_config: TerrainSimConfig = field(
        default_factory=TerrainSimConfig,
        metadata={"help": "Simulation properties (friction, restitution)."}
    )

    def __post_init__(self):
        if self.terrain_obs_num_samples is None:
            self.terrain_obs_num_samples = self.num_samples_per_axis**2


@dataclass
class ComplexTerrainConfig(TerrainConfig):
    """Configuration for complex procedural terrain."""
    
    num_terrains: int = field(
        default=7,
        metadata={"help": "Number of terrain variations.", "min": 1}
    )
    num_levels: int = field(
        default=7,
        metadata={"help": "Number of difficulty levels.", "min": 1}
    )
    terrain_proportions: List[float] = field(
        default_factory=lambda: [0.2, 0.1, 0.1, 0.1, 0.05, 0.0, 0.0, 0.45],
        metadata={"help": "Proportions for different terrain types."}
    )
    minimal_humanoid_spacing: float = field(
        default=0.0,
        metadata={"help": "Minimum spacing between humanoids.", "min": 0.0}
    )
