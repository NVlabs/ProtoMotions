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
from dataclasses import dataclass
from typing import Literal, Tuple, List, Dict, Optional, Any
import torch
import re

from protomotions.utils.config_builder import ConfigBuilder


def get_matching_indices(
    names: List[str],
    names_to_match: Optional[List[str]] = None,
    indices_to_match: Optional[List[int]] = None,
) -> List[int]:
    """
    Get the indices of the names that match the given names or indices.

    Args:
        names: List of all available names
        names_to_match: List of regex patterns to match against names
        indices_to_match: List of indices to return directly

    Returns:
        List of indices where names match the regex patterns
    """
    assert (
        names_to_match is not None or indices_to_match is not None
    ), "Either names_to_match or indices_to_match must be provided"
    assert (
        names_to_match is None or indices_to_match is None
    ), "Only one of names_to_match or indices_to_match must be provided"

    if names_to_match is not None:
        # Set to store unique matching names (avoid duplicates from multiple regex)
        matching_names = set()

        # Go over all regex patterns
        for regex_pattern in names_to_match:
            # Find all names that match the current regex
            for i, name in enumerate(names):
                if re.fullmatch(regex_pattern, name):
                    assert (
                        i not in matching_names
                    ), f"Multiple regex patterns match the same name {name}"
                    matching_names.add(i)

        # Get indices for all unique matching names
        return list(matching_names)

    return indices_to_match


@dataclass
class MarkerConfig(ConfigBuilder):
    """Configuration for a single marker instance."""

    size: Literal["tiny", "small", "regular"]


@dataclass
class VisualizationMarkerConfig(ConfigBuilder):
    """Configuration for a group of visualization markers."""

    type: Literal["sphere", "arrow"]
    color: Tuple[float, float, float]  # RGB values
    markers: List[MarkerConfig]


@dataclass
class MarkerState(ConfigBuilder):
    """Represents the state of a marker in 3D space."""

    translation: torch.Tensor  # Translation vector (position)
    orientation: torch.Tensor  # Orientation quaternion
    color: Optional[Tuple[float, float, float]] = None  # RGB values


@dataclass
class ActionNoiseDomainRandomizationConfig(ConfigBuilder):
    """Configuration for action noise."""

    action_noise_range: Tuple[float, float]
    dof_names: Optional[List[str]] = None
    dof_indices: Optional[List[int]] = None

    def __post_init__(self):
        """Validate that dof_names and dof_indices are not both provided."""
        if self.dof_names is not None and self.dof_indices is not None:
            raise ValueError("Only one of dof_names or dof_indices must be provided.")
        if self.dof_names is None and self.dof_indices is None:
            raise ValueError("Either dof_names or dof_indices must be provided.")
        if self.action_noise_range is None:
            raise ValueError("action_noise_range must be provided.")
        if self.action_noise_range[0] >= self.action_noise_range[1]:
            raise ValueError(
                "action_noise_range must be a tuple of two values where the first value is less than the second value."
            )


@dataclass
class FrictionDomainRandomizationConfig(ConfigBuilder):
    """Configuration for friction."""

    num_buckets: int
    static_friction_range: Tuple[float, float]
    dynamic_friction_range: Tuple[float, float]
    restitution_range: Tuple[float, float]
    body_names: Optional[List[str]] = None
    body_indices: Optional[List[int]] = None

    def __post_init__(self):
        """Validate that body_names and body_indices are not both provided."""
        if self.body_names is not None and self.body_indices is not None:
            raise ValueError("Only one of body_names or body_indices must be provided.")
        if self.body_names is None and self.body_indices is None:
            raise ValueError("Either body_names or body_indices must be provided.")


@dataclass
class CenterOfMassDomainRandomizationConfig(ConfigBuilder):
    """Configuration for center of mass."""

    com_range: Dict[str, Tuple[float, float]]
    body_names: Optional[List[str]] = None
    body_indices: Optional[List[int]] = None

    def __post_init__(self):
        """Validate that com_range is a dictionary with valid keys."""
        if self.com_range is None:
            raise ValueError("com_range must be a dictionary with valid keys.")
        if not all(key in ["x", "y", "z"] for key in self.com_range.keys()):
            raise ValueError("com_range must be a dictionary with valid keys.")
        if self.body_names is None and self.body_indices is None:
            raise ValueError("Either body_names or body_indices must be provided.")
        if self.body_names is not None and self.body_indices is not None:
            raise ValueError("Only one of body_names or body_indices must be provided.")


@dataclass
class DomainRandomizationConfig(ConfigBuilder):
    """Configuration for domain randomization."""

    action_noise: Optional[ActionNoiseDomainRandomizationConfig] = None
    friction: Optional[FrictionDomainRandomizationConfig] = None
    center_of_mass: Optional[CenterOfMassDomainRandomizationConfig] = None


@dataclass
class SimParams(ConfigBuilder):
    """Configuration for core simulation parameters."""

    fps: int
    decimation: int


@dataclass
class SimulatorConfig(ConfigBuilder):
    """Main configuration class for the simulator."""

    _target_: str = None  # Path to the simulator class
    w_last: bool = None  # quaternion format (xyzw vs wxyz)
    headless: bool = None
    num_envs: int = None
    sim: SimParams = None
    experiment_name: str = None
    camera: Optional[Any] = None
    record_viewer: bool = False
    viewer_record_dir: str = "output/recordings/viewer"
    domain_randomization: Optional[DomainRandomizationConfig] = None
    """Domain randomization configuration.

    Both IsaacGym and IsaacLab use "average" friction combine mode (PhysX default).
    The effective friction between robot and terrain is computed as:
        effective_friction = (robot_friction + terrain_friction) / 2

    For friction randomization, the effective range will be:
        effective_min = (friction_range_min + terrain.sim_config.static_friction) / 2
        effective_max = (friction_range_max + terrain.sim_config.static_friction) / 2

    Example: With terrain.sim_config.static_friction=1.0 and robot range [0.0, 2.2]:
        - Effective friction range: [0.5, 1.6]
    """

    def __post_init__(self):
        assert self._target_ is not None, "SimulatorConfig._target_ must be provided"
        assert self.w_last is not None, "SimulatorConfig.w_last must be provided"
        assert self.headless is not None, "SimulatorConfig.headless must be provided"
        assert self.num_envs is not None, "SimulatorConfig.num_envs must be provided"
        assert self.sim is not None, "SimulatorConfig.sim must be provided"
        assert (
            self.experiment_name is not None
        ), "SimulatorConfig.experiment_name must be provided"


@dataclass
class SimBodyOrdering(ConfigBuilder):
    """Configuration for the ordering of bodies in the simulation."""

    body_names: List[str]
    dof_names: List[str]
