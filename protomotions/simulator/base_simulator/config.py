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
"""Configuration classes for base simulator and domain randomization."""

from typing import Literal, Tuple, List, Dict, Optional, Any
import torch
import re
from dataclasses import dataclass, field


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
class MarkerConfig:
    """Configuration for a single marker instance."""

    size: Literal["tiny", "small", "regular"] = field(
        default="regular",
        metadata={"help": "Marker size for visualization."}
    )


@dataclass
class VisualizationMarkerConfig:
    """Configuration for a group of visualization markers."""

    type: Literal["sphere", "arrow"] = field(
        default="sphere",
        metadata={"help": "Marker geometry type."}
    )
    color: Tuple[float, float, float] = field(
        default=(1.0, 0.0, 0.0),
        metadata={"help": "RGB color values (0-1)."}
    )
    markers: List[MarkerConfig] = field(
        default_factory=list,
        metadata={"help": "List of marker configurations."}
    )


@dataclass
class MarkerState:
    """Represents the state of a marker in 3D space."""

    translation: torch.Tensor = field(
        default=None,
        metadata={"help": "Translation vector (position)."}
    )
    orientation: torch.Tensor = field(
        default=None,
        metadata={"help": "Orientation quaternion."}
    )
    color: Optional[Tuple[float, float, float]] = field(
        default=None,
        metadata={"help": "Optional RGB color override."}
    )


@dataclass
class ActionNoiseDomainRandomizationConfig:
    """Configuration for action noise domain randomization."""

    action_noise_range: Tuple[float, float] = field(
        default=None,
        metadata={"help": "Range (min, max) for action noise."}
    )
    dof_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "DOF names to apply noise to (regex patterns)."}
    )
    dof_indices: Optional[List[int]] = field(
        default=None,
        metadata={"help": "DOF indices to apply noise to."}
    )

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
class FrictionDomainRandomizationConfig:
    """Configuration for friction domain randomization."""

    num_buckets: int = field(
        default=10,
        metadata={"help": "Number of friction buckets for environments.", "min": 1}
    )
    static_friction_range: Tuple[float, float] = field(
        default=(0.5, 1.5),
        metadata={"help": "Range (min, max) for static friction."}
    )
    dynamic_friction_range: Tuple[float, float] = field(
        default=(0.5, 1.5),
        metadata={"help": "Range (min, max) for dynamic friction."}
    )
    restitution_range: Tuple[float, float] = field(
        default=(0.0, 0.1),
        metadata={"help": "Range (min, max) for restitution."}
    )
    body_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Body names to apply randomization to (regex patterns)."}
    )
    body_indices: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Body indices to apply randomization to."}
    )

    def __post_init__(self):
        """Validate that body_names and body_indices are not both provided."""
        if self.body_names is not None and self.body_indices is not None:
            raise ValueError("Only one of body_names or body_indices must be provided.")
        if self.body_names is None and self.body_indices is None:
            raise ValueError("Either body_names or body_indices must be provided.")


@dataclass
class CenterOfMassDomainRandomizationConfig:
    """Configuration for center of mass domain randomization."""

    com_range: Dict[str, Tuple[float, float]] = field(
        default_factory=dict,
        metadata={"help": "Range per axis: {'x': (-0.1, 0.1), 'y': (-0.1, 0.1), 'z': (-0.1, 0.1)}"}
    )
    body_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Body names to apply randomization to (regex patterns)."}
    )
    body_indices: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Body indices to apply randomization to."}
    )

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
class ObservationNoiseDomainRandomizationConfig:
    """Configuration for observation noise domain randomization.
    
    Adds Gaussian noise to observation variables to simulate sensor noise.
    When enabled, regular state variables have noise applied while
    privileged_* versions remain clean for asymmetric actor-critic training.
    
    Noise values are standard deviations for additive Gaussian noise.
    
    Noise is applied hierarchically:
    - DOF noise: applied to joint positions and velocities
    - Root noise: applied to root body orientation and angular velocity
    - Anchor noise: applied to anchor body orientation and angular velocity
    - Whole-body noise: applied to all rigid body positions, rotations, velocities
    
    Root and anchor noise are applied on top of clean (privileged) data,
    not on already-noisy whole-body data.
    """

    # DOF-level noise
    dof_pos_noise: float = field(
        default=0.0,
        metadata={"help": "Noise std for DOF positions (radians)."}
    )
    dof_vel_noise: float = field(
        default=0.0,
        metadata={"help": "Noise std for DOF velocities (rad/s)."}
    )
    
    # Root body noise
    root_rot_noise: float = field(
        default=0.0,
        metadata={"help": "Noise std for root orientation quaternion."}
    )
    root_ang_vel_noise: float = field(
        default=0.0,
        metadata={"help": "Noise std for root angular velocity (rad/s)."}
    )
    
    # Anchor body noise
    anchor_rot_noise: float = field(
        default=0.0,
        metadata={"help": "Noise std for anchor body orientation quaternion."}
    )
    anchor_ang_vel_noise: float = field(
        default=0.0,
        metadata={"help": "Noise std for anchor body angular velocity (rad/s)."}
    )
    
    # Whole-body noise (all rigid bodies)
    body_pos_noise: float = field(
        default=0.0,
        metadata={"help": "Noise std for all rigid body positions (meters)."}
    )
    body_rot_noise: float = field(
        default=0.0,
        metadata={"help": "Noise std for all rigid body orientations (quaternion)."}
    )
    body_vel_noise: float = field(
        default=0.0,
        metadata={"help": "Noise std for all rigid body linear velocities (m/s)."}
    )
    body_ang_vel_noise: float = field(
        default=0.0,
        metadata={"help": "Noise std for all rigid body angular velocities (rad/s)."}
    )
    
    # Environment observation noise
    ground_height_noise: float = field(
        default=0.0,
        metadata={"help": "Noise std for ground height observations (meters)."}
    )

    def has_noise(self) -> bool:
        """Check if any noise is configured."""
        return (
            self.dof_pos_noise > 0.0
            or self.dof_vel_noise > 0.0
            or self.root_rot_noise > 0.0
            or self.root_ang_vel_noise > 0.0
            or self.anchor_rot_noise > 0.0
            or self.anchor_ang_vel_noise > 0.0
            or self.body_pos_noise > 0.0
            or self.body_rot_noise > 0.0
            or self.body_vel_noise > 0.0
            or self.body_ang_vel_noise > 0.0
            or self.ground_height_noise > 0.0
        )


@dataclass
class PushDomainRandomizationConfig:
    """Configuration for push/perturbation domain randomization.
    
    Applies random velocity impulses to the robot at random intervals to
    simulate external disturbances (bumps, pushes) for sim-to-real transfer.
    
    Push velocities are sampled uniformly from [-max, +max] for each component.
    Push is enabled when any velocity component is non-zero.
    """
    
    push_interval_range: Tuple[float, float] = field(
        default=(1.0, 3.0),
        metadata={"help": "Range (min, max) in seconds between pushes."}
    )
    max_linear_velocity: Tuple[float, float, float] = field(
        default=(0.0, 0.0, 0.0),
        metadata={"help": "Max linear velocity impulse (x, y, z) in m/s."}
    )
    max_angular_velocity: Tuple[float, float, float] = field(
        default=(0.0, 0.0, 0.0),
        metadata={"help": "Max angular velocity impulse (roll, pitch, yaw) in rad/s."}
    )

    def __post_init__(self):
        if self.push_interval_range[0] <= 0 or self.push_interval_range[1] <= 0:
            raise ValueError("push_interval_range values must be positive.")
        if self.push_interval_range[0] > self.push_interval_range[1]:
            raise ValueError("push_interval_range[0] must be <= push_interval_range[1].")

    def has_push(self) -> bool:
        """Check if any push velocity is configured (non-zero)."""
        return (
            any(v != 0.0 for v in self.max_linear_velocity)
            or any(v != 0.0 for v in self.max_angular_velocity)
        )


@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization."""

    action_noise: Optional[ActionNoiseDomainRandomizationConfig] = field(
        default=None,
        metadata={"help": "Action noise configuration."}
    )
    friction: Optional[FrictionDomainRandomizationConfig] = field(
        default=None,
        metadata={"help": "Friction randomization configuration."}
    )
    center_of_mass: Optional[CenterOfMassDomainRandomizationConfig] = field(
        default=None,
        metadata={"help": "Center of mass randomization configuration."}
    )
    observation_noise: Optional[ObservationNoiseDomainRandomizationConfig] = field(
        default=None,
        metadata={"help": "Observation noise configuration for sim-to-real transfer."}
    )
    push: Optional[PushDomainRandomizationConfig] = field(
        default=None,
        metadata={"help": "Push/perturbation randomization for sim-to-real transfer."}
    )


@dataclass
class SimParams:
    """Configuration for core simulation parameters."""

    fps: int = field(
        default=60,
        metadata={"help": "Simulation frames per second.", "min": 1}
    )
    decimation: int = field(
        default=4,
        metadata={"help": "Number of physics steps per control step.", "min": 1}
    )


@dataclass
class SimulatorConfig:
    """Main configuration class for the simulator."""

    _target_: str = field(
        default=None,
        metadata={"help": "Path to the simulator class."}
    )
    w_last: bool = field(
        default=None,
        metadata={"help": "Quaternion format: True for xyzw, False for wxyz."}
    )
    headless: bool = field(
        default=None,
        metadata={"help": "Run without GUI visualization."}
    )
    num_envs: int = field(
        default=None,
        metadata={"help": "Number of parallel environments.", "min": 1}
    )
    sim: SimParams = field(
        default=None,
        metadata={"help": "Simulation parameters (fps, decimation)."}
    )
    experiment_name: str = field(
        default=None,
        metadata={"help": "Name for this experiment (used for logging)."}
    )
    camera: Optional[Any] = field(
        default=None,
        metadata={"help": "Camera configuration for rendering."}
    )
    record_viewer: bool = field(
        default=False,
        metadata={"help": "Record viewer output to video."}
    )
    viewer_record_dir: str = field(
        default="output/recordings/viewer",
        metadata={"help": "Directory for viewer recordings."}
    )
    domain_randomization: Optional[DomainRandomizationConfig] = field(
        default=None,
        metadata={"help": "Domain randomization configuration for sim-to-real transfer."}
    )

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
class SimBodyOrdering:
    """Configuration for the ordering of bodies in the simulation."""

    body_names: List[str] = field(
        default_factory=list,
        metadata={"help": "Ordered list of rigid body names."}
    )
    dof_names: List[str] = field(
        default_factory=list,
        metadata={"help": "Ordered list of DOF (joint) names."}
    )
