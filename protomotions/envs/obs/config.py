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
"""Configuration classes for observation components.

This module contains all configuration dataclasses for observation-related functionality,
co-located with the observation implementations in the same directory.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any, Union
from enum import Enum
from protomotions.utils.config_builder import ConfigBuilder
from protomotions.robot_configs.base import RobotConfig, abstract_names_to_body_names


@dataclass
class SelfObsConfig(ConfigBuilder):
    """Configuration for reduced coordinates self observations (DOF-based).

    Uses joint positions, velocities, root angular velocity, and projected gravity.
    More compact than max coords but loses some spatial information.
    """

    enabled: bool = False
    num_historical_steps: int = 1


@dataclass
class MaxCoordsSelfObsConfig(ConfigBuilder):
    """Configuration for maximal coordinates self observations (full body state).

    Uses full 3D positions, rotations, velocities, and angular velocities for all bodies.
    Provides rich spatial information but results in larger observation space.
    """

    enabled: bool = True
    num_historical_steps: int = 1
    local_obs: bool = True
    root_height_obs: bool = True
    observe_contacts: bool = False


@dataclass
class ActionHistoryConfig(ConfigBuilder):
    """Configuration for action history."""

    enabled: bool = False
    num_historical_steps: int = 1


@dataclass
class HumanoidObsConfig(ConfigBuilder):
    """Configuration for humanoid observations."""

    max_coords_obs: MaxCoordsSelfObsConfig = field(
        default_factory=MaxCoordsSelfObsConfig
    )
    reduced_coords_obs: SelfObsConfig = field(default_factory=SelfObsConfig)
    action_history: ActionHistoryConfig = field(default_factory=ActionHistoryConfig)


@dataclass
class SceneObsConfig(ConfigBuilder):
    """Configuration for scene observations."""

    enabled: bool = False
    obs_object_index: Optional[int] = None


@dataclass
class SteeringObsConfig(ConfigBuilder):
    """Configuration for steering task observations and parameters.

    The steering observation provides the target direction and speed in the robot's local frame.
    Observation is a 3D vector: [local_dir_x, local_dir_y, tar_speed].

    Also contains parameters for how the target heading and speed change during training.
    """

    enabled: bool = False

    # Heading change interval (in simulation steps)
    heading_change_steps_min: int = 40
    heading_change_steps_max: int = 150

    # Heading randomization
    random_heading_probability: float = (
        0.2  # Probability of fully random heading vs incremental change
    )
    standard_heading_change: float = (
        1.57  # Max incremental heading change (~90 degrees)
    )

    # Speed parameters
    tar_speed_min: float = 1.2
    tar_speed_max: float = 6.0
    standard_speed_change: float = 0.3  # Max incremental speed change

    # Stop behavior
    stop_probability: float = 0.05  # Probability of setting target speed to 0


@dataclass
class PathGeneratorConfig(ConfigBuilder):
    """Configuration for path generation in path following environment."""

    num_verts: int = 101
    dtheta_max: float = 2.0
    sharp_turn_prob: float = 0.02
    accel_max: float = 2.0
    speed_max: float = 5.0
    speed_min: float = 0.0
    fixed_path: bool = False
    slow: bool = False
    height_conditioned: bool = False
    start_speed_max: float = 1.0
    speed_z_max: float = 0.5
    accel_z_max: float = 0.2
    head_height_max: float = 1.5
    head_height_min: float = 0.4
    use_naive_path_generator: bool = False
    use_forward_path_only: bool = False


@dataclass
class PathObsConfig(ConfigBuilder):
    """Configuration for path following observations.

    The path observation provides future waypoints along the path in the robot's local frame.
    Each waypoint is a 2D or 3D offset from the robot's head position.

    Observation size = num_traj_samples * (2 if not height_conditioned else 3).
    """

    enabled: bool = False

    # Path sampling parameters
    num_traj_samples: int = 10  # Number of future waypoints to sample
    traj_sample_timestep: float = 0.5  # Time interval between samples (seconds)

    # Termination parameters
    enable_path_termination: bool = True
    fail_dist: float = 4.0  # Max horizontal distance from path before termination
    fail_height_dist: float = 0.5  # Max height difference from path before termination

    # Path generator config
    path_generator: PathGeneratorConfig = field(default_factory=PathGeneratorConfig)


@dataclass
class MimicPhaseObsConfig(ConfigBuilder):
    """Configuration for mimic phase observations."""

    enabled: bool = False


@dataclass
class MimicTimeLeftObsConfig(ConfigBuilder):
    """Configuration for mimic time left observations."""

    enabled: bool = False


class FuturePoseType(Enum):
    """Enum for specifying the type of future pose representation."""

    MAX_COORDS = "max-coords"
    MAX_COORDS_FUTURE_REL = "max-coords-future-rel"
    MAX_COORDS_SIMPLE = "max-coords-simple"

    @classmethod
    def from_str(cls, value: str) -> "FuturePoseType":
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


@dataclass
class MimicTargetPoseConfig(ConfigBuilder):
    """Configuration for mimic target pose observations."""

    enabled: bool = False
    type: FuturePoseType = FuturePoseType.MAX_COORDS
    with_time: bool = True
    with_contacts: bool = False
    with_velocities: bool = False
    # future_steps is either a list of future frames, or int.
    # when int, it creates a range 1, 2, ..., future_steps
    future_steps: Union[int, List[int]] = (
        1  # Accepts int (range of future frames) or list of ints (specifying which future frames to use)
    )
    # Automatically populated. Do not set.
    num_future_steps: Optional[int] = field(init=False)  # Computed from future_steps

    def __post_init__(self):
        self.num_future_steps = (
            len(self.future_steps)
            if isinstance(self.future_steps, list)
            else self.future_steps
        )


@dataclass
class JointMaskingConfig(ConfigBuilder):
    """Configuration for joint masking in masked mimic."""

    masked_mimic_fixed_conditioning: Optional[List[Any]] = (
        None  # A list of body-parts to condition on. When set, this will override the randomization of the body sampling and will define the visible bodies in each frame.
    )
    force_max_conditioned_bodies_prob: float = 0.1  # Probability of ensuring the maximum number of conditioned bodies is visible.
    force_small_num_conditioned_bodies_prob: float = 0.1  # Probability of ensuring the minimum number of conditioned bodies is visible.
    visible_target_pose_prob: float = 0.8
    masked_mimic_repeat_mask_probability: float = 0.8


@dataclass
class TimeSamplingConfig(ConfigBuilder):
    """Configuration for sequential time sampling in masked mimic."""

    # Beta distribution parameters for sequential sampling from last conditioned frame
    # Beta(alpha, beta) distribution bounded between 0 and 1, then scaled to remaining motion time
    # alpha=2, beta=5 favors samples closer to the last conditioned frame
    # Higher alpha favors later in remaining time, higher beta favors earlier in remaining time
    alpha: float = 2.0  # Shape parameter alpha for beta distribution
    beta: float = 5.0  # Shape parameter beta for beta distribution


@dataclass
class MaskedMimicMaskingConfig(ConfigBuilder):
    """Configuration for masked mimic masking."""

    joint_masking: JointMaskingConfig = field(default_factory=JointMaskingConfig)
    time_sampling: TimeSamplingConfig = field(default_factory=TimeSamplingConfig)


@dataclass
class MaskedMimicTargetPoseConfig(ConfigBuilder):
    """Configuration for masked mimic target poses."""

    num_future_steps: int = 5  # Number of future poses to condition on


@dataclass
class MaskedMimicHistoricalObsConfig(ConfigBuilder):
    """Configuration for masked mimic historical observations."""

    # We subsample from the agent's obs historical saved obs to condition on
    num_historical_conditioned_steps: int = (
        15  # Number of historical steps to condition on.
    )
    use_reduced_coords_obs: bool = False


@dataclass
class MaskedMimicObsConfig(ConfigBuilder):
    """Configuration for masked mimic functionality."""

    enabled: bool = False
    masked_mimic_masking: MaskedMimicMaskingConfig = field(
        default_factory=MaskedMimicMaskingConfig
    )
    masked_mimic_target_pose: MaskedMimicTargetPoseConfig = field(
        default_factory=MaskedMimicTargetPoseConfig
    )
    historical_obs: MaskedMimicHistoricalObsConfig = field(
        default_factory=MaskedMimicHistoricalObsConfig
    )


@dataclass
class MimicObsConfig(ConfigBuilder):
    """Configuration for mimic observations."""

    enabled: bool = False

    mimic_phase_obs: MimicPhaseObsConfig = field(default_factory=MimicPhaseObsConfig)
    mimic_time_left_obs: MimicTimeLeftObsConfig = field(
        default_factory=MimicTimeLeftObsConfig
    )
    mimic_target_pose: MimicTargetPoseConfig = field(
        default_factory=MimicTargetPoseConfig
    )
