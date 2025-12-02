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
"""Base robot configuration classes.

This module defines the configuration dataclasses for robot morphology, control
parameters, and simulator-specific settings. All robot configurations (SMPL, G1, etc.)
inherit from RobotConfig and specify their robot-specific parameters.

Key Classes:
    - RobotConfig: Main robot configuration
    - RobotAssetConfig: Robot asset and physics properties
    - RobotControlConfig: PD control parameters
    - ControlType: Enum for control modes

Key Features:
    - Multi-simulator support
    - Configurable PD gains and action scaling
    - Asset file management
    - Body name mappings for observations
    - Kinematic structure extraction
"""

from protomotions.utils.config_builder import ConfigBuilder
from protomotions.components.pose_lib import ControlInfo, KinematicInfo
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import IsaacLabSimParams
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams

from typing import List, Optional, Dict, Union
from enum import Enum
from dataclasses import field, dataclass
import os
import torch


@dataclass
class SimulatorParams(ConfigBuilder):
    isaacgym: IsaacGymSimParams = field(default_factory=IsaacGymSimParams)
    isaaclab: IsaacLabSimParams = field(default_factory=IsaacLabSimParams)
    genesis: GenesisSimParams = field(default_factory=GenesisSimParams)
    newton: NewtonSimParams = field(default_factory=NewtonSimParams)


@dataclass
class InitState(ConfigBuilder):
    """Configuration for robot initial state."""

    pos: Optional[List[float]]  # [x, y, z] in meters


class ControlType(Enum):
    """Enum defining the available control types for the robot.

    BUILT_IN_PD: Built-in PD controller (e.g. Isaac Gym's internal PD controller)
    VELOCITY: Velocity control using custom PD controller
    TORQUE: Direct torque control
    PROPORTIONAL: Proportional control using custom PD controller
    """

    BUILT_IN_PD = "built_in_pd"
    TORQUE = "torque"
    PROPORTIONAL = "proportional"

    @classmethod
    def from_str(cls, value: str) -> "ControlType":
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
class RobotAssetConfig(ConfigBuilder):
    """Configuration for robot asset properties."""

    # Optional fields with defaults
    asset_root: str = "protomotions/data/assets"
    self_collisions: bool = True

    # Optional fields
    asset_file_name: str = None
    usd_asset_file_name: str = None
    usd_bodies_root_prim_path: str = None
    replace_cylinder_with_capsule: Optional[bool] = None
    thickness: Optional[float] = None
    max_angular_velocity: Optional[float] = None
    max_linear_velocity: Optional[float] = None
    density: Optional[float] = None
    angular_damping: Optional[float] = None
    linear_damping: Optional[float] = None
    disable_gravity: Optional[bool] = None
    fix_base_link: Optional[bool] = None

    def __post_init__(self):
        """Validate that asset_file_name is set."""

        if not self.asset_file_name or not self.asset_file_name.endswith(".xml"):
            raise ValueError(
                f"RobotAssetConfig.asset_file_name ('{self.asset_file_name}') "
                f"must be a valid path to an .xml MJCF file to extract kinematic info. "
                "if you are using URDF, convert it to MJCF first"
            )


@dataclass
class ControlConfig(ConfigBuilder):
    """Configuration for robot control parameters."""

    # Control info overrides for specific joints instead of the values from the MJCF asset
    override_control_info: Optional[Dict[str, ControlInfo]] = None

    # Can be "built_in_pd" or "proportional"/"velocity"/"torque" for Proportional, Velocity, Torque control
    control_type: ControlType = ControlType.BUILT_IN_PD
    # If control_type is proportional, this defines the scale beyond the pd-range.
    # so that motor does not lose strength as it approaches the joint limits
    # ref. build_pd_action_offset_scale()
    action_scale: float = 1.0

    # clamps the actions to be within [-clamp_actions, clamp_actions]
    # the clamped actions are provided to the simulator
    clamp_actions: float = 1.0

    # the positional limits used for rewards
    soft_pos_limit: float = 0.9

    # The following field is loaded post-init and populated from the MJCF asset
    control_info: Dict[str, ControlInfo] = field(init=False)

    def __post_init__(self):
        """Validate that override_control_info is a dictionary."""
        if self.override_control_info is not None:
            override_control_info = {}
            for key, value in self.override_control_info.items():
                if isinstance(value, dict):
                    override_control_info[key] = ControlInfo.from_dict(value)
                else:
                    override_control_info[key] = value
            self.override_control_info = override_control_info

    def initialize_control_info(self, asset: RobotAssetConfig):
        """Initialize control info from asset configuration."""
        if not hasattr(self, "control_info"):
            from protomotions.components.pose_lib import extract_control_info

            self.control_info = extract_control_info(
                mjcf_path=os.path.join(asset.asset_root, asset.asset_file_name),
                override_control_info=self.override_control_info,
            )


@dataclass
class RobotConfig(ConfigBuilder):
    """Configuration for robot morphology and control parameters.

    Defines all robot-specific parameters including asset files, body names,
    control settings, and physical properties. Each robot (SMPL, G1, etc.)
    should subclass this and provide robot-specific values.

    Key configuration areas:

    - **Asset**: Robot mesh/URDF files and physical properties
    - **Body Names**: Named body parts for tracking and observations
    - **Control**: PD gains, action scales, and control modes
    - **Kinematic Info**: Joint structure extracted from MJCF/URDF
    - **Simulator Params**: Simulator-specific overrides

    Example::

        config = RobotConfig(
            asset=RobotAssetConfig(asset_file_name="robot.xml"),
            control=ControlConfig(control_type=ControlType.BUILT_IN_PD)
        )
    """

    # Required nested config
    asset: RobotAssetConfig

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {}
    )

    default_root_height: float = 1
    default_dof_pos: Optional[Union[List[float], torch.Tensor]] = (
        None  # Default joint positions for resets (if None, uses zeros)
    )

    contact_bodies: Optional[Union[List[str], str]] = (
        None  # "all" means all bodies. Contact sensors are expensive to simulate, so we only spawn them when required.
    )
    trackable_bodies_subset: Union[List[str], str] = "all"

    non_termination_contact_bodies: Union[List[str], str] = "all"
    init_state: Optional[InitState] = None
    mimic_small_marker_bodies: Optional[List[str]] = None

    # Optional with default
    contact_pairs_multiplier: int = 16
    control: ControlConfig = field(default_factory=ControlConfig)

    # The following fields are loaded post-init and populated from the MJCF asset
    kinematic_info: KinematicInfo = field(init=False)
    number_of_actions: int = field(init=False)

    # Dictionary of simulator-specific simulation parameters
    simulation_params: SimulatorParams = field(default_factory=SimulatorParams)

    def __post_init__(self):
        """Compute derived fields after initialization."""

        from protomotions.components.pose_lib import extract_kinematic_info

        self.kinematic_info = extract_kinematic_info(
            os.path.join(self.asset.asset_root, self.asset.asset_file_name)
        )

        # Initialize control info in the control config
        self.control.initialize_control_info(self.asset)

        self.number_of_actions = self.kinematic_info.num_dofs

        # Initialize default_dof_pos: use provided values or zeros
        if self.default_dof_pos is None:
            # Default to zero joint positions
            self.default_dof_pos = torch.zeros(
                self.kinematic_info.num_dofs, dtype=torch.float32
            )
        else:
            # Convert to tensor if needed and validate
            if not isinstance(self.default_dof_pos, torch.Tensor):
                self.default_dof_pos = torch.tensor(
                    self.default_dof_pos, dtype=torch.float32
                )
            assert (
                len(self.default_dof_pos) == self.kinematic_info.num_dofs
            ), f"default_dof_pos length {len(self.default_dof_pos)} != num_dofs {self.kinematic_info.num_dofs}"

        self.mimic_small_marker_bodies = abstract_names_to_body_names(
            self.mimic_small_marker_bodies, self
        )
        self.contact_bodies = abstract_names_to_body_names(self.contact_bodies, self)
        self.trackable_bodies_subset = abstract_names_to_body_names(
            self.trackable_bodies_subset, self
        )

        required_abstract_names = [
            "all_left_foot_bodies",
            "all_right_foot_bodies",
            "all_left_hand_bodies",
            "all_right_hand_bodies",
            "head_body_name",
            "torso_body_name",
        ]
        for name in required_abstract_names:
            assert (
                name in self.common_naming_to_robot_body_names.keys()
            ), f"RobotConfig.common_naming_to_robot_body_names must contain {name}"

    def update_fields(self, **kwargs):
        """Update robot config fields and reprocess derived fields.

        This is useful when you need to modify fields after initialization
        without manually calling __post_init__.

        Args:
            **kwargs: Fields to update on the robot config
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"RobotConfig has no field '{key}'")
            setattr(self, key, value)

        # Reprocess fields that depend on the updated values
        self.mimic_small_marker_bodies = abstract_names_to_body_names(
            self.mimic_small_marker_bodies, self
        )
        self.contact_bodies = abstract_names_to_body_names(self.contact_bodies, self)
        self.trackable_bodies_subset = abstract_names_to_body_names(
            self.trackable_bodies_subset, self
        )


def abstract_names_to_body_names(
    names: Union[List[str], str], robot_config: RobotConfig
):
    if names is None:
        return None

    if names == "all":
        return robot_config.kinematic_info.body_names

    if isinstance(names, list):
        parsed_names = []
        for name in names:
            parsed_names.extend(abstract_names_to_body_names(name, robot_config))
        return parsed_names

    if names == "root":
        return [robot_config.kinematic_info.body_names[0]]
    elif names in robot_config.common_naming_to_robot_body_names.keys():
        return robot_config.common_naming_to_robot_body_names[names]
    else:
        return [names]
