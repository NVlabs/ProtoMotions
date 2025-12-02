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
from protomotions.robot_configs.base import (
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
    SimulatorParams,
)
from protomotions.simulator.isaacgym.config import (
    IsaacGymSimParams,
    IsaacGymPhysXParams,
)
from protomotions.simulator.isaaclab.config import (
    IsaacLabSimParams,
    IsaacLabPhysXParams,
)
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import List, Dict
from dataclasses import dataclass, field


# Armature constants for different joint types based on torque capabilities
ARMATURE_300 = 0.040  # High torque joints (knees - 300 N⋅m)
ARMATURE_200 = 0.030  # Medium-high torque joints (hips, torso - 200 N⋅m)
ARMATURE_60 = 0.010  # Medium torque joints (ankle pitch - 60 N⋅m)
ARMATURE_40 = (
    0.005  # Medium-low torque joints (shoulder pitch/roll, ankle roll - 40 N⋅m)
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_300 = ARMATURE_300 * NATURAL_FREQ**2
STIFFNESS_200 = ARMATURE_200 * NATURAL_FREQ**2
STIFFNESS_60 = ARMATURE_60 * NATURAL_FREQ**2
STIFFNESS_40 = ARMATURE_40 * NATURAL_FREQ**2

DAMPING_300 = 2.0 * DAMPING_RATIO * ARMATURE_300 * NATURAL_FREQ
DAMPING_200 = 2.0 * DAMPING_RATIO * ARMATURE_200 * NATURAL_FREQ
DAMPING_60 = 2.0 * DAMPING_RATIO * ARMATURE_60 * NATURAL_FREQ
DAMPING_40 = 2.0 * DAMPING_RATIO * ARMATURE_40 * NATURAL_FREQ


@dataclass
class H1_2RobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["left_ankle_roll_link"],
            "all_right_foot_bodies": ["right_ankle_roll_link"],
            "all_left_hand_bodies": ["left_wrist_yaw_link"],
            "all_right_hand_bodies": ["right_wrist_yaw_link"],
            "head_body_name": ["head_aux"],
            "torso_body_name": ["torso_link"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "torso_link",
            "head_aux",
            "right_ankle_roll_link",
            "left_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
    )

    default_root_height: float = 1.03

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/h1_2_box_feet.xml",
            usd_asset_file_name="usd/h1_2_box_feet/h1_2_box_feet.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/pelvis/",
            replace_cylinder_with_capsule=True,
            thickness=0.01,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            density=0.001,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                # Hip joints (200 N⋅m, 23 rad/s)
                ".*_hip_(yaw|pitch|roll)_joint": ControlInfo(
                    stiffness=STIFFNESS_200,
                    damping=DAMPING_200,
                    effort_limit=200,
                    velocity_limit=50,
                    armature=ARMATURE_200,
                ),
                # Knee joints (300 N⋅m, 14 rad/s)
                ".*_knee_joint": ControlInfo(
                    stiffness=STIFFNESS_300,
                    damping=DAMPING_300,
                    effort_limit=300,
                    velocity_limit=50,
                    armature=ARMATURE_300,
                ),
                # Ankle pitch joints (60 N⋅m, 9 rad/s)
                ".*_ankle_pitch_joint": ControlInfo(
                    stiffness=2 * STIFFNESS_60,
                    damping=2 * DAMPING_60,
                    effort_limit=60,
                    velocity_limit=50,
                    armature=2 * ARMATURE_60,
                ),
                # Ankle roll joints (40 N⋅m, 9 rad/s)
                ".*_ankle_roll_joint": ControlInfo(
                    stiffness=2 * STIFFNESS_40,
                    damping=2 * DAMPING_40,
                    effort_limit=40,
                    velocity_limit=50,
                    armature=2 * ARMATURE_40,
                ),
                # Torso joint (200 N⋅m, 23 rad/s)
                "torso_joint": ControlInfo(
                    stiffness=STIFFNESS_200,
                    damping=DAMPING_200,
                    effort_limit=200,
                    velocity_limit=50,
                    armature=ARMATURE_200,
                ),
                # Shoulder pitch/roll joints (40 N⋅m, 9 rad/s)
                ".*_shoulder_(pitch|roll)_joint": ControlInfo(
                    stiffness=STIFFNESS_40,
                    damping=DAMPING_40,
                    effort_limit=40,
                    velocity_limit=50,
                    armature=ARMATURE_40,
                ),
                # Shoulder yaw joints (18 N⋅m, 20 rad/s)
                ".*_shoulder_yaw_joint": ControlInfo(
                    stiffness=STIFFNESS_40,
                    damping=DAMPING_40,
                    effort_limit=18,
                    velocity_limit=50,
                    armature=ARMATURE_40,
                ),
                # Elbow joints (18 N⋅m, 20 rad/s)
                ".*_elbow_joint": ControlInfo(
                    stiffness=STIFFNESS_40,
                    damping=DAMPING_40,
                    effort_limit=18,
                    velocity_limit=50,
                    armature=ARMATURE_40,
                ),
                # Wrist joints (19 N⋅m, 31.4 rad/s)
                ".*_wrist_(roll|pitch|yaw)_joint": ControlInfo(
                    stiffness=STIFFNESS_40,
                    damping=DAMPING_40,
                    effort_limit=19,
                    velocity_limit=50,
                    armature=ARMATURE_40,
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=100,
                decimation=2,
                substeps=2,
                physx=IsaacGymPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            isaaclab=IsaacLabSimParams(
                fps=200,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            genesis=GenesisSimParams(
                fps=100,
                decimation=2,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=200,
                decimation=4,
            ),
        )
    )
