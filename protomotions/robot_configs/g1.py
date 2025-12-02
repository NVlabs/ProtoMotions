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


# Parameters from BeyondMimic (https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/robots/g1.py)
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


@dataclass
class G1RobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["left_ankle_roll_link"],
            "all_right_foot_bodies": ["right_ankle_roll_link"],
            "all_left_hand_bodies": ["left_rubber_hand"],
            "all_right_hand_bodies": ["right_rubber_hand"],
            "head_body_name": ["head"],
            "torso_body_name": ["torso_link"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "torso_link",
            "head",
            "right_ankle_roll_link",
            "left_ankle_roll_link",
            "left_rubber_hand",
            "right_rubber_hand",
        ]
    )

    default_root_height: float = 0.8

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/g1_bm_box_feet.xml",
            usd_asset_file_name="usd/g1_bm_box_feet/g1_bm_box_feet.usda",
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
                ".*_hip_(pitch|yaw)_joint": ControlInfo(
                    stiffness=STIFFNESS_7520_14,
                    damping=DAMPING_7520_14,
                    effort_limit=88,
                    velocity_limit=32,
                    armature=ARMATURE_7520_14,
                ),
                ".*_hip_roll_joint": ControlInfo(
                    stiffness=STIFFNESS_7520_22,
                    damping=DAMPING_7520_22,
                    effort_limit=139,
                    velocity_limit=20,
                    armature=ARMATURE_7520_22,
                ),
                ".*_knee_joint": ControlInfo(
                    stiffness=STIFFNESS_7520_22,
                    damping=DAMPING_7520_22,
                    effort_limit=139,
                    velocity_limit=20,
                    armature=ARMATURE_7520_22,
                ),
                ".*_ankle_.*": ControlInfo(
                    stiffness=2 * STIFFNESS_5020,
                    damping=2 * DAMPING_5020,
                    effort_limit=50,
                    velocity_limit=37,
                    armature=2 * ARMATURE_5020,
                ),
                "waist_yaw_joint": ControlInfo(
                    stiffness=STIFFNESS_7520_14,
                    damping=DAMPING_7520_14,
                    effort_limit=88,
                    velocity_limit=32,
                    armature=ARMATURE_7520_14,
                ),
                "waist_(roll|pitch)_joint": ControlInfo(
                    stiffness=2.0 * STIFFNESS_5020,
                    damping=2.0 * DAMPING_5020,
                    effort_limit=50,
                    velocity_limit=37,
                    armature=2.0 * ARMATURE_5020,
                ),
                ".*_(shoulder|elbow)_.*": ControlInfo(
                    stiffness=STIFFNESS_5020,
                    damping=DAMPING_5020,
                    effort_limit=25,
                    velocity_limit=37,
                    armature=ARMATURE_5020,
                ),
                ".*_wrist_roll_joint": ControlInfo(
                    stiffness=STIFFNESS_5020,
                    damping=DAMPING_5020,
                    effort_limit=25,
                    velocity_limit=37,
                    armature=ARMATURE_5020,
                ),
                ".*_wrist_pitch_joint": ControlInfo(
                    stiffness=STIFFNESS_4010,
                    damping=DAMPING_4010,
                    effort_limit=5,
                    velocity_limit=22,
                    armature=ARMATURE_4010,
                ),
                ".*_wrist_yaw_joint": ControlInfo(
                    stiffness=STIFFNESS_4010,
                    damping=DAMPING_4010,
                    effort_limit=5,
                    velocity_limit=22,
                    armature=ARMATURE_4010,
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
