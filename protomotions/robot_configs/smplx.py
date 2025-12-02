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
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import (
    IsaacLabSimParams,
    IsaacLabPhysXParams,
)
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class SMPLXRobotConfig(RobotConfig):
    non_termination_contact_bodies: List[str] = field(
        default_factory=lambda: ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"]
    )

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["L_Ankle", "L_Toe"],
            "all_right_foot_bodies": ["R_Ankle", "R_Toe"],
            "all_left_hand_bodies": [
                "L_Wrist",
                "L_Index1",
                "L_Index2",
                "L_Index3",
                "L_Middle1",
                "L_Middle2",
                "L_Middle3",
                "L_Pinky1",
                "L_Pinky2",
                "L_Pinky3",
                "L_Ring1",
                "L_Ring2",
                "L_Ring3",
                "L_Thumb1",
                "L_Thumb2",
                "L_Thumb3",
            ],
            "all_right_hand_bodies": [
                "R_Wrist",
                "R_Index1",
                "R_Index2",
                "R_Index3",
                "R_Middle1",
                "R_Middle2",
                "R_Middle3",
                "R_Pinky1",
                "R_Pinky2",
                "R_Pinky3",
                "R_Ring1",
                "R_Ring2",
                "R_Ring3",
                "R_Thumb1",
                "R_Thumb2",
                "R_Thumb3",
            ],
            "head_body_name": ["Head"],
            "torso_body_name": ["Torso"],
        }
    )

    default_root_height: float = 0.95

    # SMPLX-specific configuration
    contact_bodies: List[str] = field(
        default_factory=lambda: [
            "L_Index1",
            "L_Index2",
            "L_Index3",
            "L_Middle1",
            "L_Middle2",
            "L_Middle3",
            "L_Pinky1",
            "L_Pinky2",
            "L_Pinky3",
            "L_Ring1",
            "L_Ring2",
            "L_Ring3",
            "L_Thumb1",
            "L_Thumb2",
            "L_Thumb3",
            "R_Index1",
            "R_Index2",
            "R_Index3",
            "R_Middle1",
            "R_Middle2",
            "R_Middle3",
            "R_Pinky1",
            "R_Pinky2",
            "R_Pinky3",
            "R_Ring1",
            "R_Ring2",
            "R_Ring3",
            "R_Thumb1",
            "R_Thumb2",
            "R_Thumb3",
        ]
    )

    contact_pairs_multiplier: int = 20

    mimic_small_marker_bodies: List[str] = field(
        default_factory=lambda: [
            "L_Wrist",
            "R_Wrist",
            "L_Index1",
            "L_Index2",
            "L_Index3",
            "L_Middle1",
            "L_Middle2",
            "L_Middle3",
            "L_Pinky1",
            "L_Pinky2",
            "L_Pinky3",
            "L_Ring1",
            "L_Ring2",
            "L_Ring3",
            "L_Thumb1",
            "L_Thumb2",
            "L_Thumb3",
            "R_Index1",
            "R_Index2",
            "R_Index3",
            "R_Middle1",
            "R_Middle2",
            "R_Middle3",
            "R_Pinky1",
            "R_Pinky2",
            "R_Pinky3",
            "R_Ring1",
            "R_Ring2",
            "R_Ring3",
            "R_Thumb1",
            "R_Thumb2",
            "R_Thumb3",
        ]
    )

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/smplx_humanoid.xml",
            usd_asset_file_name="usd/smplx_humanoid.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                ".*_(Hip|Knee|Ankle)_.*": ControlInfo(
                    stiffness=800, damping=80, effort_limit=500, velocity_limit=100
                ),
                ".*_Toe_.*": ControlInfo(
                    stiffness=500, damping=50, effort_limit=500, velocity_limit=100
                ),
                "(Torso|Spine|Chest)_.*": ControlInfo(
                    stiffness=1000, damping=100, effort_limit=500, velocity_limit=100
                ),
                "(Neck|Head|.*_Thorax|.*_Shoulder)_.*": ControlInfo(
                    stiffness=500, damping=50, effort_limit=500, velocity_limit=100
                ),
                ".*_(Elbow|Wrist)_.*": ControlInfo(
                    stiffness=300, damping=30, effort_limit=300, velocity_limit=100
                ),
                ".*_(Index|Middle|Pinky|Ring|Thumb)[123]_[xyz]": ControlInfo(
                    stiffness=10, damping=1, effort_limit=10, velocity_limit=5
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            isaaclab=IsaacLabSimParams(
                fps=120,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=4,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            genesis=GenesisSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=120,
                decimation=4,
            ),
        )
    )
