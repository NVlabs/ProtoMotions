# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
from protomotions.simulator.isaaclab.config import IsaacLabSimParams
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class Soma23RobotConfig(RobotConfig):
    """
    23-body humanoid derived from SOMASkeleton30 by dropping 7 leaf end-effector
    bodies that have no actuators:
        Jaw, LeftEye, RightEye,
        LeftHandThumbEnd, LeftHandMiddleEnd,
        RightHandThumbEnd, RightHandMiddleEnd

    Remaining 23 bodies (22 joints × 3-DOF hinge = 66 DOFs):
        Hips (root), Spine1, Spine2, Chest,
        Neck1, Neck2, Head,
        RightShoulder, RightArm, RightForeArm, RightHand,
        LeftShoulder, LeftArm, LeftForeArm, LeftHand,
        RightLeg, RightShin, RightFoot, RightToeBase,
        LeftLeg, LeftShin, LeftFoot, LeftToeBase

    See mjcf/soma23_humanoid.xml for definition.
    """

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["LeftFoot", "LeftToeBase"],
            "all_right_foot_bodies": ["RightFoot", "RightToeBase"],
            "all_left_hand_bodies": ["LeftHand"],
            "all_right_hand_bodies": ["RightHand"],
            "head_body_name": ["Head"],
            "torso_body_name": ["Chest"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "Chest",
            "Head",
            "LeftFoot",
            "RightFoot",
            "LeftHand",
            "RightHand",
        ]
    )

    default_root_height: float = 0.95

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/soma23_humanoid.xml",
            usd_asset_file_name="usd/soma23_humanoid_flat/soma23_humanoid_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/Hips/",
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                "(Spine|Chest|Neck|Head).*": ControlInfo(
                    effort_limit=300, velocity_limit=100
                ),
                ".*(Shoulder|Arm|Hand)_.*": ControlInfo(
                    effort_limit=150, velocity_limit=100
                ),
                ".*(Leg|Shin|Foot|ToeBase)_.*": ControlInfo(
                    effort_limit=300, velocity_limit=100
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
