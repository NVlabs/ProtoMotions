# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
class Rigv1RobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["LeftFoot", "LeftToeBase"],
            "all_right_foot_bodies": ["RightFoot", "RightToeBase"],
            "all_left_hand_bodies": ["LeftHand"],
            "all_right_hand_bodies": ["RightHand"],
            "head_body_name": ["Head"],
            "torso_body_name": ["Spine3"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "Spine3",
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
            asset_file_name="mjcf/rigv1_humanoid.xml",
            usd_asset_file_name="usd/rigv1_humanoid.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/bodies/",
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                "(Spine|Neck|Head).*": ControlInfo(
                    effort_limit=300, velocity_limit=100
                ),
                ".*(Shoulder|Arm|Hand)_.*": ControlInfo(
                    effort_limit=150, velocity_limit=100
                ),
                ".*(Leg|Foot|ToeBase)_.*": ControlInfo(
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
