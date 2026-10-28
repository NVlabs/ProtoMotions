# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.robot_configs.base import (
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
)
from protomotions.robot_configs.motor_classes import (
    H1_2_300,
    H1_2_200,
    H1_2_60,
    H1_2_40,
)
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class H1_2RobotConfig(RobotConfig):
    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)
    # The MJCF imu site has identity axes on torso_link.
    imu_body_name: str = "torso_link"

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
                # Gains follow from the motor rating; scale=2 is two motors in
                # parallel (ankle pitch/roll). effort_limit overrides the motor
                # default for joints whose actuator differs from its gain class.
                ".*_hip_(yaw|pitch|roll)_joint": H1_2_200.control_info(),
                ".*_knee_joint": H1_2_300.control_info(),
                ".*_ankle_pitch_joint": H1_2_60.control_info(scale=2),
                ".*_ankle_roll_joint": H1_2_40.control_info(scale=2),
                "torso_joint": H1_2_200.control_info(),
                ".*_shoulder_(pitch|roll)_joint": H1_2_40.control_info(),
                ".*_shoulder_yaw_joint": H1_2_40.control_info(effort_limit=18),
                ".*_elbow_joint": H1_2_40.control_info(effort_limit=18),
                ".*_wrist_(roll|pitch|yaw)_joint": H1_2_40.control_info(
                    effort_limit=19
                ),
            },
        )
    )

    # 50 Hz policy; simulation_params defaults to the humanoid physics for 50 Hz
    # (isaacgym 100, isaaclab/newton 200, mujoco 1000 fps).
    control_hz: int = 50
