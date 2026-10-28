# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.robot_configs.base import (
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
)
from protomotions.robot_configs.motor_classes import (
    G1_5020,
    G1_7520_14,
    G1_7520_22,
    G1_4010,
)
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


# Default standing pose (radians), from BeyondMimic.
# Regex patterns resolved against DOF names in RobotConfig.__post_init__.
DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.312,
    ".*_knee_joint": 0.669,
    ".*_ankle_pitch_joint": -0.363,
    ".*_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
}


@dataclass
class G1RobotConfig(RobotConfig):
    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)

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
    default_dof_pos: Dict[str, float] = field(default_factory=lambda: DEFAULT_JOINT_POS)
    anchor_body_name: Optional[str] = "torso_link"
    # Torso orientation is reconstructed through the waist from the pelvis IMU.
    imu_body_name: str = "pelvis"

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/g1_holo_compat.xml",
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
                # A joint's gains follow from its motor; scale=2 is two motors
                # in parallel (ankle, waist roll/pitch). effort_limit overrides
                # the motor default where the joint carries a different actuator.
                ".*_hip_(pitch|yaw)_joint": G1_7520_14.control_info(),
                ".*_hip_roll_joint": G1_7520_22.control_info(),
                ".*_knee_joint": G1_7520_22.control_info(),
                ".*_ankle_.*": G1_5020.control_info(scale=2, effort_limit=50),
                "waist_yaw_joint": G1_7520_14.control_info(),
                "waist_(roll|pitch)_joint": G1_5020.control_info(
                    scale=2, effort_limit=50
                ),
                ".*_(shoulder|elbow)_.*": G1_5020.control_info(),
                ".*_wrist_roll_joint": G1_5020.control_info(),
                ".*_wrist_pitch_joint": G1_4010.control_info(),
                ".*_wrist_yaw_joint": G1_4010.control_info(),
            },
        )
    )

    # 50 Hz policy; simulation_params defaults to the humanoid physics for 50 Hz
    # (isaacgym 100, isaaclab/newton 200, mujoco 1000 fps).
    control_hz: int = 50
