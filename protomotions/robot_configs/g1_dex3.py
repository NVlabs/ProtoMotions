# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 with Dex3-1 hands: the 29 body joints of ``g1`` plus 7 finger
joints per hand (43 DOF). The body, its gains, the hand frame (the wrist link)
and the tracked bodies are the ``g1`` robot's; the asset grafts Unitree's Dex3
hands onto it in place of the rubber-hand marker, so a policy that tracks the
body transfers and the fingers are extra joints. The finger gains are the
GEAR-SONIC HOI experts' (kp 0.5, thumb root 2.0; kd 0.1; armature 0.001)."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from protomotions.robot_configs.base import (
    ControlConfig,
    ControlInfo,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
)
from protomotions.robot_configs.g1 import DEFAULT_JOINT_POS, body_control_overrides

FINGER = ControlInfo(
    stiffness=0.5, damping=0.1, armature=0.001, effort_limit=1.4, velocity_limit=10.0
)
THUMB_ROOT = ControlInfo(
    stiffness=2.0, damping=0.1, armature=0.001, effort_limit=2.45, velocity_limit=10.0
)


def control_overrides():
    """Body gains from ``g1``; every finger joint is a Dex3 motor."""
    return {
        **body_control_overrides(),
        ".*_hand_(index|middle)_[01]_joint": FINGER,
        ".*_hand_thumb_[12]_joint": FINGER,
        ".*_hand_thumb_0_joint": THUMB_ROOT,
    }


@dataclass
class G1Dex3RobotConfig(RobotConfig):
    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["left_ankle_roll_link"],
            "all_right_foot_bodies": ["right_ankle_roll_link"],
            "all_left_hand_bodies": ["left_wrist_yaw_link"],
            "all_right_hand_bodies": ["right_wrist_yaw_link"],
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
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ]
    )

    default_root_height: float = 0.8
    # Fingers rest open (zero), the body pose is the g1 standing pose.
    default_dof_pos: Dict[str, float] = field(default_factory=lambda: DEFAULT_JOINT_POS)
    anchor_body_name: Optional[str] = "torso_link"
    imu_body_name: str = "pelvis"

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/g1_dex3.xml",
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
            override_control_info=control_overrides(),
        )
    )

    control_hz: int = 50
