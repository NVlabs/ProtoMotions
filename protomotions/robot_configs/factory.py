# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from protomotions.robot_configs.base import RobotConfig


def robot_config(robot_name: str, **updates) -> RobotConfig:
    """Factory function to create robot configuration based on robot type.

    Args:
        robot_name: Name of the robot type (smpl, smplx, amp, g1, h1_2, soma23, rigv1)
        **updates: Optional field updates to apply to the robot config

    Returns:
        RobotConfig: Robot configuration object

    Raises:
        ValueError: If robot_name is not recognized
    """
    if robot_name == "smpl":
        from protomotions.robot_configs.smpl import SmplRobotConfig

        config = SmplRobotConfig()
    elif robot_name == "smplx":
        from protomotions.robot_configs.smplx import SMPLXRobotConfig

        config = SMPLXRobotConfig()
    elif robot_name == "amp":
        from protomotions.robot_configs.amp import AMPRobotConfig

        config = AMPRobotConfig()
    elif robot_name == "g1":
        from protomotions.robot_configs.g1 import G1RobotConfig

        config = G1RobotConfig()
    elif robot_name == "h1_2":
        from protomotions.robot_configs.h1_2 import H1_2RobotConfig

        config = H1_2RobotConfig()
    elif robot_name == "rigv1":
        from protomotions.robot_configs.rigv1 import Rigv1RobotConfig

        config = Rigv1RobotConfig()
    elif robot_name == "soma23":
        from protomotions.robot_configs.soma23 import Soma23RobotConfig

        config = Soma23RobotConfig()
    else:
        raise ValueError(f"Invalid robot name: {robot_name}")

    # Apply any updates
    if updates:
        config.update_fields(**updates)

    return config
