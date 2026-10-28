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
from protomotions.simulator.mujoco.config import MujocoSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class AMPRobotConfig(RobotConfig):
    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_right_foot_bodies": ["right_foot"],
            "all_left_foot_bodies": ["left_foot"],
            "all_left_hand_bodies": ["left_hand"],
            "all_right_hand_bodies": ["right_hand"],
            "head_body_name": ["head"],
            "torso_body_name": ["torso"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "torso",
            "head",
            "left_foot",
            "right_foot",
            "left_hand",
            "right_hand",
        ]
    )

    default_root_height: float = 0.8

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/amp_humanoid.xml",
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            # The AMP MJCF declares no actuator force/velocity ranges, so every
            # DOF gets an explicit torque/speed limit here (the backends have no
            # silent fallback). The gains (stiffness/damping/armature) stay the
            # MJCF values. ".*" claims every joint.
            override_control_info={
                ".*": ControlInfo(effort_limit=1000, velocity_limit=100)
            },
        )
    )

    control_hz: int = 30
    # Each passed block replaces (not merges with) the humanoid template for its
    # backend, so fields it omits are class defaults, not the humanoid tuning
    # (isaacgym and isaaclab here omit physx, so both get class-default PhysX);
    # newton falls back to the humanoid default (120 fps). implicitfast treats
    # the PD damping implicitly (Euler only does that for joint damping), which
    # keeps stiff gains on light links stable.
    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams.humanoid(
            30,
            isaacgym=IsaacGymSimParams(fps=60),
            isaaclab=IsaacLabSimParams(fps=120),
            mujoco=MujocoSimParams(fps=960, integrator="implicitfast"),
        )
    )
