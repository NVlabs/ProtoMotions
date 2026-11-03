# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
    SimulatorParams,
)
from protomotions.robot_configs.motor_classes import MotorClass
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import (
    IsaacLabPhysXParams,
    IsaacLabSimParams,
)
from protomotions.simulator.mujoco.config import MujocoSimParams

# Motor classes, written out here so this robot does not move when another
# robot's config does. Provenance: Unitree G1 actuator classes (7520-14,
# 7520-22, 5020, 4010, and a 50 Nm ankle on the 5020 rotor) with 25% more
# capacity, the rotor armature and the torque limit scaling together like a
# stronger motor of the same design, and a 1.5x speed limit. The speed
# headroom exists because the anatomical corpus is retargeted under a 20 rad/s
# joint-speed budget: at 1x the slowest class (20 rad/s, on the knees and the
# hip roll hinges) would sit exactly at the reference peak and the hard PhysX velocity
# clamp would leave the policy no room to catch up after an error. These are
# simulation design choices, not measured human actuator properties.
CAPACITY = 1.25
SPEED = 1.5
WAIST = MotorClass(
    armature=CAPACITY * 0.010177520,
    effort_limit=CAPACITY * 88,
    velocity_limit=SPEED * 32,
)
HIP_KNEE = MotorClass(
    armature=CAPACITY * 0.025101925,
    effort_limit=CAPACITY * 139,
    velocity_limit=SPEED * 20,
)
ARM = MotorClass(
    armature=CAPACITY * 0.003609725,
    effort_limit=CAPACITY * 25,
    velocity_limit=SPEED * 37,
)
WRIST = MotorClass(
    armature=CAPACITY * 0.00425, effort_limit=CAPACITY * 5, velocity_limit=SPEED * 22
)
ANKLE = MotorClass(
    armature=CAPACITY * 0.003609725,
    effort_limit=CAPACITY * 50,
    velocity_limit=SPEED * 37,
)


@dataclass
class AnatomicSomaRobotConfig(RobotConfig):
    """52 serial hinges, 23 semantic parts and 39 primitive colliders.

    The SOMA23 skeleton with its joints expanded into serial hinges on their
    own carrier links: 15 of the 22 joints get three hinges (pitch, roll and
    twist), the knees, elbows and toes one hinge each, and Neck2 a single yaw
    (15 * 3 + 7 = 52). That gives 53 bodies, of which the 30 intermediate links
    are non-colliding hinge carriers sitting on their semantic body. Joint ranges are engineering limits,
    not a medically validated human range-of-motion model. Motion files must be
    retargeted to this skeleton; SOMA23 packs (66 DOFs, 23 bodies) do not load.

    See mjcf/anatomic_soma.xml for the definition.
    """

    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0)
    anchor_body_name: str = "Chest"

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
            asset_file_name="mjcf/anatomic_soma.xml",
            self_collisions=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            # Class assignments are not homologous axes: every torso axis uses
            # the waist-yaw class, and every wrist axis uses the weak wrist
            # class. Each ankle carries two motors (scale=2 doubles armature and
            # gains, not the speed limit).
            override_control_info={
                "(Spine[12]|Chest)_.*": WAIST.control_info(),
                "(Left|Right)Leg_[xz]": WAIST.control_info(),
                "(Left|Right)(Leg_y|Shin_x)": HIP_KNEE.control_info(),
                "(Left|Right)Foot_.*": ANKLE.control_info(scale=2),
                "(Neck1|(Left|Right)(Shoulder|Arm|ForeArm|ToeBase))_.*": ARM.control_info(),
                "(Neck2|Head|(Left|Right)Hand)_.*": WRIST.control_info(),
            },
        )
    )

    control_hz: int = 30
    # Same blocks as SOMA23 except IsaacLab, which gets more contact-solver
    # resolution for the serial hinge chains: 240 Hz physics with 8 position
    # iterations. Each passed block replaces the humanoid template for its
    # backend; the decision rate stays 30 Hz everywhere.
    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams.humanoid(
            30,
            isaacgym=IsaacGymSimParams(fps=60),
            isaaclab=IsaacLabSimParams(
                fps=240,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            mujoco=MujocoSimParams(fps=960, integrator="implicitfast"),
        )
    )
