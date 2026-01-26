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


# Updated constants based on PAL Robotics KANGAROO specifications
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

# Armature values calculated from (Factor * Inertia * Gear_Ratio^2)
ARMATURE_S_PLUS = 0.01265   # For Arm 1,2 and Pelvis 1,2
ARMATURE_S_MINUS = 0.00663  # For Arm 3,4
ARMATURE_LEGS = 0.01        # Leg armature (assumed as it's a simple model with non linear transmission)

# Stiffness/Damping for Arms & Pelvis (armature * NATURAL_FREQ^2 and 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ)
STIFFNESS_S_PLUS = 49.94
DAMPING_S_PLUS = 3.179

STIFFNESS_S_MINUS = 26.177
DAMPING_S_MINUS = 1.666

# Leg-specific logic (_calc_leg_params)
def get_leg_damping(stiffness):
    return round(2.0 * DAMPING_RATIO * stiffness / NATURAL_FREQ, 3)


@dataclass
class KangarooRobotConfig(RobotConfig):
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["leg_left_foot_link"],
            "all_right_foot_bodies": ["leg_right_foot_link"],
            "all_left_hand_bodies": ["ee_left_dummy_link"],
            "all_right_hand_bodies": ["ee_right_dummy_link"],
            "head_body_name": ["head_dummy_link"],
            "torso_body_name": ["pelvis_2_link"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "pelvis_2_link",
            "head_dummy_link",
            "leg_right_foot_link",
            "leg_left_foot_link",
            "ee_left_dummy_link",
            "ee_right_dummy_link",
        ]
    )

    default_root_height: float = 0.9

    # TODO: add initial joint positions?

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/kangaroo.xml",
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
                # Pelvis / Torso (S_PLUS)
                "pelvis_[12]_joint": ControlInfo(
                    stiffness=STIFFNESS_S_PLUS,
                    damping=DAMPING_S_PLUS,
                    effort_limit=50.0,
                    velocity_limit=3.14,
                    armature=ARMATURE_S_PLUS,
                ),
                # Legs 1, 2, 3 (Stiffness 100)
                "leg_.*_[123]_joint": ControlInfo(
                    stiffness=100.0,
                    damping=get_leg_damping(100.0),
                    effort_limit=230.0, # Using max of the group for safety or specific per joint
                    velocity_limit=3.87,
                    armature=ARMATURE_LEGS,
                ),
                # Legs 4, 5 (Stiffness 30)
                "leg_.*_[45]_joint": ControlInfo(
                    stiffness=30.0,
                    damping=get_leg_damping(30.0),
                    effort_limit=140.0,
                    velocity_limit=3.87,
                    armature=ARMATURE_LEGS,
                ),
                # Leg Length
                "leg_.*_length_joint": ControlInfo(
                    stiffness=1600.0,
                    damping=get_leg_damping(1600.0),
                    effort_limit=1100.0,
                    velocity_limit=10.0,
                    armature=ARMATURE_LEGS,
                ),
                # Arm 1, 2 (S_PLUS)
                "arm_.*_[12]_joint": ControlInfo(
                    stiffness=STIFFNESS_S_PLUS,
                    damping=DAMPING_S_PLUS,
                    effort_limit=50.0,
                    velocity_limit=1.95,
                    armature=ARMATURE_S_PLUS,
                ),
                # Arm 3, 4 (S_MINUS)
                "arm_.*_[34]_joint": ControlInfo(
                    stiffness=STIFFNESS_S_MINUS,
                    damping=DAMPING_S_MINUS,
                    effort_limit=25.0,
                    velocity_limit=3.25,
                    armature=ARMATURE_S_MINUS,
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
                fps=200, # TODO: verify if in newton similarly to mujoco solref >= 2*dt 
                decimation=4,
            ),
        )
    )
