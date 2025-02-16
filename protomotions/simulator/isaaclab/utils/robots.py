# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg


SMPLX_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="protomotions/data/assets/usd/smplx_humanoid.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.65, 0.1, 1.0), metallic=0.5
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_Hip_.",
                "R_Hip_.",
                "L_Knee_.",
                "R_Knee_.",
                "L_Ankle_.",
                "R_Ankle_.",
                "L_Toe_.",
                "R_Toe_.",
            ],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "L_Hip_.": 800,
                "R_Hip_.": 800,
                "L_Knee_.": 800,
                "R_Knee_.": 800,
                "L_Ankle_.": 800,
                "R_Ankle_.": 800,
                "L_Toe_.": 500,
                "R_Toe_.": 500,
            },
            damping={
                "L_Hip_.": 80,
                "R_Hip_.": 80,
                "L_Knee_.": 80,
                "R_Knee_.": 80,
                "L_Ankle_.": 80,
                "R_Ankle_.": 80,
                "L_Toe_.": 50,
                "R_Toe_.": 50,
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=[
                "Torso_.",
                "Spine_.",
                "Chest_.",
                "Neck_.",
                "Head_.",
                "L_Thorax_.",
                "R_Thorax_.",
            ],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "Torso_.": 1000,
                "Spine_.": 1000,
                "Chest_.": 1000,
                "Neck_.": 500,
                "Head_.": 500,
                "L_Thorax_.": 500,
                "R_Thorax_.": 500,
            },
            damping={
                "Torso_.": 100,
                "Spine_.": 100,
                "Chest_.": 100,
                "Neck_.": 50,
                "Head_.": 50,
                "L_Thorax_.": 50,
                "R_Thorax_.": 50,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_Shoulder_.",
                "R_Shoulder_.",
                "L_Elbow_.",
                "R_Elbow_.",
                "L_Wrist_.",
                "R_Wrist_.",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "L_Shoulder_.": 500,
                "R_Shoulder_.": 500,
                "L_Elbow_.": 300,
                "R_Elbow_.": 300,
                "L_Wrist_.": 300,
                "R_Wrist_.": 300,
            },
            damping={
                "L_Shoulder_.": 50,
                "R_Shoulder_.": 50,
                "L_Elbow_.": 30,
                "R_Elbow_.": 30,
                "L_Wrist_.": 30,
                "R_Wrist_.": 30,
            },
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_Index1_.",
                "L_Index2_.",
                "L_Index3_.",
                "L_Middle1_.",
                "L_Middle2_.",
                "L_Middle3_.",
                "L_Pinky1_.",
                "L_Pinky2_.",
                "L_Pinky3_.",
                "L_Ring1_.",
                "L_Ring2_.",
                "L_Ring3_.",
                "L_Thumb1_.",
                "L_Thumb2_.",
                "L_Thumb3_.",
                "R_Index1_.",
                "R_Index2_.",
                "R_Index3_.",
                "R_Middle1_.",
                "R_Middle2_.",
                "R_Middle3_.",
                "R_Pinky1_.",
                "R_Pinky2_.",
                "R_Pinky3_.",
                "R_Ring1_.",
                "R_Ring2_.",
                "R_Ring3_.",
                "R_Thumb1_.",
                "R_Thumb2_.",
                "R_Thumb3_.",
            ],
            effort_limit=5,
            velocity_limit=5.0,
            stiffness={
                "L_Index1_.": 10,
                "L_Index2_.": 10,
                "L_Index3_.": 10,
                "L_Middle1_.": 10,
                "L_Middle2_.": 10,
                "L_Middle3_.": 10,
                "L_Pinky1_.": 10,
                "L_Pinky2_.": 10,
                "L_Pinky3_.": 10,
                "L_Ring1_.": 10,
                "L_Ring2_.": 10,
                "L_Ring3_.": 10,
                "L_Thumb1_.": 10,
                "L_Thumb2_.": 10,
                "L_Thumb3_.": 10,
                "R_Index1_.": 10,
                "R_Index2_.": 10,
                "R_Index3_.": 10,
                "R_Middle1_.": 10,
                "R_Middle2_.": 10,
                "R_Middle3_.": 10,
                "R_Pinky1_.": 10,
                "R_Pinky2_.": 10,
                "R_Pinky3_.": 10,
                "R_Ring1_.": 10,
                "R_Ring2_.": 10,
                "R_Ring3_.": 10,
                "R_Thumb1_.": 10,
                "R_Thumb2_.": 10,
                "R_Thumb3_.": 10,
            },
            damping={
                "L_Index1_.": 1,
                "L_Index3_.": 1,
                "L_Middle1_.": 1,
                "L_Middle3_.": 1,
                "L_Pinky1_.": 1,
                "L_Pinky3_.": 1,
                "L_Ring1_.": 1,
                "L_Ring3_.": 1,
                "L_Thumb1_.": 1,
                "L_Thumb3_.": 1,
                "R_Index1_.": 1,
                "R_Index3_.": 1,
                "R_Middle1_.": 1,
                "R_Middle3_.": 1,
                "R_Pinky1_.": 1,
                "R_Pinky3_.": 1,
                "R_Ring1_.": 1,
                "R_Ring3_.": 1,
                "R_Thumb1_.": 1,
                "R_Thumb3_.": 1,
            },
        ),
    },
)


SMPL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="protomotions/data/assets/usd/smpl_humanoid.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.65, 0.1, 1.0), metallic=0.5
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(
            0.0,
            0.0,
            0.95,
        ),  # Default initial state of SMPL with the upright configuration
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_Hip_.",
                "R_Hip_.",
                "L_Knee_.",
                "R_Knee_.",
                "L_Ankle_.",
                "R_Ankle_.",
                "L_Toe_.",
                "R_Toe_.",
            ],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "L_Hip_.": 800,
                "R_Hip_.": 800,
                "L_Knee_.": 800,
                "R_Knee_.": 800,
                "L_Ankle_.": 800,
                "R_Ankle_.": 800,
                "L_Toe_.": 500,
                "R_Toe_.": 500,
            },
            damping={
                "L_Hip_.": 80,
                "R_Hip_.": 80,
                "L_Knee_.": 80,
                "R_Knee_.": 80,
                "L_Ankle_.": 80,
                "R_Ankle_.": 80,
                "L_Toe_.": 50,
                "R_Toe_.": 50,
            },
        ),
        "torso": ImplicitActuatorCfg(
            joint_names_expr=[
                "Torso_.",
                "Spine_.",
                "Chest_.",
                "Neck_.",
                "Head_.",
                "L_Thorax_.",
                "R_Thorax_.",
            ],
            effort_limit=500,
            velocity_limit=100.0,
            stiffness={
                "Torso_.": 1000,
                "Spine_.": 1000,
                "Chest_.": 1000,
                "Neck_.": 500,
                "Head_.": 500,
                "L_Thorax_.": 500,
                "R_Thorax_.": 500,
            },
            damping={
                "Torso_.": 100,
                "Spine_.": 100,
                "Chest_.": 100,
                "Neck_.": 50,
                "Head_.": 50,
                "L_Thorax_.": 50,
                "R_Thorax_.": 50,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "L_Shoulder_.",
                "R_Shoulder_.",
                "L_Elbow_.",
                "R_Elbow_.",
                "L_Wrist_.",
                "R_Wrist_.",
                "L_Hand_.",
                "R_Hand_.",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "L_Shoulder_.": 500,
                "R_Shoulder_.": 500,
                "L_Elbow_.": 300,
                "R_Elbow_.": 300,
                "L_Wrist_.": 300,
                "R_Wrist_.": 300,
                "L_Hand_.": 300,
                "R_Hand_.": 300,
            },
            damping={
                "L_Shoulder_.": 50,
                "R_Shoulder_.": 50,
                "L_Elbow_.": 30,
                "R_Elbow_.": 30,
                "L_Wrist_.": 30,
                "R_Wrist_.": 30,
            },
        ),
    },
)


H1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="protomotions/data/assets/usd/h1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(
            0.0,
            0.0,
            1.05,
        ),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.28,  # -16 degrees
            ".*_knee_joint": 0.79,  # 45 degrees
            ".*_ankle_joint": -0.52,  # -30 degrees
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.28,
            ".*_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # https://github.com/LeCAR-Lab/HumanoidVerse/blob/master/humanoidverse/simulator/isaacsim/isaaclab_cfg.py
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 200.0,
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 300.0,
                "torso_joint": 200.0,
            },
            velocity_limit={
                ".*_hip_yaw_joint": 23.0,
                ".*_hip_roll_joint": 23.0,
                ".*_hip_pitch_joint": 23.0,
                ".*_knee_joint": 14.0,
                "torso_joint": 23.0,
            },
            stiffness=0,
            damping=0,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_joint"],
            effort_limit=40,
            velocity_limit=9.0,
            stiffness=0,
            damping=0,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 18.0,
                ".*_elbow_joint": 18.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 9.0,
                ".*_shoulder_roll_joint": 9.0,
                ".*_shoulder_yaw_joint": 20.0,
                ".*_elbow_joint": 20.0,
            },
            stiffness=0,
            damping=0,
        ),
    },
)
