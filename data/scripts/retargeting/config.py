# Adapted from https://github.com/zhengyiluo/phc/tree/h1_phc

from easydict import EasyDict


def h1_mapping():
    #### Config for extension
    extend_config = [
        {
            "joint_name": "left_arm_end_effector",
            "parent_name": "left_elbow_link",
            "pos": [0.3, 0.0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "right_arm_end_effector",
            "parent_name": "right_elbow_link",
            "pos": [0.3, 0.0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "head",
            "parent_name": "pelvis",
            "pos": [0.0, 0.0, 0.6],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
    ]

    base_link = "torso_link"
    joint_matches = [
        ["pelvis", "Pelvis"],
        ["left_hip_yaw_link", "L_Hip"],
        ["left_knee_link", "L_Knee"],
        ["left_ankle_link", "L_Ankle"],
        ["right_hip_yaw_link", "R_Hip"],
        ["right_knee_link", "R_Knee"],
        ["right_ankle_link", "R_Ankle"],
        ["left_shoulder_roll_link", "L_Shoulder"],
        ["left_elbow_link", "L_Elbow"],
        ["left_arm_end_effector", "L_Hand"],
        ["right_shoulder_roll_link", "R_Shoulder"],
        ["right_elbow_link", "R_Elbow"],
        ["right_arm_end_effector", "R_Hand"],
        ["head", "Head"],
    ]

    smpl_pose_modifier = [
        {"Pelvis": "[np.pi/2, 0, np.pi/2]"},
        {"L_Shoulder": "[0, 0, -np.pi/2]"},
        {"R_Shoulder": "[0, 0, np.pi/2]"},
        {"L_Elbow": "[0, -np.pi/2, 0]"},
        {"R_Elbow": "[0, np.pi/2, 0]"},
    ]

    asset_file = "phys_anim/data/assets/mjcf/h1_no_head_no_hands.xml"

    return EasyDict(
        extend_config=extend_config,
        base_link=base_link,
        joint_matches=joint_matches,
        smpl_pose_modifier=smpl_pose_modifier,
        asset_file=asset_file,
    )


def h1_no_head_mapping():
    #### Config for extension
    extend_config = [
        {
            "joint_name": "left_arm_end_effector",
            "parent_name": "left_elbow_link",
            "pos": [0.3, 0.0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "right_arm_end_effector",
            "parent_name": "right_elbow_link",
            "pos": [0.3, 0.0, 0.0],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
    ]

    base_link = "torso_link"
    joint_matches = [
        ["pelvis", "Pelvis"],
        ["left_hip_yaw_link", "L_Hip"],
        ["left_knee_link", "L_Knee"],
        ["left_ankle_link", "L_Ankle"],
        ["right_hip_yaw_link", "R_Hip"],
        ["right_knee_link", "R_Knee"],
        ["right_ankle_link", "R_Ankle"],
        ["left_shoulder_roll_link", "L_Shoulder"],
        ["left_elbow_link", "L_Elbow"],
        ["left_arm_end_effector", "L_Hand"],
        ["right_shoulder_roll_link", "R_Shoulder"],
        ["right_elbow_link", "R_Elbow"],
        ["right_arm_end_effector", "R_Hand"],
    ]

    smpl_pose_modifier = [
        {"Pelvis": "[np.pi/2, 0, np.pi/2]"},
        {"L_Shoulder": "[0, 0, -np.pi/2]"},
        {"R_Shoulder": "[0, 0, np.pi/2]"},
        {"L_Elbow": "[0, -np.pi/2, 0]"},
        {"R_Elbow": "[0, np.pi/2, 0]"},
    ]

    asset_file = "phys_anim/data/assets/mjcf/h1_no_head_no_hands.xml"

    return EasyDict(
        extend_config=extend_config,
        base_link=base_link,
        joint_matches=joint_matches,
        smpl_pose_modifier=smpl_pose_modifier,
        asset_file=asset_file,
    )


def h1_no_head_no_hands_mapping():
    #### Config for extension
    extend_config = []

    base_link = "torso_link"
    joint_matches = [
        ["pelvis", "Pelvis"],
        ["left_hip_yaw_link", "L_Hip"],
        ["left_knee_link", "L_Knee"],
        ["left_ankle_link", "L_Ankle"],
        ["right_hip_yaw_link", "R_Hip"],
        ["right_knee_link", "R_Knee"],
        ["right_ankle_link", "R_Ankle"],
        ["left_shoulder_roll_link", "L_Shoulder"],
        ["left_elbow_link", "L_Elbow"],
        ["right_shoulder_roll_link", "R_Shoulder"],
        ["right_elbow_link", "R_Elbow"],
    ]

    smpl_pose_modifier = [
        {"Pelvis": "[np.pi/2, 0, np.pi/2]"},
        {"L_Shoulder": "[0, 0, -np.pi/2]"},
        {"R_Shoulder": "[0, 0, np.pi/2]"},
        {"L_Elbow": "[0, -np.pi/2, 0]"},
        {"R_Elbow": "[0, np.pi/2, 0]"},
    ]

    asset_file = "phys_anim/data/assets/mjcf/h1_no_head_no_hands.xml"

    return EasyDict(
        extend_config=extend_config,
        base_link=base_link,
        joint_matches=joint_matches,
        smpl_pose_modifier=smpl_pose_modifier,
        asset_file=asset_file,
    )


def get_config(humanoid_type: str):
    if humanoid_type == "h1":
        return h1_mapping()
    elif humanoid_type == "h1_no_head":
        return h1_no_head_mapping()
    elif humanoid_type == "h1_no_head_no_hands":
        return h1_no_head_no_hands_mapping()
    else:
        raise NotImplementedError
