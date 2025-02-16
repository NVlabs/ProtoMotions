# Adapted from https://github.com/zhengyiluo/phc/tree/h1_phc

from easydict import EasyDict


def h1_mapping():
    #### Config for extension
    extend_config = [
        {
            "joint_name": "left_arm_end_effector",
            "parent_name": "left_elbow_link",
            "pos": [0.2605, 0, -0.0185],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "right_arm_end_effector",
            "parent_name": "right_elbow_link",
            "pos": [0.2605, 0, -0.0185],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "head",
            "parent_name": "pelvis",
            "pos": [0.0, 0.0, 0.6],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "left_foot_link",
            "parent_name": "left_ankle_link",
            "pos": [0.15, 0, -0.05],
            "rot": [1.0, 0.0, 0.0, 0.0],
        },
        {
            "joint_name": "right_foot_link",
            "parent_name": "right_ankle_link",
            "pos": [0.15, 0, -0.05],
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

    asset_file = "protomotions/data/assets/mjcf/h1_original.xml"

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

    asset_file = "protomotions/data/assets/mjcf/h1_original.xml"

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

    asset_file = "protomotions/data/assets/mjcf/h1_original.xml"

    return EasyDict(
        extend_config=extend_config,
        base_link=base_link,
        joint_matches=joint_matches,
        smpl_pose_modifier=smpl_pose_modifier,
        asset_file=asset_file,
    )


def smplx_with_limits_mapping():
    #### Config for extension
    extend_config = []

    base_link = "Pelvis"

    smplx_joints = [
        "Pelvis",
        "L_Hip",
        "L_Knee",
        "L_Ankle",
        "L_Toe",
        "R_Hip",
        "R_Knee",
        "R_Ankle",
        "R_Toe",
        "Torso",
        "Spine",
        "Chest",
        "Neck",
        "Head",
        "L_Thorax",
        "L_Shoulder",
        "L_Elbow",
        "L_Wrist",
        "L_Index1",
        "L_Index2",
        "L_Index3",
        "L_Middle1",
        "L_Middle2",
        "L_Middle3",
        "L_Pinky1",
        "L_Pinky2",
        "L_Pinky3",
        "L_Ring1",
        "L_Ring2",
        "L_Ring3",
        "L_Thumb1",
        "L_Thumb2",
        "L_Thumb3",
        "R_Thorax",
        "R_Shoulder",
        "R_Elbow",
        "R_Wrist",
        "R_Index1",
        "R_Index2",
        "R_Index3",
        "R_Middle1",
        "R_Middle2",
        "R_Middle3",
        "R_Pinky1",
        "R_Pinky2",
        "R_Pinky3",
        "R_Ring1",
        "R_Ring2",
        "R_Ring3",
        "R_Thumb1",
        "R_Thumb2",
        "R_Thumb3",
    ]
    joint_matches = [[joint, joint] for joint in smplx_joints]

    smpl_pose_modifier = []

    asset_file = "protomotions/data/assets/mjcf/smplx_humanoid_with_limits.xml"

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
    elif humanoid_type == "smplx_humanoid_with_limits":
        return smplx_with_limits_mapping()
    else:
        raise NotImplementedError
