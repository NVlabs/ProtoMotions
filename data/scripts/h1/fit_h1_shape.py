# This file is adapted from https://github.com/LeCAR-Lab/human2humanoid/

import os

from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
)

from torch.autograd import Variable
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES
from data.scripts.h1.h1_humanoid_batch import Humanoid_Batch_H1

h1_joint_names = [
    "pelvis",
    "left_hip_yaw_link",
    "left_hip_roll_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_link",
    "right_hip_yaw_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
]
#### Define corresonpdances between h1 and smpl joints
h1_joint_names_augment = [
    "pelvis",
    "left_hip_yaw_link",
    "left_hip_roll_link",
    "left_hip_pitch_link",
    "left_knee_link",
    "left_ankle_link",
    "right_hip_yaw_link",
    "right_hip_roll_link",
    "right_hip_pitch_link",
    "right_knee_link",
    "right_ankle_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_arm_end_effector",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_arm_end_effector",
]
h1_joint_pick = [
    "pelvis",
    "left_knee_link",
    "left_ankle_link",
    "right_knee_link",
    "right_ankle_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_arm_end_effector",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_arm_end_effector",
]
smpl_joint_pick = [
    "Pelvis",
    "L_Knee",
    "L_Ankle",
    "R_Knee",
    "R_Ankle",
    "L_Shoulder",
    "L_Elbow",
    "L_Hand",
    "R_Shoulder",
    "R_Elbow",
    "R_Hand",
]
h1_joint_pick_idx = [h1_joint_names_augment.index(j) for j in h1_joint_pick]
smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]


def create_h1_shape():
    h1_fk = Humanoid_Batch_H1(
        extend_hand=True, extend_head=False
    )  # load forward kinematics model

    #### Preparing fitting varialbes
    device = torch.device("cpu")
    pose_aa_h1 = np.repeat(
        np.repeat(
            sRot.identity().as_rotvec()[
                None,
                None,
                None,
            ],
            22,
            axis=2,
        ),
        1,
        axis=1,
    )
    pose_aa_h1 = torch.from_numpy(pose_aa_h1).float()

    dof_pos = torch.zeros((1, 19))
    pose_aa_h1 = torch.cat(
        [
            torch.zeros((1, 1, 3)),
            h1_fk.rotation_axis * dof_pos[..., None],
            torch.zeros((1, 2, 3)),
        ],
        axis=1,
    )

    ###### prepare SMPL default pause for H1
    pose_aa_stand = np.zeros((1, 72))
    rotvec = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).as_rotvec()
    pose_aa_stand[:, :3] = rotvec
    pose_aa_stand = pose_aa_stand.reshape(-1, 24, 3)
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index("L_Shoulder")] = sRot.from_euler(
        "xyz", [0, 0, -np.pi / 2], degrees=False
    ).as_rotvec()
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index("R_Shoulder")] = sRot.from_euler(
        "xyz", [0, 0, np.pi / 2], degrees=False
    ).as_rotvec()
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index("L_Elbow")] = sRot.from_euler(
        "xyz", [0, -np.pi / 2, 0], degrees=False
    ).as_rotvec()
    pose_aa_stand[:, SMPL_BONE_ORDER_NAMES.index("R_Elbow")] = sRot.from_euler(
        "xyz", [0, np.pi / 2, 0], degrees=False
    ).as_rotvec()
    pose_aa_stand = torch.from_numpy(pose_aa_stand.reshape(-1, 72))

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")

    ###### Shape fitting
    trans = torch.zeros([1, 3])
    beta = torch.zeros([1, 10])
    _, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, beta, trans)
    offset = joints[:, 0] - trans
    root_trans_offset = trans + offset

    pose_aa_h1 = pose_aa_h1.unsqueeze(0).expand(-1, 2, -1, -1)
    root_trans_offset = root_trans_offset.unsqueeze(0).expand(-1, 2, -1)

    fk_return = h1_fk.fk_batch(pose_aa_h1, root_trans_offset)

    shape_new = Variable(torch.zeros([1, 10]).to(device), requires_grad=True)
    scale = Variable(torch.ones([1]).to(device), requires_grad=True)
    optimizer_shape = torch.optim.Adam([shape_new, scale], lr=0.1)

    for iteration in range(1000):
        _, joints = smpl_parser_n.get_joints_verts(pose_aa_stand, shape_new, trans[0:1])
        root_pos = joints[:, 0]
        joints = (joints - joints[:, 0]) * scale + root_pos
        diff = (
            fk_return.global_translation[:, :, h1_joint_pick_idx]
            - joints[:, smpl_joint_pick_idx]
        )
        loss_g = diff.norm(dim=-1).mean()
        loss = loss_g
        if iteration % 100 == 0:
            print(iteration, loss.item() * 1000)

        optimizer_shape.zero_grad()
        loss.backward()
        optimizer_shape.step()

    # Create the directory if it doesn't exist
    os.makedirs("data/h1/", exist_ok=True)

    torch.save(
        (shape_new.detach().cpu(), scale.detach().cpu()),
        "data/h1/shape_optimized_v1.pt",
    )


if __name__ == "__main__":
    create_h1_shape()
