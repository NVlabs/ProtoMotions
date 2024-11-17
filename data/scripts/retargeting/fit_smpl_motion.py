# Adapted from https://github.com/zhengyiluo/phc/tree/h1_phc

import typer
import glob
import os
from pathlib import Path
from smpl_sim.utils import torch_utils
from scipy.spatial.transform import Rotation as sRot
import numpy as np
import torch
from smpl_sim.smpllib.smpl_parser import (
    SMPL_Parser,
    SMPLH_Parser,
    SMPLX_Parser,
)

import joblib
import torch
from torch.autograd import Variable
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_MUJOCO_NAMES,
    SMPL_BONE_ORDER_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from data.scripts.retargeting.torch_humanoid_batch import Humanoid_Batch
from smpl_sim.utils.smoothing_utils import gaussian_kernel_1d, gaussian_filter_1d_batch
from data.scripts.retargeting.config import get_config
from typing import List, Tuple


def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    if "mocap_framerate" not in entry_data:
        return
    framerate = entry_data["mocap_framerate"]

    root_trans = entry_data["trans"]
    pose_aa = np.concatenate(
        [entry_data["poses"][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1
    )
    betas = entry_data["betas"]
    gender = entry_data["gender"]
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans,
        "betas": betas,
        "fps": framerate,
    }


def process_motion(motion_file, cfg, smpl_parser_n, shape_new, scale):
    device = torch.device("cpu")

    humanoid_fk = Humanoid_Batch(cfg)  # load forward kinematics model
    num_augment_joint = len(cfg.extend_config)

    #### Define corresonpdances between h1 and smpl joints
    robot_joint_names_augment = humanoid_fk.body_names_augment
    robot_joint_pick = [i[0] for i in cfg.joint_matches]
    smpl_joint_pick = [i[1] for i in cfg.joint_matches]
    robot_joint_pick_idx = [
        robot_joint_names_augment.index(j) for j in robot_joint_pick
    ]
    smpl_joint_pick_idx = [SMPL_BONE_ORDER_NAMES.index(j) for j in smpl_joint_pick]

    amass_data = load_amass_data(motion_file)

    skip = int(amass_data["fps"] // 30)
    trans = torch.from_numpy(amass_data["trans"][::skip])
    N = trans.shape[0]
    pose_aa_walk = torch.from_numpy(amass_data["pose_aa"][::skip]).float()

    with torch.no_grad():
        verts, joints = smpl_parser_n.get_joints_verts(pose_aa_walk, shape_new, trans)
        default_verts, default_joints = smpl_parser_n.get_joints_verts(
            pose_aa_walk, shape_new * 0, trans
        )
        root_pos = joints[:, 0:1]
        joints = (joints - joints[:, 0:1]) * scale.detach() + root_pos
    verts_min_z = verts[0, :, 2].min().item()
    joints[..., 2] -= verts_min_z
    verts[..., 2] -= verts_min_z

    default_min_joint_each_frame = default_joints[..., 2].min(dim=-1).values
    default_min_joint_each_frame -= default_min_joint_each_frame.min()

    offset = joints[:, 0] - trans
    root_trans_offset = (trans + offset).clone()

    gt_root_rot_quat = torch.from_numpy(
        (
            sRot.from_rotvec(pose_aa_walk[:, :3])
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        ).as_quat()
    ).float()  # can't directly use this
    gt_root_rot = torch.from_numpy(
        sRot.from_quat(torch_utils.calc_heading_quat(gt_root_rot_quat)).as_rotvec()
    ).float()  # so only use the heading.

    # def dof_to_pose_aa(dof_pos):
    dof_pos = torch.zeros((1, N, humanoid_fk.num_dof, 1))

    dof_pos_new = Variable(dof_pos.clone(), requires_grad=True)
    root_rot_new = Variable(gt_root_rot.clone(), requires_grad=True)
    root_pos_offset = Variable(
        torch.zeros(root_rot_new.shape[0], 3), requires_grad=True
    )
    optimizer_pose = torch.optim.Adadelta([dof_pos_new], lr=100)
    optimizer_root = torch.optim.Adam([root_rot_new, root_pos_offset], lr=0.01)

    kernel_size = 5  # Size of the Gaussian kernel
    sigma = 0.75  # Standard deviation of the Gaussian kernel
    # Maximum number of iterations to continue without improvement before early stopping
    early_stopping_grace_max = 100
    # Counter for remaining iterations without improvement
    early_stopping_grace_left = early_stopping_grace_max
    # Minimum required improvement in loss to reset early stopping counter
    early_stopping_threshold = 0.001
    # Store the best loss value seen so far for early stopping comparison
    early_stopping_last_loss_value = 100000000

    B, T, J, D = dof_pos_new.shape

    for iteration in range(cfg.get("fitting_iterations", 10000)):
        pose_aa_h1_new = torch.cat(
            [
                root_rot_new[None, :, None],
                humanoid_fk.dof_axis * dof_pos_new,
                torch.zeros((1, N, num_augment_joint, 3)).to(device),
            ],
            axis=2,
        )
        fk_return = humanoid_fk.fk_batch(
            pose_aa_h1_new, root_trans_offset[None,] + root_pos_offset
        )

        diff = (
            fk_return.global_translation[:, :, robot_joint_pick_idx]
            - joints[:, smpl_joint_pick_idx]
        )

        loss_g = diff.norm(dim=-1).mean()
        loss = loss_g

        optimizer_pose.zero_grad()
        optimizer_root.zero_grad()
        loss.backward()
        optimizer_pose.step()
        optimizer_root.step()

        dof_pos_new.data.clamp_(
            humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None]
        )

        current_loss = loss.item()
        if current_loss < early_stopping_last_loss_value - early_stopping_threshold:
            early_stopping_grace_left = early_stopping_grace_max
            early_stopping_last_loss_value = current_loss
        else:
            early_stopping_grace_left -= 1
            if early_stopping_grace_left == 0:
                print(
                    f"Early stopping at iteration {iteration} with loss {loss.item() * 1000:.3f}"
                )
                break

        if iteration % 100 == 0:
            print(f"{motion_file}-Iter: {iteration} \t {loss.item() * 1000:.3f}")
        dof_pos_new.data = gaussian_filter_1d_batch(
            dof_pos_new.squeeze().transpose(1, 0)[None,], kernel_size, sigma
        ).transpose(2, 1)[..., None]

    dof_pos_new.data.clamp_(
        humanoid_fk.joints_range[:, 0, None], humanoid_fk.joints_range[:, 1, None]
    )
    pose_aa_h1_new = torch.cat(
        [
            root_rot_new[None, :, None],
            humanoid_fk.dof_axis * dof_pos_new,
            torch.zeros((1, N, num_augment_joint, 3)).to(device),
        ],
        axis=2,
    )

    # root_pos_offset = gaussian_filter_1d_batch(root_pos_offset.data, kernel_size, sigma)
    root_trans_offset_dump = (root_trans_offset + root_pos_offset).clone()

    combined_mesh = humanoid_fk.mesh_fk(
        pose_aa_h1_new[:, :1].detach(), root_trans_offset_dump[None, :1].detach()
    )

    height_diff = np.asarray(combined_mesh.vertices)[..., 2].min()
    root_trans_offset_dump[..., 2] -= height_diff

    fk_return = humanoid_fk.fk_batch(
        pose_aa_h1_new.detach(),
        (root_trans_offset_dump[None,] + root_pos_offset).detach(),
        return_full=True,
    )
    h1_min_joint_each_frame = fk_return.global_translation[..., 2].min(dim=-1).values
    per_frame_height_fix = (
        h1_min_joint_each_frame - default_min_joint_each_frame
    ).unsqueeze(-1)

    fk_return = humanoid_fk.fk_batch(
        pose_aa_h1_new.detach(),
        (root_trans_offset_dump[None,] + root_pos_offset).detach()
        - per_frame_height_fix,
        return_full=True,
    )

    fk_return_proper = humanoid_fk.convert_to_proper_kinematic(fk_return)

    curr_motion = {
        k: v.squeeze().detach().cpu() if torch.is_tensor(v) else v
        for k, v in fk_return_proper.items()
    }
    return curr_motion


def convert_motions(humanoid_type: str, in_out_files_paths: List[Tuple[Path, Path]]):
    cfg = get_config(humanoid_type)

    smpl_parser_n = SMPL_Parser(model_path="data/smpl", gender="neutral")
    shape_new, scale = torch.load(f"data/{humanoid_type}/shape_optimized_v1.pt")

    for file_path, out_path in in_out_files_paths:
        try:
            curr_motion = process_motion(
                file_path, cfg, smpl_parser_n, shape_new, scale
            )
            torch.save(curr_motion, out_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")


def convert_motion(humanoid_type: str, file_path: Path, out_path: Path):
    convert_motions(humanoid_type, [(file_path, out_path)])


if __name__ == "__main__":
    typer.run(convert_motion)
