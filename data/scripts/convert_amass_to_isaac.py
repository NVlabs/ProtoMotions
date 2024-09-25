# This code is adapted from https://github.com/zhengyiluo/phc/ and generalized to work with any humanoid.
# https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/convert_amass_isaac.py

import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import ipdb
import numpy as np
import torch
import typer
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from tqdm import tqdm

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree

TMP_SMPL_DIR = "/tmp/smpl"


def main(
    amass_root_dir: Path,
    humanoid_type: str = "smpl",
    force_remake: bool = False,
    force_neutral_body: bool = True,
    generate_flipped: bool = False,
    not_upright_start: bool = False,  # By default, let's start upright (for consistency across all models).
    humanoid_mjcf_path: Optional[str] = None,
):
    assert humanoid_type in [
        "smpl",
        "smplx",
        "smplh",
    ], "Humanoid type must be one of smpl, smplx, smplh"
    append_name = humanoid_type
    upright_start = not not_upright_start

    if humanoid_type == "smpl":
        mujoco_joint_names = SMPL_MUJOCO_NAMES
        joint_names = SMPL_BONE_ORDER_NAMES
    elif humanoid_type == "smplx" or humanoid_type == "smplh":
        mujoco_joint_names = SMPLH_MUJOCO_NAMES
        joint_names = SMPLH_BONE_ORDER_NAMES
    else:
        raise NotImplementedError

    left_to_right_index = []
    for idx, entry in enumerate(mujoco_joint_names):
        # swap text "R_" and "L_"
        if entry.startswith("R_"):
            left_to_right_index.append(mujoco_joint_names.index("L_" + entry[2:]))
        elif entry.startswith("L_"):
            left_to_right_index.append(mujoco_joint_names.index("R_" + entry[2:]))
        else:
            left_to_right_index.append(idx)

    folder_names = [
        f.path.split("/")[-1] for f in os.scandir(amass_root_dir) if f.is_dir()
    ]

    robot_cfg = {
        "mesh": False,
        "rel_joint_lm": True,
        "upright_start": upright_start,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": True,
        "masterfoot": False,
        "big_ankle": True,
        "freeze_hand": False,
        "box_body": False,
        "master_range": 50,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "model": humanoid_type,
        "sim": "isaacgym",
    }

    smpl_local_robot = SMPL_Robot(
        robot_cfg,
        data_dir="data/smpl",
    )

    if humanoid_mjcf_path is not None:
        skeleton_tree = SkeletonTree.from_mjcf(humanoid_mjcf_path)
    else:
        skeleton_tree = None

    uuid_str = uuid.uuid4()

    for folder_name in folder_names:
        if "retarget" in folder_name or "smpl" in folder_name:
            # Ignore folders where we store motions retargeted to AMP
            continue
        if not force_remake and f"{folder_name}-{append_name}" in folder_names:
            continue

        data_dir = amass_root_dir / folder_name
        output_dir = amass_root_dir / f"{folder_name}-{append_name}"

        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

        print(f"Processing subset {folder_name}")
        os.mkdir(output_dir)

        files = [
            f
            for f in Path(data_dir).glob("**/*.npz")
            if (f.name != "shape.npz" and "stagei.npz" not in f.name)
        ]
        print(f"Processing {len(files)} files")

        files.sort()

        for filename in tqdm(files):
            print(f"Processing {filename}")
            try:
                relative_path_dir = filename.relative_to(data_dir).parent
                relative_path_dir.mkdir(exist_ok=True, parents=True)

                outpath = (
                    output_dir
                    / relative_path_dir
                    / filename.name.replace(".npz", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                )

                motion_data = np.load(filename)

                betas = motion_data["betas"]
                gender = motion_data["gender"]
                amass_pose = motion_data["poses"]
                amass_trans = motion_data["trans"]
                if "mocap_framerate" in motion_data:
                    mocap_fr = motion_data["mocap_framerate"]
                else:
                    mocap_fr = motion_data["mocap_frame_rate"]

                pose_aa = torch.tensor(amass_pose)
                amass_trans = torch.tensor(amass_trans)
                betas = torch.from_numpy(betas)

                if force_neutral_body:
                    betas[:] = 0
                    gender = "neutral"

                motion_data = {
                    "pose_aa": pose_aa.numpy(),
                    "trans": amass_trans.numpy(),
                    "beta": betas.numpy(),
                    "gender": gender,
                }

                smpl_2_mujoco = [
                    joint_names.index(q) for q in mujoco_joint_names if q in joint_names
                ]
                batch_size = motion_data["pose_aa"].shape[0]

                if humanoid_type == "smpl":
                    pose_aa = np.concatenate(
                        [
                            motion_data["pose_aa"][:, :66],
                            np.zeros((batch_size, 6))
                        ],
                        axis=1,
                    )  # TODO: need to extract correct handle rotations instead of zero
                    pose_aa_mj = pose_aa.reshape(batch_size, 24, 3)[:, smpl_2_mujoco]
                    pose_quat = (
                        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
                        .as_quat()
                        .reshape(batch_size, 24, 4)
                    )
                else:
                    pose_aa = np.concatenate(
                        [
                            motion_data["pose_aa"][:, :66],
                            motion_data["pose_aa"][:, 75:],
                        ],
                        axis=-1,
                    )
                    pose_aa_mj = pose_aa.reshape(batch_size, 52, 3)[:, smpl_2_mujoco]
                    pose_quat = (
                        sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
                        .as_quat()
                        .reshape(batch_size, 52, 4)
                    )

                if isinstance(gender, np.ndarray):
                    gender = gender.item()

                if isinstance(gender, bytes):
                    gender = gender.decode("utf-8")
                if gender == "neutral":
                    gender_number = [0]
                elif gender == "male":
                    gender_number = [1]
                elif gender == "female":
                    gender_number = [2]
                else:
                    ipdb.set_trace()
                    raise Exception("Gender Not Supported!!")

                if skeleton_tree is None:
                    smpl_local_robot.load_from_skeleton(
                        betas=betas[None,], gender=gender_number, objs_info=None
                    )
                    smpl_local_robot.write_xml(
                        f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml"
                    )
                    skeleton_tree = SkeletonTree.from_mjcf(
                        f"{TMP_SMPL_DIR}/smpl_humanoid_{uuid_str}.xml"
                    )

                root_trans_offset = (
                    torch.from_numpy(motion_data["trans"])
                    + skeleton_tree.local_translation[0]
                )

                sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here.
                    torch.from_numpy(pose_quat),
                    root_trans_offset,
                    is_local=True,
                )

                if generate_flipped:
                    formats = ["regular", "flipped"]
                else:
                    formats = ["regular"]

                for format in formats:
                    if robot_cfg["upright_start"]:
                        B = pose_aa.shape[0]
                        pose_quat_global = (
                            (
                                sRot.from_quat(
                                    sk_state.global_rotation.reshape(-1, 4).numpy()
                                )
                                * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
                            )
                            .as_quat()
                            .reshape(B, -1, 4)
                        )
                    else:
                        pose_quat_global = sk_state.global_rotation.numpy()

                    trans = root_trans_offset.clone()
                    if format == "flipped":
                        pose_quat_global = pose_quat_global[:, left_to_right_index]
                        pose_quat_global[..., 0] *= -1
                        pose_quat_global[..., 2] *= -1
                        trans[..., 1] *= -1

                    new_sk_state = SkeletonState.from_rotation_and_root_translation(
                        skeleton_tree,
                        torch.from_numpy(pose_quat_global),
                        trans,
                        is_local=False,
                    )

                    new_sk_motion = SkeletonMotion.from_skeleton_state(
                        new_sk_state, fps=mocap_fr
                    )

                    if format == "flipped":
                        outpath = outpath.with_name(
                            outpath.stem + "_flipped" + outpath.suffix
                        )
                    print(f"Saving to {outpath}")
                    new_sk_motion.to_file(str(outpath))
            except Exception as e:
                print(f"Error processing {filename}")
                print(f"Error: {e}")
                print(f"Line: {e.__traceback__.tb_lineno}")
                continue


if __name__ == "__main__":
    with torch.no_grad():
        typer.run(main)
