# This code is adapted from https://github.com/zhengyiluo/phc/ and generalized to work with any humanoid.
# https://github.com/ZhengyiLuo/PHC/blob/master/scripts/data_process/convert_amass_isaac.py

import os
import sys
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/home/cizinsky/ProtoMotions")

import ipdb
import yaml
import numpy as np
import torch
import typer
from scipy.spatial.transform import Rotation as sRot
import pickle
from smpl_sim.smpllib.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from tqdm import tqdm

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
import time
from datetime import timedelta

TMP_SMPL_DIR = "/tmp/smpl"


def main(
    amass_root_dir: Path,
    robot_type: str = None,
    humanoid_type: str = "smpl",
    force_remake: bool = False,
    force_neutral_body: bool = True,
    generate_flipped: bool = False,
    not_upright_start: bool = False,  # By default, let's start upright (for consistency across all models).
    humanoid_mjcf_path: Optional[str] = None,
    force_retarget: bool = False,
    output_dir: Path = None,
):
    if output_dir is None:
        output_dir = amass_root_dir

    if robot_type is None:
        robot_type = humanoid_type
    elif robot_type in ["h1", "g1"]:
        assert (
            force_retarget
        ), f"Data is either SMPL or SMPL-X. The {robot_type} robot must use the retargeting pipeline."
    assert humanoid_type in [
        "smpl",
        "smplx",
        "smplh",
    ], "Humanoid type must be one of smpl, smplx, smplh"
    append_name = robot_type
    if force_retarget:
        append_name += "_retargeted"
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

    # Count total number of files that need processing
    start_time = time.time()
    total_files = 0
    total_files_to_process = 0
    processed_files = 0
    for folder_name in folder_names:
        if "retarget" in folder_name or "smpl" in folder_name or "h1" in folder_name:
            continue
        data_dir = amass_root_dir / folder_name
        save_dir = output_dir / f"{folder_name}-{append_name}"

        all_files_in_folder = [
            f
            for f in Path(data_dir).glob("**/*.[np][pk][lz]")
            if (f.name != "shape.npz" and "stagei.npz" not in f.name)
        ]

        if not force_remake:
            # Only count files that don't already have outputs
            files_to_process = [
                f
                for f in all_files_in_folder
                if not (
                    save_dir
                    / f.relative_to(data_dir).parent
                    / f.name.replace(".npz", ".npy")
                    .replace(".pkl", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                ).exists()
            ]
        else:
            files_to_process = all_files_in_folder
        print(
            f"Processing {len(files_to_process)}/{len(all_files_in_folder)} files in {folder_name}"
        )
        total_files_to_process += len(files_to_process)
        total_files += len(all_files_in_folder)

    print(f"Total files to process: {total_files_to_process}/{total_files}")

    for folder_name in folder_names:
        if "retarget" in folder_name or "smpl" in folder_name or "h1" in folder_name:
            # Ignore folders where we store motions retargeted to AMP
            continue

        data_dir = amass_root_dir / folder_name
        save_dir = amass_root_dir / f"{folder_name}-{append_name}"

        print(f"Processing subset {folder_name}")
        os.makedirs(save_dir, exist_ok=True)

        files = [
            f
            for f in Path(data_dir).glob("**/*.[np][pk][lz]")
            if (f.name != "shape.npz" and "stagei.npz" not in f.name)
        ]
        print(f"Processing {len(files)} files")

        files.sort()

        for filename in tqdm(files):
            try:
                relative_path_dir = filename.relative_to(data_dir).parent
                outpath = (
                    save_dir
                    / relative_path_dir
                    / filename.name.replace(".npz", ".npy")
                    .replace(".pkl", ".npy")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                )

                # Check if the output file already exists
                if not force_remake and outpath.exists():
                    # print(f"Skipping {filename} as it already exists.")
                    continue

                # Create the output directory if it doesn't exist
                os.makedirs(save_dir / relative_path_dir, exist_ok=True)

                print(f"Processing {filename}")
                if filename.suffix == ".npz":
                    motion_data = np.load(filename)

                    betas = motion_data["betas"]
                    gender = motion_data["gender"]
                    amass_pose = motion_data["poses"]
                    amass_trans = motion_data["trans"]
                    if humanoid_type == "smplx":
                        # Load the fps from the yaml file
                        fps_yaml_path = Path("data/yaml_files/motion_fps_amassx.yaml")
                        with open(fps_yaml_path, "r") as f:
                            fps_dict = yaml.safe_load(f)

                        # Convert filename to match yaml format
                        yaml_key = (
                            folder_name
                            + "/"
                            + str(
                                relative_path_dir
                                / filename.name.replace(".npz", ".npy")
                                .replace("-", "_")
                                .replace(" ", "_")
                                .replace("(", "_")
                                .replace(")", "_")
                            )
                        )

                        if yaml_key in fps_dict:
                            mocap_fr = fps_dict[yaml_key]
                        elif "mocap_framerate" in motion_data:
                            mocap_fr = motion_data["mocap_framerate"]
                        elif "mocap_frame_rate" in motion_data:
                            mocap_fr = motion_data["mocap_frame_rate"]
                        else:
                            raise Exception(f"FPS not found for {yaml_key}")
                    else:
                        if "mocap_framerate" in motion_data:
                            mocap_fr = motion_data["mocap_framerate"]
                        else:
                            mocap_fr = motion_data["mocap_frame_rate"]
                else:
                    print(f"Skipping {filename} as it is not a valid file")
                    continue

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
                        [motion_data["pose_aa"][:, :66], np.zeros((batch_size, 6))],
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

                    if force_retarget:
                        from data.scripts.retargeting.mink_retarget import (
                            retarget_motion,
                        )

                        print("Force retargeting motion using mink retargeter...")
                        # Convert to 30 fps to speedup Mink retargeting
                        skip = int(mocap_fr // 30)
                        new_sk_state = SkeletonState.from_rotation_and_root_translation(
                            skeleton_tree,
                            torch.from_numpy(pose_quat_global[::skip]),
                            trans[::skip],
                            is_local=False,
                        )
                        new_sk_motion = SkeletonMotion.from_skeleton_state(
                            new_sk_state, fps=30
                        )

                        if robot_type in ["smpl", "smplx", "smplh"]:
                            robot_type = f"{robot_type}_humanoid"
                        new_sk_motion = retarget_motion(
                            motion=new_sk_motion, robot_type=robot_type, render=False
                        )

                    if format == "flipped":
                        outpath = outpath.with_name(
                            outpath.stem + "_flipped" + outpath.suffix
                        )
                    print(f"Saving to {outpath}")
                    if robot_type in ["h1", "g1"]:
                        torch.save(new_sk_motion, str(outpath))
                    else:
                        new_sk_motion.to_file(str(outpath))

                    processed_files += 1
                    elapsed_time = time.time() - start_time
                    avg_time_per_file = elapsed_time / processed_files
                    remaining_files = total_files_to_process - processed_files
                    estimated_time_remaining = avg_time_per_file * remaining_files

                    print(
                        f"\nProgress: {processed_files}/{total_files_to_process} files"
                    )
                    print(
                        f"Average time per file: {timedelta(seconds=int(avg_time_per_file))}"
                    )
                    print(
                        f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}"
                    )
                    print(
                        f"Estimated completion time: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_time_remaining))}\n"
                    )
            except Exception as e:
                print(f"Error processing {filename}")
                print(f"Error: {e}")
                print(f"Line: {e.__traceback__.tb_lineno}")
                continue


if __name__ == "__main__":
    with torch.no_grad():
        typer.run(main)
