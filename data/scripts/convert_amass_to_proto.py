# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# https://github.com/user-attachments/files/19629778/LeftTurn01_stageii.npz.zip
# Here's a file from AMASS-X.
# Place it in /home/amassx/tmp/
# You can run
# python data/scripts/convert_amass_to_proto.py /home/amassx --humanoid-type=smplx --robot-type=g1

import os
from pathlib import Path
import pickle
from typing import List

import yaml
import numpy as np
import torch
import typer
from scipy.spatial.transform import Rotation as sRot

from data.smpl.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)

from tqdm import tqdm

from protomotions.utils.rotations import (
    matrix_to_quaternion,
    quat_mul,
    quaternion_to_matrix,
)

import time
from datetime import timedelta

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_from_transforms_with_velocities,
    extract_qpos_from_transforms,
    compute_angular_velocity,
    compute_forward_kinematics_from_transforms,
    compute_joint_rot_mats_from_global_mats,
)

from contact_detection import compute_contact_labels_from_pos_and_vel

TMP_SMPL_DIR = "/tmp/smpl"


app = typer.Typer(pretty_exceptions_enable=False)


def closest_divisor_larger_than_target(rounded_fps, target_fps):
    # Find divisors of rounded_fps
    divisors = [i for i in range(1, rounded_fps + 1) if rounded_fps % i == 0]
    # Filter divisors that are larger than target_fps
    larger_divisors = [d for d in divisors if d >= target_fps]
    # Return the smallest divisor that is larger than or equal to target_fps
    if larger_divisors:
        return min(larger_divisors)
    else:
        return None


def load_motion_configs(yaml_files):
    """Load motion configurations from YAML files and create timing dictionary."""
    motion_timings = {}

    for yaml_file in yaml_files:
        print(f"Loading motion config from {yaml_file}")
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f)

        for motion in config.get("motions", []):
            file_path = motion["file"]
            # Assume single sub_motion per motion - take the first one
            sub_motions = motion.get("sub_motions", [])
            if sub_motions:
                timings = sub_motions[0].get("timings", {})
                start_time = timings.get("start", 0.0)
                end_time = timings.get("end", None)

                # Store single timing entry per motion
                motion_timings[file_path] = {"start": start_time, "end": end_time}

    return motion_timings


def slice_motion_data(pose_aa, amass_trans, start_time, end_time, fps):
    """Slice motion data based on start and end times."""
    if start_time == 0.0 and end_time is None:
        # No slicing needed
        return pose_aa, amass_trans

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time is not None else pose_aa.shape[0]

    # Ensure we don't go out of bounds
    start_frame = max(0, start_frame)
    end_frame = min(pose_aa.shape[0], end_frame)

    print(
        f"Slicing motion from frame {start_frame} to {end_frame} (time {start_time}s to {end_time}s)"
    )

    return pose_aa[start_frame:end_frame], amass_trans[start_frame:end_frame]


def convert_amass_to_motion(
    pose_aa,
    amass_trans,
    mocap_fr,
    output_fps,
    humanoid_type,
    joint_names,
    mujoco_joint_names,
    kinematic_info,
    device,
    dtype,
):
    """
    Convert AMASS motion data to processed motion object.

    Args:
        pose_aa: AMASS pose in axis-angle format
        amass_trans: AMASS root translations
        mocap_fr: Motion capture frame rate
        output_fps: Desired output frame rate
        humanoid_type: Type of humanoid (smpl or smplx)
        joint_names: List of joint names
        mujoco_joint_names: List of MuJoCo joint names
        kinematic_info: Kinematic info object
        device: torch device
        dtype: torch dtype

    Returns:
        motion: Processed motion object
        current_output_fps: Actual output FPS after downsampling
    """
    # downsample motion if needed
    largest_divisor = closest_divisor_larger_than_target(mocap_fr, output_fps)
    if largest_divisor is not None:
        downsample_factor = mocap_fr // largest_divisor
        amass_trans = amass_trans[::downsample_factor]  # (T, 3)
        pose_aa = pose_aa[::downsample_factor]  # (T, N)
        current_output_fps = output_fps
    else:
        current_output_fps = mocap_fr

    smpl_2_mujoco = [
        joint_names.index(q) for q in mujoco_joint_names if q in joint_names
    ]
    batch_size = pose_aa.shape[0]

    if humanoid_type == "smpl":
        pose_aa = np.concatenate(
            [pose_aa[:, :66], np.zeros((batch_size, 6))],
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
                pose_aa[:, :66],
                pose_aa[:, 75:],
            ],
            axis=-1,
        )
        pose_aa_mj = pose_aa.reshape(batch_size, 52, 3)[:, smpl_2_mujoco]
        pose_quat = (
            sRot.from_rotvec(pose_aa_mj.reshape(-1, 3))
            .as_quat()
            .reshape(batch_size, 52, 4)
        )

    amass_trans = torch.from_numpy(amass_trans).to(device, dtype)
    pose_quat = torch.from_numpy(pose_quat).to(device, dtype)
    local_rot_mats = quaternion_to_matrix(pose_quat, w_last=True)

    _, world_rot_mat = compute_forward_kinematics_from_transforms(
        kinematic_info, amass_trans, local_rot_mats
    )
    global_quat = matrix_to_quaternion(world_rot_mat, w_last=True)

    rot1 = sRot.from_euler("xyz", np.array([-np.pi / 2, -np.pi / 2, 0]), degrees=False)
    rot1_quat = (
        torch.from_numpy(rot1.as_quat())
        .to(device, dtype)
        .expand(amass_trans.shape[0], -1)
    )

    n_j = 23 if humanoid_type == "smpl" else 51  # smplx has 51 non-root joints
    for i in range(0, n_j + 1):
        global_quat[:, i, :] = quat_mul(global_quat[:, i, :], rot1_quat, w_last=True)

    local_rot_mats_rotated = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=quaternion_to_matrix(global_quat, w_last=True),
    )

    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=amass_trans,
        joint_rot_mats=local_rot_mats_rotated,
        fps=current_output_fps,
        compute_velocities=True,
    )

    pose_quat_rotated = matrix_to_quaternion(local_rot_mats_rotated, w_last=True)
    # caching local rotation to disk file, in case anyone needs it later
    motion.local_rigid_body_rot = pose_quat_rotated.clone()

    # calc dof pos and dof vel
    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=amass_trans,
        joint_rot_mats=local_rot_mats_rotated,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]

    local_angular_vels = compute_angular_velocity(
        batched_robot_rot_mats=local_rot_mats_rotated[:, 1:, :, :],
        fps=current_output_fps,
    )

    assert local_angular_vels.shape[1] == n_j  # (T, 23, 3)
    # because we know all joints are 3 dof exp_map joints...
    motion.dof_vel = local_angular_vels.reshape(-1, n_j * 3)

    motion.fix_height(height_offset=foot_offsets_dict[humanoid_type])

    # compute contacts using position and velocity thresholds
    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)

    return motion, current_output_fps


def save_motion(motion, outpath):
    """Save motion object to disk."""
    # Create output directory if it doesn't exist
    os.makedirs(outpath.parent, exist_ok=True)

    print(f"Saving to {outpath}")
    torch.save(motion.to_dict(), str(outpath))


def process_motion_segment(
    pose_aa,
    amass_trans,
    mocap_fr,
    output_fps,
    humanoid_type,
    joint_names,
    mujoco_joint_names,
    kinematic_info,
    device,
    dtype,
    outpath,
):
    """Process a motion segment and save it to disk."""
    motion, _ = convert_amass_to_motion(
        pose_aa,
        amass_trans,
        mocap_fr,
        output_fps,
        humanoid_type,
        joint_names,
        mujoco_joint_names,
        kinematic_info,
        device,
        dtype,
    )
    save_motion(motion, outpath)
    return motion


foot_offsets_dict = {
    "smpl": 0.015,
    "smplx": 0.017,
}


@app.command()
def main(
    amass_root_dir: Path,
    humanoid_type: str = "smpl",
    force_remake: bool = False,
    output_fps: int = 30,
    motion_configs: List[str] = typer.Option(
        None, "--motion-config", help="YAML files containing motion configurations"
    ),
):
    device = torch.device("cpu")  # cuda does not seem faster?
    dtype = torch.float32

    # Load motion configurations if provided
    motion_timings = {}
    if motion_configs:
        motion_timings = load_motion_configs(motion_configs)
        print(f"Loaded timing configurations for {len(motion_timings)} motions")

    assert humanoid_type in [
        "smpl",
        "smplx",
    ], "Humanoid type must be one of smpl, smplx"

    if humanoid_type == "smpl":
        mujoco_joint_names = SMPL_MUJOCO_NAMES
        joint_names = SMPL_BONE_ORDER_NAMES
    elif humanoid_type == "smplx":
        mujoco_joint_names = SMPLH_MUJOCO_NAMES
        joint_names = SMPLH_BONE_ORDER_NAMES
    else:
        raise NotImplementedError

    folder_names = [
        f.path.split("/")[-1] for f in os.scandir(amass_root_dir) if f.is_dir()
    ]

    kinematic_info = extract_kinematic_info(
        f"protomotions/data/assets/mjcf/{humanoid_type}_humanoid.xml"
    )

    # Count total number of files that need processing
    start_time = time.time()
    total_files = 0
    total_files_to_process = 0
    processed_files = 0
    for folder_name in folder_names:
        if "smpl" in folder_name:
            continue
        data_dir = amass_root_dir / folder_name
        output_dir = amass_root_dir / f"{folder_name}"

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
                    output_dir
                    / f.relative_to(data_dir).parent
                    / f.name.replace(".npz", ".motion")
                    .replace(".pkl", ".motion")
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
        if "smpl" in folder_name:
            # Ignore folders where we store converted motions
            continue

        data_dir = amass_root_dir / folder_name
        output_dir = amass_root_dir / f"{folder_name}"

        print(f"Processing subset {folder_name}")
        os.makedirs(output_dir, exist_ok=True)

        files = [
            f
            for f in Path(data_dir).glob("**/*.[np][pk][lz]")
            if (f.name != "shape.npz" and "stagei.npz" not in f.name)
        ]
        print(f"Processing {len(files)} files")

        files.sort()

        for filename in tqdm(files):
            # try:
            relative_path_dir = filename.relative_to(data_dir).parent
            outpath = (
                output_dir
                / relative_path_dir
                / filename.name.replace(".npz", ".motion")
                .replace(".pkl", ".motion")
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
            os.makedirs(output_dir / relative_path_dir, exist_ok=True)

            print(f"Processing {filename}")
            if filename.suffix == ".npz" and "samp" not in str(filename):
                motion_data = np.load(filename)

                # gender = "neutral"      # assume neutral gender with beta = 0
                pose_aa = motion_data["poses"]
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
                            / filename.name.replace(".npz", ".motion")
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

            elif filename.suffix == ".pkl" and "samp" in str(filename):
                with open(filename, "rb") as f:
                    motion_data = pickle.load(f, encoding="latin1")

                pose_aa = motion_data["pose_est_fullposes"]
                amass_trans = motion_data["pose_est_trans"]
                mocap_fr = motion_data["mocap_framerate"]

            else:
                print(f"Skipping {filename} as it is not a valid file")
                continue

            mocap_fr = np.round(mocap_fr).astype(int)

            # Check if this motion should be sliced based on YAML configs
            if motion_timings:
                # Create motion key to match with YAML config
                motion_relative_path = str(
                    relative_path_dir
                    / filename.name.replace(".npz", ".motion")
                    .replace(".pkl", ".motion")
                    .replace("-", "_")
                    .replace(" ", "_")
                    .replace("(", "_")
                    .replace(")", "_")
                )
                motion_key = folder_name + "/" + motion_relative_path

                if motion_key in motion_timings:
                    print(f"Found timing config for {motion_key}")
                    # Get the timing configuration
                    timing_config = motion_timings[motion_key]
                    start_time = timing_config["start"]
                    end_time = timing_config["end"]

                    # Slice the motion data
                    sliced_pose_aa, sliced_amass_trans = slice_motion_data(
                        pose_aa, amass_trans, start_time, end_time, mocap_fr
                    )

                    # Process this segment with original filename
                    process_motion_segment(
                        sliced_pose_aa,
                        sliced_amass_trans,
                        mocap_fr,
                        output_fps,
                        humanoid_type,
                        joint_names,
                        mujoco_joint_names,
                        kinematic_info,
                        device,
                        dtype,
                        outpath,
                    )

                    # Skip the normal processing since we handled the segment
                    continue
                else:
                    print(f"No timing config found for {motion_key}")
                    continue

            # Process the full motion using the same function
            process_motion_segment(
                pose_aa,
                amass_trans,
                mocap_fr,
                output_fps,
                humanoid_type,
                joint_names,
                mujoco_joint_names,
                kinematic_info,
                device,
                dtype,
                outpath,
            )

            processed_files += 1
            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / processed_files
            remaining_files = total_files_to_process - processed_files
            estimated_time_remaining = avg_time_per_file * remaining_files

            print(f"\nProgress: {processed_files}/{total_files_to_process} files")
            print(f"Average time per file: {timedelta(seconds=int(avg_time_per_file))}")
            print(
                f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}"
            )
            print(
                f"Estimated completion time: {time.strftime('%H:%M:%S', time.localtime(time.time() + estimated_time_remaining))}\n"
            )
        # except Exception as e:
        #     print(f"Error processing {filename}")
        #     print(f"Error: {e}")
        #     print(f"Line: {e.__traceback__.tb_lineno}")
        #     continue


if __name__ == "__main__":
    with torch.no_grad():
        app()
