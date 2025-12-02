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
import os
from pathlib import Path
import glob
from typing import Optional
import torch
import typer
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_from_transforms_with_velocities,
    compute_cartesian_velocity,
    extract_transforms_from_qpos,
    extract_qpos_from_transforms,
)
from protomotions.robot_configs.factory import robot_config
from motion_filter import passes_exclude_motion_filter

app = typer.Typer(pretty_exceptions_enable=False)


def euler_to_quaternion_wxyz(euler_xyz_degrees):
    """
    Convert Euler angles (XYZ order, in degrees) to quaternion (w, x, y, z).

    Args:
        euler_xyz_degrees: numpy array of shape (N, 3) with Euler angles in degrees

    Returns:
        numpy array of shape (N, 4) with quaternions in (w, x, y, z) order
    """
    # Convert degrees to radians
    euler_xyz_radians = np.radians(euler_xyz_degrees)

    # Create rotation objects from Euler angles (XYZ order)
    rotations = R.from_euler("xyz", euler_xyz_radians)

    # Get quaternions in (w, x, y, z) order
    quaternions_wxyz = rotations.as_quat()

    # scipy returns (x, y, z, w) but we need (w, x, y, z)
    quaternions_wxyz = quaternions_wxyz[:, [3, 0, 1, 2]]

    return quaternions_wxyz


def process_csv_file(csv_path, input_fps, output_fps, device, dtype):
    """
    Process a CSV file and extract motion data.

    Args:
        csv_path: Path to CSV file
        device: torch device
        dtype: torch data type

    Returns:
        tuple: (root_pos, root_rot_wxyz, joint_angles)
    """
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Extract root position (convert from cm to meters)
    root_pos_cm = df[["root_translateX", "root_translateY", "root_translateZ"]].values
    root_pos = root_pos_cm / 100.0  # Convert cm to meters

    # Extract root rotation (Euler angles in degrees)
    root_euler = df[["root_rotateX", "root_rotateY", "root_rotateZ"]].values

    # Convert Euler to quaternion (wxyz format)
    root_rot_wxyz = euler_to_quaternion_wxyz(root_euler)

    # Extract joint angles (all remaining columns except Frame and root pose)
    joint_angle_cols = [
        col
        for col in df.columns
        if col
        not in [
            "Frame",
            "root_translateX",
            "root_translateY",
            "root_translateZ",
            "root_rotateX",
            "root_rotateY",
            "root_rotateZ",
        ]
    ]
    joint_angles = df[joint_angle_cols].values

    # Convert to radians (assuming joint angles are in degrees)
    joint_angles = np.radians(joint_angles)

    # assume 120 fps input motion, convert to 30 fps
    assert input_fps % output_fps == 0, "Input fps must be divisible by output fps"
    factor = input_fps // output_fps
    joint_angles = joint_angles[::factor]
    root_pos = root_pos[::factor]
    root_rot_wxyz = root_rot_wxyz[::factor]

    # Convert to torch tensors
    root_pos = torch.from_numpy(root_pos).to(device, dtype)
    root_rot_wxyz = torch.from_numpy(root_rot_wxyz).to(device, dtype)
    joint_angles = torch.from_numpy(joint_angles).to(device, dtype)

    return root_pos, root_rot_wxyz, joint_angles


def process_npz_file(npz_path, input_fps, output_fps, device, dtype):
    """
    Process an NPZ file and extract motion data.

    Args:
        npz_path: Path to NPZ file
        device: torch device
        dtype: torch data type

    Returns:
        tuple: (root_pos, root_rot_wxyz, joint_angles)
    """
    data = np.load(npz_path, allow_pickle=True)

    assert input_fps % output_fps == 0, "Input fps must be divisible by output fps"
    factor = input_fps // output_fps

    # Extract and downsample the arrays (can't modify NpzFile in-place)
    base_frame_pos = data["base_frame_pos"][::factor]
    base_frame_wxyz = data["base_frame_wxyz"][::factor]
    joint_angles = data["joint_angles"][::factor]

    root_pos = torch.from_numpy(base_frame_pos).to(device, dtype)
    root_rot_wxyz = torch.from_numpy(base_frame_wxyz).to(device, dtype)
    joint_angles = torch.from_numpy(joint_angles).to(device, dtype)

    return root_pos, root_rot_wxyz, joint_angles


def apply_contact_labels_to_motion(
    motion,
    contact_labels_dir,
    motion_filename,
    input_fps,
    output_fps,
    left_foot_idx,
    right_foot_idx,
    device,
):
    """
    Load and apply contact labels from source motion to the motion object.

    Args:
        motion: Motion object to update
        contact_labels_dir: Directory containing contact label files
        motion_filename: Name of the motion file (for matching)
        input_fps: Input fps of contact labels
        output_fps: Target output fps
        left_foot_idx: Index of left foot body
        right_foot_idx: Index of right foot body
        device: torch device
    """
    # Try to load contact labels
    base_filename = os.path.splitext(motion_filename)[0]
    # Remove "_retargeted" suffix if present to match with contact file
    if base_filename.endswith("_retargeted"):
        base_filename = base_filename[: -len("_retargeted")]

    contact_labels_path = contact_labels_dir / f"{base_filename}_contacts.npz"

    if not os.path.exists(contact_labels_path):
        raise FileNotFoundError(
            f"Contact labels file not found: {contact_labels_path}\n"
            f"Please ensure contact files are generated for all motions."
        )

    # Load contact labels
    contact_data = np.load(contact_labels_path, allow_pickle=True)
    foot_contacts = contact_data["foot_contacts"]  # [K, 2] - left, right

    # Downsample if needed
    assert input_fps % output_fps == 0, "Input fps must be divisible by output fps"
    factor = input_fps // output_fps
    if factor > 1:
        foot_contacts = foot_contacts[::factor]

    # Check length matches
    motion_length = motion.rigid_body_pos.shape[0]
    contact_length = foot_contacts.shape[0]
    assert motion_length == contact_length, (
        f"Motion length ({motion_length}) does not match contact length ({contact_length}) "
        f"for {motion_filename}. Contact file: {contact_labels_path}"
    )

    # Create rigid_body_contacts tensor
    num_bodies = motion.rigid_body_pos.shape[1]
    rigid_body_contacts = np.zeros((motion_length, num_bodies), dtype=bool)

    # Set left and right foot contacts (threshold at 0.5)
    rigid_body_contacts[:, left_foot_idx] = foot_contacts[:, 0] > 0.5  # Left foot
    rigid_body_contacts[:, right_foot_idx] = foot_contacts[:, 1] > 0.5  # Right foot

    motion.rigid_body_contacts = torch.from_numpy(rigid_body_contacts).to(device)
    print(
        f"Applied contact labels from original motion before re-targeting {contact_labels_path}"
    )


@app.command()
def main(
    retargeted_motion_dir: Path = typer.Option(
        ..., help="Directory with retargeted motion files (.npz or .csv)."
    ),
    output_dir: Path = typer.Option(
        ..., help="Directory to save ProtoMotions motion files."
    ),
    input_fps: int = typer.Option(30, help="Input motion fps"),
    output_fps: int = typer.Option(30, help="Output motion fps"),
    force_remake: bool = False,
    ignore_first_n_frames: int = 0,  # ignore the first n frames of the motion
    # Motion filter options
    apply_motion_filter: bool = typer.Option(False, help="Apply motion quality filter"),
    min_height_threshold: float = typer.Option(
        -0.05, help="Minimum height threshold for motion filter"
    ),
    max_velocity_threshold: float = typer.Option(
        15.0, help="Maximum velocity threshold for motion filter"
    ),
    max_dof_vel_threshold: float = typer.Option(
        40.0, help="Maximum DOF velocity threshold for motion filter"
    ),
    duration_height_filter: float = typer.Option(
        0.1, help="Height threshold for duration filter"
    ),
    duration_height_seconds: float = typer.Option(
        0.6, help="Duration in seconds for height filter"
    ),
    robot_type: str = typer.Option("g1", help="Robot type"),
    # Contact labels from source motion
    contact_labels_dir: Optional[Path] = typer.Option(
        None,
        help="Directory with contact label files (*_contacts.npz). If provided, use these for rigid_body_contacts.",
    ),
):
    """
    Converts retargeted G1 motions (from .npz or .csv files) to the ProtoMotions motion format.
    """
    device = torch.device("cpu")
    dtype = torch.float32

    os.makedirs(output_dir, exist_ok=True)

    # Get kinematic info for the specified robot
    mjcf_path = f"protomotions/data/assets/mjcf/{robot_type}.xml"
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"MJCF file not found at {mjcf_path}")

    kinematic_info = extract_kinematic_info(mjcf_path)

    # Get robot config to find foot link names (for contact labeling)
    robot_cfg = robot_config(robot_type)
    left_foot_name = robot_cfg.common_naming_to_robot_body_names[
        "all_left_foot_bodies"
    ][0]
    right_foot_name = robot_cfg.common_naming_to_robot_body_names[
        "all_right_foot_bodies"
    ][0]

    # Find foot body indices
    body_names = kinematic_info.body_names
    left_foot_idx = body_names.index(left_foot_name)
    right_foot_idx = body_names.index(right_foot_name)

    print(f"Robot type: {robot_type}")
    print(f"Left foot: {left_foot_name} (index {left_foot_idx})")
    print(f"Right foot: {right_foot_name} (index {right_foot_idx})")

    # Find both NPZ and CSV files
    npz_files = sorted(list(glob.glob(str(retargeted_motion_dir / "*.npz"))))
    csv_files = sorted(list(glob.glob(str(retargeted_motion_dir / "*.csv"))))
    motion_files = npz_files + csv_files

    print(
        f"Found {len(motion_files)} motion files to process ({len(npz_files)} NPZ, {len(csv_files)} CSV)."
    )

    for motion_file_path in tqdm(motion_files, desc="Processing motions"):
        motion_file = Path(motion_file_path)
        outpath = (output_dir / motion_file.name.replace(" ", "_")).with_suffix(".motion")

        if not force_remake and outpath.exists():
            continue

        print(f"Processing {motion_file.name}")

        try:
            # Determine file type and process accordingly
            if motion_file.suffix.lower() == ".csv":
                root_pos, root_rot_wxyz, joint_angles = process_csv_file(
                    motion_file_path, input_fps, output_fps, device, dtype
                )
            elif motion_file.suffix.lower() == ".npz":
                root_pos, root_rot_wxyz, joint_angles = process_npz_file(
                    motion_file_path, input_fps, output_fps, device, dtype
                )
            else:
                print(f"Unsupported file format: {motion_file.suffix}")
                continue

            if ignore_first_n_frames > 0:
                root_pos = root_pos[ignore_first_n_frames:]
                root_rot_wxyz = root_rot_wxyz[ignore_first_n_frames:]
                joint_angles = joint_angles[ignore_first_n_frames:]

            qpos = torch.cat([root_pos, root_rot_wxyz, joint_angles], dim=-1)

            root_pos_from_qpos, joint_rot_mats = extract_transforms_from_qpos(
                kinematic_info, qpos
            )

            motion = fk_from_transforms_with_velocities(
                kinematic_info=kinematic_info,
                root_pos=root_pos_from_qpos,
                joint_rot_mats=joint_rot_mats,
                fps=output_fps,
                compute_velocities=True,
            )

            # to ensure joint angles falls into [-pi, pi]
            # otherwise we can simply use the joint angles as dof_pos
            qpos = extract_qpos_from_transforms(
                kinematic_info, root_pos, joint_rot_mats
            )
            motion.dof_pos = qpos[:, 7:]

            allowed_delta = [0.0, 2 * np.pi, 4 * np.pi]

            # Allow epsilon error in the difference between qpos and joint_angles
            delta = (qpos[:, 7:] - joint_angles).abs()
            epsilon = 1e-4
            # For each element, check if it is close to any allowed_delta (within epsilon)
            allowed = torch.zeros_like(delta, dtype=torch.bool)
            for d in allowed_delta:
                allowed |= (delta - d).abs() < epsilon
            assert allowed.all(), "qpos and joint_angles are not allowed (exceeds allowed delta with epsilon tolerance)"

            dof_vel = compute_cartesian_velocity(
                batched_robot_pos=joint_angles.unsqueeze(1),
                fps=output_fps,
            )
            motion.dof_vel = dof_vel.squeeze(1)

            # motion.fix_height(height_offset=0.04)

            motion.fix_height_per_frame(height_offset=0.02)
            motion.fix_height(height_offset=0.04)

            # Handle contact labels
            if contact_labels_dir is not None:
                apply_contact_labels_to_motion(
                    motion=motion,
                    contact_labels_dir=contact_labels_dir,
                    motion_filename=motion_file.name,
                    input_fps=input_fps,
                    output_fps=output_fps,
                    left_foot_idx=left_foot_idx,
                    right_foot_idx=right_foot_idx,
                    device=device,
                )
            else:
                # Default: zero contacts
                motion.rigid_body_contacts = torch.zeros(
                    motion.rigid_body_pos.shape[0],
                    motion.rigid_body_pos.shape[1],
                    device=device,
                    dtype=torch.bool,
                )

            motion.local_rigid_body_rot = (
                None  # HACK: prevent motion_lib from interpolating
            )

            # Apply motion filter if enabled
            if apply_motion_filter:
                if not passes_exclude_motion_filter(
                    motion,
                    min_height_threshold=min_height_threshold,
                    max_velocity_threshold=max_velocity_threshold,
                    max_dof_vel_threshold=max_dof_vel_threshold,
                    duration_height_filter=duration_height_filter,
                    duration_height_seconds=duration_height_seconds,
                ):
                    print(
                        f"Skipping {motion_file.name} because it does not pass motion filter"
                    )
                    continue

            print(f"motion.dof_pos: {motion.dof_pos.shape}")
            print(f"motion.dof_vel: {motion.dof_vel.shape}")
            print(f"motion.rigid_body_pos: {motion.rigid_body_pos.shape}")
            print(f"motion.rigid_body_rot: {motion.rigid_body_rot.shape}")

            print(f"Saving to {outpath}")
            torch.save(motion.to_dict(), str(outpath))

        except Exception as e:
            print(f"Error processing {motion_file.name}: {e}")
            print(f"Skipping {motion_file.name} due to error")
            import traceback

            traceback.print_exc()
            continue  # Skip this file and continue with the next one


if __name__ == "__main__":
    with torch.no_grad():
        app()
