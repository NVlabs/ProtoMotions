# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
import hashlib
import os
from pathlib import Path
from typing import Optional
import torch
import typer
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_from_transforms_with_velocities,
    compute_cartesian_velocity,
    extract_transforms_from_qpos,
    extract_qpos_from_transforms,
)
from protomotions.robot_configs.factory import robot_config
from contact_detection import compute_contact_labels_from_pos_and_vel
from motion_filter import passes_exclude_motion_filter

app = typer.Typer(pretty_exceptions_enable=False)

"""
    Converts retargeted G1 CSV files to the ProtoMotions motion format.

    Expected CSV format (with header row):
        Frame, root_translateX, root_translateY, root_translateZ,
               root_rotateX,    root_rotateY,    root_rotateZ,
               <joint>_dof, ...

    Conventions:
        - root_translate*: root position in centimeters (converted to meters)
        - root_rotate*:    root orientation as extrinsic XYZ Euler angles in degrees
                           (converted to wxyz quaternion)
        - <joint>_dof:     joint angles in degrees (converted to radians)
        - Joint column order matches the G1 MJCF joint order exactly.
"""


def euler_to_quat_wxyz(euler_deg: np.ndarray, order: str) -> np.ndarray:
    """
    Convert Euler angles (degrees) to quaternion (w, x, y, z).

    The CSV columns are always [rotX, rotY, rotZ].  The ``order`` string is
    passed directly to ``scipy.spatial.transform.Rotation.from_euler`` and
    controls both axis order and sign:

    * Uppercase letters → intrinsic rotations (e.g. "XYZ", "YXZ", "ZYX")
    * Lowercase letters → extrinsic rotations (e.g. "xyz", "zyx")
    * Axis letters refer to the *column* order, so "YXZ" means treat
      col-0 as Y, col-1 as X, col-2 as Z before composing.

    The confirmed correct convention for the G1 retargeted CSV dataset is
    "xyz" (extrinsic XYZ), which is the default.

    Args:
        euler_deg: (N, 3) array of Euler angles in degrees [rotX, rotY, rotZ]
        order: scipy-compatible Euler order string

    Returns:
        (N, 4) array of quaternions in wxyz order
    """
    rot = Rotation.from_euler(order, euler_deg, degrees=True)
    xyzw = rot.as_quat()  # scipy returns (x, y, z, w)
    wxyz = np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=-1)
    return wxyz


def process_csv_file(
    csv_path,
    input_fps,
    output_fps,
    device,
    dtype,
    euler_order,
    pos_units="cm",
    rot_format="euler_deg",
    joint_units="deg",
    has_header=True,
    has_frame_column=True,
):
    """
    Process a G1 CSV file and extract motion data.

    Supports two CSV layouts:
      1. Original retargeted format (has_header=True, has_frame_column=True):
         Frame | root_translateX/Y/Z (cm) | root_rotateX/Y/Z (deg) | joint_dofs (deg)
      2. Kimodo generated format (has_header=False, has_frame_column=False):
         root_posX/Y/Z (m) | root_quat_w/x/y/z | joint_dofs (rad)

    Args:
        pos_units: "cm" or "m" for root position units.
        rot_format: "euler_deg" (3 cols, Euler degrees) or "quat_wxyz" (4 cols, wxyz quaternion).
        joint_units: "deg" or "rad" for joint angle units.
        has_header: Whether the CSV has a header row to skip.
        has_frame_column: Whether column 0 is a frame index to skip.

    Returns:
        tuple: (root_pos, root_rot_wxyz, joint_angles)
               root_pos      : (N, 3) float tensor, meters
               root_rot_wxyz : (N, 4) float tensor, unit quaternion wxyz
               joint_angles  : (N, J) float tensor, radians
    """
    skip = 1 if has_header else 0
    data = np.loadtxt(csv_path, delimiter=",", skiprows=skip)

    col = 1 if has_frame_column else 0

    root_pos = data[:, col : col + 3]
    if pos_units == "cm":
        root_pos = root_pos / 100.0
    col += 3

    if rot_format == "euler_deg":
        root_euler_deg = data[:, col : col + 3]
        root_rot_wxyz = euler_to_quat_wxyz(root_euler_deg, euler_order)
        col += 3
    elif rot_format == "quat_wxyz":
        root_rot_wxyz = data[:, col : col + 4]
        col += 4
    else:
        raise ValueError(f"Unknown rot_format: {rot_format}")

    joint_angles = data[:, col:]
    if joint_units == "deg":
        joint_angles = np.deg2rad(joint_angles)

    factor = input_fps // output_fps
    if factor > 1:
        root_pos = root_pos[::factor]
        root_rot_wxyz = root_rot_wxyz[::factor]
        joint_angles = joint_angles[::factor]

    root_pos = torch.from_numpy(root_pos).to(device, dtype)
    root_rot_wxyz = torch.from_numpy(root_rot_wxyz).to(device, dtype)
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
    """
    base_filename = os.path.splitext(motion_filename)[0]
    if base_filename.endswith("_retargeted"):
        base_filename = base_filename[: -len("_retargeted")]

    contact_labels_path = contact_labels_dir / f"{base_filename}_contacts.npz"

    if not os.path.exists(contact_labels_path):
        raise FileNotFoundError(
            f"Contact labels file not found: {contact_labels_path}\n"
            f"Please ensure contact files are generated for all motions."
        )

    contact_data = np.load(contact_labels_path, allow_pickle=True)
    foot_contacts = contact_data["foot_contacts"]  # [K, 2] - left, right

    factor = input_fps // output_fps
    if factor > 1:
        foot_contacts = foot_contacts[::factor]

    motion_length = motion.rigid_body_pos.shape[0]
    contact_length = foot_contacts.shape[0]
    assert motion_length == contact_length, (
        f"Motion length ({motion_length}) does not match contact length ({contact_length}) "
        f"for {motion_filename}. Contact file: {contact_labels_path}"
    )

    num_bodies = motion.rigid_body_pos.shape[1]
    rigid_body_contacts = np.zeros((motion_length, num_bodies), dtype=bool)
    rigid_body_contacts[:, left_foot_idx] = foot_contacts[:, 0] > 0.5
    rigid_body_contacts[:, right_foot_idx] = foot_contacts[:, 1] > 0.5

    motion.rigid_body_contacts = torch.from_numpy(rigid_body_contacts).to(device)
    print(f"Applied contact labels from {contact_labels_path}")


@app.command()
def main(
    input_dir: Path = typer.Option(
        ..., help="Directory (or parent directory) with G1 retargeted CSV files."
    ),
    output_dir: Path = typer.Option(
        ..., help="Directory to save ProtoMotions motion files."
    ),
    input_fps: int = typer.Option(30, help="Input motion fps"),
    output_fps: int = typer.Option(30, help="Output motion fps"),
    force_remake: bool = False,
    ignore_first_n_frames: int = 0,
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
    contact_labels_dir: Optional[Path] = typer.Option(
        None,
        help="Directory with contact label files (*_contacts.npz). If provided, use these for rigid_body_contacts.",
    ),
    pos_units: str = typer.Option(
        "cm", help="Root position units in CSV: 'cm' or 'm'."
    ),
    rot_format: str = typer.Option(
        "euler_deg",
        help="Root rotation format: 'euler_deg' (3 cols) or 'quat_wxyz' (4 cols).",
    ),
    joint_units: str = typer.Option(
        "deg", help="Joint angle units in CSV: 'deg' or 'rad'."
    ),
    has_header: bool = typer.Option(True, help="CSV has a header row to skip."),
    has_frame_column: bool = typer.Option(
        True, help="CSV column 0 is a frame index to skip."
    ),
    euler_order: str = typer.Option(
        "xyz",
        help=(
            "Scipy-compatible Euler order for root rotation (only used with --rot-format euler_deg). "
            "Uppercase = intrinsic, lowercase = extrinsic."
        ),
    ),
    num_rank: int = typer.Option(
        1, help="Total number of parallel ranks (for SLURM array splitting)."
    ),
    slurm_rank: int = typer.Option(
        0, help="This rank's index (0-based). Files are assigned via SHA-256 hash of relative path."
    ),
):
    device = torch.device("cpu")
    dtype = torch.float32

    os.makedirs(output_dir, exist_ok=True)

    robot_mjcf_mapping = {
        "g1": "g1_bm_box_feet.xml",
        "h1_2": "h1_2.xml",
    }

    mjcf_filename = robot_mjcf_mapping.get(robot_type, f"{robot_type}.xml")
    mjcf_path = f"protomotions/data/assets/mjcf/{mjcf_filename}"
    if not os.path.exists(mjcf_path):
        raise FileNotFoundError(f"MJCF file not found at {mjcf_path}")

    kinematic_info = extract_kinematic_info(mjcf_path)

    robot_cfg = robot_config(robot_type)
    left_foot_name = robot_cfg.common_naming_to_robot_body_names["all_left_foot_bodies"][0]
    right_foot_name = robot_cfg.common_naming_to_robot_body_names["all_right_foot_bodies"][0]

    body_names = kinematic_info.body_names
    left_foot_idx = body_names.index(left_foot_name)
    right_foot_idx = body_names.index(right_foot_name)

    print(f"Robot type: {robot_type}")
    print(f"Left foot:  {left_foot_name} (index {left_foot_idx})")
    print(f"Right foot: {right_foot_name} (index {right_foot_idx})")

    if input_fps % output_fps != 0:
        raise ValueError(
            f"input_fps ({input_fps}) must be divisible by output_fps ({output_fps})"
        )

    if rot_format == "euler_deg":
        try:
            Rotation.from_euler(euler_order, [0, 0, 0], degrees=True)
        except Exception as exc:
            raise ValueError(
                f"--euler-order '{euler_order}' is not a valid scipy Euler order string: {exc}"
            ) from exc
        print(f"Root Euler order: {euler_order}")
    print(f"CSV format: pos_units={pos_units}, rot_format={rot_format}, joint_units={joint_units}, header={has_header}, frame_col={has_frame_column}")

    # Collect CSV files recursively, preserving relative paths
    csv_files = []
    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if f.lower().endswith(".csv"):
                csv_files.append(Path(root) / f)
    csv_files.sort()
    print(f"Found {len(csv_files)} CSV files total (num_rank={num_rank}, slurm_rank={slurm_rank}).")

    for csv_file_path in tqdm(csv_files, desc="Processing motions"):
        motion_file = Path(csv_file_path)
        rel_path = motion_file.relative_to(input_dir)

        # Deterministic file-level splitting via SHA-256 hash
        file_hash = int(hashlib.sha256(str(rel_path).encode("utf-8")).hexdigest(), 16)
        if file_hash % num_rank != slurm_rank:
            continue

        outpath = (output_dir / str(rel_path).replace(" ", "_")).with_suffix(".motion")
        outpath.parent.mkdir(parents=True, exist_ok=True)

        if not force_remake and outpath.exists():
            continue

        print(f"Processing {motion_file.name}")

        try:
            root_pos, root_rot_wxyz, joint_angles = process_csv_file(
                csv_file_path,
                input_fps,
                output_fps,
                device,
                dtype,
                euler_order,
                pos_units=pos_units,
                rot_format=rot_format,
                joint_units=joint_units,
                has_header=has_header,
                has_frame_column=has_frame_column,
            )

            expected_dofs = kinematic_info.num_dofs
            actual_cols = joint_angles.shape[-1]
            if actual_cols != expected_dofs:
                raise ValueError(
                    f"{motion_file.name}: joint angle columns ({actual_cols}) "
                    f"!= expected DOFs ({expected_dofs}) from MJCF. "
                    f"Check --rot-format (euler_deg=3 cols, quat_wxyz=4 cols) "
                    f"and --has-frame-column settings."
                )

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
                velocity_max_horizon=3,
            )

            # Re-extract qpos to ensure joint angles are in [-pi, pi]
            qpos = extract_qpos_from_transforms(
                kinematic_info, root_pos, joint_rot_mats
            )
            motion.dof_pos = qpos[:, 7:]

            allowed_delta = [0.0, 2 * np.pi, 4 * np.pi]
            delta = (qpos[:, 7:] - joint_angles).abs()
            epsilon = 1e-4
            allowed = torch.zeros_like(delta, dtype=torch.bool)
            for d in allowed_delta:
                allowed |= (delta - d).abs() < epsilon
            assert allowed.all(), (
                "qpos and joint_angles diverge beyond the allowed delta (epsilon tolerance exceeded)"
            )

            dof_vel = compute_cartesian_velocity(
                batched_robot_pos=joint_angles.unsqueeze(1),
                fps=output_fps,
            )
            motion.dof_vel = dof_vel.squeeze(1)

            # Fix height per frame, then update velocities accordingly
            translation_vecs = motion.fix_height_per_frame(height_offset=0.02)
            if motion.rigid_body_vel is not None and motion.fps is not None:
                vel_delta = torch.zeros(
                    translation_vecs.shape[0],
                    1,
                    3,
                    device=motion.rigid_body_vel.device,
                    dtype=motion.rigid_body_vel.dtype,
                )
                vel_delta[:-1] = (
                    (translation_vecs[1:] - translation_vecs[:-1]).unsqueeze(1)
                    / motion.motion_dt
                )
                motion.rigid_body_vel = motion.rigid_body_vel + vel_delta

            motion.fix_height(height_offset=0.04)

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
                motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
                    positions=motion.rigid_body_pos,
                    velocity=motion.rigid_body_vel,
                    vel_thres=0.15,
                    height_thresh=0.1,
                ).to(torch.bool)

            motion.local_rigid_body_rot = None  # prevent motion_lib from interpolating

            if apply_motion_filter:
                if not passes_exclude_motion_filter(
                    motion,
                    min_height_threshold=min_height_threshold,
                    max_velocity_threshold=max_velocity_threshold,
                    max_dof_vel_threshold=max_dof_vel_threshold,
                    duration_height_filter=duration_height_filter,
                    duration_height_seconds=duration_height_seconds,
                ):
                    print(f"Skipping {motion_file.name} (failed motion filter)")
                    continue

            print(f"  dof_pos:         {motion.dof_pos.shape}")
            print(f"  dof_vel:         {motion.dof_vel.shape}")
            print(f"  rigid_body_pos:  {motion.rigid_body_pos.shape}")
            print(f"  rigid_body_rot:  {motion.rigid_body_rot.shape}")
            print(f"  Saving to {outpath}")
            torch.save(motion.to_dict(), str(outpath))

        except Exception as e:
            print(f"Error processing {motion_file.name}: {e}")
            import traceback

            traceback.print_exc()
            continue


if __name__ == "__main__":
    with torch.no_grad():
        app()
