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
"""
    Converts SOMASkeleton77 motions (local joint_rot_mats + posed_joints)
    to ProtoMotions format for the soma23 humanoid.

    Data layout:
        posed_joints/<session>/<motion>.npy   — (T, 77, 3) joint positions, y-up
        joint_rot_mats/<session>/<motion>.npy  — (T, 77, 3, 3) LOCAL rotation matrices

    The 77-body SOMA skeleton is subselected down to the 23 bodies in
    soma23_humanoid.xml, dropping hands, finger details, face joints,
    and toe ends (leaf end-effectors without actuators).

    Coordinate system is converted from y-up to z-up.
"""
import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from scipy.spatial.transform import Rotation as R

from protomotions.utils.rotations import (
    matrix_to_quaternion,
    quat_mul,
    quat_rotate,
    quaternion_to_matrix,
)
from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_from_transforms_with_velocities,
    extract_qpos_from_transforms,
    compute_angular_velocity,
    compute_forward_kinematics_from_transforms,
)
from contact_detection import compute_contact_labels_from_pos_and_vel
from motion_filter import passes_exclude_motion_filter

app = typer.Typer(pretty_exceptions_enable=False)

# SOMASkeleton77 bone order (matches the data array axis-1 ordering)
SOMASKEL77_BONE_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest",
    "Neck1", "Neck2", "Head", "HeadEnd", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]

# MJCF body order (from soma23_humanoid.xml tree traversal)
MJCF_BODY_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest",
    "Neck1", "Neck2", "Head",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
]

# Indices into the 77-body array for each of the 23 MJCF bodies
SOMASKEL77_TO_MJCF_INDICES = [
    SOMASKEL77_BONE_NAMES.index(name) for name in MJCF_BODY_NAMES
]


def create_motion_from_soma23_global_rotations(
    global_rot_mats: torch.Tensor,  # [T, 23, 3, 3] globals from MJCF FK
    root_pos: torch.Tensor,  # [T, 3]
    kinematic_info,
    fps: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    """
    Create a RobotState from global rotations produced by running MJCF FK on
    SOMASkeleton local rotations (npy/npz pipeline). Applies rot1+rot2
    y-up → z-up correction, then derives everything from MJCF FK.
    """
    from protomotions.components.pose_lib import compute_joint_rot_mats_from_global_mats

    num_bodies = global_rot_mats.shape[1]
    num_joints = num_bodies - 1

    global_quat = matrix_to_quaternion(global_rot_mats, w_last=True)

    rot1 = R.from_euler("xyz", np.array([-np.pi / 2, 0, 0]), degrees=False)
    rot2 = R.from_euler("xyz", np.array([-np.pi / 2, np.pi, 0]), degrees=False)

    rot1_quat = (
        torch.from_numpy(rot1.as_quat())
        .to(device, dtype)
        .expand(global_quat.shape[0], -1)
    )
    rot2_quat = (
        torch.from_numpy(rot2.as_quat())
        .to(device, dtype)
        .expand(global_quat.shape[0], -1)
    )

    for i in range(num_bodies):
        global_quat[:, i, :] = quat_mul(global_quat[:, i, :], rot1_quat, w_last=True)
        global_quat[:, i, :] = quat_mul(rot2_quat, global_quat[:, i, :], w_last=True)

    root_pos = quat_rotate(rot2_quat, root_pos, w_last=True)

    local_rot_mats_rotated = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=quaternion_to_matrix(global_quat, w_last=True),
    )

    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=local_rot_mats_rotated,
        fps=fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )

    motion.local_rigid_body_rot = matrix_to_quaternion(local_rot_mats_rotated, w_last=True)

    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=local_rot_mats_rotated,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]

    local_angular_vels = compute_angular_velocity(
        batched_robot_rot_mats=local_rot_mats_rotated[:, 1:, :, :],
        fps=fps,
    )
    assert local_angular_vels.shape[1] == num_joints
    motion.dof_vel = local_angular_vels.reshape(-1, num_joints * 3)

    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)

    return motion


def create_motion_from_soma23_data(
    local_rot_mats: torch.Tensor,  # [T, 23, 3, 3] (already subselected)
    root_pos: torch.Tensor,  # [T, 3]
    kinematic_info,
    fps: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    """
    Create a RobotState motion from SOMASkeleton local rotation matrices
    that are compatible with the MJCF kinematic_info (npy / npz pipeline).

    The data is y-up; we run FK to get global rotations, then delegate to
    create_motion_from_soma23_global_rotations for the y-up → z-up conversion.
    """
    # FK in the original y-up frame to get global rotations.
    _, world_rot_mat = compute_forward_kinematics_from_transforms(
        kinematic_info, root_pos, local_rot_mats
    )

    return create_motion_from_soma23_global_rotations(
        global_rot_mats=world_rot_mat,
        root_pos=root_pos,
        kinematic_info=kinematic_info,
        fps=fps,
        device=device,
        dtype=dtype,
    )


@app.command()
def main(
    input_dir: Path = typer.Option(
        ..., help="Root directory with posed_joints/ and joint_rot_mats/ subdirs."
    ),
    output_dir: Path = typer.Option(
        ..., help="Directory to save ProtoMotions motion files."
    ),
    input_fps: int = typer.Option(120, help="Input motion fps"),
    output_fps: int = typer.Option(30, help="Output motion fps"),
    force_remake: bool = False,
    min_height_threshold: float = typer.Option(-0.05),
    max_velocity_threshold: float = typer.Option(15.0),
    max_dof_vel_threshold: float = typer.Option(40.0),
    duration_height_filter: float = typer.Option(0.1),
    duration_height_seconds: float = typer.Option(0.6),
    ignore_motion_filter: bool = False,
    num_rank: int = typer.Option(1, help="Total number of parallel ranks."),
    slurm_rank: int = typer.Option(0, help="This rank's index (0-based)."),
):
    if input_fps % output_fps != 0:
        raise ValueError(
            f"input_fps ({input_fps}) must be divisible by output_fps ({output_fps})"
        )

    device = torch.device("cpu")
    dtype = torch.float32

    kinematic_info = extract_kinematic_info(
        "protomotions/data/assets/mjcf/soma23_humanoid.xml"
    )
    print("kinematic_info:", kinematic_info)
    assert kinematic_info.num_bodies == 23
    assert kinematic_info.nq == 22 * 3 + 7

    downsample_factor = input_fps // output_fps

    posed_joints_dir = input_dir / "posed_joints"
    joint_rot_mats_dir = input_dir / "joint_rot_mats"

    for root, _dirs, files in os.walk(posed_joints_dir):
        for f in sorted(files):
            if not f.endswith(".npy"):
                continue

            rel_path = Path(root).relative_to(posed_joints_dir)

            file_hash = int(
                hashlib.sha256(str(rel_path / f).encode("utf-8")).hexdigest(), 16
            )
            if file_hash % num_rank != slurm_rank:
                continue

            pos_path = Path(root) / f
            rot_path = joint_rot_mats_dir / rel_path / f

            if not rot_path.exists():
                print(f"Skipping {f}: no matching joint_rot_mats file")
                continue

            out_subdir = output_dir / rel_path
            out_subdir.mkdir(parents=True, exist_ok=True)
            motion_filename = Path(f).stem + ".motion"
            output_file = out_subdir / motion_filename

            if not force_remake and output_file.exists():
                continue

            print(f"Processing {rel_path / f}")

            try:
                posed = np.load(pos_path)    # (T, 77, 3)
                rot = np.load(rot_path)      # (T, 77, 3, 3)

                # Subselect to MJCF bodies
                rot = rot[:, SOMASKEL77_TO_MJCF_INDICES, :, :]  # (T, 23, 3, 3)
                root_pos = posed[:, 0, :]  # (T, 3) — root position from posed_joints

                # Downsample
                rot = rot[::downsample_factor]
                root_pos = root_pos[::downsample_factor]

                root_pos = torch.from_numpy(root_pos).to(device, dtype)
                local_rot_mats = torch.from_numpy(rot).to(device, dtype)

                motion = create_motion_from_soma23_data(
                    local_rot_mats=local_rot_mats,
                    root_pos=root_pos,
                    kinematic_info=kinematic_info,
                    fps=output_fps,
                    device=device,
                    dtype=dtype,
                )

                if not ignore_motion_filter and not passes_exclude_motion_filter(
                    motion,
                    min_height_threshold=min_height_threshold,
                    max_velocity_threshold=max_velocity_threshold,
                    max_dof_vel_threshold=max_dof_vel_threshold,
                    duration_height_filter=duration_height_filter,
                    duration_height_seconds=duration_height_seconds,
                ):
                    print(f"Skipping {f} (failed motion filter)")
                    continue

                print(f"  dof_pos:         {motion.dof_pos.shape}")
                print(f"  rigid_body_pos:  {motion.rigid_body_pos.shape}")
                print(f"  Saving to {output_file}")
                torch.save(motion.to_dict(), str(output_file))

            except Exception as e:
                print(f"Error processing {f}: {e}")
                import traceback
                traceback.print_exc()
                continue


if __name__ == "__main__":
    with torch.no_grad():
        app()
