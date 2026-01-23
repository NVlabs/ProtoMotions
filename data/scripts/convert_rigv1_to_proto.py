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
from typing import Optional, Dict, Any, Tuple

import typer
import os
from pathlib import Path
import numpy as np
import torch
import hashlib
import yaml
from protomotions.utils.rotations import (
    matrix_to_quaternion,
    quat_mul,
    quat_rotate,
    quaternion_to_matrix,
)
from scipy.spatial.transform import Rotation as R

from protomotions.simulator.base_simulator.simulator_state import RobotState

from protomotions.components.pose_lib import (
    compute_angular_velocity,
    compute_joint_rot_mats_from_global_mats,
    extract_kinematic_info,
    fk_from_transforms_with_velocities,
    extract_qpos_from_transforms,
)
from contact_detection import compute_contact_labels_from_pos_and_vel
from motion_filter import passes_exclude_motion_filter
from keypoint_utils import extract_keypoints_from_motion, get_keypoint_indices

app = typer.Typer(pretty_exceptions_enable=True)


def gen_yaml_one_motion_default(
    output_motion_path: Path,
    fps: int,
    idx: int,
    additional_fields: Optional[Dict[str, Any]] = None,
):
    result = {
        "file": str(output_motion_path),
        "fps": fps,
        "weight": 1.0,
        "idx": idx,
    }
    if additional_fields is not None:
        result.update(additional_fields)
    return result


def create_motion_from_rigv1_data(
    global_rot_mats: torch.Tensor,  # [T, 27, 3, 3]
    root_pos: torch.Tensor,  # [T, 3]
    kinematic_info,
    fps: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> RobotState:
    """
    Create a RobotState motion from Rigv1 global rotation matrices and root position.

    Args:
        global_rot_mats: Global rotation matrices [T, 27, 3, 3]
        root_pos: Root position [T, 3]
        kinematic_info: Kinematic information for the robot
        fps: Frame rate
        device: Torch device
        dtype: Torch dtype

    Returns:
        RobotState motion object
    """
    # Convert to torch tensors if not already
    if not isinstance(global_rot_mats, torch.Tensor):
        global_rot_mats = torch.from_numpy(global_rot_mats)
    if not isinstance(root_pos, torch.Tensor):
        root_pos = torch.from_numpy(root_pos)

    global_rot_mats = global_rot_mats.to(device, dtype)
    root_pos = root_pos.to(device, dtype)

    # Convert to quaternions and filter joints
    global_quat = matrix_to_quaternion(global_rot_mats, w_last=True)
    # for removing 4 dead joints in the Rigv1 raw data
    joints_in_xml_order_idx = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        13,
        14,
        15,
        16,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
    ]
    global_quat = global_quat[:, joints_in_xml_order_idx]
    assert global_quat.shape[1] == 23

    # Apply coordinate system transformations
    # rot1 transforms motion rotations represented with a y-up character to a z-up character
    # but did not change the motion itself
    # rot2 changes the motion from y-up to z-up
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

    for i in range(0, 23):
        global_quat[:, i, :] = quat_mul(global_quat[:, i, :], rot1_quat, w_last=True)
        global_quat[:, i, :] = quat_mul(rot2_quat, global_quat[:, i, :], w_last=True)

    root_pos = quat_rotate(rot2_quat, root_pos, w_last=True)

    # compute vels of local rotation
    local_rot_mats = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=quaternion_to_matrix(global_quat, w_last=True),
    )

    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=local_rot_mats,
        fps=fps,
        compute_velocities=True,
        velocity_max_horizon=3,  # Use multi-horizon minimum for noise-filtered velocities
    )
    # caching local rotation to disk file, in case anyone needs it later
    motion.local_rigid_body_rot = matrix_to_quaternion(local_rot_mats, w_last=True)

    # calc dof pos and dof vel
    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_pos,
        joint_rot_mats=local_rot_mats,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]  # (T, 22 * 3)

    local_angular_vels = compute_angular_velocity(
        batched_robot_rot_mats=local_rot_mats[:, 1:, :, :],
        fps=fps,
    )
    assert local_angular_vels.shape[1] == 22  # (T, 22, 3)
    # because we know all joints are 3 dof exp_map joints...
    motion.dof_vel = local_angular_vels.reshape(-1, 22 * 3)

    # compute contacts using position and velocity thresholds
    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)

    return motion


def apply_height_augmentation(
    motion: RobotState,
    output_file: Path,
    output_path: Path,
    output_fps: int,
    height_augmentation_min: float = 0.8,
    height_augmentation_max: float = 1.2,
    n_repeat_aug: int = 1,
    output_motions_yaml: Optional[list] = None,
    output_yaml_idx: Optional[int] = None,
) -> Tuple[list, int]:
    """
    Apply height augmentation to a motion and save all variants.

    Args:
        motion: Original RobotState motion
        output_file: Output file path for the original motion
        output_path: Base output directory
        output_fps: Frame rate
        height_augmentation_min: Minimum height scale factor
        height_augmentation_max: Maximum height scale factor
        n_repeat_aug: Number of augmentations to create
        output_motions_yaml: List to append YAML entries to
        output_yaml_idx: Starting index for YAML entries

    Returns:
        Tuple of (updated_yaml_list, next_yaml_idx)
    """
    if output_motions_yaml is None:
        output_motions_yaml = []
    if output_yaml_idx is None:
        output_yaml_idx = 0

    scale_factors = torch.linspace(
        height_augmentation_min, height_augmentation_max, n_repeat_aug - 1
    )

    for aug_idx in range(n_repeat_aug):
        motion_clone: RobotState = motion.clone()

        global_translation_z = motion_clone.rigid_body_pos[:, :, 2]
        global_translation_z_min_per_frame = global_translation_z.min(
            dim=-1
        ).values  # (T, )

        # First augmentation (aug_idx=0) is the original (scale=1.0)
        # Subsequent augmentations use scale factors on a linear scale
        if aug_idx == 0:
            scale_factor = 1.0
        else:
            scale_factor = scale_factors[aug_idx - 1].item()

        floor_height = global_translation_z_min_per_frame.min().item()
        heights_per_frame = global_translation_z_min_per_frame - floor_height
        new_heights_per_frame = scale_factor * heights_per_frame
        new_global_translation_z_min_per_frame = floor_height + new_heights_per_frame
        global_translation_diff_z = (
            new_global_translation_z_min_per_frame - global_translation_z_min_per_frame
        )  # (T, )

        global_translation_diff = torch.zeros(
            len(global_translation_diff_z),
            3,
            device=motion_clone.rigid_body_pos.device,
            dtype=motion_clone.rigid_body_pos.dtype,
        )
        global_translation_diff[:, 2] = global_translation_diff_z

        motion_clone.translate(global_translation_diff)
        
        # Manually update velocities for time-series translation
        if motion_clone.rigid_body_vel is not None and motion_clone.fps is not None:
            vel_delta = torch.zeros(
                global_translation_diff.shape[0], 1, 3,
                device=motion_clone.rigid_body_vel.device,
                dtype=motion_clone.rigid_body_vel.dtype,
            )
            vel_delta[:-1] = (global_translation_diff[1:] - global_translation_diff[:-1]).unsqueeze(1) / motion_clone.motion_dt
            motion_clone.rigid_body_vel = motion_clone.rigid_body_vel + vel_delta
        
        motion_clone.fix_height()

        # Create output filename with augmentation index suffix
        if aug_idx == 0:
            curr_output_file = output_file
            curr_output_file_relative = output_file.relative_to(output_path)
        else:
            # Add augmentation index to filename before extension
            base_name = output_file.stem
            extension = output_file.suffix
            curr_output_file = output_file.with_name(
                f"{base_name}_aug{aug_idx}{extension}"
            )
            curr_output_file_relative = curr_output_file.relative_to(output_path)

        # Save the motion
        aug_motion_dict = motion_clone.to_dict()
        aug_motion_dict["scale_factor"] = scale_factor  # save the scale factor also
        torch.save(aug_motion_dict, str(curr_output_file))
        print(f"Saved to {curr_output_file} with scale factor {scale_factor:.3f}")

        if len(output_motions_yaml) >= 0:  # if yaml list is provided
            output_motions_yaml.append(
                gen_yaml_one_motion_default(
                    curr_output_file_relative,
                    output_fps,
                    output_yaml_idx,
                    additional_fields={"scale_factor": scale_factor},
                )
            )
            output_yaml_idx += 1

    return output_motions_yaml, output_yaml_idx


@app.command()
def main(
    rigv1_rot_pos_path: Path,
    output_path: Path,
    input_fps: int = 120,
    output_fps: int = 30,
    num_rank: int = 1,
    slurm_rank: int = 0,
    min_height_threshold: float = -0.05,
    max_velocity_threshold: float = 15.0,
    max_dof_vel_threshold: float = 40.0,
    duration_height_filter: float = 0.1,
    duration_height_seconds: float = 0.6,
    only_process_yaml: Optional[Path] = None,
    ignore_motion_filter: bool = False,
    height_augmentation: bool = False,
    height_augmentation_min: float = 0.8,
    height_augmentation_max: float = 1.2,
    n_repeat_aug: int = 1,
    yaml_output_name: Optional[str] = None,
    extract_keypoints: bool = False,
    keypoints_output_path: Optional[Path] = None,
    force_remake: bool = False,
):
    if yaml_output_name is not None:
        yaml_output = output_path / yaml_output_name
    else:
        yaml_output = None

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")  # cuda does not seem faster?
    dtype = torch.float32

    # Load file paths from YAML if provided
    specified_files = set()
    if only_process_yaml is not None:
        print(f"Loading specified file paths from {only_process_yaml}")
        print("disable motion filter")
        ignore_motion_filter = True
        with open(only_process_yaml, "r") as f:
            yaml_data = yaml.safe_load(f)
            if "row_mapping" in yaml_data:
                specified_files = {
                    f"{path}.motion" for path in yaml_data["row_mapping"].values()
                }

        print(f"Will only process {len(specified_files)} files specified in YAML")

    kinematic_info = extract_kinematic_info(
        "protomotions/data/assets/mjcf/rigv1_humanoid.xml"
    )
    print("kinematic_info", kinematic_info)
    assert kinematic_info.num_bodies == 23
    assert kinematic_info.nq == 22 * 3 + 7

    if extract_keypoints and keypoints_output_path is None:
        raise typer.Exit(
            "Error: --keypoints-output-path must be provided when --extract-keypoints is enabled."
        )

    if extract_keypoints:
        os.makedirs(keypoints_output_path, exist_ok=True)
        print(f"Keypoints will be saved to: {keypoints_output_path}")
        conceptual_keypoint_names, _, keypoint_indices_in_mjcf = get_keypoint_indices(
            kinematic_info
        )

    output_motions_yaml = []
    output_yaml_idx = 0

    # Walk through all subdirectories in rigv1_rot_pos_path
    for root, dirs, files in os.walk(rigv1_rot_pos_path / "posed_joints"):
        for f in files:
            if f.endswith(".npy"):
                # Get relative path from posed_joints to file
                rel_path = Path(root).relative_to(rigv1_rot_pos_path / "posed_joints")
                # Convert to .motion extension for comparison with specified_files
                output_rel_path = str(rel_path / (Path(f).stem + ".motion"))

                # If yaml_path is provided, only process files that match the paths in the YAML file
                if only_process_yaml is not None:
                    if output_rel_path not in specified_files:
                        # print(f"Skipping {output_rel_path} because it is not in the YAML file")
                        continue
                    else:
                        print(f"Found {output_rel_path} in YAML file, processing...")

                # only process the files for the current rank
                # using hash of the file name to determine the rank
                # file_hash = hash(f)   # --> not good since not deterministic
                file_hash = int(
                    hashlib.sha256(str(rel_path / f).encode("utf-8")).hexdigest(), 16
                )

                if file_hash % num_rank != slurm_rank:
                    print(f"Skipping {f} because it is not for rank {slurm_rank}")
                    continue

                # Construct input paths
                rigv1_pos_path = rigv1_rot_pos_path / "posed_joints" / rel_path / f
                rigv1_rot_path = rigv1_rot_pos_path / "global_rot_mats" / rel_path / f

                if not rigv1_pos_path.exists() or not rigv1_rot_path.exists():
                    print(
                        f"Skipping either {rigv1_pos_path} or {rigv1_rot_path} doesn't exist"
                    )
                    continue

                # Create output directory if it doesn't exist
                output_dir = output_path / rel_path
                output_dir.mkdir(parents=True, exist_ok=True)
                # Change extension from .npy to .motion for torch-saved RobotState
                motion_filename = Path(f).stem + ".motion"
                output_file_relative = rel_path / motion_filename
                output_file = output_dir / motion_filename

                if not force_remake and output_file.exists():
                    print(f"Skipping {output_file_relative} because it already exists")
                    continue

                print(f"Processing {rigv1_pos_path}")

                # Load data
                rigv1_root_pos = np.load(rigv1_pos_path)
                rigv1_root_pos = torch.from_numpy(rigv1_root_pos[:, 0, :]).to(
                    device, dtype
                )

                global_rot_mats = np.load(rigv1_rot_path)
                global_rot_mats = torch.from_numpy(global_rot_mats).to(device, dtype)

                # downsample motion if needed
                assert (
                    input_fps % output_fps == 0
                ), "input_fps must be divisible by output_fps"
                downsample_factor = input_fps // output_fps
                global_rot_mats = global_rot_mats[::downsample_factor]
                rigv1_root_pos = rigv1_root_pos[::downsample_factor]

                # Create motion using helper function
                motion = create_motion_from_rigv1_data(
                    global_rot_mats=global_rot_mats,
                    root_pos=rigv1_root_pos,
                    kinematic_info=kinematic_info,
                    fps=output_fps,
                    device=device,
                    dtype=dtype,
                )

                # Skip motion filtering if YAML file is provided
                if not ignore_motion_filter and not passes_exclude_motion_filter(
                    motion,
                    min_height_threshold=min_height_threshold,
                    max_velocity_threshold=max_velocity_threshold,
                    max_dof_vel_threshold=max_dof_vel_threshold,
                    duration_height_filter=duration_height_filter,
                    duration_height_seconds=duration_height_seconds,
                ):
                    print(
                        f"Skipping {f} because it does not pass exclude motion filter"
                    )
                    continue

                # Extract and save keypoints if enabled
                if extract_keypoints:
                    keypoint_output_dir = keypoints_output_path / rel_path
                    keypoint_output_dir.mkdir(parents=True, exist_ok=True)
                    keypoint_output_file = (
                        keypoint_output_dir / f"{Path(f).stem}_keypoints.npy"
                    )

                    if not force_remake and keypoint_output_file.exists():
                        print(
                            f"Skipping keypoint extraction for {f}, file already exists."
                        )
                    else:
                        keypoint_data = extract_keypoints_from_motion(
                            all_body_positions=motion.rigid_body_pos,
                            all_body_rotations_quat=motion.rigid_body_rot,
                            keypoint_indices_in_mjcf=keypoint_indices_in_mjcf,
                            conceptual_keypoint_names=conceptual_keypoint_names,
                            device=device,
                            flat_feet=True,
                            aux_points=True,
                            contacts=motion.rigid_body_contacts,
                            kinematic_info=kinematic_info,
                        )
                        keypoint_data_to_save = {
                            "positions": keypoint_data["positions"].cpu().numpy(),
                            "orientations": keypoint_data["orientations"].cpu().numpy(),
                            "left_foot_contacts": keypoint_data["left_foot_contacts"],
                            "right_foot_contacts": keypoint_data["right_foot_contacts"],
                        }
                        np.save(str(keypoint_output_file), keypoint_data_to_save)
                        print(f"Saved keypoints to {keypoint_output_file}")

                if not height_augmentation:
                    torch.save(motion.to_dict(), str(output_file))
                    print(f"Saved to {output_file}")

                    if yaml_output is not None:
                        output_motions_yaml.append(
                            gen_yaml_one_motion_default(
                                output_file_relative, output_fps, output_yaml_idx
                            )
                        )
                        output_yaml_idx += 1
                else:
                    # Apply height augmentation
                    output_motions_yaml, output_yaml_idx = apply_height_augmentation(
                        motion=motion,
                        output_file=output_file,
                        output_path=output_path,
                        output_fps=output_fps,
                        height_augmentation_min=height_augmentation_min,
                        height_augmentation_max=height_augmentation_max,
                        n_repeat_aug=n_repeat_aug,
                        output_motions_yaml=output_motions_yaml
                        if yaml_output is not None
                        else None,
                        output_yaml_idx=output_yaml_idx,
                    )

    if yaml_output is not None:
        with open(yaml_output, "w") as f:
            yaml.dump({"motions": output_motions_yaml}, f)
        print(f"Saved motions list to {yaml_output}")


if __name__ == "__main__":
    app()
