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
Convert rigv1 npz motion files to ProtoMotions format.

Usage:
    python data/scripts/convert_rigv1_npz_to_proto.py <input_dir> <output_dir> [options]

Example:
    python data/scripts/convert_rigv1_npz_to_proto.py \
            examples/data/rigv1/ \
            examples/data/rigv1/proto \
            --input-fps 30 \
            --output-fps 30
"""
from typing import Optional
import typer
import os
from pathlib import Path
import numpy as np
import torch
import yaml

from protomotions.components.pose_lib import extract_kinematic_info

from convert_rigv1_to_proto import (
    passes_exclude_motion_filter,
    gen_yaml_one_motion_default,
    create_motion_from_rigv1_data,
)
from keypoint_utils import extract_keypoints_from_motion, get_keypoint_indices

app = typer.Typer(pretty_exceptions_enable=True)


@app.command()
def main(
    input_dir: Path,
    output_dir: Path,
    input_fps: int = 120,
    output_fps: int = 30,
    # Motion filter options
    ignore_motion_filter: bool = False,
    min_height_threshold: float = -0.05,
    max_velocity_threshold: float = 15.0,
    max_dof_vel_threshold: float = 40.0,
    duration_height_filter: float = 0.2,
    duration_height_seconds: float = 1.0,
    # Output options
    yaml_output_name: Optional[str] = None,
    extract_keypoints: bool = False,
    keypoints_output_path: Optional[Path] = None,
    force_remake: bool = False,
):
    """Convert rigv1 npz motion files to ProtoMotions format."""
    if yaml_output_name is not None:
        yaml_output = output_dir / yaml_output_name
    else:
        yaml_output = None

    device = torch.device("cpu")
    dtype = torch.float32

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

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_fps % output_fps != 0:
        raise ValueError(
            f"input_fps ({input_fps}) must be divisible by output_fps ({output_fps})"
        )

    # Process all .npz files in input_dir (flat, no recursion)
    npz_files = sorted(input_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} npz files in {input_dir}")

    for npz_file in npz_files:
        file_name = npz_file.stem + ".motion"
        output_file = output_dir / file_name

        if not force_remake and output_file.exists():
            print(f"Skipping {file_name} because it already exists")
            continue

        print(f"Processing {npz_file}")

        data = np.load(npz_file, allow_pickle=True)

        # Check required fields
        if "global_rot_mats" not in data or "root_positions" not in data:
            raise ValueError(f"{npz_file} doesn't contain required fields 'global_rot_mats' and 'root_positions'")

        # Load and squeeze batch dimension (assume first dim is size 1)
        global_rot_mats = data["global_rot_mats"]
        root_pos = data["root_positions"]

        if global_rot_mats.ndim == 5:
            global_rot_mats = global_rot_mats[0]  # (1, T, 27, 3, 3) -> (T, 27, 3, 3)
        if root_pos.ndim == 3:
            root_pos = root_pos[0]  # (1, T, 3) -> (T, 3)

        print(f"  global_rot_mats: {global_rot_mats.shape}")
        print(f"  root_pos: {root_pos.shape}")

        # Validate shapes
        if (
            global_rot_mats.ndim != 4
            or global_rot_mats.shape[1] != 27
            or global_rot_mats.shape[2:] != (3, 3)
        ):
            print(
                f"Skipping {npz_file} because global_rot_mats has wrong shape: "
                f"{global_rot_mats.shape}, expected: [T, 27, 3, 3]"
            )
            continue

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            print(
                f"Skipping {npz_file} because root_pos has wrong shape: "
                f"{root_pos.shape}, expected: [T, 3]"
            )
            continue

        if global_rot_mats.shape[0] != root_pos.shape[0]:
            print(
                f"Skipping {npz_file} because global_rot_mats and root_pos "
                "have different sequence lengths"
            )
            continue


        downsample_factor = input_fps // output_fps
        global_rot_mats = global_rot_mats[::downsample_factor]
        root_pos = root_pos[::downsample_factor]

        # Create motion using helper function
        motion = create_motion_from_rigv1_data(
            global_rot_mats=global_rot_mats,
            root_pos=root_pos,
            kinematic_info=kinematic_info,
            fps=output_fps,
            device=device,
            dtype=dtype,
        )

        # Apply motion filtering
        if not ignore_motion_filter and not passes_exclude_motion_filter(
            motion,
            min_height_threshold=min_height_threshold,
            max_velocity_threshold=max_velocity_threshold,
            max_dof_vel_threshold=max_dof_vel_threshold,
            duration_height_filter=duration_height_filter,
            duration_height_seconds=duration_height_seconds,
        ):
            print(f"Skipping {npz_file.name} because it does not pass motion filter")
            continue

        # Extract and save keypoints for pyroki retargeting if enabled
        if extract_keypoints:
            keypoint_output_file = keypoints_output_path / f"{npz_file.stem}_keypoints.npy"

            if not force_remake and keypoint_output_file.exists():
                print(f"Skipping keypoint extraction for {npz_file.name}, file already exists.")
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

        # Save motion
        torch.save(motion.to_dict(), str(output_file))
        print(f"Saved to {output_file}")

        if yaml_output is not None:
            output_motions_yaml.append(
                gen_yaml_one_motion_default(file_name, output_fps, output_yaml_idx)
            )
            output_yaml_idx += 1

    if yaml_output is not None:
        with open(yaml_output, "w") as f:
            yaml.dump({"motions": output_motions_yaml}, f)
        print(f"Saved motions list to {yaml_output}")


if __name__ == "__main__":
    app()

