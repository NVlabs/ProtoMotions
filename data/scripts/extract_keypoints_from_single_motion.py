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
"""
# Keypoint Extraction from a Single .motion File

This script extracts keypoints from a single ProtoMotions .motion file for use
with PyRoki retargeting. It produces output compatible with the batch retargeting
pipeline.

## Usage

```bash
python data/scripts/extract_keypoints_from_single_motion.py /path/to/motion.motion \\
    --skeleton-format smpl \\
    --output-path /path/to/output_dir
```

## Parameters

- `motion_file`: Path to the .motion file to process.
- `--skeleton-format`: Skeleton format: 'rigv1' or 'smpl' (default: smpl).
- `--output-path`: Directory to save the extracted keypoints. If not provided,
                   defaults to the same directory as the input file.

## Output

Creates a numpy file with:
- `positions`: shape `(traj_length, 18, 3)` - XYZ coordinates
- `orientations`: shape `(traj_length, 18, 3, 3)` - rotation matrices
- `left_foot_contacts`: shape `(traj_length, 2)` - contact labels for left ankle and toebase
- `right_foot_contacts`: shape `(traj_length, 2)` - contact labels for right ankle and toebase
"""

import os
from pathlib import Path
import numpy as np
import torch
import typer
from typing import Optional

from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)
from protomotions.components.pose_lib import extract_kinematic_info

from keypoint_utils import (
    extract_keypoints_from_motion,
    get_keypoint_indices,
    get_mjcf_path,
)

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    motion_file: Path = typer.Argument(
        ..., help="Path to the .motion file to process."
    ),
    output_path: Optional[Path] = typer.Option(
        None,
        help="Directory to save the extracted keypoints. If not provided, uses the same directory as the input file.",
    ),
    skeleton_format: str = typer.Option(
        "smpl", "--skeleton-format", help="Skeleton format: 'rigv1' or 'smpl'."
    ),
    force_remake: bool = typer.Option(
        False, "--force-remake", help="Force reprocessing even if output exists."
    ),
):
    """Extract keypoints from a single .motion file for PyRoki retargeting."""
    device = torch.device("cpu")

    # Validate input file
    if not motion_file.is_file():
        print(f"Error: Motion file not found: {motion_file}")
        raise typer.Exit(code=1)

    if motion_file.suffix != ".motion":
        print(f"Error: Input must be a .motion file. Got: {motion_file}")
        raise typer.Exit(code=1)

    if skeleton_format not in ["rigv1", "smpl"]:
        print(
            f"Error: skeleton_format must be 'rigv1' or 'smpl'. Got: {skeleton_format}"
        )
        raise typer.Exit(code=1)

    # Set output path
    if output_path is None:
        output_path = motion_file.parent

    os.makedirs(output_path, exist_ok=True)

    # Create output filename
    identifier_stem = f"{motion_file.parent.name}_{motion_file.stem}"
    keypoint_output_file = output_path / f"{identifier_stem}_keypoints.npy"

    if not force_remake and keypoint_output_file.exists():
        print(f"Output file already exists: {keypoint_output_file}")
        print("Use --force-remake to overwrite.")
        raise typer.Exit(code=0)

    print(f"Processing: {motion_file}")
    print(f"Output: {keypoint_output_file}")

    # Load MJCF kinematic info
    mjcf_file_path = get_mjcf_path(skeleton_format)
    kinematic_info = extract_kinematic_info(mjcf_file_path)

    # Get keypoint indices
    conceptual_keypoint_names, mjcf_target_body_names, keypoint_indices_in_mjcf = (
        get_keypoint_indices(kinematic_info, skeleton_format)
    )

    if len(keypoint_indices_in_mjcf) != len(conceptual_keypoint_names):
        missing = set(mjcf_target_body_names) - set(kinematic_info.body_names)
        print(
            f"Error: Could not find all target keypoint body names in MJCF. Missing: {missing}"
        )
        raise typer.Exit(code=1)

    print(f"Targeting {len(keypoint_indices_in_mjcf)} keypoints: {mjcf_target_body_names}")

    try:
        # Load the motion file
        motion_data = torch.load(motion_file, weights_only=False)
        motion = RobotState.from_dict(motion_data, state_conversion=StateConversion.COMMON)

        # Get positions and rotations
        all_body_positions = motion.rigid_body_pos  # [T, N_bodies, 3]
        all_body_rotations_quat = motion.rigid_body_rot  # [T, N_bodies, 4]

        # Validate body count
        if all_body_positions.shape[1] != kinematic_info.num_bodies:
            print(
                f"Warning: Body count mismatch. "
                f"Expected {kinematic_info.num_bodies} (MJCF bodies), "
                f"but found {all_body_positions.shape[1]} in motion file."
            )
            raise typer.Exit(code=1)

        # Get contact data if available
        contacts = None
        if hasattr(motion, 'rigid_body_contacts') and motion.rigid_body_contacts is not None:
            contacts = motion.rigid_body_contacts  # [T, N_bodies]
            print(f"Found contact data with shape: {contacts.shape}")
        else:
            print("No contact data found in motion file. Contacts will be zeros.")
            # Create zero contacts
            contacts = torch.zeros(
                all_body_positions.shape[0],
                all_body_positions.shape[1],
                dtype=torch.bool,
                device=device,
            )

        # Extract keypoints
        keypoint_data = extract_keypoints_from_motion(
            all_body_positions=all_body_positions,
            all_body_rotations_quat=all_body_rotations_quat,
            keypoint_indices_in_mjcf=keypoint_indices_in_mjcf,
            conceptual_keypoint_names=conceptual_keypoint_names,
            device=device,
            skeleton_format=skeleton_format,
            flat_feet=True,
            aux_points=True,
            contacts=contacts,
            kinematic_info=kinematic_info,
        )

        # Prepare data for saving
        keypoint_data_to_save = {
            "positions": keypoint_data["positions"].cpu().numpy(),
            "orientations": keypoint_data["orientations"].cpu().numpy(),
            "left_foot_contacts": keypoint_data.get(
                "left_foot_contacts",
                np.zeros((all_body_positions.shape[0], 2), dtype=int)
            ),
            "right_foot_contacts": keypoint_data.get(
                "right_foot_contacts",
                np.zeros((all_body_positions.shape[0], 2), dtype=int)
            ),
        }

        print(f"keypoint_positions.shape: {keypoint_data_to_save['positions'].shape}")
        print(f"keypoint_orientations.shape: {keypoint_data_to_save['orientations'].shape}")
        print(f"left_foot_contacts.shape: {keypoint_data_to_save['left_foot_contacts'].shape}")
        print(f"right_foot_contacts.shape: {keypoint_data_to_save['right_foot_contacts'].shape}")

        # Save
        np.save(str(keypoint_output_file), keypoint_data_to_save)
        print(f"Saved keypoints to: {keypoint_output_file}")

    except Exception as e:
        print(f"Error processing motion file: {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()

