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
Convert SOMASkeleton30 NPZ motion files to ProtoMotions format for soma23.

NPZ files contain data from the SOMA pipeline with 30-body SOMASkeleton30:
    local_rot_mats  — (T, 30, 3, 3)  local rotation matrices
    root_positions  — (T, 3)          root translation

A legacy batch dimension (B=1) is also accepted and squeezed automatically.

The 30 bodies are subselected to the 23 MJCF bodies by dropping 7 leaf
end-effectors: Jaw, LeftEye, RightEye, LeftHandThumbEnd, LeftHandMiddleEnd,
RightHandThumbEnd, RightHandMiddleEnd.

Usage:
    python data/scripts/convert_soma23_npz_to_proto.py \
        --input-dir data/soma-kimodo-generated/ \
        --output-dir data/soma-kimodo-generated/proto \
        --input-fps 30 --output-fps 30
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer

from protomotions.components.pose_lib import extract_kinematic_info

from convert_soma23_to_proto import (
    MJCF_BODY_NAMES,
    create_motion_from_soma23_data,
)
from contact_detection import compute_contact_labels_from_pos_and_vel
from motion_filter import passes_exclude_motion_filter

app = typer.Typer(pretty_exceptions_enable=False)

# SOMASkeleton30 bone order (from SOMASkeleton30.bone_order_names_with_parents)
SOMASKEL30_BONE_NAMES = [
    "Hips", "Spine1", "Spine2", "Chest",
    "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumbEnd", "LeftHandMiddleEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumbEnd", "RightHandMiddleEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase",
]

SOMASKEL30_TO_MJCF_INDICES = [
    SOMASKEL30_BONE_NAMES.index(name) for name in MJCF_BODY_NAMES
]


@app.command()
def main(
    input_dir: Path = typer.Option(..., help="Directory containing .npz files."),
    output_dir: Path = typer.Option(..., help="Directory to save .motion files."),
    input_fps: int = typer.Option(30, help="Input motion fps"),
    output_fps: int = typer.Option(30, help="Output motion fps"),
    force_remake: bool = False,
    ignore_motion_filter: bool = False,
    min_height_threshold: float = typer.Option(-0.05),
    max_velocity_threshold: float = typer.Option(15.0),
    max_dof_vel_threshold: float = typer.Option(40.0),
    duration_height_filter: float = typer.Option(0.1),
    duration_height_seconds: float = typer.Option(0.6),
    yaml_output_name: Optional[str] = None,
):
    """Convert SOMASkeleton30 NPZ motion files to ProtoMotions format."""
    device = torch.device("cpu")
    dtype = torch.float32

    kinematic_info = extract_kinematic_info(
        "protomotions/data/assets/mjcf/soma23_humanoid.xml"
    )
    print("kinematic_info:", kinematic_info)
    assert kinematic_info.num_bodies == 23
    assert kinematic_info.nq == 22 * 3 + 7

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_fps % output_fps != 0:
        raise ValueError(
            f"input_fps ({input_fps}) must be divisible by output_fps ({output_fps})"
        )
    downsample_factor = input_fps // output_fps

    npz_files = sorted(input_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} npz files in {input_dir}")

    output_motions_yaml = []

    for npz_file in npz_files:
        motion_filename = npz_file.stem + ".motion"
        output_file = output_dir / motion_filename

        if not force_remake and output_file.exists():
            print(f"Skipping {motion_filename} (already exists)")
            continue

        print(f"Processing {npz_file}")

        try:
            data = np.load(npz_file, allow_pickle=True)

            if "local_rot_mats" not in data or "root_positions" not in data:
                print(
                    f"Skipping {npz_file}: missing 'local_rot_mats' or 'root_positions'"
                )
                continue

            local_rot_mats = data["local_rot_mats"]
            root_pos = data["root_positions"]

            # Squeeze leading batch dimension if present
            if local_rot_mats.ndim == 5:
                local_rot_mats = local_rot_mats[0]
            if root_pos.ndim == 3:
                root_pos = root_pos[0]

            if local_rot_mats.shape[1] != 30:
                print(
                    f"Skipping {npz_file}: expected 30 bodies, "
                    f"got {local_rot_mats.shape[1]}"
                )
                continue

            print(f"  local_rot_mats: {local_rot_mats.shape}")
            print(f"  root_pos: {root_pos.shape}")

            # Subselect 30→23 MJCF bodies
            local_rot_mats = local_rot_mats[:, SOMASKEL30_TO_MJCF_INDICES, :, :]

            # Downsample
            local_rot_mats = local_rot_mats[::downsample_factor]
            root_pos = root_pos[::downsample_factor]

            local_rot_mats = torch.from_numpy(local_rot_mats).to(device, dtype)
            root_pos = torch.from_numpy(root_pos).to(device, dtype)

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
                print(f"Skipping {npz_file.name} (failed motion filter)")
                continue

            print(f"  dof_pos:         {motion.dof_pos.shape}")
            print(f"  rigid_body_pos:  {motion.rigid_body_pos.shape}")
            print(f"  Saving to {output_file}")
            torch.save(motion.to_dict(), str(output_file))

            if yaml_output_name is not None:
                output_motions_yaml.append(
                    {"file": motion_filename, "fps": output_fps}
                )

        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if yaml_output_name is not None:
        import yaml

        yaml_output = output_dir / yaml_output_name
        with open(yaml_output, "w") as f:
            yaml.dump({"motions": output_motions_yaml}, f)
        print(f"Saved motions list to {yaml_output}")


if __name__ == "__main__":
    with torch.no_grad():
        app()
