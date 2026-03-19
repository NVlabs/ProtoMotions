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
Convert SOMA NPZ motion files to ProtoMotions format for soma23.

NPZ files contain the full 77-joint SOMA skeleton:
    local_rot_mats  — (T, 77, 3, 3)  local rotation matrices
    posed_joints    — (T, 77, 3)     global joint positions (root pos = index 0)
    global_rot_mats — (T, 77, 3, 3)  global rotation matrices (unused by converter)
    foot_contacts   — (T, 4)         foot contact labels (unused by converter)

The 77 joints are subselected to the 23 MJCF bodies using SOMASKEL77_TO_MJCF_INDICES
(same mapping as the npy and BVH pipelines).

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
    SOMASKEL77_TO_MJCF_INDICES,
    create_motion_from_soma23_data,
)
from motion_filter import passes_exclude_motion_filter

app = typer.Typer(pretty_exceptions_enable=False)


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
    """Convert SOMA NPZ motion files (77 joints) to ProtoMotions format."""
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

            for required in ("local_rot_mats", "posed_joints"):
                if required not in data:
                    print(f"Skipping {npz_file}: missing '{required}'")
                    continue

            local_rot_mats = data["local_rot_mats"]  # (T, 77, 3, 3)
            root_pos = data["posed_joints"][:, 0, :]  # (T, 3)

            if local_rot_mats.shape[1] != 77:
                print(
                    f"Skipping {npz_file}: expected 77 joints, "
                    f"got {local_rot_mats.shape[1]}"
                )
                continue

            print(f"  local_rot_mats: {local_rot_mats.shape}")
            print(f"  root_pos: {root_pos.shape}")

            # Subselect to 23 MJCF bodies
            local_rot_mats = local_rot_mats[:, SOMASKEL77_TO_MJCF_INDICES, :, :]

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
