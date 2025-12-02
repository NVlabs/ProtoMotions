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
from typing import Optional
import pickle
import typer
import os
from pathlib import Path
import numpy as np
import torch
import hashlib
import yaml

from protomotions.components.pose_lib import (
    extract_kinematic_info,
)

# Import helper functions from the original script
from convert_rigv1_to_proto import (
    passes_exclude_motion_filter,
    gen_yaml_one_motion_default,
    create_motion_from_rigv1_data,
)
from keypoint_utils import extract_keypoints_from_motion, get_keypoint_indices

app = typer.Typer(pretty_exceptions_enable=True)


@app.command()
def main(
    rigv1_pkl_path: Path,
    output_path: Path,
    input_fps: int = 120,
    output_fps: int = 30,
    num_rank: int = 1,
    slurm_rank: int = 0,
    min_height_threshold: float = -0.05,
    max_velocity_threshold: float = 15.0,
    max_dof_vel_threshold: float = 40.0,
    duration_height_filter: float = 0.2,
    duration_height_seconds: float = 1.0,
    only_process_yaml: Optional[Path] = None,
    ignore_motion_filter: bool = False,
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
                    f"{path}.pkl" for path in yaml_data["row_mapping"].values()
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

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Walk through rigv1_pkl_path and process .pkl files
    for root, dirs, files in os.walk(rigv1_pkl_path):
        for f in files:
            if f.endswith(".pkl"):
                # Get relative path from rigv1_pkl_path to file
                rel_path = Path(root).relative_to(rigv1_pkl_path)
                full_rel_path = str(rel_path / f)

                # If yaml_path is provided, only process files that match the paths in the YAML file
                if only_process_yaml is not None:
                    if full_rel_path not in specified_files:
                        # print(f"Skipping {full_rel_path} because it is not in the YAML file")
                        continue
                    else:
                        print(f"Found {full_rel_path} in YAML file, processing...")

                # only process the files for the current rank
                # using hash of the file name to determine the rank
                file_hash = int(
                    hashlib.sha256(str(rel_path / f).encode("utf-8")).hexdigest(), 16
                )

                if file_hash % num_rank != slurm_rank:
                    print(f"Skipping {f} because it is not for rank {slurm_rank}")
                    continue

                # Construct input path
                pkl_file_path = Path(root) / f

                if not pkl_file_path.exists():
                    print(f"Skipping {pkl_file_path} doesn't exist")
                    continue

                # Create output directory structure if it doesn't exist
                output_dir = output_path / rel_path
                output_dir.mkdir(parents=True, exist_ok=True)
                file_name = f.replace(".pkl", ".motion").replace(" ", "_")
                output_file_relative = rel_path / file_name  # Save as .motion file
                output_file = output_dir / file_name  # save as .motion file

                if not force_remake and output_file.exists():
                    print(f"Skipping {output_file_relative} because it already exists")
                    continue

                print(f"Processing {pkl_file_path}")

                try:
                    # Load data from pkl file
                    with open(pkl_file_path, "rb") as pkl_file:
                        data = pickle.load(pkl_file)

                    if "joints_pos" in data and "root_pos" not in data:
                        data["root_pos"] = data["joints_pos"][:, 0, :]

                    # # print data keys
                    # print("data keys", data.keys())

                    # Extract global_rot_mats and root_pos from the pkl data
                    if "global_rot_mats" not in data or "root_pos" not in data:
                        print(
                            f"Skipping {pkl_file_path} because it doesn't contain required fields 'global_rot_mats' and 'root_pos'"
                        )
                        continue

                    global_rot_mats = data["global_rot_mats"]  # Expected: [T, 27, 3, 3]
                    root_pos = data["root_pos"]  # Expected: [T, 3]

                    print("global_rot_mats", global_rot_mats.shape)
                    print("root_pos", root_pos.shape)

                    # Convert to numpy arrays if needed
                    if isinstance(global_rot_mats, list):
                        global_rot_mats = np.array(global_rot_mats)
                    if isinstance(root_pos, list):
                        root_pos = np.array(root_pos)

                    # Validate shapes
                    if (
                        global_rot_mats.ndim != 4
                        or global_rot_mats.shape[1] != 27
                        or global_rot_mats.shape[2:] != (3, 3)
                    ):
                        print(
                            f"Skipping {pkl_file_path} because global_rot_mats has wrong shape: {global_rot_mats.shape}, expected: [T, 27, 3, 3]"
                        )
                        continue

                    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
                        print(
                            f"Skipping {pkl_file_path} because root_pos has wrong shape: {root_pos.shape}, expected: [T, 3]"
                        )
                        continue

                    if global_rot_mats.shape[0] != root_pos.shape[0]:
                        print(
                            f"Skipping {pkl_file_path} because global_rot_mats and root_pos have different sequence lengths"
                        )
                        continue

                except Exception as e:
                    print(f"Error loading {pkl_file_path}: {e}")
                    continue

                # downsample motion if needed
                assert (
                    input_fps % output_fps == 0
                ), "input_fps must be divisible by output_fps"
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

                # Save motion (no height augmentation for pkl script)
                torch.save(motion.to_dict(), str(output_file))
                print(f"Saved to {output_file}")

                if yaml_output is not None:
                    output_motions_yaml.append(
                        gen_yaml_one_motion_default(
                            output_file_relative, output_fps, output_yaml_idx
                        )
                    )
                    output_yaml_idx += 1

    if yaml_output is not None:
        with open(yaml_output, "w") as f:
            yaml.dump({"motions": output_motions_yaml}, f)
        print(f"Saved motions list to {yaml_output}")


if __name__ == "__main__":
    app()
