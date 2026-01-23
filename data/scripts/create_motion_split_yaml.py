#!/usr/bin/env python3
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
import yaml
import os
import argparse
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import joblib
import numpy as np
import torch

from data.smpl.smpl_joint_names import (
    SMPL_BONE_ORDER_NAMES,
    SMPL_MUJOCO_NAMES,
    SMPLH_BONE_ORDER_NAMES,
    SMPLH_MUJOCO_NAMES,
)

from protomotions.components.pose_lib import (
    extract_kinematic_info,
)

# Import motion processing functions from convert_amass_to_proto
import sys

sys.path.insert(0, os.path.dirname(__file__))
from convert_amass_to_proto import convert_amass_to_motion, foot_offsets_dict


@dataclass
class ProcessingOptions:
    ignore_occlusions: bool
    occlusion_bound: int = 0
    occlusion: int = 0


def closest_divisor_larger_than_target(rounded_fps, target_fps):
    """Find divisors of rounded_fps that are larger than target_fps."""
    divisors = [i for i in range(1, rounded_fps + 1) if rounded_fps % i == 0]
    larger_divisors = [d for d in divisors if d >= target_fps]
    if larger_divisors:
        return min(larger_divisors)
    else:
        return None


def amass_to_amassx(file_path):
    file_path = file_path.replace("_poses", "_stageii")
    file_path = file_path.replace("SSM_synced", "SSM")
    file_path = file_path.replace("MPI_HDM05", "HDM05")
    file_path = file_path.replace("MPI_mosh", "MoSh")
    file_path = file_path.replace("MPI_Limits", "PosePrior")
    file_path = file_path.replace("TCD_handMocap", "TCDHands")
    file_path = file_path.replace("Transitions_mocap", "Transitions")
    file_path = file_path.replace("DFaust_67", "DFaust")
    file_path = file_path.replace("BioMotionLab_NTroje", "BMLrub")
    file_path = file_path.replace("Eyes_Japan_Dataset", "EyesJapanDataset")
    return file_path


def check_floating_and_suggest_bound(
    rigid_body_pos,
    fps,
    bad_height_threshold=0.15,
    bad_duration_threshold=0.3,
    good_height_threshold=0.05,
    foot_offset=0.0,
):
    """
    Check if motion has floating issues and suggest a trim bound.

    Detects: (1) floating segments exceeding duration threshold, (2) motion ending mid-air.
    Returns suggested trim point at last grounded frame before the problematic segment.
    """
    effective_bad_threshold = bad_height_threshold + foot_offset
    effective_good_threshold = good_height_threshold + foot_offset

    min_heights = rigid_body_pos[:, :, 2].min(dim=1).values  # (T,)
    num_frames = min_heights.shape[0]
    is_floating = min_heights > effective_bad_threshold

    if not is_floating.any():
        return False, None, 0.0, 0.0, 0.0, None

    # Find floating segment runs via transition detection
    padded = torch.cat([torch.tensor([False], device=is_floating.device), is_floating])
    transitions = padded[1:] != padded[:-1]
    transition_indices = torch.where(transitions)[0]

    max_consecutive_frames = 0
    max_start_frame = 0
    max_end_frame = 0
    starts_floating = is_floating[0].item()

    if len(transition_indices) == 0:
        if starts_floating:
            max_consecutive_frames = num_frames
            max_start_frame = 0
            max_end_frame = num_frames - 1
    else:
        idx_list = transition_indices.tolist()
        if starts_floating:
            idx_list = [0] + idx_list
        if is_floating[-1].item():
            idx_list.append(num_frames)

        # idx_list has pairs: [start1, end1, start2, end2, ...]
        for i in range(0, len(idx_list) - 1, 2):
            start = idx_list[i]
            end = idx_list[i + 1] - 1
            run_length = end - start + 1
            if run_length > max_consecutive_frames:
                max_consecutive_frames = run_length
                max_start_frame = start
                max_end_frame = end

    max_consecutive_duration = max_consecutive_frames / fps
    has_bad_floating = max_consecutive_duration >= bad_duration_threshold

    ends_floating = is_floating[-1].item()
    end_floating_start_frame = None

    if ends_floating:
        non_floating_frames = torch.where(~is_floating)[0]
        if len(non_floating_frames) > 0:
            end_floating_start_frame = non_floating_frames[-1].item() + 1
        else:
            end_floating_start_frame = 0

    suggested_bound_time = None
    trim_start_frame = None
    trim_reason = None

    if has_bad_floating:
        trim_start_frame = max_start_frame
        trim_reason = "duration_exceeded"
    elif ends_floating and end_floating_start_frame is not None:
        trim_start_frame = end_floating_start_frame
        trim_reason = "ends_floating"
        has_bad_floating = True

    if trim_start_frame is not None:
        if trim_start_frame > 0:
            search_region = min_heights[:trim_start_frame]
            good_frames = torch.where(search_region <= effective_good_threshold)[0]
            if len(good_frames) > 0:
                suggested_bound_frame = good_frames[-1].item() + 1
            else:
                suggested_bound_frame = 0
        else:
            suggested_bound_frame = 0

        suggested_bound_time = suggested_bound_frame / fps

    return (
        has_bad_floating,
        suggested_bound_time,
        max_consecutive_duration,
        max_start_frame / fps,
        max_end_frame / fps,
        trim_reason,
    )


def is_valid_motion(
    occlusion_data: dict,
    motion_name: str,
    options: ProcessingOptions,
):
    """
    Check if a motion is valid based on occlusion data.
    Returns (is_valid, bound) where bound is the index where occlusion occurs.
    """

    # Try different key formats for occlusion data lookup
    occlusion_keys = [
        motion_name,  # Just the motion name
        # Remove humanoid type and transform separators
        "_".join(
            motion_name.replace("-smplx", "")
            .replace("-smpl", "")
            .replace(".motion", "")
            .split("/")
        ),
    ]

    motion_occlusion_data = {}
    for key in occlusion_keys:
        if key in occlusion_data:
            motion_occlusion_data = occlusion_data[key]
            break

    # Match PHC filtering logic exactly:
    # - If motion is in occlusion data with "sitting"/"airborne" + idxes → trim at bound
    # - If motion is in occlusion data with any other issue → skip (irrecoverable)
    # - If motion is NOT in occlusion data → include (no bound)
    if not options.ignore_occlusions and len(motion_occlusion_data) > 0:
        issue = motion_occlusion_data.get("issue", "")
        if (issue == "sitting" or issue == "airborne") and "idxes" in motion_occlusion_data:
            bound = motion_occlusion_data["idxes"][0]  # Bound assumes 30 FPS
            if bound < 10:
                options.occlusion_bound += 1
                print("bound too small", motion_name, bound)
                return False, 0
            else:
                return True, bound
        else:
            options.occlusion += 1
            print("issue irrecoverable", motion_name, issue)
            return False, 0

    return True, None


def load_yaml_file(file_path: Path) -> dict:
    """Load YAML file and return its contents"""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml_file(data: dict, file_path: Path) -> None:
    """Save data to a YAML file"""
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def create_motion_entry(
    motion_file: str, full_path: str, end_time: float, fps: float, idx: int
) -> dict:
    """Create a motion entry for the YAML file"""
    motion_entry = {
        "file": motion_file,
        "fps": float(fps),
        "weight": 1.0,
        "idx": idx,
    }
    if end_time is not None:
        end_time = float(end_time)

    if end_time is not None:
        sub_motions = [{"timings": {"start": 0.0, "end": end_time}}]
        motion_entry["sub_motions"] = sub_motions
    return motion_entry


def process_dataset_split(
    motion_dir: str,
    split_folders: List[str],
    humanoid_type: str,
    occlusion_data: dict,
    options: ProcessingOptions,
    split_name: str,
    motion_fps_data: Optional[dict] = None,
    check_floating: bool = False,
    floating_height_threshold: float = 0.15,
    floating_duration_threshold: float = 0.3,
    floating_cut_threshold: float = 0.05,
    kinematic_info=None,
    joint_names=None,
    mujoco_joint_names=None,
    device=None,
    dtype=None,
) -> dict:
    """Process motion files for a specific dataset split"""
    print(f"Processing {split_name} split with folders: {split_folders}")

    yaml_data = {"motions": []}
    motion_count = 0
    filtered_count = 0
    flipped_count = 0
    floating_count = 0
    adjusted_bound_count = 0

    # Process each specific folder
    for folder in split_folders:
        if humanoid_type == "smplx":
            folder = amass_to_amassx(folder)
        folder_path = os.path.join(motion_dir, folder)
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"Warning: Folder {folder_path} not found, skipping")
            continue

        print(f"Processing folder: {folder_path}")

        # Walk through the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                if (
                    file.endswith(".npz")
                    and not file.endswith("stagei.npz")
                    and not file.endswith("shape.npz")
                ):
                    # Skip files with '_flipped' in the name
                    if "_flipped.motion" in file:
                        flipped_count += 1
                        continue

                    # Create the relative path from the motion directory
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, motion_dir)
                    rel_path_motion = (
                        rel_path.replace(".npz", ".motion")
                        .replace("-", "_")
                        .replace(" ", "_")
                        .replace("(", "_")
                        .replace(")", "_")
                    )

                    # Check if motion is valid
                    is_valid, bound = is_valid_motion(
                        occlusion_data, rel_path_motion, options
                    )
                    if not is_valid:
                        filtered_count += 1
                        continue

                    motion = np.load(full_path)

                    if motion_fps_data is not None and rel_path_motion in motion_fps_data:
                        mocap_fr = motion_fps_data[rel_path_motion]
                    elif "mocap_framerate" in motion:
                        mocap_fr = motion["mocap_framerate"]
                    elif "mocap_frame_rate" in motion:
                        mocap_fr = motion["mocap_frame_rate"]
                    else:
                        raise Exception(f"FPS not found for {rel_path_motion}")

                    # Calculate duration in seconds
                    duration_seconds = motion["poses"].shape[0] / mocap_fr

                    # If there's an occlusion bound, adjust the end time
                    if bound is None:
                        end_time = float(duration_seconds)
                    else:
                        end_time = (
                            bound * 1.0 / 30.0
                        )  # Convert to seconds. Bounds assume 30 fps.

                    if end_time < 0.1:
                        filtered_count += 1
                        print(
                            f"Skipping {rel_path} because it has less than 0.1 seconds"
                        )
                        continue

                    # Check for floating if enabled
                    if check_floating:
                        try:
                            # Get the motion data to check (apply bound if exists)
                            pose_aa_to_check = motion["poses"]
                            amass_trans_to_check = motion["trans"]

                            # Apply slicing if bound exists
                            if bound is not None:
                                start_frame = 0
                                end_frame = int(
                                    bound * mocap_fr / 30.0
                                )  # Convert bound from 30fps to actual fps
                                end_frame = min(end_frame, pose_aa_to_check.shape[0])
                                pose_aa_to_check = pose_aa_to_check[
                                    start_frame:end_frame
                                ]
                                amass_trans_to_check = amass_trans_to_check[
                                    start_frame:end_frame
                                ]

                            # Convert motion using the refactored function
                            motion_obj, motion_fps = convert_amass_to_motion(
                                pose_aa=pose_aa_to_check.copy(),
                                amass_trans=amass_trans_to_check.copy(),
                                mocap_fr=int(mocap_fr),
                                humanoid_type=humanoid_type,
                                joint_names=joint_names,
                                mujoco_joint_names=mujoco_joint_names,
                                kinematic_info=kinematic_info,
                                device=device,
                                dtype=dtype,
                                output_fps=30,
                            )

                            (
                                has_bad_floating,
                                suggested_bound_time,
                                max_duration,
                                bad_start_time,
                                bad_end_time,
                                trim_reason,
                            ) = check_floating_and_suggest_bound(
                                motion_obj.rigid_body_pos,
                                motion_fps,
                                bad_height_threshold=floating_height_threshold,
                                bad_duration_threshold=floating_duration_threshold,
                                good_height_threshold=floating_cut_threshold,
                                foot_offset=foot_offsets_dict.get(humanoid_type, 0.0),
                            )

                            if has_bad_floating:
                                if (
                                    suggested_bound_time is not None
                                    and suggested_bound_time >= 0.3
                                ):  # ~10 frames
                                    old_end_time = end_time if end_time else duration_seconds
                                    end_time = suggested_bound_time

                                    reason_str = (
                                        "ends floating"
                                        if trim_reason == "ends_floating"
                                        else f"floating {max_duration:.2f}s"
                                    )
                                    print(f"✂️  Adjusting {rel_path_motion} ({reason_str}):")
                                    if trim_reason == "duration_exceeded":
                                        print(f"   - Bad segment: {bad_start_time:.2f}s-{bad_end_time:.2f}s")
                                    print(f"   - Trimmed: {old_end_time:.2f}s -> {end_time:.2f}s")

                                    adjusted_bound_count += 1
                                else:
                                    floating_count += 1
                                    reason_str = (
                                        "ends floating with no grounded portion"
                                        if trim_reason == "ends_floating"
                                        else f"unsalvageable floating: {max_duration:.2f}s"
                                    )
                                    print(f"⚠️  Skipping {rel_path_motion}: {reason_str}")
                                    continue

                        except Exception as e:
                            print(f"Warning: Could not check floating for {rel_path_motion}: {e}")

                    # Create motion entry
                    motion_entry = create_motion_entry(
                        rel_path_motion, full_path, end_time, mocap_fr, motion_count
                    )

                    # Add to YAML data
                    yaml_data["motions"].append(motion_entry)
                    motion_count += 1

    print(f"{split_name} split statistics:")
    print(f"  - Total motions: {motion_count}")
    print(f"  - Filtered motions due to occlusions: {filtered_count}")
    print(f"  - Skipped flipped motions: {flipped_count}")
    if check_floating:
        print(f"  - Adjusted bounds due to floating: {adjusted_bound_count}")
        print(f"  - Skipped motions due to floating: {floating_count}")

    return yaml_data


def main():
    parser = argparse.ArgumentParser(
        description="Create YAML files for train/validation/test splits"
    )
    parser.add_argument(
        "--motion_dir",
        type=str,
        required=True,
        help="Directory containing motion files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/yaml_files",
        help="Directory to save output YAML files",
    )
    parser.add_argument(
        "--occlusion_data",
        type=str,
        default="data/amass/amass_copycat_occlusion_v3.pkl",
        help="Path to occlusion data",
    )
    parser.add_argument(
        "--humanoid_type",
        type=str,
        default="smpl",
        help="Type of humanoid (smpl, etc.)",
    )
    parser.add_argument(
        "--ignore_occlusions", action="store_true", help="Ignore occlusion filtering"
    )
    parser.add_argument(
        "--motion_fps", type=str, default=None, help="Motion FPS yaml file"
    )
    parser.add_argument(
        "--check_floating",
        action="store_true",
        help="Check for floating issues in motions",
    )
    parser.add_argument(
        "--floating_height_threshold",
        type=float,
        default=0.1,
        help="Height threshold (m) for detecting floating. Default: 0.1 (10cm)",
    )
    parser.add_argument(
        "--floating_duration_threshold",
        type=float,
        default=1,
        help="Duration threshold (s) for bad floating. Default: 1",
    )
    parser.add_argument(
        "--floating_cut_threshold",
        type=float,
        default=0.05,
        help="Height threshold (m) for finding clean cut points. Default: 0.05 (5cm)",
    )

    args = parser.parse_args()

    # Dataset split definition
    dataset_splits = {
        "validation": ["HumanEva", "MPI_HDM05", "SFU", "MPI_mosh"],
        "test": ["Transitions_mocap", "SSM_synced"],
        "train": [
            "CMU",
            "MPI_Limits",
            "TotalCapture",
            "KIT",
            "EKUT",
            "TCD_handMocap",
            "BMLhandball",
            "DanceDB",
            "ACCAD",
            "BMLmovi",
            "BioMotionLab_NTroje",
            "Eyes_Japan_Dataset",
            "DFaust_67",
        ],
    }

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load occlusion data
    print(f"Loading occlusion data: {args.occlusion_data}")
    occlusion_data = joblib.load(args.occlusion_data)
    renamed_occlusion_data = {}
    for key, value in occlusion_data.items():
        if args.humanoid_type == "smpl":
            skip = 2
        else:
            skip = 0
        renamed_key = (
            key[skip:]
            .replace(".npz", ".motion")
            .replace(".pkl", ".motion")
            .replace("-", "_")
            .replace(" ", "_")
            .replace("(", "_")
            .replace(")", "_")
        )
        if args.humanoid_type == "smplx":
            renamed_key = amass_to_amassx(renamed_key)
        renamed_occlusion_data[renamed_key] = value

    # Load motion FPS data
    motion_fps_data = None
    if args.motion_fps:
        print(f"Loading motion FPS data: {args.motion_fps}")
        motion_fps_data = load_yaml_file(args.motion_fps)

    # Create processing options
    options = ProcessingOptions(ignore_occlusions=args.ignore_occlusions)

    # Process each split
    # Initialize floating check parameters if needed
    kinematic_info = None
    joint_names = None
    mujoco_joint_names = None
    device = None
    dtype = None

    if args.check_floating:
        print("Initializing floating detection...")
        device = torch.device("cpu")
        dtype = torch.float32

        # Set up joint names based on humanoid type
        if args.humanoid_type == "smpl":
            mujoco_joint_names = SMPL_MUJOCO_NAMES
            joint_names = SMPL_BONE_ORDER_NAMES
            mjcf_path = "protomotions/data/assets/mjcf/smpl_humanoid.xml"
        elif args.humanoid_type == "smplx":
            mujoco_joint_names = SMPLH_MUJOCO_NAMES
            joint_names = SMPLH_BONE_ORDER_NAMES
            mjcf_path = "protomotions/data/assets/mjcf/smplx_humanoid.xml"
        else:
            raise NotImplementedError(
                f"Humanoid type {args.humanoid_type} not supported"
            )

        # Extract kinematic info
        kinematic_info = extract_kinematic_info(mjcf_path)
        print("Floating detection initialized")

    for split_name, split_folders in dataset_splits.items():
        yaml_data = process_dataset_split(
            args.motion_dir,
            split_folders,
            args.humanoid_type,
            renamed_occlusion_data,
            options,
            split_name,
            motion_fps_data,
            check_floating=args.check_floating,
            floating_height_threshold=args.floating_height_threshold,
            floating_duration_threshold=args.floating_duration_threshold,
            floating_cut_threshold=args.floating_cut_threshold,
            kinematic_info=kinematic_info,
            joint_names=joint_names,
            mujoco_joint_names=mujoco_joint_names,
            device=device,
            dtype=dtype,
        )

        # Save the YAML file
        output_file = os.path.join(
            args.output_dir, f"amass_{args.humanoid_type}_{split_name}.yaml"
        )
        print(f"Saving {split_name} YAML to: {output_file}")
        save_yaml_file(yaml_data, Path(output_file))

    print("Processing complete.")
    print("Occlusion statistics across all splits:")
    print(f"  - Occlusion bound too small: {options.occlusion_bound}")
    print(f"  - Irrecoverable occlusion issues: {options.occlusion}")


if __name__ == "__main__":
    main()
