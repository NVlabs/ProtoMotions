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

"""
Automates the AMASS to MotionLib conversion pipeline.

This script runs the two-step process:
1. Convert AMASS .npz files to ProtoMotions .motion files (using motion-config YAMLs)
2. Package each motion-config into its own .pt MotionLib file

Usage:
    python data/scripts/convert_amass_to_motionlib.py /path/to/amass_root /path/to/output_dir \
        --motion-config train.yaml --motion-config test.yaml --motion-config val.yaml

This will create:
    - output_dir/train.pt
    - output_dir/test.pt
    - output_dir/val.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Convert AMASS data to packaged MotionLib .pt files (one per motion-config)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create train/test/val splits from YAML configs
  python data/scripts/convert_amass_to_motionlib.py /data/amass output/ \\
      --motion-config configs/train.yaml \\
      --motion-config configs/test.yaml \\
      --motion-config configs/val.yaml

  # Use SMPL-X humanoid
  python data/scripts/convert_amass_to_motionlib.py /data/amass output/ \\
      --humanoid-type smplx \\
      --motion-config configs/train.yaml
        """,
    )

    # Required arguments
    parser.add_argument(
        "amass_root_dir",
        type=Path,
        help="Root directory containing AMASS subfolders with .npz files",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for the packaged .pt MotionLib files",
    )

    # Motion config files (required - defines the splits)
    parser.add_argument(
        "--motion-config",
        type=Path,
        action="append",
        dest="motion_configs",
        required=True,
        help="YAML file(s) containing motion configurations. Each creates a separate .pt file. "
        "Can be specified multiple times.",
    )

    # Optional arguments for convert_amass_to_proto.py
    parser.add_argument(
        "--humanoid-type",
        type=str,
        default="smpl",
        choices=["smpl", "smplx"],
        help="Humanoid type: smpl (24 joints) or smplx (52 joints). Default: smpl",
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=30,
        help="Target output FPS for motion files. Default: 30",
    )
    parser.add_argument(
        "--force-remake",
        action="store_true",
        help="Overwrite existing .motion files",
    )

    # Optional arguments for motion_lib.py
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for packaging. Default: cpu",
    )

    args = parser.parse_args()

    # Validate input path
    if not args.amass_root_dir.exists():
        print(f"Error: AMASS root directory does not exist: {args.amass_root_dir}")
        sys.exit(1)

    if not args.amass_root_dir.is_dir():
        print(f"Error: AMASS root path is not a directory: {args.amass_root_dir}")
        sys.exit(1)

    # Validate motion config files exist
    for config in args.motion_configs:
        if not config.exists():
            print(f"Error: Motion config file does not exist: {config}")
            sys.exit(1)

    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get the project root (assuming this script is in data/scripts/)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    # Step 1: Convert AMASS to ProtoMotions .motion files (all configs at once)
    print("=" * 60)
    print("Step 1: Converting AMASS .npz files to ProtoMotions .motion files")
    print("=" * 60)

    convert_script = project_root / "data" / "scripts" / "convert_amass_to_proto.py"
    convert_cmd = [
        sys.executable,
        str(convert_script),
        str(args.amass_root_dir),
        "--humanoid-type",
        args.humanoid_type,
        "--output-fps",
        str(args.output_fps),
    ]

    if args.force_remake:
        convert_cmd.append("--force-remake")

    for config in args.motion_configs:
        convert_cmd.extend(["--motion-config", str(config)])

    print(f"Running: {' '.join(convert_cmd)}")
    result = subprocess.run(convert_cmd, cwd=project_root)

    if result.returncode != 0:
        print(f"Error: AMASS conversion failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print("\nStep 1 complete: .motion files created")

    # Step 2: Package each motion-config into its own .pt file
    print("\n" + "=" * 60)
    print("Step 2: Packaging each motion-config into MotionLib .pt files")
    print("=" * 60)

    motionlib_script = project_root / "protomotions" / "components" / "motion_lib.py"

    for config_path in args.motion_configs:
        config_name = config_path.stem  # e.g., "train" from "train.yaml"
        output_file = args.output_dir / f"{config_name}.pt"

        print(f"\nPackaging {config_name} from {config_path}...")

        # Create a temporary YAML with paths resolved relative to amass_root_dir
        # Read the original config and update paths to be absolute
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Update file paths to be absolute (relative to amass_root_dir)
        for motion in config.get("motions", []):
            original_file = motion["file"]
            # The file paths in the config are relative to amass_root_dir
            absolute_path = args.amass_root_dir / original_file
            motion["file"] = str(absolute_path)

        # Write updated config to temp file
        temp_yaml = args.output_dir / f".tmp_{config_name}.yaml"
        with open(temp_yaml, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        package_cmd = [
            sys.executable,
            str(motionlib_script),
            "--motion-path",
            str(temp_yaml),
            "--output-file",
            str(output_file),
            "--device",
            args.device,
        ]

        print(f"Running: {' '.join(package_cmd)}")
        result = subprocess.run(package_cmd, cwd=project_root)

        # Clean up temp file
        temp_yaml.unlink()

        if result.returncode != 0:
            print(
                f"Error: MotionLib packaging for {config_name} failed with return code {result.returncode}"
            )
            sys.exit(result.returncode)

        print(f"Created: {output_file}")

    # Done
    print("\n" + "=" * 60)
    print("Conversion complete!")
    print(f"Output files in: {args.output_dir}")
    for config_path in args.motion_configs:
        config_name = config_path.stem
        print(f"  - {config_name}.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
