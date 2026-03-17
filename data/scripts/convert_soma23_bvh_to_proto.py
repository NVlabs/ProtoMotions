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
Convert SOMASkeleton77 BVH motions to ProtoMotions format for soma23.

Pipeline:
  1. Parse BVH → local_rot_mats (T, 77, 3, 3) + root_trans (T, 3)
  2. change_tpose → re-express in standard T-pose convention using
     precomputed global rotation offsets (BVH zero-pose != T-pose)
  3. Subselect 77→23 MJCF bodies
  4. Feed into create_motion_from_soma23_data (same as npy pipeline:
     MJCF FK + rot1/rot2 y-up→z-up)

Usage:
    python data/scripts/convert_soma23_bvh_to_proto.py \
        --input-dir /path/to/bvh/files \
        --output-dir /path/to/output \
        --input-fps 120 --output-fps 30
"""
import hashlib
import os
from pathlib import Path

import torch
import typer

from protomotions.components.pose_lib import extract_kinematic_info

from convert_soma23_to_proto import (
    SOMASKEL77_TO_MJCF_INDICES,
    create_motion_from_soma23_data,
)
from bvh import SkeletonBvh, load_bvh_animation, change_tpose
from motion_filter import passes_exclude_motion_filter

app = typer.Typer(pretty_exceptions_enable=False)

TPOSE_OFFSETS_PATH = "data/soma/standard_t_pose_global_offsets_rots.p"


@app.command()
def main(
    input_dir: Path = typer.Option(
        ..., help="Root directory to search for .bvh files (recursive)."
    ),
    output_dir: Path = typer.Option(
        ..., help="Directory to save ProtoMotions .motion files."
    ),
    input_fps: int = typer.Option(120, help="Input motion fps"),
    output_fps: int = typer.Option(30, help="Output motion fps"),
    force_remake: bool = False,
    ignore_motion_filter: bool = False,
    min_height_threshold: float = typer.Option(-0.05),
    max_velocity_threshold: float = typer.Option(15.0),
    max_dof_vel_threshold: float = typer.Option(40.0),
    duration_height_filter: float = typer.Option(0.1),
    duration_height_seconds: float = typer.Option(0.6),
    num_rank: int = typer.Option(1, help="Total number of parallel ranks."),
    slurm_rank: int = typer.Option(0, help="This rank's index (0-based)."),
):
    """Convert SOMASkeleton77 BVH motions to ProtoMotions format."""
    if input_fps % output_fps != 0:
        raise ValueError(
            f"input_fps ({input_fps}) must be divisible by output_fps ({output_fps})"
        )

    device = torch.device("cpu")
    dtype = torch.float32

    kinematic_info = extract_kinematic_info(
        "protomotions/data/assets/mjcf/soma23_humanoid.xml"
    )
    assert kinematic_info.num_bodies == 23
    assert kinematic_info.nq == 22 * 3 + 7

    # Per-body global rotation offsets: BVH zero-pose → standard T-pose (77 bodies)
    global_rot_offsets = torch.load(TPOSE_OFFSETS_PATH, weights_only=False)
    assert global_rot_offsets.shape == (77, 3, 3)
    print(f"Loaded T-pose offsets from {TPOSE_OFFSETS_PATH}")

    downsample_factor = input_fps // output_fps

    bvh_skeleton = None
    bvh_parent_indices = None

    for root, _dirs, files in os.walk(input_dir):
        for f in sorted(files):
            if not f.endswith(".bvh"):
                continue

            rel_path = Path(root).relative_to(input_dir)

            file_hash = int(
                hashlib.sha256(str(rel_path / f).encode("utf-8")).hexdigest(), 16
            )
            if file_hash % num_rank != slurm_rank:
                continue

            bvh_path = Path(root) / f

            out_subdir = output_dir / rel_path
            out_subdir.mkdir(parents=True, exist_ok=True)
            motion_filename = Path(f).stem + ".motion"
            output_file = out_subdir / motion_filename

            if not force_remake and output_file.exists():
                continue

            print(f"Processing {rel_path / f}")

            try:
                if bvh_skeleton is None:
                    bvh_skeleton = SkeletonBvh()
                    bvh_skeleton.load_from_bvh(str(bvh_path), exclude_bones={"Root"})
                    bvh_parent_indices = bvh_skeleton.get_parent_indices()

                root_trans, local_rot_mats = load_bvh_animation(
                    str(bvh_path), bvh_skeleton
                )
                root_trans *= 0.01  # cm → m
                root_trans = torch.tensor(root_trans)
                local_rot_mats = torch.tensor(local_rot_mats)

                if local_rot_mats.shape[1] != 77:
                    print(
                        f"Skipping {f}: expected 77 bodies, "
                        f"got {local_rot_mats.shape[1]}"
                    )
                    continue

                # Convert BVH local rots to standard T-pose convention
                tpose_local_rots, _ = change_tpose(
                    local_rot_mats, global_rot_offsets, bvh_parent_indices
                )

                # Subselect 77→23 MJCF bodies
                tpose_local_rots = tpose_local_rots[:, SOMASKEL77_TO_MJCF_INDICES, :, :]

                # Downsample
                tpose_local_rots = tpose_local_rots[::downsample_factor]
                root_trans = root_trans[::downsample_factor]

                tpose_local_rots = tpose_local_rots.to(device, dtype)
                root_trans = root_trans.to(device, dtype)

                motion = create_motion_from_soma23_data(
                    local_rot_mats=tpose_local_rots,
                    root_pos=root_trans,
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
