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

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import typer

from tqdm import tqdm

import time
from datetime import timedelta

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_batch_mjcf_with_velocities,
)

from contact_detection import compute_contact_labels_from_pos_and_vel

app = typer.Typer(pretty_exceptions_enable=False)


def convert_phuma_quaternion_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert quaternion from PHUMA format (x, y, z, w) to MuJoCo format (w, x, y, z).
    
    Args:
        quat_xyzw: (T, 4) quaternion in (x, y, z, w) format
        
    Returns:
        quat_wxyz: (T, 4) quaternion in (w, x, y, z) format
    """
    return np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1)


def compute_dof_velocity(dof_pos: torch.Tensor, fps: int) -> torch.Tensor:
    """
    Compute joint velocities using finite differences.
    
    Args:
        dof_pos: (T, num_dof) joint positions
        fps: frames per second
        
    Returns:
        dof_vel: (T, num_dof) joint velocities
    """
    dt = 1.0 / fps
    
    # Central difference for interior points
    dof_vel = torch.zeros_like(dof_pos)
    
    if dof_pos.shape[0] > 2:
        # Central difference for middle frames
        dof_vel[1:-1] = (dof_pos[2:] - dof_pos[:-2]) / (2 * dt)
        # Forward difference for first frame
        dof_vel[0] = (dof_pos[1] - dof_pos[0]) / dt
        # Backward difference for last frame
        dof_vel[-1] = (dof_pos[-1] - dof_pos[-2]) / dt
    elif dof_pos.shape[0] == 2:
        dof_vel[0] = (dof_pos[1] - dof_pos[0]) / dt
        dof_vel[1] = dof_vel[0]
    else:
        # Single frame - zero velocity
        pass
    
    return dof_vel


def convert_phuma_to_motion(
    phuma_data: dict,
    kinematic_info,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Convert PHUMA motion data to ProtoMotions format.
    
    Args:
        phuma_data: Dictionary containing PHUMA motion data
        kinematic_info: Kinematic info from G1 MJCF file
        device: torch device
        dtype: torch dtype
        
    Returns:
        motion: RobotState object with all motion data
    """
    # Extract data from PHUMA file
    root_trans = phuma_data['root_trans']  # (T, 3)
    root_ori = phuma_data['root_ori']      # (T, 4) in (x, y, z, w) format
    dof_pos = phuma_data['dof_pos']        # (T, 29)
    fps = int(phuma_data['fps'])
    
    T = root_trans.shape[0]
    
    # Convert quaternion from PHUMA (xyzw) to MuJoCo (wxyz) format
    root_quat_wxyz = convert_phuma_quaternion_to_wxyz(root_ori)  # (T, 4)
    
    # Build qpos: [root_pos(3), root_quat_wxyz(4), dof_pos(29)] = (T, 36)
    qpos = np.concatenate([root_trans, root_quat_wxyz, dof_pos], axis=-1)
    qpos = torch.from_numpy(qpos).to(device=device, dtype=dtype)
    
    # Convert dof_pos to tensor
    dof_pos_tensor = torch.from_numpy(dof_pos).to(device=device, dtype=dtype)
    
    # Perform forward kinematics to get rigid body positions and rotations
    motion = fk_batch_mjcf_with_velocities(
        kinematic_info=kinematic_info,
        qpos=qpos,
        fps=fps,
        compute_velocities=True,
    )
    
    # Store dof_pos (already have it)
    motion.dof_pos = dof_pos_tensor
    
    # Compute dof_vel using finite differences
    motion.dof_vel = compute_dof_velocity(dof_pos_tensor, fps)
    
    # Compute contact labels based on position and velocity
    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.05,
    ).to(torch.bool)
    
    return motion


def save_motion(motion, outpath: Path):
    """Save motion object to disk."""
    os.makedirs(outpath.parent, exist_ok=True)
    print(f"Saving to {outpath}")
    torch.save(motion.to_dict(), str(outpath))


@app.command()
def main(
    phuma_root_dir: Path = typer.Argument(..., help="Root directory containing PHUMA .npy files"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Output directory for .motion files"),
    humanoid_type: str = typer.Option("g1", "--humanoid-type", help="Humanoid type (g1, h1_2)"),
    force_remake: bool = typer.Option(False, "--force-remake", help="Overwrite existing .motion files"),
):
    """
    Convert PHUMA motion data to ProtoMotions .motion format.
    """
    device = torch.device("cpu")
    dtype = torch.float32
    
    # Select MJCF file based on robot type
    mjcf_path_map = {
        "g1": "protomotions/data/assets/mjcf/g1_bm.xml",
        "h1_2": "protomotions/data/assets/mjcf/h1_2.xml",
    }
    
    if humanoid_type not in mjcf_path_map:
        print(f"Error: Unknown humanoid type '{humanoid_type}'. Available: {list(mjcf_path_map.keys())}")
        raise typer.Exit(1)
    
    mjcf_path = mjcf_path_map[humanoid_type]
    print(f"Using MJCF file: {mjcf_path}")
    
    # Extract kinematic info from MJCF
    kinematic_info = extract_kinematic_info(mjcf_path)
    print(f"Loaded kinematic info: {kinematic_info.num_bodies} bodies, {kinematic_info.nq} DOFs")
    
    # Find all .npy files in the directory
    all_files = list(Path(phuma_root_dir / humanoid_type).glob("**/*.npy"))
    print(f"Found {len(all_files)} .npy files")
    
    # Filter out files that already have outputs
    if not force_remake:
        files_to_process = []
        for f in all_files:
            if output_dir is not None:
                # Calculate relative path from phuma_root_dir/humanoid_type
                relative_path = f.relative_to(phuma_root_dir / humanoid_type)
                out_path = output_dir / humanoid_type / relative_path.with_suffix(".motion")
            else:
                out_path = f.with_suffix(".motion")
            if not out_path.exists():
                files_to_process.append(f)
    else:
        files_to_process = all_files
    
    print(f"Files to process: {len(files_to_process)}/{len(all_files)}")
    
    if len(files_to_process) == 0:
        print("No files to process. Use --force-remake to reprocess existing files.")
        return
    
    # Process files
    start_time = time.time()
    processed_files = 0
    
    for filename in tqdm(files_to_process):
        try:
            # Load PHUMA data
            phuma_data = np.load(filename, allow_pickle=True).item()
            
            # Verify required keys
            required_keys = ['root_trans', 'root_ori', 'dof_pos', 'fps']
            missing_keys = [k for k in required_keys if k not in phuma_data]
            if missing_keys:
                print(f"Skipping {filename}: missing keys {missing_keys}")
                continue
            
            # Determine output path
            if output_dir is not None:
                # Calculate relative path from phuma_root_dir/humanoid_type
                relative_path = filename.relative_to(phuma_root_dir / humanoid_type)
                outpath = output_dir / humanoid_type / relative_path.with_suffix(".motion")
            else:
                outpath = filename.with_suffix(".motion")
            
            # Convert to ProtoMotions format
            motion = convert_phuma_to_motion(
                phuma_data=phuma_data,
                kinematic_info=kinematic_info,
                device=device,
                dtype=dtype,
            )
            
            # Save motion
            save_motion(motion, outpath)
            
            processed_files += 1
            
            # Progress reporting
            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / processed_files
            remaining_files = len(files_to_process) - processed_files
            estimated_time_remaining = avg_time_per_file * remaining_files
            
            if processed_files % 100 == 0:
                print(f"\nProgress: {processed_files}/{len(files_to_process)} files")
                print(f"Average time per file: {timedelta(seconds=int(avg_time_per_file))}")
                print(f"Estimated time remaining: {timedelta(seconds=int(estimated_time_remaining))}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nDone! Processed {processed_files} files in {timedelta(seconds=int(time.time() - start_time))}")


if __name__ == "__main__":
    with torch.no_grad():
        app()
