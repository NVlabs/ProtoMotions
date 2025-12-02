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
Recompute contact labels from packaged MotionLib

This script loads a packaged MotionLib (.pt file) and recomputes contact labels
using position and velocity thresholds, then saves the updated MotionLib.

Usage:
    python data/scripts/recompute_contacts_from_packaged_motionlib.py /path/to/input_motion_lib.pt --output-file /path/to/output_motion_lib.pt

The script will:
1. Load the existing MotionLib
2. Recompute contact labels for all motions using position/velocity heuristics
3. Save the updated MotionLib with new contact data
"""

import os
from pathlib import Path
import torch
import typer

from protomotions.components.motion_lib import MotionLib
from contact_detection import compute_contact_labels_from_pos_and_vel

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    input_motion_lib_file: Path = typer.Argument(
        ..., help="Path to the input .pt file containing the packaged MotionLib data."
    ),
    output_file: Path = typer.Option(
        ..., help="Path to save the updated MotionLib with recomputed contacts."
    ),
    vel_thres: float = typer.Option(
        0.1, help="Velocity threshold for contact detection (m/s)."
    ),
    height_thresh: float = typer.Option(
        0.08, help="Height threshold for contact detection (m)."
    ),
    device: str = typer.Option(
        "cpu", help="Device to use for processing (cpu or cuda)."
    ),
    force_overwrite: bool = typer.Option(
        False, "--force-overwrite", help="Force overwrite existing output file."
    ),
):
    """
    Recompute contact labels from a packaged MotionLib (.pt file).
    """
    device = torch.device(device)

    if not input_motion_lib_file.is_file() or input_motion_lib_file.suffix != ".pt":
        print(f"Error: Input must be a .pt file. Got: {input_motion_lib_file}")
        raise typer.Exit(code=1)

    if output_file.exists() and not force_overwrite:
        print(
            f"Error: Output file {output_file} already exists. Use --force-overwrite to overwrite."
        )
        raise typer.Exit(code=1)

    # Create output directory if it doesn't exist
    os.makedirs(output_file.parent, exist_ok=True)

    print(f"Loading MotionLib from: {input_motion_lib_file}")
    print(f"Will save updated MotionLib to: {output_file}")
    print(
        f"Contact detection thresholds: vel={vel_thres} m/s, height={height_thresh} m"
    )

    # Load MotionLib
    from protomotions.components.motion_lib import MotionLibConfig

    motion_lib = MotionLib(
        config=MotionLibConfig(motion_file=str(input_motion_lib_file)),
        device=str(device),
    )
    print(f"Loaded MotionLib with {motion_lib.num_motions()} motions.")

    # Get the original contact tensor shape for verification
    original_contacts_shape = motion_lib.contacts.shape
    print(f"Original contacts shape: {original_contacts_shape}")

    # Recompute contacts for all frames at once
    print("Recomputing contact labels...")
    new_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion_lib.gts,  # [total_frames, N_bodies, 3]
        velocity=motion_lib.gvs,  # [total_frames, N_bodies, 3]
        vel_thres=vel_thres,
        height_thresh=height_thresh,
    ).to(torch.bool)

    print(f"New contacts shape: {new_contacts.shape}")

    # Verify shapes match
    if new_contacts.shape != original_contacts_shape:
        print(
            f"Warning: Contact shape mismatch. Original: {original_contacts_shape}, New: {new_contacts.shape}"
        )
        raise typer.Exit(code=1)

    # Update the contact data in the motion lib
    motion_lib.contacts = new_contacts

    # Save updated MotionLib
    print(f"Saving updated MotionLib to: {output_file}")
    motion_lib.save_to_file(output_file)

    # Report statistics
    total_contact_frames = torch.sum(new_contacts).item()
    total_frames = new_contacts.numel()
    contact_percentage = (total_contact_frames / total_frames) * 100

    print("\nContact Statistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Contact frames: {total_contact_frames}")
    print(f"  Contact percentage: {contact_percentage:.2f}%")
    print("Successfully updated MotionLib with new contact labels!")


if __name__ == "__main__":
    app()
