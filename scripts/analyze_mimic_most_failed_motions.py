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
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path


def analyze_env_weights(checkpoint_path: str):
    """
    Analyze motion weights from an environment checkpoint file.

    Args:
        checkpoint_path: Path to the environment checkpoint file
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")

    # Load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Extract motion weights
    if "motion_manager" not in checkpoint:
        print("Error: No motion_manager found in checkpoint")
        print(f"Available keys: {list(checkpoint.keys())}")
        return

    motion_manager_state = checkpoint["motion_manager"]

    if "motion_weights" not in motion_manager_state:
        print("Error: No motion_weights found in motion_manager state")
        print(f"Available keys: {list(motion_manager_state.keys())}")
        return

    motion_weights = motion_manager_state["motion_weights"]
    motion_file_name = motion_manager_state.get("motion_file_name", "Unknown")

    print(f"Motion file: {motion_file_name}")
    print(f"Total number of motions: {len(motion_weights)}")
    print(f"Weight tensor shape: {motion_weights.shape}")
    print(f"Weight tensor dtype: {motion_weights.dtype}")

    # Convert to numpy for analysis
    weights_np = motion_weights.cpu().numpy()

    # Basic statistics
    print("\nWeight Statistics:")
    print(f"  Min weight: {weights_np.min():.6f}")
    print(f"  Max weight: {weights_np.max():.6f}")
    print(f"  Mean weight: {weights_np.mean():.6f}")
    print(f"  Std weight: {weights_np.std():.6f}")
    print(f"  Sum of weights: {weights_np.sum():.6f}")

    # Top 120 weights
    print("\nTop 120 Motion Weights:")
    print("=" * 50)

    # Get indices of top 120 weights
    top_indices = np.argsort(weights_np)[-120:][::-1]  # Sort descending

    print(f"{'Rank':<6} {'Motion ID':<10} {'Weight':<12}")
    print("-" * 30)

    for rank, motion_id in enumerate(top_indices, 1):
        weight = weights_np[motion_id]
        print(f"{rank:<6} {motion_id:<10} {weight:<12.6f}")

    # Print top 120 motion IDs as a list for subset_method
    print("\nTop 120 Motion IDs for subset_method:")
    print("=" * 50)
    motion_id_list = top_indices.tolist()

    # Format as a list that can be copied directly
    list_str = "[" + ", ".join(map(str, motion_id_list)) + "]"
    print(f'motion_manager.subset_method="{list_str}"')

    # Also print in chunks for readability
    print("\nFormatted for readability:")
    print('motion_manager.subset_method="[')
    for i in range(0, len(motion_id_list), 10):
        chunk = motion_id_list[i : i + 10]
        chunk_str = ", ".join(map(str, chunk))
        if i + 10 < len(motion_id_list):
            print(f"  {chunk_str},")
        else:
            print(f"  {chunk_str}")
    print(']"')

    print(f"\nNote: Total {len(motion_id_list)} motion IDs listed above.")

    # Create histogram
    plt.figure(figsize=(12, 8))

    # Create two subplots
    plt.subplot(2, 1, 1)
    plt.hist(weights_np, bins=100, alpha=0.7, edgecolor="black")
    plt.title(f"Distribution of Motion Weights (Total: {len(weights_np)} motions)")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

    # Log scale histogram for better visualization of long tail
    plt.subplot(2, 1, 2)
    plt.hist(weights_np, bins=100, alpha=0.7, edgecolor="black")
    plt.title("Distribution of Motion Weights (Log Scale)")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the histogram
    output_path = (
        checkpoint_path.parent / f"{checkpoint_path.stem}_weights_histogram.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nHistogram saved to: {output_path}")

    # Show the plot
    plt.show()

    # Additional analysis: find motions with zero or very low weights
    zero_weights = np.sum(weights_np == 0)
    very_low_weights = np.sum(weights_np < 1e-6)

    print("\nAdditional Analysis:")
    print(f"  Motions with zero weights: {zero_weights}")
    print(f"  Motions with very low weights (< 1e-6): {very_low_weights}")

    if zero_weights > 0:
        zero_indices = np.where(weights_np == 0)[0]
        print(
            f"  Zero weight motion IDs: {zero_indices[:20].tolist()}"
        )  # Show first 20
        if len(zero_indices) > 20:
            print(f"    ... and {len(zero_indices) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze motion weights from environment checkpoint"
    )
    parser.add_argument(
        "checkpoint_path",
        default="tmp/env_retargeted_95_0722_isaac.pt.ckpt",
        nargs="?",
        help="Path to the environment checkpoint file",
    )

    args = parser.parse_args()

    analyze_env_weights(args.checkpoint_path)


if __name__ == "__main__":
    main()
