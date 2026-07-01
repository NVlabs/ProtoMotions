# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect goal-state frames in SAMP motion data.

For each SAMP motion clip, identifies the longest contiguous segment where the
character is static (low body velocity) and marks the middle frame of that
segment as the goal state. This is stored as a per-frame binary mask
``goal_states`` in the packaged .pt file.

Usage:
    python scripts/samp_detect_goal_states.py \
        --input ~/protomotions_assets/samp/samp_motions.pt \
        --output ~/protomotions_assets/samp/samp_motions.pt \
        --velocity-threshold 0.15

The script can also operate on per-rank sharded files (samp_motions_0.pt, etc.).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def detect_static_segment(body_vel: torch.Tensor, threshold: float) -> tuple[int, int]:
    """Find the longest contiguous segment where mean body speed < threshold.

    Args:
        body_vel: [num_frames, num_bodies, 3] velocity tensor.
        threshold: Speed threshold in m/s.

    Returns:
        (start_frame, end_frame) of the longest static segment.
        Returns (0, 0) if no static frames found.
    """
    # Mean speed across all bodies per frame
    speed_per_body = body_vel.norm(dim=-1)  # [num_frames, num_bodies]
    mean_speed = speed_per_body.mean(dim=-1)  # [num_frames]

    is_static = mean_speed < threshold  # [num_frames]

    if not is_static.any():
        return (0, 0)

    # Find longest contiguous run of True values
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0

    for i in range(len(is_static)):
        if is_static[i]:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    return (best_start, best_start + best_len)


def annotate_goal_states(
    data: dict, velocity_threshold: float, verbose: bool = True
) -> torch.Tensor:
    """Create goal_states mask for a packaged motion library.

    Args:
        data: Dict loaded from a .pt motion library file.
        velocity_threshold: Speed threshold for static detection.
        verbose: Print per-motion diagnostics.

    Returns:
        Boolean tensor [total_frames] with True at goal-state frames.
    """
    gvs = data["gvs"]  # [total_frames, num_bodies, 3]
    length_starts = data["length_starts"]
    motion_num_frames = data["motion_num_frames"]
    motion_lengths = data["motion_lengths"]
    num_motions = len(motion_lengths)
    total_frames = gvs.shape[0]

    goal_states = torch.zeros(total_frames, dtype=torch.bool)

    num_with_goals = 0
    for m in range(num_motions):
        start = length_starts[m].item()
        n_frames = motion_num_frames[m].item()
        vel = gvs[start : start + n_frames]  # [n_frames, num_bodies, 3]

        seg_start, seg_end = detect_static_segment(vel, velocity_threshold)
        seg_len = seg_end - seg_start

        if seg_len > 0:
            mid = seg_start + seg_len // 2
            goal_states[start + mid] = True
            num_with_goals += 1

            if verbose:
                t_mid = (mid / max(n_frames - 1, 1)) * motion_lengths[m].item()
                motion_file = ""
                if "motion_files" in data:
                    motion_file = f" ({data['motion_files'][m]})"
                print(
                    f"  Motion {m}{motion_file}: static segment "
                    f"[{seg_start}..{seg_end}) ({seg_len} frames), "
                    f"goal frame {mid} (t={t_mid:.2f}s)"
                )
        elif verbose:
            motion_file = ""
            if "motion_files" in data:
                motion_file = f" ({data['motion_files'][m]})"
            print(f"  Motion {m}{motion_file}: no static segment found")

    print(
        f"\nGoal states annotated: {num_with_goals}/{num_motions} motions "
        f"({100 * num_with_goals / max(num_motions, 1):.1f}%)"
    )
    return goal_states


def main():
    parser = argparse.ArgumentParser(
        description="Detect goal-state frames in SAMP motion data"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input .pt motion library file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: overwrite input)",
    )
    parser.add_argument(
        "--velocity-threshold",
        type=float,
        default=0.15,
        help="Mean body speed threshold (m/s) for static detection (default: 0.15)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-motion output",
    )
    args = parser.parse_args()

    output_path = args.output or args.input
    input_path = Path(args.input)

    print(f"Loading {input_path}...")
    data = torch.load(input_path, weights_only=False, map_location="cpu")

    print(
        f"Loaded {len(data['motion_lengths'])} motions, "
        f"{data['gvs'].shape[0]} total frames"
    )
    print(f"Velocity threshold: {args.velocity_threshold} m/s\n")

    goal_states = annotate_goal_states(
        data, args.velocity_threshold, verbose=not args.quiet
    )
    data["goal_states"] = goal_states

    print(f"\nSaving to {output_path}...")
    torch.save(data, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
