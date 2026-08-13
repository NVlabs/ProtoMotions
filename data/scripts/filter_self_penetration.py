# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Filter nova23 .motion files for heavy self-penetration.

Uses pure sphere/capsule geometry checks (no MuJoCo) against the geoms
defined in nova23_humanoid.xml.  For each motion file the script:

  1. Loads rigid_body_pos and rigid_body_rot from the .motion dict.
  2. Transforms every geom to world space (batched over all frames).
  3. Computes pairwise distances for non-adjacent body pairs.
  4. Rejects the motion if:
       a) Any frame has penetration > --max-depth  (instant reject), or
       b) More than --max-penetrating-frames frames have penetration
          > --penetration-threshold  (sustained reject).

Surviving motions are symlinked (or copied) into --output-dir, preserving
subdirectory structure so motion_lib.py can re-package them directly.

Usage:
    python data/scripts/filter_self_penetration.py \
        --input-dir /path/to/chunk_22/ \
        --output-dir /path/to/chunk_22_filtered/ \
        --penetration-threshold 0.03 \
        --max-penetrating-frames 10 \
        --max-depth 0.05 \
        --subsample 3
"""

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import typer
from torch import Tensor

app = typer.Typer(pretty_exceptions_enable=False)

# ---------------------------------------------------------------------------
# Quaternion rotation (standalone, avoids importing protomotions)
# ---------------------------------------------------------------------------

def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vectors v by quaternions q.  Both in xyzw convention.

    Args:
        q: (..., 4) quaternions  (xyzw)
        v: (..., 3) vectors

    Returns:
        (..., 3) rotated vectors
    """
    q_vec = q[..., :3]
    q_w = q[..., 3:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (q_vec * v).sum(dim=-1, keepdim=True) * 2.0
    return a + b + c


# ---------------------------------------------------------------------------
# Batched distance primitives
# ---------------------------------------------------------------------------

def _point_to_segment_dist_sq(
    p: Tensor, a: Tensor, b: Tensor
) -> Tensor:
    """Squared distance from points p to segments a-b.  All shapes (..., 3)."""
    ab = b - a
    ap = p - a
    t = (ap * ab).sum(-1, keepdim=True) / (
        (ab * ab).sum(-1, keepdim=True).clamp(min=1e-12)
    )
    t = t.clamp(0.0, 1.0)
    closest = a + t * ab
    diff = p - closest
    return (diff * diff).sum(-1)


def point_to_segment_dist(p: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Distance from points to segments.  All (..., 3)."""
    return _point_to_segment_dist_sq(p, a, b).sqrt()


def segment_to_segment_dist(
    a1: Tensor, a2: Tensor, b1: Tensor, b2: Tensor
) -> Tensor:
    """Minimum distance between two line segments a1-a2 and b1-b2.

    Uses the standard clamp-based algorithm, fully batched.
    Shapes: all (..., 3).  Returns (...,).
    """
    d1 = a2 - a1  # direction of segment A
    d2 = b2 - b1  # direction of segment B
    r = a1 - b1

    a = (d1 * d1).sum(-1)  # ||d1||^2
    e = (d2 * d2).sum(-1)  # ||d2||^2
    f = (d2 * r).sum(-1)

    EPS = 1e-12
    both_degen = (a < EPS) & (e < EPS)

    # --- case: both degenerate (points) ---
    dist_sq_degen = (r * r).sum(-1)

    # --- general case ---
    b_val = (d1 * d2).sum(-1)
    c = (d1 * r).sum(-1)
    denom = (a * e - b_val * b_val).clamp(min=EPS)

    s = ((b_val * f - c * e) / denom).clamp(0.0, 1.0)
    t = (b_val * s + f) / e.clamp(min=EPS)

    # Clamp t then recompute s
    t_clamped = t.clamp(0.0, 1.0)
    s_recalc = ((b_val * t_clamped - c) / a.clamp(min=EPS)).clamp(0.0, 1.0)

    # If segment A is degenerate, project point onto B
    a_degen = a < EPS
    s_recalc = torch.where(a_degen, torch.zeros_like(s_recalc), s_recalc)
    t_for_a_degen = (f / e.clamp(min=EPS)).clamp(0.0, 1.0)
    t_clamped = torch.where(a_degen, t_for_a_degen, t_clamped)

    # If segment B is degenerate, project point onto A
    b_degen = e < EPS
    t_clamped = torch.where(b_degen, torch.zeros_like(t_clamped), t_clamped)
    s_for_b_degen = ((-c) / a.clamp(min=EPS)).clamp(0.0, 1.0)
    s_recalc = torch.where(b_degen, s_for_b_degen, s_recalc)

    closest_a = a1 + s_recalc.unsqueeze(-1) * d1
    closest_b = b1 + t_clamped.unsqueeze(-1) * d2
    diff = closest_a - closest_b
    dist_sq_general = (diff * diff).sum(-1)

    dist_sq = torch.where(both_degen, dist_sq_degen, dist_sq_general)
    return dist_sq.clamp(min=0.0).sqrt()


# ---------------------------------------------------------------------------
# Geom definitions from nova23_humanoid.xml
# ---------------------------------------------------------------------------

@dataclass
class SphereGeom:
    body_idx: int
    name: str
    radius: float
    local_center: Tuple[float, float, float]


@dataclass
class CapsuleGeom:
    body_idx: int
    name: str
    radius: float
    local_p1: Tuple[float, float, float]
    local_p2: Tuple[float, float, float]


SPHERE_GEOMS: List[SphereGeom] = [
    SphereGeom(0, "Hips", 0.08, (0, -0.03, 0)),
    SphereGeom(1, "Spine1", 0.06, (0, -0.03, 0.05)),
    SphereGeom(2, "Spine2", 0.07, (0, -0.04, 0.06)),
    SphereGeom(3, "Chest", 0.11, (0, -0.04, 0.12)),
]

CAPSULE_GEOMS: List[CapsuleGeom] = [
    CapsuleGeom(4, "Neck1", 0.04, (0, -0.03, 0), (0, -0.03, 0.07)),
    CapsuleGeom(5, "Neck2", 0.04, (0, 0, 0), (0, -0.01, 0.05)),
    CapsuleGeom(7, "RightShoulder", 0.045, (-0.045, 0.03, 0), (-0.11, 0.03, 0)),
    CapsuleGeom(8, "RightArm", 0.045, (-0.045, 0, 0), (-0.24, 0, 0)),
    CapsuleGeom(9, "RightForeArm", 0.035, (-0.04, 0, 0), (-0.23, 0, 0)),
    CapsuleGeom(10, "RightHand", 0.05, (-0.02, 0, 0), (-0.050, -0.011, -0.008)),
    CapsuleGeom(11, "LeftShoulder", 0.045, (0.045, 0.03, 0), (0.11, 0.03, 0)),
    CapsuleGeom(12, "LeftArm", 0.045, (0.045, 0, 0), (0.24, 0, 0)),
    CapsuleGeom(13, "LeftForeArm", 0.035, (0.04, 0, 0), (0.23, 0, 0)),
    CapsuleGeom(14, "LeftHand", 0.05, (0.02, 0, 0), (0.050, -0.011, -0.008)),
    CapsuleGeom(15, "RightLeg", 0.06, (0, 0, 0), (0, 0, -0.37)),
    CapsuleGeom(16, "RightShin", 0.05, (0, 0, -0.05), (0, 0, -0.37)),
    CapsuleGeom(19, "LeftLeg", 0.06, (0, 0, 0), (0, 0, -0.37)),
    CapsuleGeom(20, "LeftShin", 0.05, (0, 0, -0.05), (0, 0, -0.37)),
]

ALL_GEOM_BODY_INDICES = sorted(
    {g.body_idx for g in SPHERE_GEOMS} | {g.body_idx for g in CAPSULE_GEOMS}
)

# Pairs to skip: parent-child + MJCF <contact><exclude>.
# Uses body indices from MJCF_BODY_NAMES order.
EXCLUDED_PAIRS = {
    # Parent-child (kinematic tree)
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (3, 7), (7, 8), (8, 9), (9, 10),
    (3, 11), (11, 12), (12, 13), (13, 14),
    (0, 15), (15, 16),
    (0, 19), (19, 20),
    # MJCF <contact><exclude> (near-adjacent bodies)
    (0, 2),   # Hips-Spine2
    (1, 3),   # Spine1-Chest
    (2, 4),   # Spine2-Neck1
    (3, 5),   # Chest-Neck2
    (1, 4),   # Spine1-Neck1
    (2, 5),   # Spine2-Neck2
    (3, 8),   # Chest-RightArm
    (3, 12),  # Chest-LeftArm
    (2, 8),   # Spine2-RightArm
    (2, 12),  # Spine2-LeftArm
}


def _build_check_pairs():
    """Build list of geom pairs to check (excluding adjacent/excluded)."""
    all_geoms = []
    for g in SPHERE_GEOMS:
        all_geoms.append(("sphere", g))
    for g in CAPSULE_GEOMS:
        all_geoms.append(("capsule", g))

    pairs = []
    for i in range(len(all_geoms)):
        for j in range(i + 1, len(all_geoms)):
            ti, gi = all_geoms[i]
            tj, gj = all_geoms[j]
            bi, bj = gi.body_idx, gj.body_idx
            key = (min(bi, bj), max(bi, bj))
            if key in EXCLUDED_PAIRS:
                continue
            pairs.append((ti, gi, tj, gj))
    return pairs


CHECK_PAIRS = _build_check_pairs()


# ---------------------------------------------------------------------------
# Per-motion self-penetration check
# ---------------------------------------------------------------------------

def check_motion(
    motion_path: Path,
    penetration_threshold: float,
    max_penetrating_frames: int,
    max_depth: float,
    subsample: int,
) -> Tuple[bool, dict]:
    """Check a single .motion file for self-penetration.

    Returns (passes, info_dict).
    """
    data = torch.load(str(motion_path), map_location="cpu", weights_only=False)
    body_pos = data["rigid_body_pos"]  # [T, 23, 3]
    body_rot = data["rigid_body_rot"]  # [T, 23, 4]  xyzw

    if subsample > 1:
        body_pos = body_pos[::subsample]
        body_rot = body_rot[::subsample]

    T = body_pos.shape[0]

    # Pre-transform all geom positions to world space
    sphere_world = {}  # body_idx -> (T, 3)
    capsule_world = {}  # body_idx -> ((T, 3), (T, 3))

    for g in SPHERE_GEOMS:
        lc = torch.tensor(g.local_center, dtype=torch.float32).unsqueeze(0)  # (1, 3)
        bp = body_pos[:, g.body_idx]  # (T, 3)
        br = body_rot[:, g.body_idx]  # (T, 4)
        sphere_world[g.body_idx] = bp + quat_rotate(br, lc.expand(T, -1))

    for g in CAPSULE_GEOMS:
        lp1 = torch.tensor(g.local_p1, dtype=torch.float32).unsqueeze(0)
        lp2 = torch.tensor(g.local_p2, dtype=torch.float32).unsqueeze(0)
        bp = body_pos[:, g.body_idx]
        br = body_rot[:, g.body_idx]
        wp1 = bp + quat_rotate(br, lp1.expand(T, -1))
        wp2 = bp + quat_rotate(br, lp2.expand(T, -1))
        capsule_world[g.body_idx] = (wp1, wp2)

    worst_depth_global = 0.0
    worst_pair_name = ""
    penetrating_frame_count = 0
    reject_reason = None

    # Per-frame min distance across all pairs -> max penetration
    max_pen_per_frame = torch.zeros(T)

    for type_i, gi, type_j, gj in CHECK_PAIRS:
        ri = gi.radius
        rj = gj.radius
        sum_r = ri + rj

        if type_i == "sphere" and type_j == "sphere":
            ci = sphere_world[gi.body_idx]
            cj = sphere_world[gj.body_idx]
            dist = torch.linalg.norm(ci - cj, dim=-1)  # (T,)
            gap = dist - sum_r

        elif type_i == "sphere" and type_j == "capsule":
            ci = sphere_world[gi.body_idx]
            p1, p2 = capsule_world[gj.body_idx]
            dist = point_to_segment_dist(ci, p1, p2)
            gap = dist - sum_r

        elif type_i == "capsule" and type_j == "sphere":
            cj = sphere_world[gj.body_idx]
            p1, p2 = capsule_world[gi.body_idx]
            dist = point_to_segment_dist(cj, p1, p2)
            gap = dist - sum_r

        else:  # capsule-capsule
            a1, a2 = capsule_world[gi.body_idx]
            b1, b2 = capsule_world[gj.body_idx]
            dist = segment_to_segment_dist(a1, a2, b1, b2)
            gap = dist - sum_r

        penetration = (-gap).clamp(min=0.0)  # (T,)
        max_pen_per_frame = torch.maximum(max_pen_per_frame, penetration)

        pair_worst = penetration.max().item()
        if pair_worst > worst_depth_global:
            worst_depth_global = pair_worst
            worst_pair_name = f"{gi.name}-{gj.name}"

    # Apply rejection criteria
    frames_above_threshold = (max_pen_per_frame > penetration_threshold).sum().item()

    if worst_depth_global > max_depth:
        reject_reason = f"instant_reject: max_depth={worst_depth_global:.4f}m > {max_depth}m at pair {worst_pair_name}"
    elif frames_above_threshold > max_penetrating_frames:
        reject_reason = (
            f"sustained_reject: {frames_above_threshold} frames > {penetration_threshold}m "
            f"(limit {max_penetrating_frames}) worst pair {worst_pair_name}"
        )

    passes = reject_reason is None

    info = {
        "file": str(motion_path),
        "num_frames_checked": T,
        "subsample": subsample,
        "worst_depth": round(worst_depth_global, 5),
        "worst_pair": worst_pair_name,
        "frames_above_threshold": int(frames_above_threshold),
        "passes": passes,
        "reject_reason": reject_reason,
    }
    return passes, info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    input_dir: Path = typer.Option(..., help="Directory of .motion files (one chunk)."),
    output_dir: Path = typer.Option(
        ..., help="Directory to place surviving motions (symlinks)."
    ),
    penetration_threshold: float = typer.Option(
        0.03, help="Depth threshold (m) for counting a frame as penetrating."
    ),
    max_penetrating_frames: int = typer.Option(
        10, help="Max frames allowed above penetration-threshold."
    ),
    max_depth: float = typer.Option(
        0.05, help="Instant-reject depth (m). Any frame deeper than this rejects the motion."
    ),
    subsample: int = typer.Option(
        3, help="Check every Nth frame (default 3 → 10fps at 30fps data)."
    ),
    copy_mode: bool = typer.Option(
        False, "--copy", help="Copy files instead of symlinking."
    ),
):
    """Filter .motion files for self-penetration and place survivors in output-dir."""

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    motion_files = sorted(input_dir.rglob("*.motion"))
    if not motion_files:
        print(f"No .motion files found in {input_dir}")
        raise typer.Exit(code=1)

    print(f"Found {len(motion_files)} motion files in {input_dir}")
    print(f"Checking {len(CHECK_PAIRS)} geom pairs per frame")
    print(f"Thresholds: penetration={penetration_threshold}m, "
          f"max_frames={max_penetrating_frames}, instant_reject={max_depth}m")
    print(f"Subsample: every {subsample} frames")
    print()

    os.makedirs(output_dir, exist_ok=True)

    total = len(motion_files)
    passed = 0
    rejected = 0
    rejected_infos = []

    for idx, mf in enumerate(motion_files):
        passes, info = check_motion(
            mf,
            penetration_threshold=penetration_threshold,
            max_penetrating_frames=max_penetrating_frames,
            max_depth=max_depth,
            subsample=subsample,
        )

        if passes:
            passed += 1
            rel = mf.relative_to(input_dir)
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if copy_mode:
                shutil.copy2(str(mf), str(dest))
            else:
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                dest.symlink_to(mf)
        else:
            rejected += 1
            rejected_infos.append(info)
            print(f"  REJECT [{rejected}] {mf.name}: {info['reject_reason']}")

        if (idx + 1) % 200 == 0 or idx == total - 1:
            print(f"  Progress: {idx + 1}/{total}  passed={passed} rejected={rejected}")

    # Write reports
    rejected_txt = output_dir / "rejected_motions.txt"
    with open(rejected_txt, "w") as f:
        for info in rejected_infos:
            f.write(f"{info['file']}\t{info['reject_reason']}\n")

    stats = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_motions": total,
        "passed": passed,
        "rejected": rejected,
        "rejection_rate": round(rejected / max(total, 1) * 100, 2),
        "penetration_threshold": penetration_threshold,
        "max_penetrating_frames": max_penetrating_frames,
        "max_depth": max_depth,
        "subsample": subsample,
        "num_check_pairs": len(CHECK_PAIRS),
    }
    stats_path = output_dir / "filter_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("=" * 60)
    print(f"Done.  {passed}/{total} passed, {rejected}/{total} rejected "
          f"({stats['rejection_rate']}%)")
    print(f"Rejected list: {rejected_txt}")
    print(f"Stats: {stats_path}")
    print(f"Surviving motions in: {output_dir}")


if __name__ == "__main__":
    app()
