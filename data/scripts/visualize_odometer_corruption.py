# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Visualize proposed odometer corruption for soft-tracking experiments.

Loads a packaged MotionLib (.pt), extracts root XY trajectories, applies the
proposed per-episode affine + log-space corruption, and produces a multi-panel
figure showing:

1. Sampled trajectories: clean vs corrupted side-by-side
2. Drift statistics: position error as % of true offset, bucketed by distance
3. Direction preservation: histogram of angular error between clean and corrupted
4. Magnitude ratio: distribution of corrupted/clean magnitude
5. What the policy sees: log-magnitude observation values

Corruption model
----------------
Per episode, sample constant parameters:
    scale    ~ Uniform(scale_lo, scale_hi)     — systematic calibration error
    yaw_bias ~ Uniform(-yaw_max_deg, +yaw_max_deg)  — systematic heading error

Per step, apply:
    xy_affine    = scale * Rotate2D(yaw_bias) @ xy_offset_raw
    mag          = ||xy_affine||
    direction    = normalize(xy_affine)
    log_mag      = log(1 + mag)
    noise_weight = mag / (mag + soft_threshold)     — smooth 0→1
    noisy_log    = log_mag + N(0, log_noise_std) * noise_weight
    noisy_mag    = max(exp(noisy_log) - 1, 0)
    xy_corrupted = direction * noisy_mag

The noise_weight smoothly suppresses noise at small offsets (where the
odometer is more accurate and the signal carries less directional info)
and ramps to full noise at large offsets (where drift accumulates).
No hard deadzone — the transition is continuous.

Usage::

    python data/scripts/visualize_odometer_corruption.py \
        --motion-file data/motion_for_trackers/amass_smpl_train_g1_subset200.pt

    # Tune corruption parameters:
    python data/scripts/visualize_odometer_corruption.py \
        --motion-file data/motion_for_trackers/amass_smpl_train_g1_subset200.pt \
        --scale-range 0.6 1.4 --yaw-range-deg 15 --log-noise-std 0.10 \
        --soft-threshold 0.15
"""

import argparse
import math
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion-file", type=str, required=True, help="Packaged MotionLib .pt file")
    p.add_argument("--scale-range", type=float, nargs=2, default=[0.7, 1.3], help="Per-episode scale Uniform(lo, hi)")
    p.add_argument("--yaw-range-deg", type=float, default=12.0, help="Per-episode yaw bias ±deg")
    p.add_argument("--log-noise-std", type=float, default=0.12, help="Per-step noise std in log-magnitude space")
    p.add_argument("--soft-threshold", type=float, default=0.15,
                   help="Characteristic length for smooth noise ramp (meters). "
                        "At this offset magnitude, noise is at 50%% strength.")
    p.add_argument("--num-traj", type=int, default=12, help="Number of trajectories to show in panel 1")
    p.add_argument("--num-trials", type=int, default=10, help="Corruption trials per motion for statistics")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output", type=str, default=None, help="Output image path (default: <motion_file>.odom_corruption.png)")
    return p.parse_args()


def load_motion_xy(motion_file: str):
    """Load root XY offsets (from frame 0) for each motion."""
    d = torch.load(motion_file, weights_only=False, map_location="cpu")
    gts = d["gts"]  # [total_frames, num_bodies, 3]
    length_starts = d["length_starts"]
    motion_num_frames = d["motion_num_frames"]
    motion_dt = d["motion_dt"]

    offsets = []
    durations = []
    for i in range(len(length_starts)):
        s = length_starts[i].item()
        nf = motion_num_frames[i].item()
        root_xy = gts[s : s + nf, 0, :2]
        offsets.append(root_xy - root_xy[0:1])
        durations.append(nf * motion_dt[i].item())
    return offsets, durations


def corrupt_trajectory(xy_offset, scale, yaw_rad, log_noise_std, soft_threshold):
    """Apply affine + smoothly-weighted log-space corruption to a trajectory."""
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    rot = torch.tensor([[cos_y, -sin_y], [sin_y, cos_y]], dtype=xy_offset.dtype)

    xy_affine = scale * (xy_offset @ rot.T)
    mag = torch.norm(xy_affine, dim=-1, keepdim=True).clamp(min=1e-8)
    direction = xy_affine / mag
    log_mag = torch.log(1 + mag)
    # Smooth noise ramp: 0 at small offsets, 1 at large offsets
    noise_weight = mag / (mag + soft_threshold)
    noisy_log_mag = log_mag + torch.randn_like(log_mag) * log_noise_std * noise_weight
    noisy_mag = (torch.exp(noisy_log_mag) - 1).clamp(min=0)
    return direction * noisy_mag


def sample_corruption_params(scale_range, yaw_max_deg):
    scale = scale_range[0] + (scale_range[1] - scale_range[0]) * torch.rand(1).item()
    yaw_rad = math.radians(yaw_max_deg * (2 * torch.rand(1).item() - 1))
    return scale, yaw_rad


def compute_statistics(offsets, args):
    """Run many corruption trials and collect error statistics."""
    buckets = {"0–0.3 m": (0.03, 0.3), "0.3–1 m": (0.3, 1.0), "1–3 m": (1.0, 3.0), "3+ m": (3.0, 1e6)}
    bucket_pct_errs = {k: [] for k in buckets}
    all_dir_errs = []
    all_mag_ratios = []
    all_log_clean = []
    all_log_noisy = []

    for xy_offset in offsets:
        T = xy_offset.shape[0]
        if T < 5:
            continue
        for _ in range(args.num_trials):
            scale, yaw_rad = sample_corruption_params(args.scale_range, args.yaw_range_deg)
            xy_corr = corrupt_trajectory(xy_offset, scale, yaw_rad, args.log_noise_std, args.soft_threshold)

            # Log-magnitude values
            mag_clean = torch.norm(xy_offset, dim=-1)
            log_c = torch.log(1 + mag_clean)
            all_log_clean.append(log_c)

            mag_corr = torch.norm(xy_corr, dim=-1)
            log_n = torch.log(1 + mag_corr)
            all_log_noisy.append(log_n)

            for f in range(5, T, max(1, T // 15)):
                cd = mag_clean[f].item()
                if cd < 0.03:
                    continue
                pos_err = torch.norm(xy_corr[f] - xy_offset[f]).item()
                pct = pos_err / cd * 100

                # Direction error
                c_dir = xy_offset[f] / cd
                cr_mag = mag_corr[f].item()
                if cr_mag < 1e-8:
                    dir_err = 90.0
                else:
                    cr_dir = xy_corr[f] / cr_mag
                    cos_a = torch.clamp(torch.dot(c_dir, cr_dir), -1, 1).item()
                    dir_err = math.degrees(math.acos(cos_a))
                all_dir_errs.append(dir_err)
                all_mag_ratios.append(cr_mag / cd if cd > 1e-8 else 1.0)

                for bname, (blo, bhi) in buckets.items():
                    if blo <= cd < bhi:
                        bucket_pct_errs[bname].append(pct)
                        break

    return bucket_pct_errs, all_dir_errs, all_mag_ratios, all_log_clean, all_log_noisy


def make_figure(offsets, durations, args):
    torch.manual_seed(args.seed)

    # ---- Compute statistics ----
    bucket_pcts, dir_errs, mag_ratios, log_clean_all, log_noisy_all = compute_statistics(offsets, args)
    dir_errs_t = torch.tensor(dir_errs)
    mag_ratios_t = torch.tensor(mag_ratios)
    log_clean_cat = torch.cat(log_clean_all)
    log_noisy_cat = torch.cat(log_noisy_all)

    # ---- Select diverse trajectories for panel 1 ----
    max_offsets = [torch.norm(o, dim=-1).max().item() for o in offsets]
    sorted_idx = sorted(range(len(offsets)), key=lambda i: max_offsets[i])
    # Pick from different quantiles to show the full range
    n = min(args.num_traj, len(offsets))
    pick_indices = [int(i * (len(sorted_idx) - 1) / (n - 1)) for i in range(n)]
    selected = [sorted_idx[i] for i in pick_indices]

    # ---- Figure layout ----
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        f"Odometer Corruption Analysis\n"
        f"scale={args.scale_range}, yaw=±{args.yaw_range_deg}°, "
        f"log_σ={args.log_noise_std}, soft_thresh={args.soft_threshold}m  |  "
        f"{len(offsets)} motions, {args.num_trials} trials each",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.35, top=0.93, bottom=0.05, left=0.06, right=0.97)

    # ================================================================
    # Panel 1: Trajectory comparisons (top row, spans 3 columns)
    # ================================================================
    # Each trajectory is drawn in its own normalized subplot cell
    cols_per_row = min(6, n)
    rows_traj = math.ceil(n / cols_per_row)
    gs_traj = gs[0, :3].subgridspec(rows_traj, cols_per_row, hspace=0.45, wspace=0.3)
    cmap = plt.cm.tab10

    for plot_i, mi in enumerate(selected):
        row_t = plot_i // cols_per_row
        col_t = plot_i % cols_per_row
        ax = fig.add_subplot(gs_traj[row_t, col_t])

        xy = offsets[mi]
        # Generate 3 corruption trials to show spread
        lines_corr = []
        for trial in range(3):
            scale, yaw_rad = sample_corruption_params(args.scale_range, args.yaw_range_deg)
            lines_corr.append(corrupt_trajectory(xy, scale, yaw_rad, args.log_noise_std, args.soft_threshold))

        color = cmap(plot_i % 10)
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.0, alpha=0.9, zorder=3)
        for t_i, xy_c in enumerate(lines_corr):
            ax.plot(xy_c[:, 0], xy_c[:, 1], color=color, linewidth=0.8, linestyle="--", alpha=0.35, zorder=2)

        # Start and end markers
        ax.plot(xy[0, 0], xy[0, 1], "o", color=color, markersize=5, zorder=5)
        if xy.shape[0] > 2:
            ax.annotate(
                "",
                xy=(xy[-1, 0].item(), xy[-1, 1].item()),
                xytext=(xy[-3, 0].item(), xy[-3, 1].item()),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
                zorder=4,
            )

        mx = max_offsets[mi]
        dur = durations[mi]
        ax.set_title(f"#{mi}  {dur:.0f}s  {mx:.1f}m", fontsize=7, pad=2)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.15)

    # Overall title for the trajectory region (use first subplot axes)
    fig.text(0.37, 0.935, "Trajectories: solid=clean, dashed=3 corruption trials",
             fontsize=10, ha="center", style="italic")

    # ================================================================
    # Panel 2: Dataset offset distribution (top-right)
    # ================================================================
    ax_dist = fig.add_subplot(gs[0, 3])
    all_max = sorted(max_offsets)
    ax_dist.hist(all_max, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    for p, ls in [(50, "-"), (90, "--"), (95, ":")]:
        idx = min(int(len(all_max) * p / 100), len(all_max) - 1)
        val = all_max[idx]
        ax_dist.axvline(val, color="red", linestyle=ls, linewidth=1.2, label=f"p{p}={val:.1f}m")
    ax_dist.set_title("Max XY Offset per Motion", fontsize=11)
    ax_dist.set_xlabel("Max offset from start (m)")
    ax_dist.set_ylabel("Count")
    ax_dist.legend(fontsize=8)

    # ================================================================
    # Panel 3: Drift % by distance bucket (middle-left)
    # ================================================================
    ax_drift = fig.add_subplot(gs[1, 0:2])
    bucket_names = [k for k in bucket_pcts if bucket_pcts[k]]
    bp_data = [bucket_pcts[k] for k in bucket_names]
    if bp_data:
        bp = ax_drift.boxplot(
            bp_data,
            tick_labels=bucket_names,
            patch_artist=True,
            showfliers=False,
            whiskerprops=dict(linewidth=1.2),
            medianprops=dict(color="red", linewidth=2),
        )
        colors = ["#a8d8ea", "#6cb4ee", "#3a86c8", "#1a5276"]
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        # Overlay real odometer range
        ax_drift.axhspan(10, 20, alpha=0.12, color="green", label="Expected real G1 (~10-20%)")
        ax_drift.axhspan(20, 35, alpha=0.08, color="orange", label="Pessimistic margin (20-35%)")
    ax_drift.set_title("Position Drift by True Offset Distance", fontsize=11)
    ax_drift.set_ylabel("Error / True Offset (%)")
    ax_drift.set_ylim(0, min(150, max(t.item() for bp_d in bp_data for t in [torch.tensor(bp_d).quantile(0.95)]) + 20) if bp_data else 100)
    ax_drift.legend(fontsize=8, loc="upper right")
    ax_drift.grid(True, axis="y", alpha=0.3)

    # ================================================================
    # Panel 4: Direction error histogram (middle-right-left)
    # ================================================================
    ax_dir = fig.add_subplot(gs[1, 2])
    ax_dir.hist(dir_errs, bins=50, range=(0, 90), color="coral", edgecolor="white", alpha=0.8)
    med_dir = dir_errs_t.median().item()
    p90_dir = dir_errs_t.quantile(0.9).item()
    ax_dir.axvline(med_dir, color="red", linewidth=2, label=f"median={med_dir:.1f}°")
    ax_dir.axvline(p90_dir, color="darkred", linewidth=1.5, linestyle="--", label=f"p90={p90_dir:.1f}°")
    frac_20 = (dir_errs_t < 20).float().mean().item()
    ax_dir.set_title(f"Direction Error  ({frac_20:.0%} < 20°)", fontsize=11)
    ax_dir.set_xlabel("Angular error (degrees)")
    ax_dir.set_ylabel("Count")
    ax_dir.legend(fontsize=8)

    # ================================================================
    # Panel 5: Magnitude ratio histogram (middle-right-right)
    # ================================================================
    ax_mag = fig.add_subplot(gs[1, 3])
    # Clip for display
    mr_clip = mag_ratios_t.clamp(0, 3)
    ax_mag.hist(mr_clip.tolist(), bins=60, range=(0, 3), color="mediumpurple", edgecolor="white", alpha=0.8)
    ax_mag.axvline(1.0, color="black", linewidth=1.5, linestyle="-", alpha=0.5, label="perfect (1.0)")
    med_mr = mag_ratios_t.median().item()
    ax_mag.axvline(med_mr, color="red", linewidth=2, label=f"median={med_mr:.2f}")
    p25_mr = mag_ratios_t.quantile(0.25).item()
    p75_mr = mag_ratios_t.quantile(0.75).item()
    ax_mag.axvspan(p25_mr, p75_mr, alpha=0.15, color="purple", label=f"IQR=[{p25_mr:.2f}, {p75_mr:.2f}]")
    ax_mag.set_title("Magnitude Ratio (corrupted / clean)", fontsize=11)
    ax_mag.set_xlabel("Ratio")
    ax_mag.set_ylabel("Count")
    ax_mag.legend(fontsize=8)

    # ================================================================
    # Panel 6: Log-magnitude obs distribution (bottom-left)
    # ================================================================
    ax_log = fig.add_subplot(gs[2, 0:2])
    bins = torch.linspace(0, log_clean_cat.max().item() * 1.1, 60).tolist()
    ax_log.hist(log_clean_cat.tolist(), bins=bins, alpha=0.6, color="steelblue", edgecolor="white", label="clean log(1+||xy||)")
    ax_log.hist(log_noisy_cat.clamp(min=0).tolist(), bins=bins, alpha=0.4, color="coral", edgecolor="white", label="noisy (what policy sees)")
    ax_log.set_title("Log-Magnitude Observation Distribution", fontsize=11)
    ax_log.set_xlabel("log(1 + magnitude)")
    ax_log.set_ylabel("Count (across all frames & trials)")
    ax_log.legend(fontsize=9)

    # ================================================================
    # Panel 7: Noise weight curve + per-episode param scatter (bottom-middle)
    # ================================================================
    gs_bottom_mid = gs[2, 2].subgridspec(2, 1, hspace=0.5)

    # Top: noise weight curve
    ax_nw = fig.add_subplot(gs_bottom_mid[0])
    x_range = torch.linspace(0, 3.0, 200)
    nw = x_range / (x_range + args.soft_threshold)
    ax_nw.plot(x_range.tolist(), nw.tolist(), color="teal", linewidth=2)
    ax_nw.axvline(args.soft_threshold, color="red", linestyle="--", linewidth=1,
                  label=f"soft_thresh={args.soft_threshold}m")
    ax_nw.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax_nw.set_title("Noise Weight: mag / (mag + thresh)", fontsize=9)
    ax_nw.set_xlabel("||xy_offset|| (m)", fontsize=8)
    ax_nw.set_ylabel("noise weight", fontsize=8)
    ax_nw.set_ylim(-0.05, 1.05)
    ax_nw.tick_params(labelsize=7)
    ax_nw.legend(fontsize=7)
    ax_nw.grid(True, alpha=0.2)

    # Bottom: per-episode param scatter
    ax_params = fig.add_subplot(gs_bottom_mid[1])
    n_show = 200
    scales_show = [args.scale_range[0] + (args.scale_range[1] - args.scale_range[0]) * torch.rand(1).item() for _ in range(n_show)]
    yaws_show = [args.yaw_range_deg * (2 * torch.rand(1).item() - 1) for _ in range(n_show)]
    ax_params.scatter(scales_show, yaws_show, s=12, alpha=0.5, color="teal")
    ax_params.set_xlim(args.scale_range[0] - 0.1, args.scale_range[1] + 0.1)
    ax_params.set_ylim(-args.yaw_range_deg - 5, args.yaw_range_deg + 5)
    ax_params.axhline(0, color="gray", linewidth=0.5)
    ax_params.axvline(1.0, color="gray", linewidth=0.5)
    ax_params.set_title("Per-Episode (scale, yaw) Samples", fontsize=9)
    ax_params.set_xlabel("Scale factor", fontsize=8)
    ax_params.set_ylabel("Yaw bias (°)", fontsize=8)
    ax_params.tick_params(labelsize=7)
    ax_params.grid(True, alpha=0.2)

    # ================================================================
    # Panel 8: Summary stats text (bottom-right-right)
    # ================================================================
    ax_txt = fig.add_subplot(gs[2, 3])
    ax_txt.axis("off")
    lines = [
        "CORRUPTION PARAMETERS",
        f"  scale:       Uniform({args.scale_range[0]}, {args.scale_range[1]})",
        f"  yaw bias:    Uniform(±{args.yaw_range_deg}°)",
        f"  log noise σ: {args.log_noise_std}",
        f"  soft thresh: {args.soft_threshold} m",
        "",
        "SUMMARY STATISTICS",
        f"  Direction error:",
        f"    median {med_dir:.1f}°, p90 {p90_dir:.1f}°",
        f"    {frac_20:.0%} within 20°",
        f"  Magnitude ratio:",
        f"    median {med_mr:.2f}",
        f"    IQR [{p25_mr:.2f}, {p75_mr:.2f}]",
    ]
    # Per-bucket drift
    lines.append("")
    lines.append("  Drift % (median):")
    for bname in bucket_pcts:
        if bucket_pcts[bname]:
            med = torch.tensor(bucket_pcts[bname]).median().item()
            lines.append(f"    {bname:12s}: {med:.0f}%")

    lines.extend([
        "",
        "DESIGN INTENT",
        "  Direction is reliable (>98% < 20°)",
        "  Magnitude is noisy (~20-40% drift)",
        "  Smooth noise ramp (no hard deadzone)",
        "  Same corruption in sim & on real robot",
    ])
    ax_txt.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax_txt.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    return fig


def main():
    args = parse_args()
    offsets, durations = load_motion_xy(args.motion_file)
    print(f"Loaded {len(offsets)} motions from {args.motion_file}")

    fig = make_figure(offsets, durations, args)

    if args.output is None:
        stem = Path(args.motion_file).stem
        args.output = str(Path(args.motion_file).parent / f"{stem}.odom_corruption.png")
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
