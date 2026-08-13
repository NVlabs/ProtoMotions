# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert original SAMP pkl files to SOMA23 .motion files.

Pipeline:
    1. Load SAMP MoSh++ pkl → SMPL-X pose params (y-up, 120fps)
    2. Extract SMPL body joints (first 66 params + 6 zero hand padding)
    3. Run smplx forward pass → posed vertices (y-up)
    4. SOMA-X PoseInversion → absolute rotations (y-up, 78 joints)
    5. Convert absolute → relative via remove_joint_orient_local
    6. Subsample 78 → 23 MJCF joints
    7. Convert to ProtoMotions .motion via create_motion_from_soma23_data (y-up → z-up)

Usage::

    python data/scripts/convert_samp_pkl_to_soma23.py \\
        /path/to/samp/pkl/armchair001_stageII.pkl \\
        /path/to/output/soma23_test/ \\
        --batch-size 256

    # Directory mode
    python data/scripts/convert_samp_pkl_to_soma23.py \\
        /path/to/samp/pkl/ \\
        /path/to/output/soma23_test/ \\
        --batch-size 256
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import smplx
import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from soma import SOMALayer
from soma.pose_inversion import PoseInversion

sys.path.insert(0, str(Path(__file__).parent))
from convert_soma23_to_proto import SOMASKEL77_TO_MJCF_INDICES
from contact_detection import compute_contact_labels_from_pos_and_vel

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_from_transforms_with_velocities,
    extract_qpos_from_transforms,
    compute_angular_velocity,
    compute_joint_rot_mats_from_global_mats,
)
from protomotions.utils.rotations import (
    matrix_to_quaternion,
)

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

SMPL_PKL = Path.home() / (
    ".cache/huggingface/hub/models--nvidia--soma-x/snapshots/"
    "6db1b4e8d737db240e680ca5358ae2897f2d9427/SMPL/SMPL_NEUTRAL.pkl"
)

# Output FPS (downsample from 120fps MoSh++ to 30fps)
OUTPUT_FPS = 30


def convert_pkl(
    pkl_path: Path,
    soma: SOMALayer,
    inv: PoseInversion,
    smpl_model,
    kinematic_info,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Convert a single SAMP pkl to SOMA23 .motion dict."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    fullposes = data["pose_est_fullposes"]  # [T, 165] SMPL-X axis-angle
    trans = data["pose_est_trans"]  # [T, 3]
    betas = data["shape_est_betas"]  # [num_betas]
    mocap_fps = data["mocap_framerate"]  # 120.0

    # Downsample to output FPS
    downsample = int(mocap_fps / OUTPUT_FPS)
    fullposes = fullposes[::downsample]
    trans = trans[::downsample]
    T = fullposes.shape[0]

    # Extract SMPL body params from SMPL-X: first 3 = global_orient, next 63 = body (21 joints)
    global_orient = torch.from_numpy(fullposes[:, :3]).float().to(device)
    # Body pose: joints 1-21 (63 params) + 2 zero hand joints (6 params) = 69
    body_pose_21 = fullposes[:, 3:66]  # 21 joints
    body_pose_23 = np.concatenate(
        [body_pose_21, np.zeros((T, 6), dtype=fullposes.dtype)], axis=1
    )  # 23 joints = 69
    body_pose = torch.from_numpy(body_pose_23).float().to(device)

    # Betas (use first 10 for SMPL)
    betas_10 = torch.from_numpy(betas[:10]).float().to(device).unsqueeze(0)
    inv.prepare_identity(betas_10)

    transl = torch.from_numpy(trans).float().to(device)

    # --- Batch: smplx forward + SOMA-X inversion ---
    all_rotations = []
    all_root_trans = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        with torch.no_grad():
            smpl_out = smpl_model(
                body_pose=body_pose[start:end],
                global_orient=global_orient[start:end],
                betas=betas_10.expand(end - start, -1),
                transl=transl[start:end],
            )
        result = inv.fit(
            smpl_out.vertices,
            body_iters=2,
            full_iters=1,
            autograd_iters=10,
            autograd_lr=5e-3,
        )
        all_rotations.append(result["rotations"].cpu())
        all_root_trans.append(result["root_translation"].cpu())

    rotations = torch.cat(all_rotations, dim=0)  # [T, 78, 3, 3]
    root_trans = torch.cat(all_root_trans, dim=0)  # [T, 3]

    # --- Use absolute rotations directly (skip remove_joint_orient_local) ---
    # Subsample 78 → 23 MJCF absolute rotations
    somax_to_mjcf = [i + 1 for i in SOMASKEL77_TO_MJCF_INDICES]
    soma23_abs = rotations[:, somax_to_mjcf]  # [T, 23, 3, 3]

    # SOMA-X absolute rotations are pose-only (T-pose ≈ identity).
    # The SOMA23 MJCF has identity body orientations.
    # So we can extract locals directly from these globals using MJCF kinematic_info.
    # Then rotate root position from the source coordinate frame to z-up.
    local_zup = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=soma23_abs,
    )
    # Root position: source data is z-up (SAMP MoSh++ convention)
    root_pos_zup = root_trans
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos_zup,
        joint_rot_mats=local_zup,
        fps=OUTPUT_FPS,
        compute_velocities=True,
        velocity_max_horizon=3,
    )
    motion.local_rigid_body_rot = matrix_to_quaternion(local_zup, w_last=True)
    num_joints = 22
    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_pos_zup,
        joint_rot_mats=local_zup,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]
    local_ang_vel = compute_angular_velocity(
        batched_robot_rot_mats=local_zup[:, 1:],
        fps=OUTPUT_FPS,
    )
    motion.dof_vel = local_ang_vel.reshape(-1, num_joints * 3)
    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)

    out = motion.to_dict()
    out["fps"] = OUTPUT_FPS
    return out


@app.command()
def main(
    input_path: Path = typer.Argument(..., help="Single .pkl file or directory"),
    output_dir: Path = typer.Argument(..., help="Output directory for .motion files"),
    batch_size: int = typer.Option(256, "--batch-size"),
    device_str: str = typer.Option("cuda", "--device"),
) -> None:
    """Convert SAMP MoSh++ pkl files to SOMA23 .motion."""
    device = torch.device(device_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        pkl_files = [input_path]
    elif input_path.is_dir():
        pkl_files = sorted(input_path.glob("*.pkl"))
    else:
        console.print(f"[red]ERROR[/]: {input_path} not found")
        raise typer.Exit(1)

    if not pkl_files:
        console.print(f"[red]ERROR[/]: no .pkl files in {input_path}")
        raise typer.Exit(1)

    console.print(f"Converting {len(pkl_files)} files | batch_size={batch_size}")

    # --- Initialize models ---
    console.print("Loading SMPL model...")
    smpl_model = smplx.create(
        model_type="smpl",
        model_path=str(SMPL_PKL),
        use_pca=False,
        flat_hand_mean=True,
        batch_size=1,
    ).to(device)

    console.print("Loading SOMA-X...")
    soma = SOMALayer(
        data_root=None,
        identity_model_type="smpl",
        device=device_str,
        mode="warp",
    )
    inv = PoseInversion(soma, low_load=True)

    console.print("Loading SOMA23 kinematic info...")
    mjcf_path = (
        Path(__file__).parent.parent.parent
        / "protomotions"
        / "data"
        / "assets"
        / "mjcf"
        / "soma23_humanoid.xml"
    )
    kinematic_info = extract_kinematic_info(str(mjcf_path))

    successes = 0
    failures = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
    ) as progress:
        task = progress.add_task("Converting", total=len(pkl_files), status="")
        for pf in pkl_files:
            name = pf.stem
            progress.update(task, status=f"[cyan]{name[:40]}[/]")
            out_path = output_dir / f"{name}.motion"
            if out_path.exists():
                progress.console.print(f"  [dim]SKIP[/] {name}")
                progress.advance(task)
                successes += 1
                continue
            try:
                out_dict = convert_pkl(
                    pf, soma, inv, smpl_model, kinematic_info, batch_size, device
                )
                torch.save(out_dict, out_path)
                successes += 1
            except Exception as e:
                failures.append((name, str(e)))
                progress.console.print(f"  [red]FAIL[/] {name}: {e}")
            progress.advance(task)

    console.print(f"\n[green]Done[/]: {successes}/{len(pkl_files)} converted")
    if failures:
        console.print(f"[red]Failures ({len(failures)}):[/]")
        for name, err in failures:
            console.print(f"  {name}: {err}")


if __name__ == "__main__":
    app()
