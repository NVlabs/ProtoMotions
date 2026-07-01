# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert SOMA23 .motion files to SMPL .motion format via optimization.

Both skeletons derive from the SMPL body model but have different bone lengths
and topology (SOMA23: 23 bodies/66 DOFs, SMPL: 24 bodies/69 DOFs).  A direct
transfer of local rotations would accumulate position drift at extremities.

Instead, we optimise SMPL joint angles (axis-angle, 3 per joint) so that the
SMPL FK output best matches the SOMA23 global body positions and rotations
for corresponding bodies.

Body correspondence (SOMA23 idx → SMPL MJCF idx):
    Hips(0)→Pelvis(0)  Spine1(1)→Torso(9)  Spine2(2)→Spine(10)
    Chest(3)→Chest(11)  Neck2(5)→Neck(12)  Head(6)→Head(13)
    RShoulder(7)→R_Thorax(19)  RArm(8)→R_Shoulder(20)
    RForeArm(9)→R_Elbow(21)  RHand(10)→R_Wrist(22)
    LShoulder(11)→L_Thorax(14)  LArm(12)→L_Shoulder(15)
    LForeArm(13)→L_Elbow(16)  LHand(14)→L_Wrist(17)
    RLeg(15)→R_Hip(5)  RShin(16)→R_Knee(6)
    RFoot(17)→R_Ankle(7)  RToeBase(18)→R_Toe(8)
    LLeg(19)→L_Hip(1)  LShin(20)→L_Knee(2)
    LFoot(21)→L_Ankle(3)  LToeBase(22)→L_Toe(4)

SOMA23 Neck1(4) is absorbed into Neck2→Neck.
SMPL L_Hand(18)/R_Hand(23) have no SOMA23 counterpart → identity local rot.

Usage::

    # Single file
    python data/scripts/convert_soma23_motion_to_smpl.py \\
        ~/protomotions_assets/beyond/motions_soma23/Chimney_Climb_001_a.motion \\
        /tmp/beyond_smpl/

    # Directory
    python data/scripts/convert_soma23_motion_to_smpl.py \\
        ~/protomotions_assets/beyond/motions_soma23/ \\
        /tmp/beyond_smpl/
"""

from __future__ import annotations

import sys
from pathlib import Path

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

sys.path.insert(0, str(Path(__file__).parent))
from contact_detection import compute_contact_labels_from_pos_and_vel

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    compute_angular_velocity,
)
from protomotions.utils.rotations import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

# ── Body correspondence: soma23_idx → smpl_mjcf_idx ──────────────────────
# fmt: off
SOMA23_TO_SMPL = {
    0: 0,    # Hips → Pelvis
    1: 9,    # Spine1 → Torso
    2: 10,   # Spine2 → Spine
    3: 11,   # Chest → Chest
    # 4: Neck1 — absorbed into Neck2→Neck
    5: 12,   # Neck2 → Neck
    6: 13,   # Head → Head
    7: 19,   # RightShoulder → R_Thorax
    8: 20,   # RightArm → R_Shoulder
    9: 21,   # RightForeArm → R_Elbow
    10: 22,  # RightHand → R_Wrist
    11: 14,  # LeftShoulder → L_Thorax
    12: 15,  # LeftArm → L_Shoulder
    13: 16,  # LeftForeArm → L_Elbow
    14: 17,  # LeftHand → L_Wrist
    15: 5,   # RightLeg → R_Hip
    16: 6,   # RightShin → R_Knee
    17: 7,   # RightFoot → R_Ankle
    18: 8,   # RightToeBase → R_Toe
    19: 1,   # LeftLeg → L_Hip
    20: 2,   # LeftShin → L_Knee
    21: 3,   # LeftFoot → L_Ankle
    22: 4,   # LeftToeBase → L_Toe
}
# fmt: on

# SMPL bodies without SOMA23 counterpart (identity local rotation)
SMPL_UNMAPPED = [18, 23]  # L_Hand, R_Hand


def _axis_angle_to_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle [..., 3] → rotation matrix [..., 3, 3]."""
    angle = aa.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / angle
    K = torch.zeros(*aa.shape[:-1], 3, 3, device=aa.device, dtype=aa.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    sin = angle.unsqueeze(-1).sin()
    cos = angle.unsqueeze(-1).cos()
    return eye + sin * K + (1 - cos) * (K @ K)


def _smpl_fk(
    root_pos: torch.Tensor,
    joint_aa: torch.Tensor,
    kinematic_info,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched SMPL FK from root position + axis-angle joint params.

    Args:
        root_pos: [B, 3] root translation.
        joint_aa: [B, 23, 3] axis-angle per joint (joints 1..23).
        kinematic_info: SMPL kinematic info from MJCF.

    Returns:
        global_pos: [B, 24, 3] world-frame body positions.
        global_rot: [B, 24, 3, 3] world-frame body rotations.
    """
    B = root_pos.shape[0]
    device = root_pos.device
    num_bodies = len(kinematic_info.body_names)
    parent_indices = kinematic_info.parent_indices
    local_offsets = kinematic_info.local_pos.to(device)  # [24, 3]
    rest_rot_mats = kinematic_info.local_rot_ref_mat.to(device)  # [24, 3, 3]

    # Joint rotation matrices from axis-angle
    joint_rot_mats = _axis_angle_to_matrix(joint_aa)  # [B, 23, 3, 3]

    body_pos = [None] * num_bodies
    body_rot = [None] * num_bodies
    body_pos[0] = root_pos
    body_rot[0] = rest_rot_mats[0].unsqueeze(0).expand(B, -1, -1)

    for i in range(1, num_bodies):
        p = parent_indices[i]
        local_rot = rest_rot_mats[i].unsqueeze(0) @ joint_rot_mats[:, i - 1]
        body_rot[i] = body_rot[p] @ local_rot
        offset_world = (body_rot[p] @ local_offsets[i].unsqueeze(-1)).squeeze(-1)
        body_pos[i] = body_pos[p] + offset_world

    return torch.stack(body_pos, dim=1), torch.stack(body_rot, dim=1)


def _smpl_fk_with_root_rot(
    root_pos: torch.Tensor,
    root_rot_mat: torch.Tensor,
    joint_aa: torch.Tensor,
    kinematic_info,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FK with explicit root rotation (not optimized — taken from SOMA23 data).

    Args:
        root_pos: [B, 3]
        root_rot_mat: [B, 3, 3] global root rotation.
        joint_aa: [B, 23, 3] axis-angle per joint.
        kinematic_info: SMPL kinematic info.

    Returns:
        global_pos: [B, 24, 3], global_rot: [B, 24, 3, 3]
    """
    device = root_pos.device
    num_bodies = len(kinematic_info.body_names)
    parent_indices = kinematic_info.parent_indices
    local_offsets = kinematic_info.local_pos.to(device)
    rest_rot_mats = kinematic_info.local_rot_ref_mat.to(device)

    joint_rot_mats = _axis_angle_to_matrix(joint_aa)

    body_pos = [None] * num_bodies
    body_rot = [None] * num_bodies
    body_pos[0] = root_pos
    body_rot[0] = root_rot_mat

    for i in range(1, num_bodies):
        p = parent_indices[i]
        local_rot = rest_rot_mats[i].unsqueeze(0) @ joint_rot_mats[:, i - 1]
        body_rot[i] = body_rot[p] @ local_rot
        offset_world = (body_rot[p] @ local_offsets[i].unsqueeze(-1)).squeeze(-1)
        body_pos[i] = body_pos[p] + offset_world

    return torch.stack(body_pos, dim=1), torch.stack(body_rot, dim=1)


def _init_from_global_rotations(
    soma23_grs: torch.Tensor,
    smpl_ki,
) -> torch.Tensor:
    """Compute initial SMPL joint axis-angle from SOMA23 global rotations.

    Maps SOMA23 global rotations to SMPL body order, extracts SMPL local
    rotations via the kinematic tree, then converts to axis-angle.

    Args:
        soma23_grs: [B, 23, 4] SOMA23 global rotations (xyzw, z-up).
        smpl_ki: SMPL kinematic info.

    Returns:
        init_aa: [B, 23, 3] initial axis-angle per SMPL joint.
    """
    from protomotions.components.pose_lib import compute_joint_rot_mats_from_global_mats

    B = soma23_grs.shape[0]
    device = soma23_grs.device
    soma_rot_mat = quaternion_to_matrix(soma23_grs, w_last=True)

    # Map SOMA23 globals to SMPL body order
    smpl_globals = (
        torch.eye(3, device=device)
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(B, 24, -1, -1)
        .clone()
    )
    for soma_idx, smpl_idx in SOMA23_TO_SMPL.items():
        smpl_globals[:, smpl_idx] = soma_rot_mat[:, soma_idx]
    # Unmapped: L_Hand(18) and R_Hand(23) use parent global
    smpl_globals[:, 18] = smpl_globals[:, 17]
    smpl_globals[:, 23] = smpl_globals[:, 22]

    # Extract SMPL local rotations from globals
    local_rot_mats = compute_joint_rot_mats_from_global_mats(
        kinematic_info=smpl_ki, global_rot_mats=smpl_globals
    )

    # Factor out rest rotation: joint_rot = rest^-1 @ local
    rest_rot = smpl_ki.local_rot_ref_mat.to(device)
    joint_rot_mats = rest_rot[1:].unsqueeze(0).transpose(-1, -2) @ local_rot_mats[:, 1:]

    # Convert to axis-angle via scipy
    from scipy.spatial.transform import Rotation as Rot

    init_aa = (
        torch.from_numpy(
            Rot.from_matrix(joint_rot_mats.cpu().reshape(-1, 3, 3).numpy()).as_rotvec()
        )
        .float()
        .reshape(B, 23, 3)
        .to(device)
    )
    return init_aa


# Per-body position weights for optimization.
# Focus on end-effectors (pelvis, head, hands, feet), then elbows/knees,
# low weight for spine/shoulder chain (let bone length differences absorb there).
_SMPL_POS_WEIGHTS = torch.ones(24)
# Pelvis(0), Head(13)
_SMPL_POS_WEIGHTS[[0, 13]] = 10.0
# Hands: L_Wrist(17), R_Wrist(22)
_SMPL_POS_WEIGHTS[[17, 22]] = 20.0
# Feet: L_Ankle(3), R_Ankle(7), L_Toe(4), R_Toe(8)
_SMPL_POS_WEIGHTS[[3, 7, 4, 8]] = 20.0
# Elbows/Knees: L_Elbow(16), R_Elbow(21), L_Knee(2), R_Knee(6)
_SMPL_POS_WEIGHTS[[16, 21, 2, 6]] = 8.0
# Spine/shoulders get low weight — let bone length mismatch absorb here
_SMPL_POS_WEIGHTS[[9, 10, 11, 12, 14, 15, 19, 20]] = 0.2


def optimize_frame_batch(
    soma23_gts: torch.Tensor,
    soma23_grs: torch.Tensor,
    smpl_ki,
    num_lbfgs_iters: int = 50,
    pos_weight: float = 50.0,
    rot_weight: float = 5.0,
    smooth_weight: float = 5.0,
    prev_joint_aa: torch.Tensor | None = None,
) -> torch.Tensor:
    """Optimize SMPL joint angles for a batch of frames using L-BFGS.

    Initializes from SOMA23 global rotation transfer, then optimizes
    position + rotation + temporal smoothness.

    Position weighting focuses on end-effectors (pelvis, head, hands, feet),
    then elbows/knees, with low weight on spine/shoulders.

    Args:
        soma23_gts: [B, 23, 3] SOMA23 global positions (z-up).
        soma23_grs: [B, 23, 4] SOMA23 global rotations (xyzw, z-up).
        smpl_ki: SMPL kinematic info.
        num_lbfgs_iters: Number of L-BFGS outer iterations.
        pos_weight: Global position loss weight.
        rot_weight: Global rotation loss weight.
        smooth_weight: Temporal smoothness weight (joint angle difference to prev batch).
        prev_joint_aa: [B_prev, 23, 3] solution from previous batch (last few frames).

    Returns:
        joint_aa: [B, 23, 3] optimized axis-angle per SMPL joint.
    """
    B = soma23_gts.shape[0]
    device = soma23_gts.device

    # Build targets
    target_pos = torch.zeros(B, 24, 3, device=device)
    target_rot_mat = torch.zeros(B, 24, 3, 3, device=device)
    target_mask = torch.zeros(24, device=device)

    soma23_rot_mat = quaternion_to_matrix(soma23_grs, w_last=True)

    for soma_idx, smpl_idx in SOMA23_TO_SMPL.items():
        target_pos[:, smpl_idx] = soma23_gts[:, soma_idx]
        target_rot_mat[:, smpl_idx] = soma23_rot_mat[:, soma_idx]
        target_mask[smpl_idx] = 1.0

    root_pos = soma23_gts[:, 0]
    root_rot_mat = soma23_rot_mat[:, 0]

    # Per-body weights
    body_pos_w = _SMPL_POS_WEIGHTS.to(device) * target_mask

    # Initialize from global rotation transfer
    init_aa = _init_from_global_rotations(soma23_grs, smpl_ki)
    joint_aa = init_aa.clone().detach().requires_grad_(True)

    # Detached previous solution for smoothness (last frame of prev batch)
    prev_aa = prev_joint_aa[-1:].detach() if prev_joint_aa is not None else None

    optimizer = torch.optim.LBFGS(
        [joint_aa], lr=1.0, max_iter=20, line_search_fn="strong_wolfe"
    )

    eye = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)

    for _ in range(num_lbfgs_iters):

        def closure():
            optimizer.zero_grad()
            pred_pos, pred_rot = _smpl_fk_with_root_rot(
                root_pos, root_rot_mat, joint_aa, smpl_ki
            )
            # Position loss (end-effector weighted)
            pos_err = ((pred_pos - target_pos) ** 2).sum(dim=-1) * body_pos_w
            pos_loss = pos_err.mean()

            # Rotation loss (all mapped bodies)
            rot_diff = pred_rot.transpose(-1, -2) @ target_rot_mat
            rot_err = ((rot_diff - eye) ** 2).sum(dim=(-1, -2)) * target_mask
            rot_loss = rot_err.mean()

            loss = pos_weight * pos_loss + rot_weight * rot_loss

            # Temporal smoothness: penalize joint angle jumps between frames
            if B > 1:
                frame_diff = (joint_aa[1:] - joint_aa[:-1]) ** 2
                loss = loss + smooth_weight * frame_diff.mean()
            # Continuity with previous batch
            if prev_aa is not None:
                boundary_diff = (joint_aa[0:1] - prev_aa) ** 2
                loss = loss + smooth_weight * boundary_diff.mean()

            loss.backward()
            return loss

        optimizer.step(closure)

    return joint_aa.detach()


def convert_motion(
    soma23_path: Path,
    smpl_ki,
    device: torch.device,
    batch_size: int = 64,
    num_iters: int = 500,
) -> dict:
    """Convert a single SOMA23 .motion to SMPL .motion dict."""
    data = torch.load(soma23_path, weights_only=False, map_location=device)
    soma_gts = data["gts"].to(device)  # [T, 23, 3]
    soma_grs = data["grs"].to(device)  # [T, 23, 4] xyzw
    fps = data.get("fps", 30)
    T = soma_gts.shape[0]

    # Process in overlapping windows for temporal consistency.
    # Each window overlaps by `overlap` frames with the previous one.
    # The overlap region is blended linearly to avoid boundary discontinuities.
    overlap = batch_size // 4  # 25% overlap
    all_joint_aa = []
    prev_aa = None

    start = 0
    while start < T:
        end = min(start + batch_size, T)
        batch_gts = soma_gts[start:end]
        batch_grs = soma_grs[start:end]

        joint_aa_batch = optimize_frame_batch(
            batch_gts, batch_grs, smpl_ki, prev_joint_aa=prev_aa
        )

        if prev_aa is not None and start > 0:
            # Blend overlap region: linear ramp from prev to current
            blend_len = min(overlap, joint_aa_batch.shape[0])
            alpha = (
                torch.linspace(0, 1, blend_len, device=device)
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            prev_overlap = all_joint_aa[-1][-blend_len:].to(device)
            curr_overlap = joint_aa_batch[:blend_len]
            blended = (1 - alpha) * prev_overlap + alpha * curr_overlap
            all_joint_aa[-1] = all_joint_aa[-1][:-blend_len]
            all_joint_aa.append(blended.cpu())
            all_joint_aa.append(joint_aa_batch[blend_len:].cpu())
        else:
            all_joint_aa.append(joint_aa_batch.cpu())

        prev_aa = joint_aa_batch
        start += batch_size - overlap

    joint_aa = torch.cat(all_joint_aa, dim=0)[:T]  # trim to exact length

    # Run FK to get final positions/rotations
    root_pos = soma_gts[:, 0].cpu()
    root_rot_mat = quaternion_to_matrix(soma_grs[:, 0], w_last=True).cpu()

    # Batched FK on CPU
    with torch.no_grad():
        global_pos, global_rot = _smpl_fk_with_root_rot(
            root_pos, root_rot_mat, joint_aa, smpl_ki
        )

    global_rot_quat = matrix_to_quaternion(global_rot, w_last=True)

    # Compute local rotations for the motion file
    joint_rot_mats = _axis_angle_to_matrix(joint_aa)
    local_rest_rot = smpl_ki.local_rot_ref_mat
    num_bodies = len(smpl_ki.body_names)
    local_rot_mats = torch.zeros(T, num_bodies, 3, 3)
    local_rot_mats[:, 0] = root_rot_mat
    for i in range(1, num_bodies):
        local_rot_mats[:, i] = local_rest_rot[i].unsqueeze(0) @ joint_rot_mats[:, i - 1]
    local_rot_quat = matrix_to_quaternion(local_rot_mats, w_last=True)

    # dof_pos via axis-angle (exp_map) — 23 joints × 3 = 69
    dof_pos = joint_aa.reshape(T, -1)

    # Velocities via finite differences
    dt = 1.0 / fps
    gvs = torch.zeros_like(global_pos)
    gvs[1:] = (global_pos[1:] - global_pos[:-1]) / dt
    gvs[0] = gvs[1]

    gavs = torch.zeros(T, num_bodies, 3)
    ang_vel = compute_angular_velocity(local_rot_mats[:, 1:], fps=fps)
    dof_vel = ang_vel.reshape(T, -1)

    # Contact labels
    contacts = compute_contact_labels_from_pos_and_vel(
        positions=global_pos,
        velocity=gvs,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)

    return {
        "gts": global_pos,
        "grs": global_rot_quat,
        "gvs": gvs,
        "gavs": gavs,
        "dps": dof_pos,
        "dvs": dof_vel,
        "lrs": local_rot_quat,
        "contacts": contacts,
        "fps": fps,
    }


@app.command()
def main(
    input_path: Path = typer.Argument(
        ..., help="Single .motion file or directory of .motion files (SOMA23)"
    ),
    output_dir: Path = typer.Argument(
        ..., help="Output directory for SMPL .motion files"
    ),
    batch_size: int = typer.Option(
        64, "--batch-size", help="Frames per optimization batch"
    ),
    num_iters: int = typer.Option(
        500, "--num-iters", help="Optimization iterations per batch"
    ),
    device_str: str = typer.Option("cuda", "--device"),
) -> None:
    """Convert SOMA23 .motion files to SMPL format via FK optimization."""
    device = torch.device(device_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        motion_files = [input_path]
    elif input_path.is_dir():
        motion_files = sorted(input_path.glob("*.motion"))
    else:
        console.print(f"[red]ERROR[/]: {input_path} not found")
        raise typer.Exit(1)

    if not motion_files:
        console.print(f"[red]ERROR[/]: no .motion files in {input_path}")
        raise typer.Exit(1)

    console.print(
        f"Converting {len(motion_files)} motions | iters={num_iters} batch={batch_size}"
    )

    # Load SMPL kinematic info
    smpl_mjcf = (
        Path(__file__).parent.parent.parent
        / "protomotions"
        / "data"
        / "assets"
        / "mjcf"
        / "smpl_humanoid.xml"
    )
    smpl_ki = extract_kinematic_info(str(smpl_mjcf))

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
        task = progress.add_task("Converting", total=len(motion_files), status="")

        for mf in motion_files:
            name = mf.stem
            progress.update(task, status=f"[cyan]{name[:40]}[/]")
            out_path = output_dir / mf.name

            if out_path.exists():
                progress.console.print(f"  [dim]SKIP[/] {name} (exists)")
                progress.advance(task)
                successes += 1
                continue

            try:
                out_dict = convert_motion(mf, smpl_ki, device, batch_size, num_iters)
                torch.save(out_dict, out_path)
                successes += 1
                progress.console.print(f"  [green]OK[/] {name}")
            except Exception as e:
                failures.append((name, str(e)))
                progress.console.print(f"  [red]FAIL[/] {name}: {e}")

            progress.advance(task)

    console.print(f"\n[green]Done[/]: {successes}/{len(motion_files)} converted")
    if failures:
        console.print(f"[red]Failures ({len(failures)}):[/]")
        for name, err in failures:
            console.print(f"  {name}: {err}")


if __name__ == "__main__":
    app()
