# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert SAMP SMPL motions to SOMA23 format using SOMA-X pose inversion.

Pipeline per motion:
    1. Load SMPL .motion file (dof_pos, rigid_body_pos/rot in z-up)
    2. Reconstruct SMPL body_pose + global_orient (axis-angle)
    3. Run smplx forward pass → posed vertices [T, 6890, 3]
    4. Batch through SOMA-X PoseInversion → SOMA 78-joint rotations [T, 78, 3, 3]
    5. Subsample 78 → 23 MJCF joints
    6. Convert via create_motion_from_soma23_global_rotations (handles y-up → z-up)
    7. Transfer contact labels from SMPL motion
    8. Save as .motion file

The SMPL vertices are in SMPL's native y-up frame (smplx handles this).
SOMA-X outputs y-up rotations, which create_motion_from_soma23_global_rotations
converts to z-up.

Usage::

    python data/scripts/convert_smpl_samp_to_soma23.py \\
        ~/protomotions_assets/samp/motions/ \\
        ~/protomotions_assets/samp/motions_soma23/ \\
        --batch-size 128

    # Single file
    python data/scripts/convert_smpl_samp_to_soma23.py \\
        ~/protomotions_assets/samp/motions/armchair001_stageII.motion \\
        ~/protomotions_assets/samp/motions_soma23/ \\
        --batch-size 128
"""

from __future__ import annotations

import sys
from pathlib import Path

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
from scipy.spatial.transform import Rotation as Rot

from soma import SOMALayer
from soma.pose_inversion import PoseInversion

# Add data/scripts to path for convert_soma23_to_proto imports
sys.path.insert(0, str(Path(__file__).parent))
from convert_soma23_to_proto import (
    SOMASKEL77_TO_MJCF_INDICES,
)
from contact_detection import compute_contact_labels_from_pos_and_vel
from data.smpl.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES

from protomotions.components.pose_lib import extract_kinematic_info
from protomotions.utils.rotations import quaternion_to_matrix

# MJCF body order → AMASS body order permutation.
# smpl_2_mujoco[mj_idx] = amass_idx (used during amass→proto conversion).
# We need the inverse: for each amass_idx, which mj_idx holds that joint.
_SMPL_2_MUJOCO = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES]
_MJCF_TO_AMASS = [0] * 24
for _mj, _am in enumerate(_SMPL_2_MUJOCO):
    _MJCF_TO_AMASS[_am] = _mj

app = typer.Typer(pretty_exceptions_enable=False)
console = Console()

# SOMA-X uses 78 joints: Root (index 0) + 77 SOMASKEL77 joints (1-77).
# To get SOMA23 MJCF bodies, offset SOMASKEL77_TO_MJCF_INDICES by +1.
SOMAX78_TO_MJCF_INDICES = [i + 1 for i in SOMASKEL77_TO_MJCF_INDICES]

# SMPL pkl path (chumpy-free version cached by SOMA-X setup)
SMPL_PKL = Path.home() / (
    ".cache/huggingface/hub/models--nvidia--soma-x/snapshots/"
    "6db1b4e8d737db240e680ca5358ae2897f2d9427/SMPL/SMPL_NEUTRAL.pkl"
)

# z-up ↔ y-up rotation: rotate -90° about X
_R_ZUP_TO_YUP = torch.tensor(
    Rot.from_euler("x", -90, degrees=True).as_matrix(), dtype=torch.float32
)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T


def _quat_to_rotvec(q_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (xyzw) to axis-angle (rotvec). Shape [..., 4] → [..., 3]."""
    # scipy expects wxyz
    q_np = q_xyzw.cpu().numpy()
    shape = q_np.shape[:-1]
    q_flat = q_np.reshape(-1, 4)
    rv = Rot.from_quat(q_flat).as_rotvec()  # scipy uses xyzw
    return torch.from_numpy(rv.reshape(*shape, 3)).float()


def convert_motion(
    smpl_motion_path: Path,
    soma: SOMALayer,
    inv: PoseInversion,
    smpl_model: smplx.SMPL,
    kinematic_info,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Convert a single SMPL .motion file to SOMA23 .motion dict."""
    motion_data = torch.load(smpl_motion_path, weights_only=False, map_location="cpu")
    local_rots = motion_data[
        "local_rigid_body_rot"
    ]  # [T, 24, 4] xyzw quats, MJCF order
    rigid_body_pos = motion_data["rigid_body_pos"]  # [T, 24, 3] z-up
    fps = motion_data.get("fps", 30)
    T = local_rots.shape[0]

    # --- Step 1: Reconstruct SMPL parameters from local_rigid_body_rot ---
    # Reorder from MJCF body order → AMASS body order (what smplx expects)
    local_rots_amass = local_rots[:, _MJCF_TO_AMASS]  # [T, 24, 4] xyzw, AMASS order

    # Convert quaternions to axis-angle
    local_aa = (
        torch.from_numpy(
            Rot.from_quat(local_rots_amass.reshape(-1, 4).numpy()).as_rotvec()
        )
        .float()
        .reshape(T, 24, 3)
    )

    # Root orientation (body 0): local_rots stores the global rotation (z-up).
    # Convert to y-up for smplx.
    root_mat_zup = quaternion_to_matrix(local_rots_amass[:, 0], w_last=True)
    root_mat_yup = (
        _R_ZUP_TO_YUP.unsqueeze(0) @ root_mat_zup @ _R_ZUP_TO_YUP.T.unsqueeze(0)
    )
    global_orient = torch.from_numpy(
        Rot.from_matrix(root_mat_yup.numpy()).as_rotvec()
    ).float()  # [T, 3]

    # Body pose (joints 1-23): local rotations are parent-relative, frame-invariant
    body_pose = local_aa[:, 1:].reshape(T, 69)  # [T, 69]

    # Root translation: convert z-up → y-up
    root_pos_zup = rigid_body_pos[:, 0]  # [T, 3]
    root_pos_yup = root_pos_zup @ _R_ZUP_TO_YUP.T  # [T, 3]

    # --- Step 2: Batch smplx forward → vertices ---
    all_rotations = []
    all_root_trans = []

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        bp = body_pose[start:end].to(device)
        go = global_orient[start:end].to(device)
        tr = root_pos_yup[start:end].to(device)

        with torch.no_grad():
            smpl_out = smpl_model(
                body_pose=bp,
                global_orient=go,
                transl=tr,
            )
            verts = smpl_out.vertices  # [B, 6890, 3]

            # --- Step 3: SOMA-X pose inversion ---
            result = inv.fit(verts)
            # result["rotations"]: [B, 78, 3, 3] absolute rotations (y-up)
            # result["root_translation"]: [B, 3]
            all_rotations.append(result["rotations"].cpu())
            all_root_trans.append(result["root_translation"].cpu())

    soma_rotations = torch.cat(all_rotations, dim=0)  # [T, 78, 3, 3]
    soma_root_trans = torch.cat(all_root_trans, dim=0)  # [T, 3]

    # --- Step 4+5+6: Use SOMA-X globals directly, rotate y-up→z-up, derive locals via MJCF ---
    # SOMA-X gives global (absolute) rotations in y-up. The SOMA23 MJCF is z-up.
    # Strategy: subsample globals to 23 joints, rotate to z-up, then use MJCF
    # kinematic_info to extract local rotations and run FK for positions.
    from protomotions.components.pose_lib import (
        fk_from_transforms_with_velocities,
        extract_qpos_from_transforms,
        compute_angular_velocity,
        compute_joint_rot_mats_from_global_mats,
    )
    from protomotions.utils.rotations import (
        matrix_to_quaternion,
    )

    # Subsample 78 → 23 globals (still in y-up, pose-only ≈ identity at T-pose)
    soma23_globals = soma_rotations[:, SOMAX78_TO_MJCF_INDICES]  # [T, 23, 3, 3]

    # SOMA-X globals are pose-only (T-pose ≈ identity) in y-up.
    # The MJCF body orientations are identity (z-up T-pose).
    # Extracting locals from these globals gives pose-deviation rotations,
    # which the z-up FK applies on top of the z-up skeleton → correct z-up result.
    # Only the root position needs y-up → z-up conversion.
    R_y2z = torch.tensor(
        Rot.from_euler("x", 90, degrees=True).as_matrix(), dtype=torch.float32
    )
    root_pos_zup = soma_root_trans @ R_y2z.T  # [T, 3]

    # Extract local rotations directly (no rotation of globals needed)
    local_rot_mats = compute_joint_rot_mats_from_global_mats(
        kinematic_info=kinematic_info,
        global_rot_mats=soma23_globals,
    )

    # FK for positions and velocities
    motion = fk_from_transforms_with_velocities(
        kinematic_info=kinematic_info,
        root_pos=root_pos_zup,
        joint_rot_mats=local_rot_mats,
        fps=fps,
        compute_velocities=True,
        velocity_max_horizon=3,
    )
    motion.local_rigid_body_rot = matrix_to_quaternion(local_rot_mats, w_last=True)
    num_joints = 22
    qpos = extract_qpos_from_transforms(
        kinematic_info=kinematic_info,
        root_pos=root_pos_zup,
        joint_rot_mats=local_rot_mats,
        multi_dof_decomposition_method="exp_map",
    )
    motion.dof_pos = qpos[:, 7:]
    local_ang_vel = compute_angular_velocity(
        batched_robot_rot_mats=local_rot_mats[:, 1:], fps=fps
    )
    motion.dof_vel = local_ang_vel.reshape(-1, num_joints * 3)
    motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
        positions=motion.rigid_body_pos,
        velocity=motion.rigid_body_vel,
        vel_thres=0.15,
        height_thresh=0.1,
    ).to(torch.bool)

    # Build output dict
    out = motion.to_dict()
    out["fps"] = fps
    return out


@app.command()
def main(
    input_path: Path = typer.Argument(
        ..., help="Single .motion file or directory of .motion files"
    ),
    output_dir: Path = typer.Argument(
        ..., help="Output directory for SOMA23 .motion files"
    ),
    batch_size: int = typer.Option(128, "--batch-size", help="Frames per GPU batch"),
    device_str: str = typer.Option("cuda", "--device"),
) -> None:
    """Convert SMPL SAMP motions to SOMA23 via SOMA-X pose inversion."""
    device = torch.device(device_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    if input_path.is_file():
        motion_files = [input_path]
    elif input_path.is_dir():
        motion_files = sorted(input_path.glob("*.motion"))
    else:
        console.print(f"[red]ERROR[/]: {input_path} not found")
        raise typer.Exit(1)

    if not motion_files:
        console.print(f"[red]ERROR[/]: no .motion files found in {input_path}")
        raise typer.Exit(1)

    console.print(f"Converting {len(motion_files)} motions | batch_size={batch_size}")

    # --- Initialize models ---
    console.print("Loading SMPL model...")
    smpl_model = smplx.create(str(SMPL_PKL), model_type="smpl").to(device)

    console.print("Loading SOMA-X...")
    soma = SOMALayer(
        data_root=None,
        identity_model_type="smpl",
        device=device_str,
        mode="warp",
    )
    inv = PoseInversion(soma, low_load=True)
    inv.prepare_identity(torch.zeros(1, 10).to(device))

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

    # --- Convert ---
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
                out_dict = convert_motion(
                    mf, soma, inv, smpl_model, kinematic_info, batch_size, device
                )
                torch.save(out_dict, out_path)
                successes += 1
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
