# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retarget a SOMA23 packed MotionLib to anatomic SOMA without changing timing.

Use ProtoMotions Python for this driver and --pyroki-python for the existing
PyRoki environment. Whole clips are optimized; none are trimmed or subsampled.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from protomotions.components.pose_lib import (  # noqa: E402
    compute_cartesian_velocity,
    extract_kinematic_info,
    fk_batch_mjcf_with_velocities,
)
from data.scripts.from_soma_to_anatomical_soma.soma23_to_anatomical_soma_retargeting import (  # noqa: E402
    CANONICAL_FRAME,
    SOMA_TO_CANONICAL,
    canonical_soma_poses,
    SOMA_POSITION_WEIGHTS,
    SOMA_ORIENTATION_WEIGHTS,
    SOMA_BONE_PAIRS,
    hinge_seed,
    quality_metrics,
    save_verified,
    semantic_names,
)


@contextmanager
def solver_worker(command):
    """One JAX process per pack, not one startup and compilation per motion."""
    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:

        def run(input_path, output_path):
            request = {"input": str(input_path), "output": str(output_path)}
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            response = process.stdout.readline()
            if not response:
                raise RuntimeError(
                    f"PyRoki worker exited with code {process.wait()}; see its error above"
                )
            parsed = json.loads(response)
            if "error" in parsed:
                raise ValueError(parsed["error"])
            if parsed != {"output": str(output_path)}:
                raise RuntimeError(f"Unexpected worker response: {response}")

        yield run
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()


def retarget(
    source_path,
    output_path,
    pyroki_python,
    clip_ids=None,
    iterations=500,
    cache_dir=None,
    max_joint_velocity=20.0,
    max_joint_acceleration=300.0,
    resume_dir=None,
    allow_motion_budget_exceedance=False,
    frame_bucket_size=0,
):
    if not isinstance(frame_bucket_size, int) or frame_bucket_size < 0:
        raise ValueError("Frame bucket size must be a nonnegative integer")
    if output_path.exists() or source_path.resolve() == output_path.resolve():
        raise ValueError("Refusing to overwrite an existing output or the source")
    asset = ROOT / "protomotions/data/assets/mjcf/anatomic_soma.xml"
    source_asset = ROOT / "protomotions/data/assets/mjcf/soma23_humanoid.xml"
    kin, source_kin = (
        extract_kinematic_info(str(asset)),
        extract_kinematic_info(str(source_asset)),
    )
    names = semantic_names(asset)
    if source_kin.body_names != names:
        raise ValueError("Source semantic body order differs from SOMA23")
    data = torch.load(source_path, weights_only=True, mmap=True, map_location="cpu")
    known = {
        "gts",
        "grs",
        "gvs",
        "gavs",
        "dps",
        "dvs",
        "contacts",
        "lrs",
        "motion_num_frames",
        "motion_dt",
        "motion_lengths",
        "motion_weights",
        "motion_files",
        "length_starts",
        "goal_states",
    }
    if set(data) - known:
        raise ValueError(
            f"Unhandled source metadata fields: {sorted(set(data) - known)}"
        )
    ids = list(range(len(data["motion_files"]))) if clip_ids is None else clip_ids
    if (
        not ids
        or len(set(ids)) != len(ids)
        or min(ids) < 0
        or max(ids) >= len(data["motion_files"])
    ):
        raise ValueError("Invalid or duplicate clip indices")
    if data["gts"].shape[1:] != (len(names), 3) or data["grs"].shape[1:] != (
        len(names),
        4,
    ):
        raise ValueError("Expected SOMA23 world body positions/quaternions")
    expected_starts = torch.cat(
        (torch.zeros(1, dtype=torch.long), data["motion_num_frames"].cumsum(0)[:-1])
    )
    if not torch.equal(data["length_starts"], expected_starts) or int(
        data["motion_num_frames"].sum()
    ) != len(data["gts"]):
        raise ValueError("Invalid source clip partition")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_suffix(".audit.jsonl")
    if report_path.exists() and resume_dir is None:
        raise ValueError("Audit file already exists; use a fresh output name")
    if resume_dir is not None:
        resume_dir.mkdir(parents=True, exist_ok=True)
        contract = {
            "source": str(source_path.resolve()),
            "iterations": iterations,
            "frame_bucket_size": frame_bucket_size,
            "allow_motion_budget_exceedance": allow_motion_budget_exceedance,
            "max_joint_velocity": max_joint_velocity,
            "max_joint_acceleration": max_joint_acceleration,
            "code_hashes": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (
                    asset,
                    Path(__file__),
                    ROOT / "pyroki/from_soma_to_anatomical_soma/solve_trajectory.py",
                    ROOT / "pyroki/from_soma_to_anatomical_soma/analytic_kinematics.py",
                    ROOT
                    / "pyroki/from_soma_to_anatomical_soma/banded_trajectory_solver.py",
                    ROOT
                    / "data/scripts/from_soma_to_anatomical_soma/soma23_to_anatomical_soma_retargeting.py",
                )
            },
        }
        marker = resume_dir / "contract.json"
        if marker.exists() and json.loads(marker.read_text()) != contract:
            raise ValueError("Resume contract changed; use a fresh resume directory")
        marker.write_text(json.dumps(contract, indent=2) + "\n")
    fields = {
        key: [] for key in ("gts", "grs", "gvs", "gavs", "dps", "dvs", "contacts")
    }
    semantic_ids = [kin.body_names.index(n) for n in names]
    frames_kept = []
    command = [
        pyroki_python,
        str(ROOT / "pyroki/from_soma_to_anatomical_soma/solve_trajectory.py"),
        "--asset",
        str(asset),
        "--serve",
        "--iterations",
        str(iterations),
        "--max-joint-velocity",
        str(max_joint_velocity),
        "--max-joint-acceleration",
        str(max_joint_acceleration),
        "--frame-bucket-size",
        str(frame_bucket_size),
    ]
    if cache_dir is not None:
        command.extend(["--cache-dir", str(cache_dir)])
    if allow_motion_budget_exceedance:
        command.append("--allow-motion-budget-exceedance")
    failures = []
    worker = solver_worker(command)
    with report_path.open(
        "w" if resume_dir is not None else "x"
    ) as report, tempfile.TemporaryDirectory(
        prefix="soma-retarget-"
    ) as temporary, worker as solve:
        folder = Path(temporary)
        for index in ids:
            start, count = (
                int(data["length_starts"][index]),
                int(data["motion_num_frames"][index]),
            )
            dt = float(data["motion_dt"][index])
            if count < 1 or not np.isfinite(dt) or dt <= 0:
                raise ValueError(f"Invalid clip metadata: {index}")
            p = data["gts"][start : start + count].numpy().astype(float)
            r = data["grs"][start : start + count].numpy().astype(float)
            if (
                not np.isfinite(p).all()
                or not np.isfinite(r).all()
                or not np.allclose(np.linalg.norm(r, axis=-1), 1, atol=1e-4)
            ):
                raise ValueError(f"Invalid source pose: {index}")
            p, r = canonical_soma_poses(p, r)
            contact = (
                data["contacts"][start : start + count].numpy()
                if data.get("contacts") is not None
                else np.zeros((count, len(names)), dtype=bool)
            )
            seed, dof_names = hinge_seed(asset, r)
            if dof_names != kin.dof_names:
                raise ValueError("Seed and asset DOF order differ")
            digest = hashlib.sha256()
            for value in (p, r, contact, seed, np.asarray(dt)):
                digest.update(str((value.shape, value.dtype)).encode())
                digest.update(value.tobytes())
            input_hash = digest.hexdigest()
            cached = None if resume_dir is None else resume_dir / f"{index:08d}.npz"
            solution_path = (
                cached
                if cached is not None and cached.exists()
                else folder / "output.npz"
            )
            if solution_path != cached:
                np.savez(
                    folder / "input.npz",
                    positions=p,
                    rotations=r,
                    contacts=contact,
                    seed=seed,
                    body_names=names,
                    dof_names=dof_names,
                    dt=dt,
                )
                try:
                    solve(folder / "input.npz", solution_path)
                except ValueError as exc:
                    failures.append(index)
                    report.write(
                        json.dumps(
                            {
                                "source_index": index,
                                "motion_name": data["motion_files"][index],
                                "status": "failed",
                                "error": str(exc),
                            }
                        )
                        + "\n"
                    )
                    report.flush()
                    print(f"Rejected source clip {index}: {exc}", flush=True)
                    continue
            with np.load(solution_path, allow_pickle=False) as solution:
                q = solution["dof_pos"]
                diagnostics = json.loads(str(solution["diagnostics"]))
                if (
                    solution_path == cached
                    and str(solution["input_sha256"]) != input_hash
                ):
                    raise ValueError(f"Cached source changed for clip {index}")
            if q.shape != seed.shape or not np.isfinite(q).all():
                raise ValueError(f"Invalid solver output: {index}")

            def state_for(coordinates):
                qpos = torch.tensor(
                    np.concatenate(
                        (p[:, 0], r[:, 0, [3, 0, 1, 2]], coordinates), axis=1
                    ),
                    dtype=torch.float32,
                )
                return fk_batch_mjcf_with_velocities(
                    kin, qpos, fps=1 / dt, velocity_max_horizon=1
                )

            state = state_for(q)
            seed_state = state_for(seed)
            metrics = quality_metrics(
                p,
                r,
                state.rigid_body_pos[:, semantic_ids].numpy(),
                state.rigid_body_rot[:, semantic_ids].numpy(),
                q,
                names,
            )
            row = {
                "source_index": index,
                "motion_name": data["motion_files"][index],
                "frames": count,
                "dt": dt,
                "result": metrics,
                "solver": diagnostics,
                "seed": quality_metrics(
                    p,
                    r,
                    seed_state.rigid_body_pos[:, semantic_ids].numpy(),
                    seed_state.rigid_body_rot[:, semantic_ids].numpy(),
                    seed,
                    names,
                ),
            }
            report.write(json.dumps(row, allow_nan=False) + "\n")
            report.flush()
            if diagnostics.get("motion_budget_exceeded"):
                print(
                    f"WARNING: source clip {index} exceeds motion budgets; recovery exception enabled",
                    flush=True,
                )
            if metrics["steps_over_pi"]:
                raise ValueError(
                    f"Clip {index} retains a >pi coordinate jump; inspect {report_path}"
                )
            if cached is not None and not cached.exists():
                partial_cache = cached.with_suffix(".partial.npz")
                np.savez(
                    partial_cache,
                    dof_pos=q,
                    diagnostics=json.dumps(diagnostics),
                    input_sha256=input_hash,
                )
                partial_cache.rename(cached)
            fields["gts"].append(state.rigid_body_pos)
            fields["grs"].append(state.rigid_body_rot)
            fields["gvs"].append(
                state.rigid_body_vel
                if count > 1
                else torch.zeros_like(state.rigid_body_pos)
            )
            fields["gavs"].append(
                state.rigid_body_ang_vel
                if count > 1
                else torch.zeros_like(state.rigid_body_pos)
            )
            fields["dps"].append(torch.tensor(q, dtype=torch.float32))
            fields["dvs"].append(
                compute_cartesian_velocity(
                    fields["dps"][-1], 1 / dt, velocity_max_horizon=1
                )
                if count > 1
                else torch.zeros_like(fields["dps"][-1])
            )
            mapped_contacts = torch.zeros(
                (count, kin.num_bodies), dtype=torch.as_tensor(contact).dtype
            )
            mapped_contacts[:, semantic_ids] = torch.as_tensor(contact)
            fields["contacts"].append(mapped_contacts)
            frames_kept.append(torch.arange(start, start + count))
            print(
                f"Completed source clip {index}: {count} frames, max position error {metrics['position_max_m']:.4f} m",
                flush=True,
            )
    if failures:
        raise ValueError(
            f"Rejected {len(failures)} clips; no incomplete pack published. See {report_path}. Successful solves remain in --resume-dir when supplied."
        )
    saved = {key: torch.cat(value) for key, value in fields.items()}
    for key in ("motion_num_frames", "motion_dt", "motion_lengths", "motion_weights"):
        saved[key] = data[key][ids].clone()
    saved["motion_files"] = tuple(data["motion_files"][i] for i in ids)
    saved["length_starts"] = torch.cat(
        (torch.zeros(1, dtype=torch.long), saved["motion_num_frames"].cumsum(0)[:-1])
    )
    saved["lrs"] = None  # Scalar hinges must not enter MotionLib's exp-map path.
    if data.get("contacts") is None:
        saved["contacts"] = None
    if data.get("goal_states") is not None:
        saved["goal_states"] = data["goal_states"][torch.cat(frames_kept)].clone()
    save_verified(
        saved,
        output_path,
        lambda reloaded: reloaded["motion_files"] == saved["motion_files"]
        and torch.equal(reloaded["dps"], saved["dps"]),
    )
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "schema": "protomotions/soma-retarget/v1",
                "source": str(source_path),
                "asset_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "iterations": iterations,
                "clip_indices": ids,
                "frames": len(saved["gts"]),
                "frame_bucket_size": frame_bucket_size,
                "backend": "pyroki-structured-cr",
                "precision": "float32",
                "analytic_fk": True,
                "coordinate_system": CANONICAL_FRAME,
                "source_to_target_basis": SOMA_TO_CANONICAL.tolist(),
                "root": "fixed_basis_rotation",
                "timing": "unchanged",
                "output_clamping": False,
                "position_weight": 50.0,
                "rotation_weight": 4.0,
                "velocity_weight": 0.01,
                "body_position_weights": SOMA_POSITION_WEIGHTS,
                "body_orientation_weights": SOMA_ORIENTATION_WEIGHTS,
                "bone_pairs": SOMA_BONE_PAIRS,
                "bone_direction_weight": 1.0,
                "contact_displacement_weight": 30.0,
                "acceleration_weight": 0.002,
                "contact_position_multiplier": 5.0,
                "max_joint_velocity_rad_s": max_joint_velocity,
                "max_joint_acceleration_rad_s2": max_joint_acceleration,
                "motion_limit_weight": 100.0,
                "motion_limit_penalty_start_fraction": 0.9,
                "allow_motion_budget_exceedance": allow_motion_budget_exceedance,
                "initial_lm_damping": 1.0,
                "maximum_lm_damping": 1e12,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pyroki-python", required=True)
    parser.add_argument(
        "--clip-ids",
        type=int,
        nargs="+",
        help="Explicit pilot selection; otherwise all clips",
    )
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument(
        "--max-joint-velocity",
        type=float,
        default=20.0,
        help="Retargeting budget in rad/s, not a hardware rating",
    )
    parser.add_argument(
        "--max-joint-acceleration",
        type=float,
        default=300.0,
        help="Retargeting budget in rad/s^2",
    )
    parser.add_argument(
        "--cache-dir", type=Path, help="Persistent JAX compilation cache"
    )
    parser.add_argument(
        "--frame-bucket-size",
        type=int,
        default=0,
        help="Pad clips to a multiple of this many frames so JAX compiles fewer "
        "shapes (the released corpus used 128); 0 keeps exact clip lengths",
    )
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="Persist verified per-clip solves; reject code/config/source changes",
    )
    parser.add_argument(
        "--allow-motion-budget-exceedance",
        action="store_true",
        help="Recovery-only rate-budget exception; retain convergence, finite-value and joint-bound checks",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    retarget(
        args.input,
        args.output,
        args.pyroki_python,
        args.clip_ids,
        args.iterations,
        args.cache_dir,
        args.max_joint_velocity,
        args.max_joint_acceleration,
        args.resume_dir,
        args.allow_motion_budget_exceedance,
        args.frame_bucket_size,
    )
