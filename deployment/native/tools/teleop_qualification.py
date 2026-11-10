# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the native launcher's scripted teleop scenario under a disturbance grid.

Each cell launches ``metatron_v1_deploy_cpp`` on MuJoCo with the same operator
script and a different (friction, added hand mass, push) setting, records the
metrics stream to a file and scores it. Scores come from the record alone, so a
cell can be re-scored without re-running:

* upright: torso height above 0.5 m and tilt below 34 degrees for the final 2 s;
* hand error at hold: mean and 95th percentile of the per-hand goal error over
  the last 1.5 s of each hand segment;
* settle time: first time the error stays under 5 cm after a goal change;
* planted-foot slip: maximum support slip while hands were active;
* recovery after a push: time for the hand error to return under 5 cm.

    python tools/teleop_qualification.py --launcher build-clang/metatron_v1_deploy_cpp \
        --artifacts artifacts/metatron-ep6410 --out qualification/
"""
from __future__ import annotations

import argparse
import csv
import concurrent.futures
import itertools
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

HOLD_WINDOW_S = 1.5
SETTLE_M = 0.05


def load(path: Path) -> list[dict]:
    frames = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                frames.append(json.loads(line))
    return frames


def score(frames: list[dict], script: dict, pushes: list[dict]) -> dict:
    if not frames:
        return {"frames": 0, "upright": False}
    t0 = frames[0]["t_ns"] * 1e-9
    times = [f["t_ns"] * 1e-9 - t0 for f in frames]
    end = times[-1]
    final = [f for f, t in zip(frames, times) if t >= end - 2.0]
    upright = all((f.get("pose/torso_z") or 0) > 0.5 and (f.get("pose/tilt_deg") or 90) < 34 for f in final)
    result = {"frames": len(frames), "duration_s": round(end, 2), "upright": upright,
              "min_torso_z": round(min((f.get("pose/torso_z") or 0) for f in frames), 3),
              "max_tilt_deg": round(max((f.get("pose/tilt_deg") or 0) for f in frames), 1)}
    hand_segments = [s for s in script["segments"] if s.get("mode") == "teleop"]
    holds, settles = [], []
    slip = 0.0
    for segment in hand_segments:
        inside = [(f, t) for f, t in zip(frames, times) if segment["start"] <= t < segment["end"]]
        if not inside:
            continue
        for key in ("task/hand_error_left_m", "task/hand_error_right_m"):
            errors = [(t, f.get(key)) for f, t in inside if f.get(key) is not None]
            if not errors:
                continue
            tail = [e for t, e in errors if t >= segment["end"] - HOLD_WINDOW_S]
            if tail:
                holds.append(statistics.fmean(tail))
            settled = None
            for index, (t, e) in enumerate(errors):
                if e < SETTLE_M and all(e2 < SETTLE_M for _, e2 in errors[index:]):
                    settled = t - segment["start"]
                    break
            settles.append(settled if settled is not None else math.inf)
        slip = max([slip] + [f.get("task/support_slip_m") or 0 for f, _ in inside])
    if holds:
        holds.sort()
        result["hold_error_mean_cm"] = round(100 * statistics.fmean(holds), 1)
        result["hold_error_p95_cm"] = round(100 * holds[min(len(holds) - 1, int(0.95 * len(holds)))], 1)
    finite = [s for s in settles if math.isfinite(s)]
    result["settled_hands"] = f"{len(finite)}/{len(settles)}"
    if finite:
        result["settle_time_median_s"] = round(statistics.median(finite), 2)
    result["support_slip_max_cm"] = round(100 * slip, 1)
    recoveries = []
    for push in pushes:
        after = [(t, f) for f, t in zip(frames, times) if t >= push["time"]]
        recovered = None
        for t, f in after:
            errs = [f.get(k) for k in ("task/hand_error_left_m", "task/hand_error_right_m") if f.get(k) is not None]
            if errs and max(errs) < SETTLE_M and t > push["time"] + push.get("duration", 0.1):
                recovered = t - push["time"]
                break
        recoveries.append(recovered if recovered is not None else math.inf)
    if recoveries:
        result["push_recovery_s"] = [None if math.isinf(r) else round(r, 2) for r in recoveries]
    if "ik/offset_left_rad" in frames[-1]:
        result["ik_offset_max_rad"] = round(max(max(f.get("ik/offset_left_rad") or 0, f.get("ik/offset_right_rad") or 0) for f in frames), 4)
    result["energy_j"] = round(frames[-1].get("power/energy_j") or 0, 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--script", default=None, help="operator script; default artifacts/teleop_reach_script.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frictions", type=float, nargs="+", default=[0.6, 0.9, 1.2])
    parser.add_argument("--hand-masses", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--push-forces", type=float, nargs="+", default=[0.0, 50.0, 100.0, 200.0])
    parser.add_argument("--push-time", type=float, default=8.0, help="seconds into the script, during the first hold")
    parser.add_argument("--push-directions", type=int, default=4)
    parser.add_argument("--steps", type=int, default=950)
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--parallel", type=int, default=1, help="launchers to run at once (each is real-time 50 Hz)")
    args = parser.parse_args()
    artifacts = Path(args.artifacts)
    script_path = Path(args.script) if args.script else artifacts / "teleop_reach_script.json"
    script = json.loads(script_path.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = []
    for friction, mass, force in itertools.product(args.frictions, args.hand_masses, args.push_forces):
        directions = range(args.push_directions) if force > 0 else [None]
        for direction in directions:
            cells.append((friction, mass, force, direction))

    def run_cell(cell):
        friction, mass, force, direction = cell
        if True:
            name = f"mu{friction:g}_m{mass:g}_F{force:g}" + (f"_d{direction}" if direction is not None else "")
            metrics = out / f"{name}.jsonl"
            pushes = []
            if force > 0:
                angle = 2 * math.pi * direction / args.push_directions
                pushes = [{"time": args.push_time, "force": [force * math.cos(angle), force * math.sin(angle), 0.0], "duration": 0.1}]
            if not args.score_only or not metrics.exists():
                command = [args.launcher, "--backend", "mujoco", "--xml", str(artifacts / "g1_holo_compat_sim.xml"),
                           "--engine", str(artifacts / "model.graphsafe.fp32.plan"), "--fk-engine", str(artifacts / "fk.fp32.plan"),
                           "--manifest", str(artifacts / "manifest.json"), "--recovery-plan", str(artifacts / "recovery_plan.json"),
                           "--teleop-script", str(script_path), "--metrics-file", str(metrics), "--steps", str(args.steps)]
                if friction > 0:
                    command += ["--sim-friction", str(friction)]
                if mass > 0:
                    command += ["--sim-hand-mass", str(mass)]
                for push in pushes:
                    fx, fy, fz = push["force"]
                    command += ["--sim-push", f"{push['time']},{fx},{fy},{fz},{push['duration']}"]
                log = out / f"{name}.log"
                with log.open("w") as handle:
                    code = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT).returncode
                if code != 0:
                    print(f"{name}: launcher exit {code} (see {log})", file=sys.stderr)
            row = {"name": name, "friction": friction, "hand_mass_kg": mass, "push_N": force, "push_dir": direction}
            if metrics.exists():
                row.update(score(load(metrics), script, pushes))
            print(json.dumps(row), flush=True)
            return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        rows = list(pool.map(run_cell, cells))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("name", "friction", "hand_mass_kg", "push_N", "push_dir"), k))
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()})
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out / 'summary.csv'} ({len(rows)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
