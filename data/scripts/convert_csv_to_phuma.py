#!/usr/bin/env python3
"""
Convert Unitree G1 retargeted motion CSVs (e.g., LaFAN1 retargeted) into PHUMA .npy dict format.

Expected per-frame layout (common in retargeted G1 CSV dumps):
  Option A (37 cols): [frame_idx, root_x, root_y, root_z, qx, qy, qz, qw, 29 joint angles]
  Option B (36 cols): [root_x, root_y, root_z, qx, qy, qz, qw, 29 joint angles]

Outputs .npy files shaped for ProtoMotions PHUMA pipeline:
  {
    'root_trans': (T, 3) float32,
    'root_ori':   (T, 4) float32 (xyzw),
    'dof_pos':    (T, 29) float32,
    'fps':        float
  }
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# Joint order documented for G1 retargeted datasets (29 DoFs after root pose).
# (Root pose itself is XYZ + quaternion XYZW, which is handled separately.)
G1_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

NAME_TO_DOF_INDEX = {name: i for i, name in enumerate(G1_DOF_NAMES)}

ARM_JOINTS = [
    # left arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _read_csv_numeric(csv_path: Path) -> np.ndarray:
    """
    Robust-ish CSV reader:
      - supports header or no header
      - returns float32 numpy array
    """
    try:
        df = pd.read_csv(csv_path)
        arr = df.to_numpy()
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("Non-numeric dtype (likely header-only or mixed).")
        return arr.astype(np.float32)
    except Exception:
        # Headerless (or weird) fallback
        arr = np.loadtxt(csv_path, delimiter=",", dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr


def _strip_frame_index_if_present(arr: np.ndarray) -> np.ndarray:
    """
    If we have 37 columns, the first is commonly a frame index. Remove it.
    """
    if arr.shape[1] == 37:
        # Heuristic: first column is integer-ish monotonic
        c0 = arr[:, 0]
        intish = np.all(np.abs(c0 - np.round(c0)) < 1e-3)
        if intish:
            return arr[:, 1:]
        # If not int-ish, still probably frame index, but be conservative:
        # only strip if 36 cols expected after strip.
        if arr[:, 1:].shape[1] == 36:
            return arr[:, 1:]
    return arr


def convert_one(csv_path: Path, out_path: Path, fps: float, freeze_arm_joints: bool) -> None:
    arr = _read_csv_numeric(csv_path)
    arr = _strip_frame_index_if_present(arr)

    if arr.shape[1] != 36:
        raise ValueError(
            f"{csv_path.name}: expected 36 columns after optional frame_idx removal, got {arr.shape[1]}.\n"
            "Expected: root(7) + dof_pos(29)."
        )

    root_trans = arr[:, 0:3].astype(np.float32)
    root_ori = arr[:, 3:7].astype(np.float32)   # xyzw
    dof_pos = arr[:, 7:].astype(np.float32)

    if dof_pos.shape[1] != 29:
        raise ValueError(f"{csv_path.name}: expected 29 DoFs for G1, got {dof_pos.shape[1]}")

    if freeze_arm_joints:
        arm_idx = [NAME_TO_DOF_INDEX[n] for n in ARM_JOINTS]
        dof_pos[:, arm_idx] = dof_pos[250, arm_idx]  # freeze to 250th frame (boxing pose)

    motion = {
        "root_trans": root_trans,
        "root_ori": root_ori,
        "dof_pos": dof_pos,
        "fps": float(fps),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, motion, allow_pickle=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", type=Path, required=True, help="Directory containing G1 retargeted CSV motion files.")
    ap.add_argument("--out_root", type=Path, required=True, help="Output root directory (will write out_root/g1/*.npy).")
    ap.add_argument("--fps", type=float, default=30.0, help="Frame rate (default: 30).")
    ap.add_argument("--freeze_arms", action="store_true", help="Freeze arm joints to first frame (reduces punching).")
    args = ap.parse_args()

    csv_files = sorted(args.csv_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No .csv files found in {args.csv_dir}")

    out_dir = args.out_root
    for csv_path in csv_files:
        out_path = out_dir / (csv_path.stem + ".npy")
        convert_one(csv_path, out_path, fps=args.fps, freeze_arm_joints=args.freeze_arms)
        print(f"Wrote {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
