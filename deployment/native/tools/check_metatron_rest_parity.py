# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the native v5 rest template with the ProtoMotions Python FK path."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    fk_batch_mjcf_with_velocities,
)
from tools.metatron_v1_reference import _multimodal_style_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--fk-plan", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    info = extract_kinematic_info(str(args.xml))
    assert list(info.body_names) == manifest["robot_contract"]["body_names"]
    joints = np.zeros(29, np.float32)
    for name, value in {
        "left_hip_pitch_joint": -0.312,
        "right_hip_pitch_joint": -0.312,
        "left_knee_joint": 0.669,
        "right_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "right_ankle_pitch_joint": -0.363,
        "left_shoulder_roll_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "left_elbow_joint": 1.4,
        "right_elbow_joint": 1.4,
    }.items():
        joints[manifest["robot_contract"]["joint_names"].index(name)] = value
    qpos = np.concatenate(([0, 0, 0, 1, 0, 0, 0], joints)).astype(np.float32)
    with torch.no_grad():
        reference = fk_batch_mjcf_with_velocities(
            info,
            torch.from_numpy(qpos[None]),
            compute_velocities=False,
        )
    pos = reference.rigid_body_pos[0].numpy().copy()
    rot = reference.rigid_body_rot[0].numpy().copy()
    pos[:, 2] -= pos[:, 2].min()
    state = {
        "body_pos": pos,
        "body_rot": rot,
        "body_vel": np.zeros_like(pos),
        "body_ang_vel": np.zeros_like(pos),
    }
    style, _ = _multimodal_style_features(
        [state] * 5,
        np.asarray([16, 13, 7, 24, 32]),
    )
    native = json.loads(
        subprocess.check_output(
            [str(args.dump), str(args.fk_plan)],
            text=True,
        )
    )
    np_pos = np.asarray(native["positions"], np.float32).reshape(33, 3)
    np_rot = np.asarray(native["rotations"], np.float32).reshape(33, 4)
    np_style = np.asarray(native["style"], np.float32)
    position_error = float(np.max(np.abs(np_pos - pos)))
    rotation_error = float(
        np.max(
            np.minimum(
                np.abs(np_rot - rot),
                np.abs(np_rot + rot),
            )
        )
    )
    style_error = float(np.max(np.abs(np_style - style)))
    print(
        f"rest parity: position={position_error:.8g} m "
        f"rotation={rotation_error:.8g} style={style_error:.8g}"
    )
    if position_error > 1e-5 or rotation_error > 1e-5 or style_error > 1e-5:
        raise SystemExit("native rest pose differs from Python reference")


if __name__ == "__main__":
    main()
