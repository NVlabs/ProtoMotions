# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic reference motions for the 4-body sensor chain (``fake3_mimic``).

A real ``MotionLib`` only loads from files, so :func:`build_chain_motion_lib`
writes a few deterministic ``.motion`` clips (the ``torch.save`` dict that
``RobotState.from_dict`` reads) into a temporary directory, loads them through
a YAML descriptor (fixed clip order, unlike a directory scan) and deletes the
files again. Nothing is committed: the clips are a pure function of the clip
index, so both sides of a parity run build the same library.

Each clip is about one second at 50 Hz: a slowly translating, tilted root and
smooth sinusoidal joint angles with a per-clip frequency and phase. Body poses
and velocities come from MuJoCo's forward kinematics on the chain MJCF, with
the same body-origin velocity convention as the fake simulator.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict

import mujoco
import numpy as np
import torch
import yaml

from protomotions.tests.support.sensor_simulator import CHAIN_XML

FPS = 50.0
NUM_FRAMES = 50
NUM_BODIES = 4


def chain_clip(clip: int, num_frames: int = NUM_FRAMES, fps: float = FPS) -> Dict:
    """One clip as the dict a ``.motion`` file holds."""
    model = mujoco.MjModel.from_xml_path(str(CHAIN_XML))
    data = mujoco.MjData(model)
    times = np.arange(num_frames) / fps

    root_start = np.array([0.0, 0.5 * clip, 0.9])
    root_vel = np.array([0.3, 0.1 * (clip + 1), 0.0])
    quat_wxyz = np.array([0.95, 0.1 * clip, 0.2, 0.15])
    quat_wxyz = quat_wxyz / np.linalg.norm(quat_wxyz)
    omega = 2.0 * np.pi * (0.5 + 0.25 * clip)
    amplitude = np.array([0.6, 0.8])
    phase = np.array([0.0, 0.5 * np.pi]) + 0.3 * clip
    dof_pos = amplitude * np.sin(omega * times[:, None] + phase)
    dof_vel = amplitude * omega * np.cos(omega * times[:, None] + phase)

    body_pos = np.zeros((num_frames, NUM_BODIES, 3))
    body_rot = np.zeros((num_frames, NUM_BODIES, 4))
    body_vel = np.zeros((num_frames, NUM_BODIES, 3))
    body_ang_vel = np.zeros((num_frames, NUM_BODIES, 3))
    velocity = np.zeros(6)
    for frame, t in enumerate(times):
        data.qpos[:3] = root_start + root_vel * t
        data.qpos[3:7] = quat_wxyz
        data.qpos[7:] = dof_pos[frame]
        data.qvel[:3] = root_vel
        data.qvel[3:6] = 0.0
        data.qvel[6:] = dof_vel[frame]
        mujoco.mj_forward(model, data)
        for body in range(NUM_BODIES):
            mj_id = body + 1  # the world body is not part of the robot state
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, mj_id, velocity, 0
            )
            body_pos[frame, body] = data.xpos[mj_id]
            body_rot[frame, body] = data.xquat[mj_id][[1, 2, 3, 0]]  # xyzw
            body_ang_vel[frame, body] = velocity[:3]
            # mj_objectVelocity reports the COM velocity; common state is at
            # the body origin (same correction as the fake simulator).
            body_vel[frame, body] = velocity[3:] + np.cross(
                velocity[:3], data.xpos[mj_id] - data.xipos[mj_id]
            )

    def as_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(array, dtype=torch.float32)

    return {
        "fps": fps,
        "rigid_body_pos": as_tensor(body_pos),
        "rigid_body_rot": as_tensor(body_rot),
        "rigid_body_vel": as_tensor(body_vel),
        "rigid_body_ang_vel": as_tensor(body_ang_vel),
        "dof_pos": as_tensor(dof_pos),
        "dof_vel": as_tensor(dof_vel),
    }


def build_chain_motion_lib(num_clips: int, device="cpu"):
    """A real ``MotionLib`` holding ``num_clips`` synthetic chain clips."""
    from protomotions.components.motion_lib import MotionLib, MotionLibConfig

    with tempfile.TemporaryDirectory(prefix="parity-chain-motions-") as root_dir:
        root = Path(root_dir)
        entries = []
        for clip in range(num_clips):
            name = f"chain_{clip}.motion"
            torch.save(chain_clip(clip), root / name)
            entries.append({"file": name, "weight": 1.0})
        descriptor = root / "chain_motions.yaml"
        descriptor.write_text(yaml.safe_dump({"motions": entries}))
        return MotionLib(MotionLibConfig(motion_file=str(descriptor)), device=device)
