# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional steering-only style contract, shared with the native implementation."""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def crawling_preset() -> tuple[np.ndarray, np.ndarray]:
    path = Path(__file__).resolve().parents[1] / "assets/styles/crawling.json"
    data = json.loads(path.read_text())
    if (
        data["version"] != 2
        or data["anchorBodyIndex"] != 16
        or data["conditionableBodyIds"] != [16, 13, 7, 24, 32]
        or data["styleFrameCount"] != 5
        or data["styleMaskWidth"] != 11
    ):
        raise ValueError("unsupported crawling style contract")
    preset = data["presets"]["crawling"]
    if (
        preset["recoveryEnabled"]
        or preset["keepsTaskFootTargets"]
        or preset["controlsHeading"]
        or not preset["suppressesTaskHeading"]
    ):
        raise ValueError("unsupported crawling style task semantics")
    pose = np.asarray(preset["poseSamples"], dtype=np.float32)
    velocity = np.asarray(preset["velocitySamples"], dtype=np.float32)
    if (
        pose.shape != (225,)
        or velocity.shape != (150,)
        or not np.isfinite(pose).all()
        or not np.isfinite(velocity).all()
    ):
        raise ValueError("invalid crawling style tensors")
    pose.setflags(write=False)
    velocity.setflags(write=False)
    return pose, velocity


def apply_crawling_style(raw: dict[str, np.ndarray], heading: float) -> None:
    """Keep projected torso XY/velocity, replacing competing posture goals.

    The demo's crawling preset suppresses heading poses, upright torso/limb
    poses, and recovery. Style velocities are adjusted to the requested world
    velocity in the current torso-heading frame, preserving relative motion.
    """
    prefix = "masked_mimic."
    masks = raw[prefix + "target_bodies_masks"].reshape(5, 5, 2)
    translation = raw[prefix + "anchor_translation_mask"]
    visible = raw[prefix + "target_poses_masks"]
    keep_xy = masks[:, 0, 0] & translation[:, 0] & visible
    masks[:] = False
    masks[:, 0, 0] = keep_xy
    translation[:] = False
    translation[:, 0] = keep_xy
    raw[prefix + "anchor_rotation_mask"][:] = False
    visible[:] = keep_xy
    raw[prefix + "time_offsets"][~keep_xy] = 0.0
    for name in (
        "local_target_bodies_masks",
        "local_target_poses_masks",
        "local_time_offsets",
    ):
        raw[prefix + name][:] = 0

    pose, velocity = crawling_preset()
    raw[prefix + "style_pose_samples"] = pose.copy()
    velocity = velocity.reshape(5, 5, 6).copy()
    if raw[prefix + "velocity_command_mask"][0]:
        x, y = raw[prefix + "average_velocity"]
        c, s = math.cos(heading), math.sin(heading)
        local = np.asarray([c * x + s * y, -s * x + c * y], np.float32)
        velocity[:, :, :2] += (local - velocity[:, 0, :2])[:, None, :]
    raw[prefix + "style_velocity_samples"] = velocity.reshape(-1)
    raw[prefix + "style_condition_masks"] = np.ones((5, 11), dtype=bool)
    raw[prefix + "style_frame_masks"] = np.ones(5, dtype=bool)
