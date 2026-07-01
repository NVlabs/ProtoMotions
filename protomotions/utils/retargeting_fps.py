# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FPS metadata helpers for retargeting pipelines."""

from __future__ import annotations

import numpy as np


def fps_from_mapping(mapping, fallback_fps: float, key: str = "fps") -> float:
    """Read scalar FPS metadata from a numpy-style mapping."""
    if key not in mapping:
        return float(fallback_fps)
    return float(np.asarray(mapping[key]).item())


def fps_from_motion_dt(motion_dt) -> float:
    """Convert MotionLib dt metadata to FPS."""
    return 1.0 / float(np.asarray(motion_dt).item())


def subsampled_fps(source_fps: float, subsample_factor: int) -> float:
    """Return the FPS after keeping every Nth frame."""
    if subsample_factor <= 0:
        raise ValueError("subsample_factor must be positive")
    return float(source_fps) / int(subsample_factor)


def resolve_output_fps(input_fps: float, requested_output_fps: float | None) -> float:
    """Default output FPS to the per-motion input FPS."""
    if requested_output_fps is None:
        return float(input_fps)
    return float(requested_output_fps)


def downsample_factor(input_fps: float, output_fps: float) -> int:
    """Return integer frame stride for exact FPS downsampling."""
    input_fps = float(input_fps)
    output_fps = float(output_fps)
    ratio = input_fps / output_fps
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, rtol=1e-6, atol=1e-8):
        raise ValueError(
            f"input_fps ({input_fps:g}) must be an integer multiple of "
            f"output_fps ({output_fps:g})"
        )
    return factor
