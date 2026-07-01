# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared odometer corruption utility for training and deployment.

Single source of truth for the corruption transform applied to the XY offset
observation.  Both the torch (GPU, batched) and numpy (single-env, deployment)
implementations execute the identical algorithm so there is zero risk of
sim-to-real mismatch from reimplementation.

Algorithm (per-step):
    1. Apply per-episode affine: scale * Rotate2D(yaw_bias) @ xy_local
    2. Decompose into direction + magnitude
    3. Add proportional log-space noise: log(1+mag) + N(0, σ) * mag/(mag+threshold)
    4. Reconstruct: direction * max(exp(noisy_log) - 1, 0)

Design rationale — see ``data/scripts/visualize_odometer_corruption.py`` for
interactive parameter tuning and ``protomotions/tests/test_odom_corruption.py``
for the torch-numpy equivalence test.

Training usage::

    from protomotions.utils.odom_corruption import apply_odom_corruption_torch
    xy_corrupted = apply_odom_corruption_torch(xy_local, scale, yaw_cs, ...)

Deployment usage::

    from protomotions.utils.odom_corruption import (
        apply_odom_corruption_np, sample_odom_params_np,
    )
    scale, yaw_cs = sample_odom_params_np(scale_range=(0.7, 1.3), yaw_range_deg=6.0)
    xy_corrupted = apply_odom_corruption_np(xy_local, scale, yaw_cs, ...)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from torch import Tensor


# =============================================================================
# Torch implementation (GPU, batched — used in training)
# =============================================================================


def apply_odom_corruption_torch(
    xy_local: Tensor,
    odom_scale: Tensor,
    odom_yaw_cos_sin: Tensor,
    log_noise_std: float = 0.12,
    soft_threshold: float = 0.15,
) -> Tensor:
    """Apply odometer corruption to a heading-local XY offset (batched GPU).

    Args:
        xy_local: Heading-local XY offset [envs, 2].
        odom_scale: Per-episode multiplicative scale [envs].
        odom_yaw_cos_sin: Per-episode yaw bias as (cos, sin) [envs, 2].
        log_noise_std: Std of per-step noise in log(1+mag) space.
        soft_threshold: Smooth noise ramp characteristic length (metres).

    Returns:
        Corrupted XY offset [envs, 2].
    """
    # Step 1: per-episode affine — scale * Rotate2D(yaw) @ xy
    cos_y = odom_yaw_cos_sin[:, 0]
    sin_y = odom_yaw_cos_sin[:, 1]
    x_rot = cos_y * xy_local[:, 0] - sin_y * xy_local[:, 1]
    y_rot = sin_y * xy_local[:, 0] + cos_y * xy_local[:, 1]
    xy_affine = odom_scale.unsqueeze(-1) * torch.stack([x_rot, y_rot], dim=-1)

    # Step 2: direction + magnitude decomposition
    mag = torch.norm(xy_affine, dim=-1, keepdim=True).clamp(min=1e-8)
    direction = xy_affine / mag

    # Step 3: proportional log-space noise
    log_mag = torch.log(1.0 + mag)
    noise_weight = mag / (mag + soft_threshold)
    noisy_log_mag = log_mag + torch.randn_like(log_mag) * log_noise_std * noise_weight

    # Step 4: reconstruct
    noisy_mag = (torch.exp(noisy_log_mag) - 1.0).clamp(min=0.0)
    return direction * noisy_mag


# =============================================================================
# NumPy implementation (single-env — used in deployment)
# =============================================================================


def apply_odom_corruption_np(
    xy_local: np.ndarray,
    odom_scale: float,
    odom_yaw_cos_sin: np.ndarray,
    log_noise_std: float = 0.12,
    soft_threshold: float = 0.15,
) -> np.ndarray:
    """Apply odometer corruption to a heading-local XY offset (single-env numpy).

    Identical algorithm to ``apply_odom_corruption_torch`` — import this in
    deployment code (RoboJuDo, test_tracker_mujoco) to avoid reimplementation.

    Args:
        xy_local: Heading-local XY offset [2].
        odom_scale: Per-session multiplicative scale (scalar).
        odom_yaw_cos_sin: Per-session yaw bias as (cos, sin) array [2].
        log_noise_std: Std of per-step noise in log(1+mag) space.
        soft_threshold: Smooth noise ramp characteristic length (metres).

    Returns:
        Corrupted XY offset [2].
    """
    cos_y, sin_y = float(odom_yaw_cos_sin[0]), float(odom_yaw_cos_sin[1])

    # Step 1: per-session affine
    x_rot = cos_y * xy_local[0] - sin_y * xy_local[1]
    y_rot = sin_y * xy_local[0] + cos_y * xy_local[1]
    xy_affine = odom_scale * np.array([x_rot, y_rot], dtype=np.float32)

    # Step 2: direction + magnitude
    mag = max(float(np.linalg.norm(xy_affine)), 1e-8)
    direction = xy_affine / mag

    # Step 3: proportional log-space noise
    log_mag = math.log(1.0 + mag)
    noise_weight = mag / (mag + soft_threshold)
    noisy_log_mag = log_mag + np.random.randn() * log_noise_std * noise_weight

    # Step 4: reconstruct
    noisy_mag = max(math.exp(noisy_log_mag) - 1.0, 0.0)
    return direction * noisy_mag


def sample_odom_params_np(
    scale_range: Tuple[float, float] = (0.7, 1.3),
    yaw_range_deg: float = 6.0,
) -> Tuple[float, np.ndarray]:
    """Sample per-session odometer corruption parameters.

    Call once at session/episode start. Returns (scale, yaw_cos_sin) to pass
    to ``apply_odom_corruption_np`` each step.

    Args:
        scale_range: (lo, hi) for Uniform scale sampling.
        yaw_range_deg: Half-range in degrees for Uniform yaw bias.

    Returns:
        Tuple of (scale: float, yaw_cos_sin: ndarray[2]).
    """
    scale = float(np.random.uniform(scale_range[0], scale_range[1]))
    yaw_rad = math.radians(float(np.random.uniform(-yaw_range_deg, yaw_range_deg)))
    yaw_cos_sin = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float32)
    return scale, yaw_cos_sin
