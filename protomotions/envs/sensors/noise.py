# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measurement noise as a pure function of ideal readings and supplied draws.

Nothing here draws random numbers or keeps state. The runtime draws once per
physical sample, in the measured tensor shapes, and passes the draws in. A
``NoNoise`` channel is returned unchanged (its draw is simply unused), so
switching one channel to exact never changes another channel's stream.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch
from torch import Tensor

from protomotions.envs.sensors.config import NoNoise, SensorConfig
from protomotions.envs.sensors.measurement import SensorReadings
from protomotions.utils.odom_corruption import apply_odom_corruption_torch


@dataclass
class NoiseDraws:
    """Uniform U(-1, 1) draws shaped like the measurement, plus one N(0, 1) per row for drift."""

    measurement: SensorReadings
    localization_log: Tensor


@dataclass
class DriftState:
    """Per-row odometer drift parameters, fixed for an episode.

    ``origin`` is the ideal localization at episode start [n, 3]; drift acts on
    the horizontal displacement from it. ``scale`` [n] multiplies that
    displacement and ``yaw`` [n] rotates it (radians).
    """

    origin: Tensor
    scale: Tensor
    yaw: Tensor


def add_sensor_noise(
    readings: SensorReadings,
    config: SensorConfig,
    draws: NoiseDraws,
    drift: Optional[DriftState],
) -> SensorReadings:
    """Perturb each measured channel by draw * half-width; orientation is renormalized.

    Localization is corrupted in two steps that mirror a real odometer: the
    drift law (episode scale and yaw bias on the displacement from the start,
    then proportional log-magnitude noise) and white positional noise on top.
    """
    drift_cfg = None if config.localization is None else config.localization.drift
    if drift_cfg is not None and drift is None:
        raise ValueError(
            "drift is configured; the runtime must supply the episode DriftState"
        )

    out = {}
    for f in fields(SensorReadings):
        clean = getattr(readings, f.name)
        if clean is None:
            out[f.name] = None
            continue
        level = config.level_of(f.name)
        if level is None:
            raise ValueError(
                f"readings carry {f.name!r} but the sensor config declares no such channel"
            )
        if isinstance(level, NoNoise):
            out[f.name] = clean.clone()
        else:
            out[f.name] = clean + getattr(draws.measurement, f.name) * level

    if readings.localization is not None and drift_cfg is not None:
        cfg = drift_cfg
        displacement = readings.localization[:, :2] - drift.origin[:, :2]
        yaw_cos_sin = torch.stack([drift.yaw.cos(), drift.yaw.sin()], dim=-1)
        corrupted = apply_odom_corruption_torch(
            displacement,
            drift.scale,
            yaw_cos_sin,
            log_noise_std=0.0
            if isinstance(cfg.log_magnitude, NoNoise)
            else cfg.log_magnitude,
            soft_threshold=cfg.soft_threshold,
            standardized_draw=draws.localization_log,
        )
        # Each populated output above already owns its storage, including NoNoise.
        out["localization"][:, :2] += corrupted - displacement

    if not isinstance(config.level_of("imu_quat"), NoNoise):
        q = out["imu_quat"]
        out["imu_quat"] = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return SensorReadings(**out)
