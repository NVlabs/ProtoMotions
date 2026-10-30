# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor deployment contract; exporters serialize it with dataclasses.asdict."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IMUMetadata:
    """IMU placement and the frames of its orientation and gyro readings."""

    body_name: str
    body_index: int
    mount_sensor_to_body_xyzw: list[float]
    orientation_frame: str = "world"
    gyro_frame: str = "sensor"


@dataclass(frozen=True)
class LocalizationMetadata:
    """World position of a point fixed to the named body."""

    body_name: str
    body_index: int
    offset_body_xyz: list[float]
    position_frame: str = "world"


@dataclass(frozen=True)
class BaseVelocityMetadata:
    """Linear velocity of the named body's origin."""

    body_name: str
    body_index: int
    frame: str = "world"


@dataclass(frozen=True)
class HeightMetadata:
    """Vertical clearance of the named body's origin above local ground."""

    body_name: str
    body_index: int
    reference: str = "local_ground"
    axis: str = "world_z"


@dataclass(frozen=True)
class SensorDeploymentMetadata:
    """Physical sensor inputs and conventions expected by the exported policy.

    The host feeds readings only; the graph derives every body quantity it
    needs from them (the body poses come from the recipe's FK kernel).
    """

    imu: IMUMetadata
    anchor_body_name: str
    localization: LocalizationMetadata | None = None
    base_velocity: BaseVelocityMetadata | None = None
    height: HeightMetadata | None = None
    schema: str = "physical-sensors/v1"
    joint_coordinates: str = "scalar_hinge"
    w_last: bool = True  # xyzw
    history_initialization: str = "repeat_first_reading"
