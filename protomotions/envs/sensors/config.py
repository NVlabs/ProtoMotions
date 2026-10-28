# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declared physical sensors, one channel at a time, and their measurement accuracy.

Noise is declared per physical channel, never per derived feature: an encoder
error reaches hand positions through forward kinematics, so there is no separate
end-effector noise. A channel is either a uniform half-width in the channel's own
unit or ``NoNoise(reason)`` (measured exactly). The two mandatory channels (joint
encoders, IMU) are always present and carry noise only; their bodies and mounts
come from the robot config. Each optional channel (localization, base velocity,
height) is one ``Optional`` field, present when it is not ``None``, carrying its
own settings. Body names become indices once, in ``SensorConfig.mounts``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, AbstractSet, Dict, Optional, Tuple, Union

from protomotions.envs.sensors.metadata import (
    BaseVelocityMetadata,
    HeightMetadata,
    IMUMetadata,
    LocalizationMetadata,
    SensorDeploymentMetadata,
)

if TYPE_CHECKING:
    from protomotions.robot_configs.base import RobotConfig


@dataclass(frozen=True)
class NoNoise:
    """An exact channel. The reason is documentation and does not affect equality."""

    reason: str = field(compare=False)

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("NoNoise requires a reason")


NoiseLevel = Union[float, NoNoise]


def _check_level(level: NoiseLevel, name: str) -> None:
    if isinstance(level, NoNoise):
        return
    if (
        isinstance(level, bool)
        or not isinstance(level, (int, float))
        or not math.isfinite(level)
        or level <= 0
    ):
        raise ValueError(
            f"{name}: declare a finite positive half-width or NoNoise(reason)"
        )


@dataclass(frozen=True)
class LocalizationDrift:
    """Odometer drift: per-episode scale and yaw bias, per-step proportional log-magnitude noise.

    Horizontal only. ``scale`` and ``yaw`` (radians) are uniform half-widths
    drawn once per episode; ``log_magnitude`` is the per-step standard deviation
    in log(1 + |displacement|) space, weighted by |d| / (|d| + soft_threshold)
    so that noise vanishes near the start.
    """

    scale: NoiseLevel
    yaw: NoiseLevel
    log_magnitude: NoiseLevel
    soft_threshold: float = 0.15

    def __post_init__(self) -> None:
        for name in ("scale", "yaw", "log_magnitude"):
            _check_level(getattr(self, name), f"drift.{name}")
        if not isinstance(self.scale, NoNoise) and self.scale >= 1:
            raise ValueError("drift.scale half-width must be below one")
        if not math.isfinite(self.soft_threshold) or self.soft_threshold <= 0:
            raise ValueError("drift.soft_threshold must be positive")


@dataclass(frozen=True)
class JointEncoderSensor:
    """Per-joint position and velocity encoders (rad, rad/s). Always present."""

    pos_noise: NoiseLevel
    vel_noise: NoiseLevel

    def __post_init__(self) -> None:
        _check_level(self.pos_noise, "joint_encoders.pos_noise")
        _check_level(self.vel_noise, "joint_encoders.vel_noise")


@dataclass(frozen=True)
class IMUSensor:
    """Root-body orientation and angular velocity (xyzw component, rad/s). Always present.

    The body and mount come from robot.imu_body_name and robot.imu_mount_quat.
    Orientation noise is additive on quaternion components then renormalized, the
    law the existing BeyondMimic recipes were trained with.
    """

    quat_noise: NoiseLevel
    gyro_noise: NoiseLevel

    def __post_init__(self) -> None:
        _check_level(self.quat_noise, "imu.quat_noise")
        _check_level(self.gyro_noise, "imu.gyro_noise")


@dataclass(frozen=True)
class LocalizationSensor:
    """World position (m) of ``offset`` on ``body_name``, with optional odometer drift."""

    body_name: str
    noise: NoiseLevel
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    drift: Optional[LocalizationDrift] = None

    def __post_init__(self) -> None:
        _check_level(self.noise, "localization.noise")
        if len(self.offset) != 3 or not all(math.isfinite(v) for v in self.offset):
            raise ValueError("localization offset must be a finite xyz vector")


@dataclass(frozen=True)
class BaseVelocitySensor:
    """World linear velocity (m/s) of body 0's origin."""

    noise: NoiseLevel

    def __post_init__(self) -> None:
        _check_level(self.noise, "base_velocity.noise")


@dataclass(frozen=True)
class HeightSensor:
    """Body 0's clearance above local ground (m)."""

    noise: NoiseLevel

    def __post_init__(self) -> None:
        _check_level(self.noise, "height.noise")


@dataclass(frozen=True)
class SensorMounts:
    """Where the sensors sit, as indices resolved once from the robot asset.

    ``imu_body`` carries the IMU; ``imu_mount_quat`` (xyzw) rotates sensor axes
    into that body's axes. ``localization_body`` and ``localization_offset`` give
    the body and body-axes point the localization sensor measures, or None.
    """

    imu_body: int
    imu_mount_quat: Tuple[float, float, float, float]
    localization_body: Optional[int] = None
    localization_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)


# The declaration order channels() and the deployment host read them in.
CANONICAL_CHANNELS = (
    "joint_encoders",
    "imu",
    "localization",
    "base_velocity",
    "height",
)

# SensorReadings field -> the (channel, noise attribute) that measures it.
_READING_TO_CHANNEL = {
    "joint_pos": ("joint_encoders", "pos_noise"),
    "joint_vel": ("joint_encoders", "vel_noise"),
    "imu_quat": ("imu", "quat_noise"),
    "imu_gyro": ("imu", "gyro_noise"),
    "localization": ("localization", "noise"),
    "base_velocity": ("base_velocity", "noise"),
    "height": ("height", "noise"),
}


@dataclass(frozen=True)
class SensorConfig:
    """The physical sensors a policy reads, one channel at a time.

    The two mandatory channels are plain fields; each optional channel is an
    ``Optional`` field, present when it is not ``None``. A missing mandatory
    channel is the constructor's ``TypeError``. There is no suite to keep in
    agreement with the robot and no noise slot for a sensor that does not exist.
    """

    joint_encoders: JointEncoderSensor
    imu: IMUSensor
    localization: Optional[LocalizationSensor] = None
    base_velocity: Optional[BaseVelocitySensor] = None
    height: Optional[HeightSensor] = None
    # None derives the stream from the process seed, which training offsets per
    # rank; a fixed default would correlate every rank's noise.
    seed: Optional[int] = None

    def present(self, name: str) -> bool:
        """Whether an optional channel is declared (a mandatory one is always present)."""
        return getattr(self, name) is not None

    def channels(self) -> Dict[str, object]:
        """The present channels, keyed by name, in canonical declaration order."""
        return {
            name: getattr(self, name)
            for name in CANONICAL_CHANNELS
            if getattr(self, name) is not None
        }

    def level_of(self, reading: str) -> Optional[NoiseLevel]:
        """The noise level that measures a SensorReadings field, or None if absent."""
        channel_name, attr = _READING_TO_CHANNEL[reading]
        channel = getattr(self, channel_name)
        return None if channel is None else getattr(channel, attr)

    def for_inference(self) -> "SensorConfig":
        """Each present channel read exact (frozen, so a new config).

        Inference reads the clean sensor stream; the mounting and which channels
        exist are unchanged.
        """
        return replace(
            self, **{name: exact_channel(ch) for name, ch in self.channels().items()}
        )

    def mounts(self, robot: "RobotConfig") -> SensorMounts:
        """Resolve the IMU and optional localization bodies to indices, once.

        This is the single place sensor body names become indices, and it keeps
        the single-hinge refusal the FK relies on.
        """
        names = robot.kinematic_info.body_names
        if robot.imu_body_name is None:
            raise ValueError(
                "robot.imu_body_name must be declared; there is no implicit root IMU"
            )
        if robot.imu_body_name not in names:
            raise ValueError(
                f"IMU body {robot.imu_body_name!r} is not in the robot asset"
            )
        mount_quat = tuple(robot.imu_mount_quat)
        if len(mount_quat) != 4 or not math.isclose(
            sum(v * v for v in mount_quat), 1.0, abs_tol=1e-6
        ):
            raise ValueError("imu_mount_quat must be a unit xyzw quaternion")

        localization_body = None
        localization_offset = (0.0, 0.0, 0.0)
        if self.localization is not None:
            if self.localization.body_name not in names:
                raise ValueError(
                    f"Localization body {self.localization.body_name!r} is not in "
                    f"the robot asset"
                )
            localization_body = names.index(self.localization.body_name)
            localization_offset = tuple(self.localization.offset)

        # Sensor FK reads one scalar hinge per jointed body. Multi-axis joint
        # conventions differ between backends and are not interpreted here.
        info = robot.kinematic_info
        if (
            any(len(axes) != 1 for axes in info.hinge_axes_map.values())
            or sum(len(axes) for axes in info.hinge_axes_map.values()) != info.num_dofs
        ):
            raise ValueError(
                "Sensor FK requires single-hinge bodies; multi-axis joints are unsupported"
            )

        return SensorMounts(
            imu_body=names.index(robot.imu_body_name),
            imu_mount_quat=mount_quat,
            localization_body=localization_body,
            localization_offset=localization_offset,
        )

    def deployment_metadata(
        self,
        robot: "RobotConfig",
        mounts: Optional[SensorMounts] = None,
        channels: Optional[AbstractSet[str]] = None,
    ) -> SensorDeploymentMetadata:
        """The sensor conventions a deployment host must honour to feed this policy.

        Placement, mounting, frames and which optional channels exist are
        recorded; nothing here calibrates hardware or verifies a host. ``mounts``
        may be supplied when the caller already resolved them, to avoid a second
        name-to-index pass.

        ``channels`` are the sensor channels the exported actor actually reads,
        the set the caller derives from the bound ``clean``/``noisy`` paths (via
        ``sensors.measurement.derived_sensor_channel``). When given, an optional
        channel (localization, base velocity, height) is exported only when it is
        in that set, so a sensor the policy never reads is not listed as a host
        requirement. Left ``None`` (the default), every configured channel is
        reported, which is the full-metadata view the config's own tests use.
        """
        mounts = self.mounts(robot) if mounts is None else mounts
        names = robot.kinematic_info.body_names
        imu = IMUMetadata(
            body_name=names[mounts.imu_body],
            body_index=mounts.imu_body,
            mount_sensor_to_body_xyzw=list(mounts.imu_mount_quat),
        )

        def _export(channel: str, present: bool) -> bool:
            return present and (channels is None or channel in channels)

        localization = None
        if _export("localization", self.localization is not None):
            localization = LocalizationMetadata(
                body_name=names[mounts.localization_body],
                body_index=mounts.localization_body,
                offset_body_xyz=list(mounts.localization_offset),
            )

        base_velocity = None
        if _export("base_velocity", self.base_velocity is not None):
            base_velocity = BaseVelocityMetadata(body_name=names[0], body_index=0)

        height = None
        # No deployable observation is height-backed today: ground_heights is
        # @non_exportable and root_height derives the localization channel
        # (measurement._FIELD_SENSOR_CHANNEL maps ground_heights to None), so no
        # name derives the "height" channel and the caller's channel set never
        # contains it, leaving this branch unkept. A future height-backed
        # deployable obs would need a name that derives the height channel.
        if _export("height", self.height is not None):
            height = HeightMetadata(body_name=names[0], body_index=0)

        return SensorDeploymentMetadata(
            imu=imu,
            localization=localization,
            base_velocity=base_velocity,
            height=height,
            anchor_body_name=robot.resolved_anchor_body_name,
        )


def exact_channel(channel):
    """A copy of a channel measured exactly: every noise level NoNoise, no drift."""
    exact = NoNoise("inference")
    if isinstance(channel, JointEncoderSensor):
        return replace(channel, pos_noise=exact, vel_noise=exact)
    if isinstance(channel, IMUSensor):
        return replace(channel, quat_noise=exact, gyro_noise=exact)
    if isinstance(channel, LocalizationSensor):
        return replace(channel, noise=exact, drift=None)
    if isinstance(channel, BaseVelocitySensor):
        return replace(channel, noise=exact)
    if isinstance(channel, HeightSensor):
        return replace(channel, noise=exact)
    raise TypeError(f"not a sensor channel: {type(channel).__name__}")


def require_root_identity_imu(mounts: SensorMounts) -> None:
    """Refuse an IMU that is not on the root body with an identity mount.

    Both quaternion signs of the identity rotation pass; a rotated or non-root
    mount needs a host-side gyro conversion the tracker protocol does not carry.
    """
    identity = ((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, -1.0))
    if mounts.imu_body != 0 or tuple(mounts.imu_mount_quat) not in identity:
        raise ValueError(
            "Tracker export requires a root-body IMU with identity mount; a "
            "non-root or rotated IMU requires host-side gyro conversion"
        )
