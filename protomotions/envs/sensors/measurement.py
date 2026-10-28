# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ideal readings of the declared physical sensors, taken from simulator state."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor

from protomotions.envs.sensors.config import SensorConfig, SensorMounts
from protomotions.simulator.base_simulator.simulator_state import RobotState
from protomotions.utils import rotations as rot


@dataclass
class SensorReadings:
    """One reading per declared channel; absent channels are None.

    Leading dimensions are arbitrary, the last is the channel width.
    joint_pos/joint_vel: scalar hinges, rad and rad/s. imu_quat: world
    orientation of the sensor frame, xyzw. imu_gyro: rad/s in sensor axes.
    localization: world metres of the declared point. base_velocity: world m/s
    of body 0's origin. height: body 0 origin clearance above local ground, m.
    """

    joint_pos: Tensor
    joint_vel: Tensor
    imu_quat: Tensor
    imu_gyro: Tensor
    localization: Optional[Tensor] = None
    base_velocity: Optional[Tensor] = None
    height: Optional[Tensor] = None

    def map(self, fn: Callable[[Tensor], Tensor]) -> SensorReadings:
        return SensorReadings(
            **{
                f.name: None
                if getattr(self, f.name) is None
                else fn(getattr(self, f.name))
                for f in fields(self)
            }
        )

    @staticmethod
    def cat(items: Sequence[SensorReadings]) -> SensorReadings:
        """Concatenate along the leading dimension; every item must declare the same channels."""
        out = {}
        for f in fields(SensorReadings):
            values = [getattr(item, f.name) for item in items]
            if any((v is None) != (values[0] is None) for v in values):
                raise ValueError(
                    f"Cannot concatenate readings with and without {f.name}"
                )
            out[f.name] = None if values[0] is None else torch.cat(values)
        return SensorReadings(**out)

    def split(self, n: int) -> tuple[SensorReadings, SensorReadings]:
        return self.map(lambda t: t[:n]), self.map(lambda t: t[n:])


def measure(
    state: RobotState,
    config: SensorConfig,
    mounts: SensorMounts,
    *,
    ground_heights: Optional[Tensor] = None,
) -> SensorReadings:
    """Read the config's sensors from simulator truth. Returns owned tensors.

    ``config`` says which optional channels exist; ``mounts`` gives the bodies
    and mount they sit on. The IMU reports the orientation of its own axes,
    body ⊗ mount, and the body's angular velocity expressed in those axes.
    Localization reports the world position of the declared offset point. Height
    is the root origin's clearance above the supplied ground, which the caller
    samples under the root.
    """
    body_q = state.rigid_body_rot[..., mounts.imu_body, :]
    mount = body_q.new_tensor(mounts.imu_mount_quat).expand_as(body_q)
    sensor_q = rot.quat_mul(body_q, mount, w_last=True)
    gyro = rot.quat_rotate_inverse(
        sensor_q, state.rigid_body_ang_vel[..., mounts.imu_body, :], w_last=True
    )

    localization = None
    if config.localization is not None:
        q = state.rigid_body_rot[..., mounts.localization_body, :]
        offset = q.new_tensor(mounts.localization_offset).expand(*q.shape[:-1], 3)
        localization = state.rigid_body_pos[
            ..., mounts.localization_body, :
        ] + rot.quat_rotate(q, offset, w_last=True)

    height = None
    if config.height is not None:
        root_z = state.rigid_body_pos[..., 0, 2]
        if ground_heights is None or ground_heights.shape != root_z.shape:
            raise ValueError("height sensor needs one ground height per root reading")
        height = (root_z - ground_heights)[..., None]

    base_velocity = (
        state.rigid_body_vel[..., 0, :] if config.base_velocity is not None else None
    )
    return SensorReadings(
        state.dof_pos,
        state.dof_vel,
        sensor_q,
        gyro,
        localization,
        base_velocity,
        height,
    ).map(Tensor.clone)


# Which sensor channel provides each raw ``BodyState`` field, beside ``measure``
# which is the code that reads each channel: the joint encoders read the joint
# state, the IMU the orientation and angular velocity, localization the body
# position (and the episode odometer that rides on it), and the base-velocity
# channel the linear velocity. The height sensor also reads the root's
# ``rigid_body_pos``, but a position quantity's provider is localization, so a
# field maps to exactly one channel here.
_FIELD_SENSOR_CHANNEL = {
    "dof_pos": "joint_encoders",
    "dof_vel": "joint_encoders",
    "rigid_body_rot": "imu",
    "rigid_body_ang_vel": "imu",
    "rigid_body_pos": "localization",
    "rigid_body_vel": "base_velocity",
    "odom_start_xy": "localization",
    "odom_start_heading_inv": "localization",
    "odom_disp_start": "localization",
    # Names no sensor measures, listed explicitly (mapped to None) so that an
    # unknown name is distinguished from a known-but-unsensed one: a new name
    # nobody assigned to a channel is a refusal, not a silent None a provider
    # rule would read as "no channel". These are the unsensed BodyState fields,
    # the policy's own command history (raw/processed actions, a POLICY_HISTORY
    # provider, never a sensor reading) and ``anchor_idx``, the anchor body index
    # the export module folds in as a constant, never a graph input.
    "dof_forces": None,
    "rigid_body_contacts": None,
    "ground_heights": None,
    "raw_actions": None,
    "processed_actions": None,
    "anchor_idx": None,
}


def sensor_channel_of_field(field: str) -> Optional[str]:
    """The sensor channel that provides a raw ``BodyState`` field, or ``None``.

    ``None`` for a field no sensor measures (the joint forces, the contacts, the
    ground heights): those are never fed from a sensor at deployment. A field the
    table does not list at all -- a new ``BodyState`` field nobody assigned to a
    channel -- is a refusal by name, so PR 2 can derive the provider from this
    table without a channel silently going missing.
    """
    try:
        return _FIELD_SENSOR_CHANNEL[field]
    except KeyError:
        raise ValueError(
            f"no sensor-channel rule for BodyState field {field!r}; known fields: "
            f"{sorted(_FIELD_SENSOR_CHANNEL)}"
        ) from None


def derived_sensor_channel(leaf: str) -> Optional[str]:
    """The sensor channel a ``body_state_names`` leaf is read from, or ``None``.

    The channel of the raw field the leaf derives from
    (:func:`protomotions.envs.state_names.raw_field_of`): a root or anchor slice
    reports its parent field's channel, ``root_height`` the localization
    channel, and a plain name its own. This is the deployment ``provider`` of a
    ``clean``/``noisy`` sensor path, derived instead of declared.
    """
    from protomotions.envs.state_names import raw_field_of

    return sensor_channel_of_field(raw_field_of(leaf))
