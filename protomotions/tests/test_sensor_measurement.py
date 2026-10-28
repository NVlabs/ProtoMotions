# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""measure() is the only simulator-to-sensor boundary; MuJoCo supplies the truth."""

import numpy as np
import pytest
import torch

from protomotions.envs.sensors.config import (
    BaseVelocitySensor,
    HeightSensor,
    IMUSensor,
    JointEncoderSensor,
    LocalizationSensor,
    NoNoise,
    SensorConfig,
    SensorMounts,
)
from protomotions.envs.sensors.measurement import SensorReadings, measure
from protomotions.tests.support.sensor_chain import (
    chain_state,
    load_chain,
    random_chain_batch,
)
from protomotions.utils import rotations as rot

TORSO, HAND = 1, 2
MOUNT = tuple(
    (
        torch.tensor([0.3, -0.2, 0.5, 0.8]) / torch.tensor([0.3, -0.2, 0.5, 0.8]).norm()
    ).tolist()
)


def _config(*, localization=False, base_velocity=False, height=False):
    """A minimal SensorConfig declaring the requested optional channels.

    measure() reads only which channels are present; the noise levels are unused,
    so every channel is exact.
    """
    exact = NoNoise("measurement test")
    return SensorConfig(
        joint_encoders=JointEncoderSensor(exact, exact),
        imu=IMUSensor(exact, exact),
        localization=LocalizationSensor("hand", exact) if localization else None,
        base_velocity=BaseVelocitySensor(exact) if base_velocity else None,
        height=HeightSensor(exact) if height else None,
    )


def _mounts(
    *,
    imu_body=0,
    imu_mount_quat=(0.0, 0.0, 0.0, 1.0),
    localization_body=None,
    localization_offset=(0.0, 0.0, 0.0),
):
    return SensorMounts(
        imu_body, imu_mount_quat, localization_body, localization_offset
    )


@pytest.fixture
def chain(tmp_path):
    model, _ = load_chain(tmp_path)
    qpos, qvel = random_chain_batch(np.random.default_rng(3), n=5)
    return chain_state(model, qpos, qvel)


def test_mounted_imu_reports_sensor_orientation_and_sensor_axis_gyro(chain):
    readings = measure(chain, _config(), _mounts(imu_body=TORSO, imu_mount_quat=MOUNT))
    mount = chain.rigid_body_rot.new_tensor(MOUNT).expand(5, 4)
    expected_q = rot.quat_mul(chain.rigid_body_rot[:, TORSO], mount, w_last=True)
    torch.testing.assert_close(readings.imu_quat, expected_q)
    # The gyro measures the IMU body's world angular velocity in the sensor's own axes.
    expected_gyro = rot.quat_rotate_inverse(
        expected_q, chain.rigid_body_ang_vel[:, TORSO], w_last=True
    )
    torch.testing.assert_close(readings.imu_gyro, expected_gyro)


def test_localization_measures_the_offset_point_on_its_body(chain):
    offset = (0.05, -0.02, 0.1)
    readings = measure(
        chain,
        _config(localization=True),
        _mounts(localization_body=HAND, localization_offset=offset),
    )
    expected = chain.rigid_body_pos[:, HAND] + rot.quat_rotate(
        chain.rigid_body_rot[:, HAND],
        chain.rigid_body_pos.new_tensor(offset).expand(5, 3),
        w_last=True,
    )
    torch.testing.assert_close(readings.localization, expected)


def test_absent_sensors_are_none_not_zero(chain):
    readings = measure(chain, _config(), _mounts())
    assert readings.localization is None
    assert readings.base_velocity is None
    assert readings.height is None


def test_height_is_root_clearance_and_requires_ground(chain):
    config, mounts = _config(height=True), _mounts()
    ground = torch.linspace(-0.1, 0.1, 5)
    readings = measure(chain, config, mounts, ground_heights=ground)
    torch.testing.assert_close(
        readings.height, (chain.rigid_body_pos[:, 0, 2] - ground)[:, None]
    )
    with pytest.raises(ValueError):
        measure(chain, config, mounts)
    with pytest.raises(ValueError):
        measure(chain, config, mounts, ground_heights=ground[:3])


def test_base_velocity_is_the_root_origin_world_velocity(chain):
    readings = measure(chain, _config(base_velocity=True), _mounts())
    torch.testing.assert_close(readings.base_velocity, chain.rigid_body_vel[:, 0])


def test_readings_are_owned_copies(chain):
    readings = measure(chain, _config(), _mounts())
    readings.joint_pos.add_(1.0)
    assert not torch.equal(readings.joint_pos, chain.dof_pos)


def test_cat_split_and_map_preserve_absent_channels(chain):
    a = measure(chain, _config(base_velocity=True), _mounts())
    b = measure(chain, _config(base_velocity=True), _mounts())
    both = SensorReadings.cat([a, b])
    assert both.joint_pos.shape[0] == 10 and both.localization is None
    first, second = both.split(5)
    torch.testing.assert_close(second.base_velocity, b.base_velocity)
    doubled = first.map(lambda t: t * 2)
    torch.testing.assert_close(doubled.joint_vel, a.joint_vel * 2)
    assert doubled.height is None
    with pytest.raises(ValueError):
        SensorReadings.cat([a, measure(chain, _config(), _mounts())])
