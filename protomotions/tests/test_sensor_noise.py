# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""add_sensor_noise is pure: supplied draws in, perturbed readings out, no RNG."""

import math

import numpy as np
import pytest
import torch

from protomotions.envs.sensors.config import (
    BaseVelocitySensor,
    HeightSensor,
    IMUSensor,
    JointEncoderSensor,
    LocalizationDrift,
    LocalizationSensor,
    NoNoise,
    SensorConfig,
    SensorMounts,
)
from protomotions.envs.sensors.measurement import measure
from protomotions.envs.sensors.noise import DriftState, NoiseDraws, add_sensor_noise
from protomotions.tests.support.sensor_chain import (
    chain_state,
    load_chain,
    random_chain_batch,
)
from protomotions.utils.odom_corruption import apply_odom_corruption_np

N = 6


def _config(
    *,
    joint_pos=0.01,
    joint_vel=0.5,
    imu_quat=0.05,
    imu_gyro=0.2,
    localization=0.02,
    base_velocity=0.1,
    height=0.02,
    drift=None,
):
    """A full SensorConfig; add_sensor_noise reads the level for every channel."""
    return SensorConfig(
        joint_encoders=JointEncoderSensor(joint_pos, joint_vel),
        imu=IMUSensor(imu_quat, imu_gyro),
        localization=LocalizationSensor("pelvis", localization, drift=drift),
        base_velocity=BaseVelocitySensor(base_velocity),
        height=HeightSensor(height),
    )


@pytest.fixture
def clean(tmp_path):
    model, _ = load_chain(tmp_path)
    qpos, qvel = random_chain_batch(np.random.default_rng(11), N)
    mounts = SensorMounts(
        imu_body=1, imu_mount_quat=(0.0, 0.0, 0.0, 1.0), localization_body=0
    )
    return measure(
        chain_state(model, qpos, qvel),
        _config(),
        mounts,
        ground_heights=torch.zeros(N),
    )


def _draws(clean, generator):
    uniform = clean.map(lambda t: torch.rand(t.shape, generator=generator) * 2 - 1)
    return NoiseDraws(
        measurement=uniform, localization_log=torch.randn(N, 1, generator=generator)
    )


def test_uniform_half_width_law_and_quaternion_normalization(clean):
    draws = _draws(clean, torch.Generator().manual_seed(0))
    noisy = add_sensor_noise(clean, _config(), draws, drift=None)
    torch.testing.assert_close(
        noisy.joint_pos, clean.joint_pos + draws.measurement.joint_pos * 0.01
    )
    torch.testing.assert_close(
        noisy.base_velocity, clean.base_velocity + draws.measurement.base_velocity * 0.1
    )
    torch.testing.assert_close(
        noisy.height, clean.height + draws.measurement.height * 0.02
    )
    perturbed = clean.imu_quat + draws.measurement.imu_quat * 0.05
    torch.testing.assert_close(
        noisy.imu_quat, perturbed / perturbed.norm(dim=-1, keepdim=True)
    )
    torch.testing.assert_close(noisy.imu_quat.norm(dim=-1), torch.ones(N))


def test_no_noise_channel_is_bit_identical_and_owned(clean):
    draws = _draws(clean, torch.Generator().manual_seed(1))
    noisy = add_sensor_noise(
        clean, _config(imu_quat=NoNoise("exact imu")), draws, drift=None
    )
    assert torch.equal(noisy.imu_quat, clean.imu_quat)
    assert noisy.imu_quat.data_ptr() != clean.imu_quat.data_ptr()
    assert not torch.equal(noisy.joint_pos, clean.joint_pos)


def test_add_sensor_noise_consumes_no_global_randomness(clean):
    draws = _draws(clean, torch.Generator().manual_seed(2))
    before = torch.random.get_rng_state()
    add_sensor_noise(clean, _config(), draws, drift=None)
    assert torch.equal(before, torch.random.get_rng_state())


def test_drift_matches_the_deployment_law_with_identical_draws(clean, monkeypatch):
    drift_cfg = LocalizationDrift(
        scale=0.3, yaw=math.radians(6), log_magnitude=0.12, soft_threshold=0.15
    )
    config = _config(localization=NoNoise("isolate drift"), drift=drift_cfg)
    draws = _draws(clean, torch.Generator().manual_seed(3))
    origin = clean.localization - torch.tensor(
        [0.8, -0.4, 0.0]
    )  # robot has walked since episode start
    drift = DriftState(
        origin=origin, scale=torch.full((N,), 1.17), yaw=torch.full((N,), 0.05)
    )
    noisy = add_sensor_noise(clean, config, draws, drift)
    yaw_cs = np.array([math.cos(0.05), math.sin(0.05)])
    for i in range(N):
        disp = (clean.localization[i, :2] - origin[i, :2]).numpy()
        # The deployment law draws its own normal; feed it the draw the sensor pipeline used.
        monkeypatch.setattr(
            np.random, "randn", lambda *_, z=float(draws.localization_log[i, 0]): z
        )
        expected = origin[i, :2].numpy() + apply_odom_corruption_np(
            disp, 1.17, yaw_cs, 0.12, 0.15
        )
        np.testing.assert_allclose(
            noisy.localization[i, :2].numpy(), expected, atol=1e-5
        )
    torch.testing.assert_close(
        noisy.localization[:, 2], clean.localization[:, 2]
    )  # drift is horizontal only


def test_drift_is_zero_at_the_episode_origin(clean):
    config = _config(
        localization=NoNoise("isolate drift"), drift=LocalizationDrift(0.3, 0.1, 0.12)
    )
    draws = _draws(clean, torch.Generator().manual_seed(4))
    drift = DriftState(
        origin=clean.localization.clone(),
        scale=torch.full((N,), 1.3),
        yaw=torch.full((N,), 0.1),
    )
    noisy = add_sensor_noise(clean, config, draws, drift)
    torch.testing.assert_close(noisy.localization, clean.localization)


@pytest.mark.parametrize("level", [0.02, NoNoise("isolate drift")])
def test_localization_drift_does_not_mutate_or_alias_clean_readings(clean, level):
    before = clean.localization.clone()
    config = _config(localization=level, drift=LocalizationDrift(0.3, 0.1, 0.12))
    draws = _draws(clean, torch.Generator().manual_seed(4))
    drift = DriftState(
        origin=before - 1.0,
        scale=torch.full((N,), 1.3),
        yaw=torch.full((N,), 0.1),
    )
    noisy = add_sensor_noise(clean, config, draws, drift)
    torch.testing.assert_close(clean.localization, before, rtol=0, atol=0)
    assert noisy.localization.data_ptr() != clean.localization.data_ptr()
    noisy.localization.zero_()
    torch.testing.assert_close(clean.localization, before, rtol=0, atol=0)


def test_configured_drift_without_drift_state_is_an_error(clean):
    config = _config(drift=LocalizationDrift(0.3, 0.1, 0.12))
    with pytest.raises(ValueError):
        add_sensor_noise(
            clean, config, _draws(clean, torch.Generator().manual_seed(5)), drift=None
        )


def test_reading_without_a_matching_channel_is_refused_not_a_silent_crash(clean):
    # ``clean`` carries localization; a config that omits that channel has no
    # noise level for it, so add_sensor_noise refuses by name instead of raising
    # an opaque TypeError on clean + draw * None.
    config_without_localization = SensorConfig(
        joint_encoders=JointEncoderSensor(0.01, 0.5), imu=IMUSensor(0.05, 0.2)
    )
    draws = _draws(clean, torch.Generator().manual_seed(8))
    with pytest.raises(ValueError, match="readings carry 'localization'"):
        add_sensor_noise(clean, config_without_localization, draws, drift=None)


def test_white_localization_noise_is_added_after_drift(clean):
    drift_cfg = LocalizationDrift(
        scale=NoNoise("x"), yaw=NoNoise("x"), log_magnitude=NoNoise("x")
    )
    draws = _draws(clean, torch.Generator().manual_seed(6))
    drift = DriftState(
        origin=clean.localization - 1.0, scale=torch.ones(N), yaw=torch.zeros(N)
    )
    noisy = add_sensor_noise(clean, _config(drift=drift_cfg), draws, drift)
    torch.testing.assert_close(
        noisy.localization, clean.localization + draws.measurement.localization * 0.02
    )


def test_absent_channels_stay_absent(tmp_path):
    model, _ = load_chain(tmp_path)
    qpos, qvel = random_chain_batch(np.random.default_rng(12), N)
    clean = measure(
        chain_state(model, qpos, qvel),
        SensorConfig(
            joint_encoders=JointEncoderSensor(0.01, 0.5), imu=IMUSensor(0.05, 0.2)
        ),
        SensorMounts(imu_body=0, imu_mount_quat=(0.0, 0.0, 0.0, 1.0)),
    )
    draws = _draws(clean, torch.Generator().manual_seed(7))
    noisy = add_sensor_noise(
        clean,
        SensorConfig(
            joint_encoders=JointEncoderSensor(0.01, 0.5), imu=IMUSensor(0.05, 0.2)
        ),
        draws,
        drift=None,
    )
    assert (
        noisy.localization is None
        and noisy.base_velocity is None
        and noisy.height is None
    )
