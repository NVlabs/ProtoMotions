# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen experts retain their declared physical sensor inputs during distillation."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from protomotions.agents.supervised.expert_utils import validate_expert_sensor_contract
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.sensors.config import (
    IMUSensor,
    JointEncoderSensor,
    LocalizationSensor,
    SensorConfig,
)


def sensor_config():
    return SensorConfig(
        joint_encoders=JointEncoderSensor(0.01, 0.5),
        imu=IMUSensor(0.05, 0.2),
        seed=9,
    )


def expert_config(source, sensors):
    return EnvConfig(
        sensors=sensors,
        observation_components={
            "actor_obs": MdpComponent(
                lambda value: value, {"value": getattr(EnvContext, source).dof_pos}
            ),
            "unused": MdpComponent(
                lambda value: value, {"value": EnvContext.noisy.dof_pos}
            ),
        },
    )


AGENT = SimpleNamespace(
    model=SimpleNamespace(actor=SimpleNamespace(in_keys=["actor_obs"]))
)


@pytest.mark.parametrize("source", ["current", "historical"])
def test_privileged_expert_ignores_sensor_differences_and_unused_components(source):
    validate_expert_sensor_contract(expert_config(source, None), AGENT, sensor_config())


@pytest.mark.parametrize("source", ["clean", "clean_historical"])
def test_clean_expert_matches_channels_not_noise_or_seed(source):
    sensors = sensor_config()
    target = replace(
        sensors,
        joint_encoders=replace(sensors.joint_encoders, pos_noise=0.02),
        seed=15,
    )
    validate_expert_sensor_contract(expert_config(source, sensors), AGENT, target)


@pytest.mark.parametrize("source", ["clean", "clean_historical"])
def test_clean_expert_matches_localization_placement(source):
    sensors = replace(
        sensor_config(),
        localization=LocalizationSensor(body_name="pelvis", noise=0.01),
    )
    # Same placement, different noise: a clean source ignores the noise.
    target = replace(
        sensors,
        localization=replace(sensors.localization, noise=0.2),
    )
    validate_expert_sensor_contract(expert_config(source, sensors), AGENT, target)
    # A different localization offset is a different placement: refused.
    moved = replace(
        sensors,
        localization=replace(sensors.localization, offset=(0.1, 0.0, 0.0)),
    )
    with pytest.raises(ValueError, match="localization.*placement"):
        validate_expert_sensor_contract(expert_config(source, sensors), AGENT, moved)


@pytest.mark.parametrize("source", ["noisy", "noisy_historical"])
def test_noisy_expert_requires_same_noise_but_not_seed(source):
    sensors = sensor_config()
    expert = expert_config(source, sensors)
    validate_expert_sensor_contract(expert, AGENT, replace(sensors, seed=15))
    with pytest.raises(ValueError, match="actor_obs.*sensor contract"):
        validate_expert_sensor_contract(
            expert,
            AGENT,
            replace(
                sensors, joint_encoders=replace(sensors.joint_encoders, pos_noise=0.02)
            ),
        )


@pytest.mark.parametrize(
    "source", ["clean", "clean_historical", "noisy", "noisy_historical"]
)
@pytest.mark.parametrize(
    "mismatch", ["missing_expert", "missing_student", "different_channels"]
)
def test_missing_or_different_channels_is_not_reinterpreted(source, mismatch):
    sensors = sensor_config()
    expert_sensors = None if mismatch == "missing_expert" else sensors
    target = None if mismatch == "missing_student" else sensors
    if mismatch == "different_channels":
        target = replace(
            sensors, localization=LocalizationSensor(body_name="pelvis", noise=0.01)
        )
    with pytest.raises(
        ValueError, match=f"actor_obs.*{source}.dof_pos.*sensor contract"
    ):
        validate_expert_sensor_contract(
            expert_config(source, expert_sensors), AGENT, target
        )
