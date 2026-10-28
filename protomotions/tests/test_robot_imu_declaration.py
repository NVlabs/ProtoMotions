# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IMU placement belongs to the robot; mounts() resolves it from the robot by name."""

import pytest

from protomotions.envs.sensors.config import (
    IMUSensor,
    JointEncoderSensor,
    LocalizationSensor,
    SensorConfig,
)
from protomotions.robot_configs.base import RobotConfig
from protomotions.tests.support.sensor_chain import ChainRobot, load_chain


def _config(localization_body_name=None, localization_offset=(0.0, 0.0, 0.0)):
    localization = None
    if localization_body_name is not None:
        localization = LocalizationSensor(
            body_name=localization_body_name, noise=0.01, offset=localization_offset
        )
    return SensorConfig(
        joint_encoders=JointEncoderSensor(0.01, 0.5),
        imu=IMUSensor(0.05, 0.2),
        localization=localization,
    )


def test_robot_config_declares_no_imu_by_default():
    assert RobotConfig.__dataclass_fields__["imu_body_name"].default is None
    assert RobotConfig.__dataclass_fields__["imu_mount_quat"].default == (
        0.0,
        0.0,
        0.0,
        1.0,
    )


def test_g1_declares_its_pelvis_imu():
    from protomotions.robot_configs.g1 import G1RobotConfig

    assert G1RobotConfig.__dataclass_fields__["imu_body_name"].default == "pelvis"


def test_mounts_resolve_names_from_the_robot(tmp_path):
    _, info = load_chain(tmp_path)
    robot = ChainRobot(info, imu_body_name="torso", imu_mount_quat=(0.0, 0.0, 0.0, 1.0))
    mounts = _config("hand", (0.1, 0.0, 0.0)).mounts(robot)
    assert (mounts.imu_body, mounts.localization_body) == (1, 2)
    assert mounts.localization_offset == (0.1, 0.0, 0.0)


def test_mounts_reject_an_undeclared_robot_imu(tmp_path):
    _, info = load_chain(tmp_path)
    with pytest.raises(ValueError, match="imu_body_name must be declared"):
        _config().mounts(ChainRobot(info, imu_body_name=None))


def test_mounts_reject_a_localization_body_not_in_the_asset(tmp_path):
    _, info = load_chain(tmp_path)
    robot = ChainRobot(info, imu_body_name="torso")
    with pytest.raises(ValueError, match="Localization body 'wing' is not"):
        _config("wing").mounts(robot)
