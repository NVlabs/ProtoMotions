# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intrinsic sensor configuration checks run at construction and at mounts(), not per step."""

from dataclasses import asdict, replace

import pytest

from protomotions.components.pose_lib import extract_kinematic_info
from protomotions.envs.sensors.config import (
    BaseVelocitySensor,
    HeightSensor,
    IMUSensor,
    JointEncoderSensor,
    LocalizationDrift,
    LocalizationSensor,
    NoNoise,
    SensorConfig,
    exact_channel,
)
from protomotions.tests.support.sensor_chain import ChainRobot, load_chain

# A body carrying a three-hinge (3-DOF) joint: the single-hinge refusal must fire.
THREE_HINGE_XML = """
<mujoco model="three_hinge">
  <compiler angle="radian"/>
  <worldbody>
    <body name="pelvis" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="5"/>
      <body name="torso" pos="0 0 0.3">
        <joint name="waist_x" type="hinge" axis="1 0 0"/>
        <joint name="waist_y" type="hinge" axis="0 1 0"/>
        <joint name="waist_z" type="hinge" axis="0 0 1"/>
        <geom type="sphere" size="0.04" mass="3"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _config(**overrides):
    base = dict(
        joint_encoders=JointEncoderSensor(0.01, 0.5),
        imu=IMUSensor(0.05, 0.2),
    )
    base.update(overrides)
    return SensorConfig(**base)


# ---------------------------- channel validation -----------------------------


def test_no_noise_requires_a_reason():
    with pytest.raises(ValueError):
        NoNoise("")


def test_no_noise_reason_does_not_affect_equality():
    assert JointEncoderSensor(NoNoise("a"), 0.5) == JointEncoderSensor(
        NoNoise("b"), 0.5
    )


def test_noise_level_must_be_finite_and_positive():
    for bad in (0.0, -0.1, float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            JointEncoderSensor(bad, 0.5)


def test_localization_offset_must_be_a_finite_xyz_vector():
    with pytest.raises(ValueError):
        LocalizationSensor("pelvis", 0.01, offset=(0.1, 0.0))
    with pytest.raises(ValueError):
        LocalizationSensor("pelvis", 0.01, offset=(0.1, float("nan"), 0.0))
    LocalizationSensor("pelvis", 0.01, offset=(0.1, 0.0, 0.0))


def test_drift_scale_half_width_must_be_below_one():
    with pytest.raises(ValueError):
        LocalizationDrift(scale=1.0, yaw=0.1, log_magnitude=0.12)
    with pytest.raises(ValueError):
        LocalizationDrift(scale=0.3, yaw=0.1, log_magnitude=0.12, soft_threshold=0.0)


def test_frozen_channel_replace_revalidates():
    imu = IMUSensor(0.05, 0.2)
    with pytest.raises(ValueError):
        replace(imu, gyro_noise=-1.0)
    assert replace(imu, gyro_noise=0.1).gyro_noise == 0.1


# ------------------------- present / channels / level_of ---------------------


def test_present_and_channels_list_declared_channels_in_canonical_order():
    config = _config(
        localization=LocalizationSensor("pelvis", 0.01), height=HeightSensor(0.02)
    )
    assert config.present("localization") and config.present("height")
    assert not config.present("base_velocity")
    assert list(config.channels()) == [
        "joint_encoders",
        "imu",
        "localization",
        "height",
    ]


def test_level_of_maps_every_reading_with_and_without_optionals():
    bare = _config()
    assert bare.level_of("joint_pos") == 0.01 and bare.level_of("joint_vel") == 0.5
    assert bare.level_of("imu_quat") == 0.05 and bare.level_of("imu_gyro") == 0.2
    for absent in ("localization", "base_velocity", "height"):
        assert bare.level_of(absent) is None
    full = _config(
        localization=LocalizationSensor("pelvis", 0.03),
        base_velocity=BaseVelocitySensor(0.1),
        height=HeightSensor(0.02),
    )
    assert full.level_of("localization") == 0.03
    assert full.level_of("base_velocity") == 0.1
    assert full.level_of("height") == 0.02


# -------------------------------- exact_channel ------------------------------


def test_exact_channel_sets_levels_exact_drops_drift_keeps_placement():
    loc = LocalizationSensor(
        "pelvis", 0.03, offset=(0.1, 0.0, 0.0), drift=LocalizationDrift(0.3, 0.1, 0.12)
    )
    exact = exact_channel(loc)
    assert isinstance(exact.noise, NoNoise) and exact.drift is None
    assert exact.body_name == "pelvis" and exact.offset == (0.1, 0.0, 0.0)
    je = exact_channel(JointEncoderSensor(0.01, 0.5))
    assert isinstance(je.pos_noise, NoNoise) and isinstance(je.vel_noise, NoNoise)
    with pytest.raises(TypeError):
        exact_channel("not a channel")


# ------------------------------ mounts() refusals ----------------------------


def test_mounts_refuse_no_imu_declaration(tmp_path):
    _, info = load_chain(tmp_path)
    with pytest.raises(ValueError, match="imu_body_name must be declared"):
        _config().mounts(ChainRobot(info, imu_body_name=None))


def test_mounts_refuse_unknown_imu_body(tmp_path):
    _, info = load_chain(tmp_path)
    with pytest.raises(ValueError, match="IMU body 'nose' is not"):
        _config().mounts(ChainRobot(info, imu_body_name="nose"))


def test_mounts_refuse_a_non_unit_mount_quaternion(tmp_path):
    _, info = load_chain(tmp_path)
    with pytest.raises(ValueError, match="unit xyzw quaternion"):
        _config().mounts(
            ChainRobot(info, imu_body_name="torso", imu_mount_quat=(0.0, 0.0, 0.0, 2.0))
        )


def test_mounts_refuse_unknown_localization_body(tmp_path):
    _, info = load_chain(tmp_path)
    with pytest.raises(ValueError, match="Localization body 'wing' is not"):
        _config(localization=LocalizationSensor("wing", 0.01)).mounts(
            ChainRobot(info, imu_body_name="torso")
        )


def test_mounts_refuse_multi_axis_joints(tmp_path):
    path = tmp_path / "three_hinge.xml"
    path.write_text(THREE_HINGE_XML)
    info = extract_kinematic_info(str(path))
    with pytest.raises(ValueError, match="single-hinge"):
        _config().mounts(ChainRobot(info, imu_body_name="torso"))


# ----------------------------- deployment_metadata ---------------------------


def test_deployment_metadata_records_placement_frames_and_optional_channels(tmp_path):
    _, info = load_chain(tmp_path)
    robot = ChainRobot(info, imu_body_name="torso", imu_mount_quat=(0.0, 0.0, 0.0, 1.0))
    meta = _config(
        localization=LocalizationSensor("hand", 0.02, offset=(0.1, 0.0, 0.0)),
        height=HeightSensor(0.02),
    ).deployment_metadata(robot)
    assert meta.imu.body_name == "torso"
    assert meta.localization.offset_body_xyz == [0.1, 0.0, 0.0]
    assert meta.base_velocity is None
    assert meta.height.reference == "local_ground"
    meta = asdict(meta)
    assert meta["schema"] == "physical-sensors/v1"
    assert meta["w_last"] is True
    assert "quaternion_order" not in meta
    assert meta["joint_coordinates"] == "scalar_hinge"
    assert meta["imu"] == {
        "body_name": "torso",
        "body_index": 1,
        "mount_sensor_to_body_xyzw": [0.0, 0.0, 0.0, 1.0],
        "orientation_frame": "world",
        "gyro_frame": "sensor",
    }
    assert meta["localization"] == {
        "body_name": "hand",
        "body_index": 2,
        "offset_body_xyz": [0.1, 0.0, 0.0],
        "position_frame": "world",
    }
    assert meta["base_velocity"] is None
    assert meta["height"] == {
        "body_name": "pelvis",
        "body_index": 0,
        "reference": "local_ground",
        "axis": "world_z",
    }
    assert meta["anchor_body_name"] == "torso"
    assert meta["derived_state_provider"] == "host_fk"
    assert meta["history_initialization"] == "repeat_first_reading"


def test_deployment_metadata_exports_only_channels_the_actor_reads(tmp_path):
    # A configured sensor the policy never reads is not listed as a host
    # requirement: an optional channel is exported only when it is in the set of
    # channels the actor reads (the exported-but-unread fix). The caller derives
    # that set from the bound clean/noisy paths; here the sets are passed directly.
    _, info = load_chain(tmp_path)
    robot = ChainRobot(info, imu_body_name="torso", imu_mount_quat=(0.0, 0.0, 0.0, 1.0))
    config = _config(
        localization=LocalizationSensor("hand", 0.02, offset=(0.1, 0.0, 0.0)),
        base_velocity=BaseVelocitySensor(0.02),
    )

    # No channels: the full-metadata view reports every configured channel.
    full = config.deployment_metadata(robot)
    assert full.localization is not None
    assert full.base_velocity is not None

    # The actor reads only localization: base velocity is dropped.
    localization_only = config.deployment_metadata(robot, channels={"localization"})
    assert localization_only.localization is not None
    assert localization_only.base_velocity is None

    # The actor reads only base velocity: localization is dropped.
    velocity_only = config.deployment_metadata(robot, channels={"base_velocity"})
    assert velocity_only.base_velocity is not None
    assert velocity_only.localization is None

    # A privileged-only actor reads no sensor channel, so nothing optional is required.
    privileged = config.deployment_metadata(robot, channels=set())
    assert privileged.localization is None and privileged.base_velocity is None

    # The IMU and anchor are always reported, filtered or not.
    assert velocity_only.imu.body_name == "torso"
    assert privileged.anchor_body_name == "torso"


@pytest.mark.parametrize("localization", [False, True])
@pytest.mark.parametrize("base_velocity", [False, True])
@pytest.mark.parametrize("height", [False, True])
def test_metadata_optional_sensors_serialize_as_plain_data(
    tmp_path, localization, base_velocity, height
):
    import json

    import yaml

    _, info = load_chain(tmp_path)
    robot = ChainRobot(info, imu_body_name="pelvis")
    root_name = info.body_names[0]
    exact = NoNoise("export test")
    config = _config(
        localization=LocalizationSensor(root_name, exact) if localization else None,
        base_velocity=BaseVelocitySensor(exact) if base_velocity else None,
        height=HeightSensor(exact) if height else None,
    )
    metadata = config.deployment_metadata(robot)
    assert (metadata.localization is not None) == localization
    assert (metadata.base_velocity is not None) == base_velocity
    assert (metadata.height is not None) == height
    if base_velocity:
        assert asdict(metadata.base_velocity) == {
            "body_name": root_name,
            "body_index": 0,
            "frame": "world",
        }
    payload = asdict(metadata)
    assert payload["w_last"] is True
    assert json.loads(json.dumps(payload)) == payload
    assert yaml.safe_load(yaml.safe_dump(payload)) == payload
