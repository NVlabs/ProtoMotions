# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""estimate_body_state reconstructs world body state from readings alone; MuJoCo is the oracle."""

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
from protomotions.envs.sensors.estimation import (
    estimate_body_state,
    with_episode_odometry,
)
from protomotions.envs.sensors.measurement import SensorReadings, measure
from protomotions.envs.state import BodyState
from protomotions.tests.support.sensor_chain import (
    chain_state,
    load_chain,
    random_chain_batch,
)
from protomotions.utils.baked_fk import BakedAnchorFK

N = 5
TORSO, HAND = 1, 2
MOUNT = tuple(
    (
        torch.tensor([0.3, -0.2, 0.5, 0.8]) / torch.tensor([0.3, -0.2, 0.5, 0.8]).norm()
    ).tolist()
)
EXACT = NoNoise("estimation test")
# The full channel set, split into presence (config) and placement (mounts).
FULL_CONFIG = dict(localization=True, base_velocity=True, height=True)
FULL_MOUNTS = dict(localization_body=HAND, localization_offset=(0.05, -0.02, 0.1))


def _config(*, localization=False, base_velocity=False, height=False):
    return SensorConfig(
        joint_encoders=JointEncoderSensor(EXACT, EXACT),
        imu=IMUSensor(EXACT, EXACT),
        localization=LocalizationSensor("hand", EXACT) if localization else None,
        base_velocity=BaseVelocitySensor(EXACT) if base_velocity else None,
        height=HeightSensor(EXACT) if height else None,
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
    model, info = load_chain(tmp_path)
    qpos, qvel = random_chain_batch(np.random.default_rng(7), N)
    fk = BakedAnchorFK(
        info, anchor_idx=0, trackable_body_indices=torch.arange(info.num_bodies)
    )
    return fk, chain_state(model, qpos, qvel)


def _assert_matches_truth(est: BodyState, truth, *, positions=True, velocities=True):
    torch.testing.assert_close(
        est.rigid_body_rot.abs(), truth.rigid_body_rot.abs(), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        est.rigid_body_ang_vel, truth.rigid_body_ang_vel, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(est.dof_pos, truth.dof_pos)
    torch.testing.assert_close(est.dof_vel, truth.dof_vel)
    if positions:
        torch.testing.assert_close(
            est.rigid_body_pos, truth.rigid_body_pos, atol=1e-5, rtol=1e-5
        )
    if velocities:
        torch.testing.assert_close(
            est.rigid_body_vel, truth.rigid_body_vel, atol=1e-4, rtol=1e-4
        )


def test_root_imu_reconstruction_matches_mujoco(chain):
    fk, truth = chain
    config, mounts = _config(**FULL_CONFIG), _mounts(imu_body=0, **FULL_MOUNTS)
    ground = torch.linspace(-0.2, 0.2, N)
    est = estimate_body_state(
        measure(truth, config, mounts, ground_heights=ground), fk, mounts
    )
    _assert_matches_truth(est, truth)
    torch.testing.assert_close(est.ground_heights, ground, atol=1e-5, rtol=1e-5)


def test_mounted_torso_imu_and_displaced_localization_match_mujoco(chain):
    fk, truth = chain
    config = _config(**FULL_CONFIG)
    mounts = _mounts(imu_body=TORSO, imu_mount_quat=MOUNT, **FULL_MOUNTS)
    est = estimate_body_state(
        measure(truth, config, mounts, ground_heights=torch.zeros(N)), fk, mounts
    )
    _assert_matches_truth(est, truth)


def test_unmeasured_quantities_are_none_never_truth(chain):
    fk, truth = chain
    config = _config()
    mounts = _mounts(imu_body=TORSO, imu_mount_quat=MOUNT)
    est = estimate_body_state(measure(truth, config, mounts), fk, mounts)
    _assert_matches_truth(est, truth, positions=False, velocities=False)
    for name in (
        "rigid_body_pos",
        "rigid_body_vel",
        "ground_heights",
        "rigid_body_contacts",
        "dof_forces",
        "odom_start_xy",
        "odom_start_heading_inv",
        "odom_disp_start",
    ):
        assert getattr(est, name) is None, name


def test_ground_height_needs_both_localization_and_height(chain):
    fk, truth = chain
    config, mounts = _config(height=True), _mounts(imu_body=0)
    est = estimate_body_state(
        measure(truth, config, mounts, ground_heights=torch.zeros(N)), fk, mounts
    )
    assert est.ground_heights is None and est.rigid_body_pos is None


def test_concatenated_clean_and_noisy_batch_is_one_call(chain):
    fk, truth = chain
    config, mounts = _config(**FULL_CONFIG), _mounts(imu_body=0, **FULL_MOUNTS)
    clean = measure(truth, config, mounts, ground_heights=torch.zeros(N))
    noisy = clean.map(lambda t: t + 0.01)
    both = estimate_body_state(SensorReadings.cat([clean, noisy]), fk, mounts)
    assert both.rigid_body_pos.shape[0] == 2 * N
    separate = estimate_body_state(noisy, fk, mounts)
    torch.testing.assert_close(both.rigid_body_pos[N:], separate.rigid_body_pos)


def test_history_shaped_readings_keep_their_leading_dims(chain):
    fk, truth = chain
    config, mounts = _config(**FULL_CONFIG), _mounts(imu_body=0, **FULL_MOUNTS)
    clean = measure(truth, config, mounts, ground_heights=torch.zeros(N))
    stacked = clean.map(lambda t: torch.stack([t, t], dim=1))  # [N, 2, ...]
    est = estimate_body_state(stacked, fk, mounts)
    assert est.rigid_body_rot.shape[:2] == (N, 2)
    torch.testing.assert_close(
        est.rigid_body_pos[:, 1], truth.rigid_body_pos, atol=1e-5, rtol=1e-5
    )


def test_episode_odometry_is_anchor_displacement_in_the_start_heading_frame(chain):
    fk, truth = chain
    config, mounts = _config(**FULL_CONFIG), _mounts(imu_body=0, **FULL_MOUNTS)
    est = estimate_body_state(
        measure(truth, config, mounts, ground_heights=torch.zeros(N)), fk, mounts
    )
    anchor = 1
    start_xy = est.rigid_body_pos[:, anchor, :2] - torch.tensor([1.0, 0.5])
    yaw = torch.linspace(-1.0, 1.0, N)
    heading_inv = torch.stack(
        [torch.zeros(N), torch.zeros(N), torch.sin(-yaw / 2), torch.cos(-yaw / 2)],
        dim=-1,
    )
    with_odom = with_episode_odometry(est, anchor, start_xy, heading_inv)
    torch.testing.assert_close(with_odom.odom_start_xy, start_xy)
    torch.testing.assert_close(with_odom.odom_start_heading_inv, heading_inv)
    # Rotating the world displacement (1.0, 0.5) by -yaw gives the start-heading-frame reading.
    c, s = torch.cos(-yaw), torch.sin(-yaw)
    expected = torch.stack([c * 1.0 - s * 0.5, s * 1.0 + c * 0.5], dim=-1)
    torch.testing.assert_close(
        with_odom.odom_disp_start, expected, atol=1e-5, rtol=1e-5
    )
    assert est.odom_disp_start is None, "the input estimate is not mutated"
    torch.testing.assert_close(with_odom.rigid_body_pos, est.rigid_body_pos)


def test_each_channel_reaches_only_its_physical_dependents(chain):
    fk, truth = chain
    config = _config(**FULL_CONFIG)
    mounts = _mounts(imu_body=TORSO, imu_mount_quat=MOUNT, **FULL_MOUNTS)
    clean = measure(truth, config, mounts, ground_heights=torch.zeros(N))
    base = estimate_body_state(clean, fk, mounts)

    def perturbed(**delta):
        return estimate_body_state(
            SensorReadings(
                **{
                    f: (getattr(clean, f) + delta[f])
                    if f in delta
                    else getattr(clean, f)
                    for f in (
                        "joint_pos",
                        "joint_vel",
                        "imu_quat",
                        "imu_gyro",
                        "localization",
                        "base_velocity",
                        "height",
                    )
                }
            ),
            fk,
            mounts,
        )

    gyro = perturbed(imu_gyro=0.3)
    torch.testing.assert_close(gyro.rigid_body_rot, base.rigid_body_rot)
    torch.testing.assert_close(gyro.rigid_body_pos, base.rigid_body_pos)
    assert not torch.allclose(gyro.rigid_body_ang_vel, base.rigid_body_ang_vel)

    loc = perturbed(localization=0.2)
    torch.testing.assert_close(
        loc.rigid_body_pos - base.rigid_body_pos,
        torch.full_like(base.rigid_body_pos, 0.2),
    )
    torch.testing.assert_close(loc.rigid_body_rot, base.rigid_body_rot)
    torch.testing.assert_close(loc.rigid_body_vel, base.rigid_body_vel)

    joints = perturbed(joint_pos=0.1)
    assert not torch.allclose(
        joints.rigid_body_pos[:, HAND], base.rigid_body_pos[:, HAND]
    )
    torch.testing.assert_close(joints.dof_vel, base.dof_vel)


def test_world_body_poses_from_the_estimated_anchor_reproduce_the_estimate(chain):
    """The in-graph FK kernel and the sensor runtime agree to float rounding.

    Deployment feeds the exported graph ``noisy.dof_pos``, ``noisy.anchor_pos``
    and ``noisy.anchor_rot``; the graph's ``BakedAnchorFK.world_body_poses``
    must then reproduce the body poses the training-time sensor runtime
    published, because both are the same FK composed on the same anchor pose.
    """
    from protomotions.envs.state_names import body_state_names

    fk, truth = chain
    config = _config(**FULL_CONFIG)
    mounts = _mounts(imu_body=TORSO, imu_mount_quat=MOUNT, **FULL_MOUNTS)
    est = estimate_body_state(
        measure(truth, config, mounts, ground_heights=torch.zeros(N)), fk, mounts
    )
    anchor = 0
    names = body_state_names(est, anchor)
    pos, rot = fk.world_body_poses(est.dof_pos, names["anchor_pos"], names["anchor_rot"])
    torch.testing.assert_close(pos, est.rigid_body_pos, atol=1e-5, rtol=1e-5)
    # Same rotation up to the quaternion sign (|q1 . q2| = 1); unlike a
    # component-wise abs() this also tells a rotation from its conjugate.
    dots = (rot * est.rigid_body_rot).sum(-1).abs()
    torch.testing.assert_close(dots, torch.ones_like(dots), atol=1e-5, rtol=0)
