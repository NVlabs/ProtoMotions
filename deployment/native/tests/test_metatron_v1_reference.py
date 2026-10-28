# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline reference functions must remain usable without a live policy host."""

import numpy as np
import pytest

from tools.metatron_v1_reference import (
    _apply_steering_targets,
    _multimodal_style_features,
    _steering_target_masks,
)


@pytest.mark.parametrize("speed", [0.0, 0.5])
def test_steering_reference_is_finite_and_has_five_horizons(speed):
    pos = np.zeros((33, 3), np.float32)
    pos[16, 2] = 0.75
    pos[13, 1], pos[7, 1] = -0.1, 0.1
    rot = np.zeros((33, 4), np.float32)
    rot[:, 3] = 1
    target_pos = np.repeat(pos[None], 5, axis=0)
    target_rot = np.repeat(rot[None], 5, axis=0)
    stationary = speed == 0
    _apply_steering_targets(
        target_pos,
        target_rot,
        pos[16],
        0.0,
        0.2,
        np.array([speed, 0], np.float32),
        0.75,
        pos,
        rot,
        0.0,
        stationary_foot_positions=pos[[13, 7]] if stationary else None,
        stationary_anchor_position=pos[16] if stationary else None,
        stationary_heading=0.0 if stationary else None,
    )
    assert np.isfinite(target_pos).all()
    assert np.isfinite(target_rot).all()
    masks, poses, translation, rotation = _steering_target_masks(
        stationary=stationary,
        feet_active=not stationary,
        torso_height=True,
    )
    assert masks.shape == (5, 5, 2)
    assert poses.shape == (5,)
    assert translation.shape == rotation.shape == (5, 2)


def test_rest_style_reference_shape_and_zero_velocity():
    rot = np.zeros((33, 4), np.float32)
    rot[:, 3] = 1
    state = {
        "body_pos": np.zeros((33, 3), np.float32),
        "body_rot": rot,
        "body_vel": np.zeros((33, 3), np.float32),
        "body_ang_vel": np.zeros((33, 3), np.float32),
    }
    pose, velocity = _multimodal_style_features(
        [state] * 5, np.array([16, 13, 7, 24, 32])
    )
    assert pose.shape == (225,)
    assert velocity.shape == (150,)
    assert np.isfinite(pose).all()
    assert not velocity.any()
