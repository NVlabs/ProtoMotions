# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from fnmatch import fnmatch

import pytest

from protomotions.simulator.newton.contact_utils import (
    get_contact_sensor_body_patterns,
    validate_contact_sensor_match,
)


def test_contact_sensor_patterns_match_newton_full_path_labels():
    patterns = get_contact_sensor_body_patterns("left_ankle_roll_link")

    assert patterns == ["left_ankle_roll_link", "*/left_ankle_roll_link"]
    assert any(
        fnmatch(
            "g1_29dof/worldbody/pelvis/left_hip_pitch_link/left_hip_roll_link/"
            "left_hip_yaw_link/left_knee_link/left_ankle_pitch_link/"
            "left_ankle_roll_link",
            pattern,
        )
        for pattern in patterns
    )


def test_contact_sensor_patterns_preserve_explicit_patterns_and_paths():
    assert get_contact_sensor_body_patterns("*/left_ankle_roll_link") == [
        "*/left_ankle_roll_link"
    ]
    assert get_contact_sensor_body_patterns(
        "g1_29dof/worldbody/pelvis/left_ankle_roll_link"
    ) == ["g1_29dof/worldbody/pelvis/left_ankle_roll_link"]


def test_validate_contact_sensor_match_accepts_nonzero_matches():
    validate_contact_sensor_match(
        "left_ankle_roll_link",
        ["left_ankle_roll_link", "*/left_ankle_roll_link"],
        matched_body_count=1,
    )


def test_validate_contact_sensor_match_raises_on_zero_matches():
    with pytest.raises(ValueError, match="matched no body labels"):
        validate_contact_sensor_match(
            "left_ankle_roll_link",
            ["left_ankle_roll_link", "*/left_ankle_roll_link"],
            matched_body_count=0,
        )
