# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from fnmatch import fnmatch

import pytest

pytest.importorskip("newton")

from protomotions.simulator.newton.simulator import NewtonSimulator


def test_contact_sensor_patterns_match_newton_full_path_labels():
    patterns = NewtonSimulator._get_contact_sensor_body_patterns("left_ankle_roll_link")

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
    assert NewtonSimulator._get_contact_sensor_body_patterns(
        "*/left_ankle_roll_link"
    ) == ["*/left_ankle_roll_link"]
    assert NewtonSimulator._get_contact_sensor_body_patterns(
        "g1_29dof/worldbody/pelvis/left_ankle_roll_link"
    ) == ["g1_29dof/worldbody/pelvis/left_ankle_roll_link"]


def test_validate_contact_sensor_match_accepts_nonzero_matches():
    NewtonSimulator._validate_contact_sensor_match(
        "left_ankle_roll_link",
        ["left_ankle_roll_link", "*/left_ankle_roll_link"],
        matched_body_count=1,
    )


def test_validate_contact_sensor_match_raises_on_zero_matches():
    with pytest.raises(ValueError, match="matched no body labels"):
        NewtonSimulator._validate_contact_sensor_match(
            "left_ankle_roll_link",
            ["left_ankle_roll_link", "*/left_ankle_roll_link"],
            matched_body_count=0,
        )
