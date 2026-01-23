# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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
"""Termination functions for environments.

Organized into:
- base: Core termination functions (fall, height, contact, episode length)
- tracking: Motion tracking terminations (tracking error, BeyondMimic style)
- task: Task-specific terminations (path following)
"""

# Base terminations
from protomotions.envs.terminations.base import (
    check_fall_contact_term,
    check_height_term,
    check_max_length_term,
    combine_fall_termination,
    threshold_termination,
    fall_termination,
    height_termination,
    contact_termination,
)

# Tracking terminations
from protomotions.envs.terminations.tracking import (
    max_joint_err,
    tracking_error_factory,
    motion_clip_done,
    anchor_pos_error,
    anchor_pos_error_factory,
    anchor_ori_error,
    anchor_ori_error_factory,
    relative_body_pos_error,
    relative_body_pos_error_factory,
)

# Task terminations
from protomotions.envs.terminations.task import (
    check_path_distance_term,
    check_path_height_term,
)

__all__ = [
    # Base functions
    "check_fall_contact_term",
    "check_height_term",
    "check_max_length_term",
    "combine_fall_termination",
    "threshold_termination",
    "fall_termination",
    "height_termination",
    "contact_termination",
    # Tracking functions
    "max_joint_err",
    "motion_clip_done",
    "anchor_pos_error",
    "anchor_ori_error",
    "relative_body_pos_error",
    # Tracking factories
    "tracking_error_factory",
    "anchor_pos_error_factory",
    "anchor_ori_error_factory",
    "relative_body_pos_error_factory",
    # Task functions
    "check_path_distance_term",
    "check_path_height_term",
]

