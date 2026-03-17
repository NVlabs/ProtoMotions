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
"""Termination compute kernels for environments.

Pure tensor functions (kernels) for computing terminations.
Use MdpComponent in experiment configs to bind kernels to context paths.

Organized into:
- base: Core termination functions (fall, height, contact, episode length)
- tracking: Motion tracking terminations (tracking error, BeyondMimic style)
- task: Task-specific terminations (path following)
"""

# Typed context views
from protomotions.envs.context_views import EnvContext

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

# Tracking termination kernels
from protomotions.envs.terminations.tracking import (
    compute_tracking_error,
    compute_anchor_pos_error_term,
    compute_anchor_ori_error_term,
    compute_relative_body_pos_error_term,
    compute_anchor_height_error_term,
    motion_clip_done,
    # Value functions (for evaluation metrics)
    mean_body_pos_error,
    max_body_pos_error,
    mean_body_rot_error,
    anchor_pos_error_value,
    anchor_ori_error_value,
    anchor_height_error_value,
    relative_body_pos_max_error,
)

# Task terminations
from protomotions.envs.terminations.task import (
    check_path_distance_term,
    check_path_height_term,
    check_steering_velocity_error,
)

__all__ = [
    # Typed context
    "EnvContext",
    # Base functions
    "check_fall_contact_term",
    "check_height_term",
    "check_max_length_term",
    "combine_fall_termination",
    "threshold_termination",
    "fall_termination",
    "height_termination",
    "contact_termination",
    # Tracking termination kernels
    "compute_tracking_error",
    "compute_anchor_pos_error_term",
    "compute_anchor_ori_error_term",
    "compute_relative_body_pos_error_term",
    "compute_anchor_height_error_term",
    "motion_clip_done",
    # Value functions
    "mean_body_pos_error",
    "max_body_pos_error",
    "mean_body_rot_error",
    "anchor_pos_error_value",
    "anchor_ori_error_value",
    "anchor_height_error_value",
    "relative_body_pos_max_error",
    # Task functions
    "check_path_distance_term",
    "check_path_height_term",
    "check_steering_velocity_error",
]
