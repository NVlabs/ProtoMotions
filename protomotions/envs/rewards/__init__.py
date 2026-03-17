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
"""Reward compute kernels for environments.

Pure tensor functions (kernels) for computing rewards.
Use MdpComponent in experiment configs to bind kernels to context paths.

Organized into:
- base: Primitive functions (MSE, norm, rotation error, power consumption)
- tracking: Motion imitation rewards (AMP/DeepMimic, BeyondMimic style)
- task: Task-specific rewards (steering, path following)
- regularization: Regularization penalties (action smoothness, limits, contacts)
"""

# Typed context views
from protomotions.envs.context_views import EnvContext

# Base primitives
from protomotions.envs.rewards.base import (
    mean_squared_error_exp,
    rotation_error_exp,
    power_consumption_exp,
    mean_squared_error,
    norm,
    delta_norm,
    delta_logmeanexp,
    rotation_error,
    absolute_difference_sum,
    power_consumption_sum,
    velocity_squared_sum,
)

# Tracking reward kernels
from protomotions.envs.rewards.tracking import (
    # Standard tracking kernels
    compute_gt_rew,
    compute_gr_rew,
    compute_gv_rew,
    compute_gav_rew,
    compute_rh_rew,
    # BeyondMimic-style kernels
    compute_global_position_error_exp,
    compute_global_anchor_pos_rew,
    compute_global_orientation_error_exp,
    compute_global_anchor_ori_rew,
    compute_relative_body_pos_rew,
    compute_relative_body_ori_rew,
    compute_global_body_lin_vel_rew,
    compute_global_body_ang_vel_rew,
)

# Task reward kernels
from protomotions.envs.rewards.task import (
    compute_heading_velocity_rew,
    compute_path_following_rew,
)

# Regularization reward kernels
from protomotions.envs.rewards.regularization import (
    compute_action_smoothness,
    compute_action_smoothness_logmeanexp,
    compute_pow_rew,
    compute_soft_pos_limit_rew,
    compute_contact_match_rew,
    compute_contact_force_change_rew,
    # Helper functions
    joint_limit_violation,
    contact_mismatch_sum,
    impact_force_penalty,
)

__all__ = [
    # Typed context
    "EnvContext",
    # Base primitives
    "mean_squared_error_exp",
    "rotation_error_exp",
    "power_consumption_exp",
    "mean_squared_error",
    "norm",
    "delta_norm",
    "delta_logmeanexp",
    "rotation_error",
    "absolute_difference_sum",
    "power_consumption_sum",
    "velocity_squared_sum",
    # Tracking reward kernels
    "compute_gt_rew",
    "compute_gr_rew",
    "compute_gv_rew",
    "compute_gav_rew",
    "compute_rh_rew",
    # BeyondMimic-style kernels
    "compute_global_position_error_exp",
    "compute_global_anchor_pos_rew",
    "compute_global_orientation_error_exp",
    "compute_global_anchor_ori_rew",
    "compute_relative_body_pos_rew",
    "compute_relative_body_ori_rew",
    "compute_global_body_lin_vel_rew",
    "compute_global_body_ang_vel_rew",
    # Task reward kernels
    "compute_heading_velocity_rew",
    "compute_path_following_rew",
    # Regularization reward kernels
    "compute_action_smoothness",
    "compute_action_smoothness_logmeanexp",
    "compute_pow_rew",
    "compute_soft_pos_limit_rew",
    "compute_contact_match_rew",
    "compute_contact_force_change_rew",
    # Regularization helper functions
    "joint_limit_violation",
    "contact_mismatch_sum",
    "impact_force_penalty",
]
