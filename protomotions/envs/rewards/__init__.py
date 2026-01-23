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
"""Reward functions for environments.

Organized into:
- base: Primitive functions (MSE, norm, rotation error, power consumption)
- tracking: Motion imitation rewards (AMP/DeepMimic, BeyondMimic style)
- task: Task-specific rewards (steering, path following)
- regularization: Regularization penalties (action smoothness, limits, contacts)
"""

# Base primitives
from protomotions.envs.rewards.base import (
    mean_squared_error_exp,
    rotation_error_exp,
    power_consumption_exp,
    mean_squared_error,
    norm,
    rotation_error,
    absolute_difference_sum,
    power_consumption_sum,
    velocity_squared_sum,
)

# Tracking rewards
from protomotions.envs.rewards.tracking import (
    # Standard tracking factories
    gt_rew_factory,
    gr_rew_factory,
    gv_rew_factory,
    gav_rew_factory,
    rh_rew_factory,
    # BeyondMimic-style
    global_position_error_exp,
    global_orientation_error_exp,
    relative_body_position_error_exp,
    relative_body_orientation_error_exp,
    global_velocity_error_exp,
    global_anchor_pos_rew_factory,
    global_anchor_ori_rew_factory,
    relative_body_pos_rew_factory,
    relative_body_ori_rew_factory,
    global_body_lin_vel_rew_factory,
    global_body_ang_vel_rew_factory,
)

# Task rewards
from protomotions.envs.rewards.task import (
    heading_velocity_reward,
    heading_velocity_reward_factory,
    path_following_reward,
    path_following_reward_factory,
)

# Regularization rewards
from protomotions.envs.rewards.regularization import (
    physical_action_rate,
    action_smoothness_factory,
    action_smoothness_physical_factory,
    pow_rew_factory,
    joint_limit_violation,
    soft_pos_limit_reward,
    soft_pos_limit_rew_factory,
    contact_mismatch_sum,
    contact_match_rew_factory,
    impact_force_penalty,
    contact_force_change_rew_factory,
)

__all__ = [
    # Base primitives
    "mean_squared_error_exp",
    "rotation_error_exp",
    "power_consumption_exp",
    "mean_squared_error",
    "norm",
    "rotation_error",
    "absolute_difference_sum",
    "power_consumption_sum",
    "velocity_squared_sum",
    # Tracking factories
    "gt_rew_factory",
    "gr_rew_factory",
    "gv_rew_factory",
    "gav_rew_factory",
    "rh_rew_factory",
    # BeyondMimic functions
    "global_position_error_exp",
    "global_orientation_error_exp",
    "relative_body_position_error_exp",
    "relative_body_orientation_error_exp",
    "global_velocity_error_exp",
    # BeyondMimic factories
    "global_anchor_pos_rew_factory",
    "global_anchor_ori_rew_factory",
    "relative_body_pos_rew_factory",
    "relative_body_ori_rew_factory",
    "global_body_lin_vel_rew_factory",
    "global_body_ang_vel_rew_factory",
    # Task functions
    "heading_velocity_reward",
    "path_following_reward",
    # Task factories
    "heading_velocity_reward_factory",
    "path_following_reward_factory",
    # Regularization functions
    "physical_action_rate",
    "joint_limit_violation",
    "soft_pos_limit_reward",
    "contact_mismatch_sum",
    "impact_force_penalty",
    # Regularization factories
    "action_smoothness_factory",
    "action_smoothness_physical_factory",
    "pow_rew_factory",
    "soft_pos_limit_rew_factory",
    "contact_match_rew_factory",
    "contact_force_change_rew_factory",
]

