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
"""Task-specific reward functions.

Provides reward functions for specific tasks:
- Steering/locomotion rewards
- Path following rewards
"""

import torch
from torch import Tensor
from typing import TYPE_CHECKING

from protomotions.utils.rotations import calc_heading_quat, quat_rotate

if TYPE_CHECKING:
    from protomotions.envs.base_env.config import RewardComponentConfig


# =============================================================================
# Steering Rewards
# =============================================================================


def heading_velocity_reward(
    root_pos: Tensor,
    prev_root_pos: Tensor,
    root_rot: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
    tar_face_dir: Tensor,
    dt: float,
) -> Tensor:
    """Reward for moving in target direction at target speed while facing that direction.

    Computes weighted combination of:
    - Direction reward: exponential penalty on velocity error and tangent velocity
    - Facing reward: alignment between robot heading and target direction

    Args:
        root_pos: Current root position [num_envs, 3]
        prev_root_pos: Previous root position [num_envs, 3]
        root_rot: Root orientation quaternions [num_envs, 4]
        tar_dir: Target movement direction [num_envs, 2]
        tar_speed: Target speed [num_envs]
        tar_face_dir: Target facing direction [num_envs, 2] (can differ from tar_dir)
        dt: Simulation timestep

    Returns:
        Reward [num_envs] in range [0, 1]
    """

    vel_err_scale = 0.25
    tangent_err_w = 0.1

    dir_reward_w = 0.7
    facing_reward_w = 0.3

    # Compute velocity in target direction
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    # Compute tangent (perpendicular) velocity
    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel
    tangent_vel_err = torch.sum(torch.square(tangent_vel), dim=-1)

    # Direction reward: penalize velocity error and tangent movement
    tar_vel_err = tar_speed - tar_dir_speed
    dir_reward = torch.exp(
        -vel_err_scale * (tar_vel_err * tar_vel_err + tangent_err_w * tangent_vel_err)
    )

    # Zero reward for moving backwards
    speed_mask = tar_dir_speed <= 0
    dir_reward[speed_mask] = 0

    # Facing reward: robot should face the target facing direction
    heading_rot = calc_heading_quat(root_rot, w_last=True)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir, w_last=True)

    facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward

    return reward


def heading_velocity_reward_factory(weight: float = 1.0) -> "RewardComponentConfig":
    """Factory for heading velocity reward (steering task).

    Args:
        weight: Reward weight multiplier.

    Returns:
        RewardComponentConfig for heading velocity reward.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig

    return RewardComponentConfig(
        function=heading_velocity_reward,
        variables={
            "root_pos": "current_state_root_pos",
            "prev_root_pos": "prev_root_pos",  # From SteeringControl.get_context()
            "root_rot": "current_state_root_rot",
            "tar_dir": "tar_dir",  # From SteeringControl.get_context()
            "tar_speed": "tar_speed",  # From SteeringControl.get_context()
            "tar_face_dir": "tar_face_dir",  # From SteeringControl.get_context()
            "dt": "dt",
        },
        weight=weight,
    )


# =============================================================================
# Path Following Rewards
# =============================================================================


def path_following_reward(
    head_pos: Tensor,
    tar_pos: Tensor,
    height_conditioned: bool,
    pos_err_scale: float = 2.0,
    height_err_scale: float = 10.0,
) -> Tensor:
    """Reward for following a path (staying close to target position).

    Computes exponential reward based on:
    - Horizontal distance to target position
    - Optionally: vertical distance to target position

    Args:
        head_pos: Current head position [num_envs, 3] (ground-relative)
        tar_pos: Target position from path [num_envs, 3] (ground-relative)
        height_conditioned: Whether to include height in reward
        pos_err_scale: Coefficient for position error
        height_err_scale: Coefficient for height error

    Returns:
        Reward [num_envs] in range [0, 1]
    """
    pos_diff = tar_pos[..., 0:2] - head_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    height_diff = tar_pos[..., 2] - head_pos[..., 2]
    height_err = height_diff * height_diff

    pos_reward = torch.exp(-pos_err_scale * pos_err)
    height_reward = torch.exp(-height_err_scale * height_err)

    if height_conditioned:
        # Multiplicative reward ensures both terms are properly met.
        reward = pos_reward * height_reward
    else:
        reward = pos_reward

    return reward


def path_following_reward_factory(weight: float = 1.0) -> "RewardComponentConfig":
    """Factory for path following reward.

    Args:
        weight: Reward weight multiplier.

    Returns:
        RewardComponentConfig for path following reward.
    """
    from protomotions.envs.base_env.config import RewardComponentConfig

    return RewardComponentConfig(
        function=path_following_reward,
        variables={
            "head_pos": "head_pos",  # From PathFollowerControl.get_context()
            "tar_pos": "tar_pos",  # From PathFollowerControl.get_context()
            "height_conditioned": "height_conditioned",  # From PathFollowerControl.get_context()
        },
        weight=weight,
    )

