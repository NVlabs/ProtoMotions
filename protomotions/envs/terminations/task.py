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
"""Task-specific termination functions.

Provides termination conditions for specific tasks:
- Path following terminations
- Steering terminations
"""

import torch
from torch import Tensor


@torch.jit.script
def check_path_distance_term(
    head_pos: Tensor,
    target_pos: Tensor,
    fail_dist: float,
    progress_buf: Tensor,
    min_progress: int = 10,
) -> Tensor:
    """Check if agent deviated too far from target path position.

    Args:
        head_pos: Agent head positions [num_envs, 3]
        target_pos: Target path positions [num_envs, 3]
        fail_dist: Maximum allowed distance from target
        progress_buf: Episode progress counter [num_envs]
        min_progress: Minimum steps before checking (avoid early termination)

    Returns:
        Boolean tensor [num_envs] indicating which agents deviated too far
    """
    tar_delta = target_pos - head_pos
    tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
    tar_fail = tar_dist_sq > fail_dist * fail_dist

    # Only check after minimum progress to avoid early termination
    tar_fail = tar_fail & (progress_buf > min_progress)

    return tar_fail


@torch.jit.script
def check_path_height_term(
    head_pos: Tensor,
    target_pos: Tensor,
    fail_height_dist: float,
    progress_buf: Tensor,
    min_progress: int = 10,
) -> Tensor:
    """Check if agent height deviated too far from target path height.

    Args:
        head_pos: Agent head positions [num_envs, 3]
        target_pos: Target path positions [num_envs, 3]
        fail_height_dist: Maximum allowed height deviation
        progress_buf: Episode progress counter [num_envs]
        min_progress: Minimum steps before checking

    Returns:
        Boolean tensor [num_envs] indicating which agents deviated in height
    """
    tar_height = target_pos[..., 2]
    height_delta = tar_height - head_pos[..., 2]
    tar_height_dist_sq = height_delta * height_delta
    tar_height_fail = tar_height_dist_sq > fail_height_dist * fail_height_dist

    # Only check after minimum progress
    tar_height_fail = tar_height_fail & (progress_buf > min_progress)

    return tar_height_fail


# ==============================================================================
# Steering Terminations / Evaluations
# ==============================================================================


@torch.jit.script
def check_steering_velocity_error(
    root_pos: Tensor,
    prev_root_pos: Tensor,
    tar_dir: Tensor,
    tar_speed: Tensor,
    dt: float,
    speed_tolerance: float,
    direction_tolerance: float,
) -> Tensor:
    """Check if agent velocity deviates from target direction/speed.

    Args:
        root_pos: Current root positions [num_envs, 3].
        prev_root_pos: Previous root positions [num_envs, 3].
        tar_dir: Target direction (unit vector) [num_envs, 2].
        tar_speed: Target speed [num_envs].
        dt: Simulation timestep.
        speed_tolerance: Acceptable speed difference (m/s).
        direction_tolerance: Acceptable direction difference (dot product, 1.0 = perfect).

    Returns:
        Boolean tensor [num_envs] indicating which agents have unacceptable velocity.
    """
    delta_pos = root_pos - prev_root_pos
    vel_xy = delta_pos[:, :2] / dt
    speed = torch.norm(vel_xy, dim=-1)

    speed_err = torch.abs(speed - tar_speed)
    speed_fail = speed_err > speed_tolerance

    vel_dir = vel_xy / (speed.unsqueeze(-1) + 1e-6)
    dir_dot = (vel_dir * tar_dir).sum(dim=-1)
    dir_fail = (dir_dot < direction_tolerance) & (speed > 0.1)

    return speed_fail | dir_fail
