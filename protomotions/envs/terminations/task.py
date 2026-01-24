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
"""Task-specific termination functions.

Provides termination conditions for specific tasks:
- Path following terminations
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

