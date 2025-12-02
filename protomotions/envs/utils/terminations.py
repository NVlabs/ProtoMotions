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
"""Reusable reset checking utilities for environments.

This module provides modular functions for checking various termination conditions
in reinforcement learning environments. These functions can be composed together
to create environment-specific reset logic.
"""

import torch
from torch import Tensor


@torch.jit.script
def check_fall_contact_term(
    contact_binary_flags: Tensor,
    non_termination_body_ids: Tensor,
    progress_buf: Tensor,
) -> Tensor:
    """Check if agent has fallen based on unwanted body contacts.

    An agent is considered fallen if any non-allowed body part is in contact
    with the ground (e.g., torso, head touching ground when only feet should).

    Args:
        contact_binary_flags: Binary contact flags [num_envs, num_bodies]
        non_termination_body_ids: IDs of bodies that are allowed to contact ground
        progress_buf: Episode progress counter [num_envs]

    Returns:
        Boolean tensor [num_envs] indicating which environments have fallen
    """
    # Create mask for all bodies, then zero out allowed contact bodies
    masked_contacts = contact_binary_flags.clone()
    masked_contacts[:, non_termination_body_ids] = False

    # Any contact on non-allowed bodies indicates a fall
    fall_contact = torch.any(masked_contacts, dim=-1)

    # First timestep can sometimes still have nonzero contact forces
    # so only check after first couple of steps
    fall_contact = fall_contact & (progress_buf > 1)

    return fall_contact


@torch.jit.script
def check_height_term(
    rigid_body_pos: Tensor,
    termination_heights: Tensor,
    non_termination_body_ids: Tensor,
) -> Tensor:
    """Check if any body parts are below termination height.

    Args:
        rigid_body_pos: Body positions [num_envs, num_bodies, 3]
        termination_heights: Height thresholds per body [num_bodies]
        non_termination_body_ids: IDs of bodies to exclude from check

    Returns:
        Boolean tensor [num_envs] indicating which environments violated height constraint
    """
    body_height = rigid_body_pos[..., 2]  # [num_envs, num_bodies]
    fall_height = body_height < termination_heights  # [num_envs, num_bodies]

    # Don't check height for allowed contact bodies (e.g., feet)
    fall_height[:, non_termination_body_ids] = False

    # Any body below threshold indicates termination
    has_height_violation = torch.any(fall_height, dim=-1)

    return has_height_violation


@torch.jit.script
def check_max_length_term(
    progress_buf: Tensor,
    max_episode_length: float,
) -> Tensor:
    """Check if episode has exceeded maximum length.

    Args:
        progress_buf: Episode progress counter [num_envs]
        max_episode_length: Maximum allowed episode length

    Returns:
        Boolean tensor [num_envs] indicating which episodes reached max length
    """
    return progress_buf >= max_episode_length - 1


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


@torch.jit.script
def combine_fall_termination(
    contact_binary_flags: Tensor,
    rigid_body_pos: Tensor,
    termination_heights: Tensor,
    non_termination_body_ids: Tensor,
    progress_buf: Tensor,
) -> Tensor:
    """Combined fall termination check (contact + height).

    Convenience function that combines contact and height checks for standard
    fall detection (agent fallen if both contact and height conditions met).

    Args:
        contact_binary_flags: Binary contact flags [num_envs, num_bodies]
        rigid_body_pos: Body positions [num_envs, num_bodies, 3]
        termination_heights: Height thresholds per body [num_bodies]
        non_termination_body_ids: IDs of bodies allowed to contact ground
        progress_buf: Episode progress counter [num_envs]

    Returns:
        Boolean tensor [num_envs] indicating which environments have fallen
    """
    fall_contact = check_fall_contact_term(
        contact_binary_flags, non_termination_body_ids, progress_buf
    )
    fall_height = check_height_term(
        rigid_body_pos, termination_heights, non_termination_body_ids
    )

    # Agent has fallen if BOTH contact and height conditions are met
    has_fallen = fall_contact & fall_height

    return has_fallen
