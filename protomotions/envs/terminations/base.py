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
"""Base termination checking utilities.

Provides core functions for checking common termination conditions:
- Fall detection (contact + height)
- Episode length limits
- Generic threshold termination
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


# ==============================================================================
# Functions for dynamic TerminationComponentConfig system
# ==============================================================================


def threshold_termination(value: Tensor, threshold: float, greater_than: bool = True) -> Tensor:
    """Terminate when value exceeds (or falls below) a threshold.
    
    Generic termination function for use with TerminationComponentConfig.
    
    Args:
        value: Values to check [num_envs] or [num_envs, ...].
        threshold: Threshold value.
        greater_than: If True, terminate when value > threshold.
                     If False, terminate when value < threshold.
    
    Returns:
        Boolean tensor [num_envs] indicating which environments should terminate.
    """
    # If value has multiple dimensions, reduce to per-env scalar
    if value.dim() > 1:
        value = value.mean(dim=tuple(range(1, value.dim())))
    
    if greater_than:
        return value > threshold
    else:
        return value < threshold


def fall_termination(
    rigid_body_pos: Tensor,
    rigid_body_contacts: Tensor,
    ground_heights: Tensor,
    termination_height: float,
    non_termination_contact_body_ids: Tensor,
    progress_buf: Tensor,
) -> Tensor:
    """Combined fall termination check for use with TerminationComponentConfig.
    
    Checks for falls using both unwanted body contacts and height violations.
    An agent has fallen if both conditions are met: unexpected contact and
    body parts below height threshold.
    
    This is a wrapper around combine_fall_termination that accepts ground_heights
    as a separate parameter for terrain adjustment.
    
    Args:
        rigid_body_pos: Body positions [num_envs, num_bodies, 3].
        rigid_body_contacts: Binary contact flags [num_envs, num_bodies].
        ground_heights: Ground height at root position [num_envs].
        termination_height: Height threshold for fall detection.
        non_termination_contact_body_ids: IDs of bodies allowed to contact ground.
        progress_buf: Episode progress counter [num_envs].
    
    Returns:
        Boolean tensor [num_envs] indicating which environments have fallen.
    """
    num_bodies = rigid_body_pos.shape[1]
    device = rigid_body_pos.device
    
    # Create per-body termination heights adjusted for terrain
    termination_heights = torch.full(
        (num_bodies,), termination_height, device=device
    )
    adjusted_termination_heights = termination_heights + ground_heights.unsqueeze(-1)
    
    return combine_fall_termination(
        contact_binary_flags=rigid_body_contacts,
        rigid_body_pos=rigid_body_pos,
        termination_heights=adjusted_termination_heights,
        non_termination_body_ids=non_termination_contact_body_ids,
        progress_buf=progress_buf,
    )


def height_termination(
    rigid_body_pos: Tensor,
    ground_heights: Tensor,
    termination_height: float,
    non_termination_body_ids: Tensor,
) -> Tensor:
    """Height-only termination check for use with TerminationComponentConfig.
    
    Checks if any body parts are below termination height (adjusted for terrain).
    
    Args:
        rigid_body_pos: Body positions [num_envs, num_bodies, 3].
        ground_heights: Ground height at root position [num_envs].
        termination_height: Height threshold for termination.
        non_termination_body_ids: IDs of bodies to exclude from check.
    
    Returns:
        Boolean tensor [num_envs] indicating which environments violated height constraint.
    """
    num_bodies = rigid_body_pos.shape[1]
    device = rigid_body_pos.device
    
    # Create per-body termination heights adjusted for terrain
    termination_heights = torch.full(
        (num_bodies,), termination_height, device=device
    )
    adjusted_termination_heights = termination_heights + ground_heights.unsqueeze(-1)
    
    return check_height_term(
        rigid_body_pos=rigid_body_pos,
        termination_heights=adjusted_termination_heights,
        non_termination_body_ids=non_termination_body_ids,
    )


def contact_termination(
    rigid_body_contacts: Tensor,
    non_termination_contact_body_ids: Tensor,
    progress_buf: Tensor,
) -> Tensor:
    """Contact-only termination check for use with TerminationComponentConfig.
    
    Checks if any non-allowed body parts are in contact with ground.
    
    Args:
        rigid_body_contacts: Binary contact flags [num_envs, num_bodies].
        non_termination_contact_body_ids: IDs of bodies allowed to contact ground.
        progress_buf: Episode progress counter [num_envs].
    
    Returns:
        Boolean tensor [num_envs] indicating which environments have unwanted contacts.
    """
    return check_fall_contact_term(
        contact_binary_flags=rigid_body_contacts,
        non_termination_body_ids=non_termination_contact_body_ids,
        progress_buf=progress_buf,
    )

