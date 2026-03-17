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
"""Observation compute kernels for masked mimic tasks.

Pure tensor functions (kernels) for computing masked mimic observations.
Use MdpComponent in experiment configs to bind kernels to context paths:

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent
    from protomotions.envs.obs.masked_mimic import (
        compute_target_poses_only,
    )
    
    observation_components = {
        "masked_target_poses": MdpComponent(
            compute_func=compute_target_poses_only,
            dynamic_vars={
                "current_state_body_pos": EnvContext.current.rigid_body_pos,
                "current_state_body_rot": EnvContext.current.rigid_body_rot,
                "masked_mimic_ref_pos": EnvContext.masked_mimic.ref_pos,
                "masked_mimic_ref_rot": EnvContext.masked_mimic.ref_rot,
                "masked_mimic_target_bodies_masks": EnvContext.masked_mimic.target_bodies_masks,
                "conditionable_body_ids": EnvContext.some_source,  # Needs proper binding
            },
            static_params={
                "future_steps": 3,
                "include_root_relative": True,
            },
        ),
    }
"""

from typing import List, Union

import torch
from torch import Tensor

from protomotions.envs.obs.target_poses import build_sparse_target_poses
from protomotions.envs.obs.utils import select_step_indices


def compute_target_poses_only(
    current_state_body_pos: Tensor,
    current_state_body_rot: Tensor,
    masked_mimic_ref_pos: Tensor,
    masked_mimic_ref_rot: Tensor,
    masked_mimic_target_bodies_masks: Tensor,
    conditionable_body_ids: Tensor,
    future_steps: Union[int, List[int]] = None,
    include_root_relative: bool = True,
) -> Tensor:
    """Compute masked target poses (hidden bodies zeroed out).
    
    Args:
        current_state_body_pos: Current body positions [num_envs, num_bodies, 3].
        current_state_body_rot: Current body rotations [num_envs, num_bodies, 4] (w-last).
        masked_mimic_ref_pos: Reference body positions [num_envs, future_steps, num_bodies, 3].
        masked_mimic_ref_rot: Reference body rotations [num_envs, future_steps, num_bodies, 4].
        masked_mimic_target_bodies_masks: Binary visibility masks [num_envs, future_steps * num_bodies * 2].
        conditionable_body_ids: List/tensor of body indices that can be conditioned on.
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
        include_root_relative: If True, output 24 features per body (body-relative + root-relative).
            If False, output 12 features per body (body-relative only: pos delta + rot delta).
    
    Returns:
        Masked target pose observations [num_envs, features].
    """
    num_envs = current_state_body_pos.shape[0]
    num_conditionable_bodies = len(conditionable_body_ids)
    
    if future_steps is not None:
        masked_mimic_ref_pos = select_step_indices(masked_mimic_ref_pos, future_steps)
        masked_mimic_ref_rot = select_step_indices(masked_mimic_ref_rot, future_steps)
        masks_per_step = num_conditionable_bodies * 2
        full_num_steps = masked_mimic_target_bodies_masks.shape[1] // masks_per_step
        masks_reshaped = masked_mimic_target_bodies_masks.view(
            num_envs, full_num_steps, masks_per_step
        )
        masks_selected = select_step_indices(masks_reshaped, future_steps)
        masked_mimic_target_bodies_masks = masks_selected.reshape(num_envs, -1)
    
    num_future_steps_actual = masked_mimic_ref_pos.shape[1]
    
    obs = build_sparse_target_poses(
        current_state_body_pos=current_state_body_pos,
        current_state_body_rot=current_state_body_rot,
        masked_mimic_ref_pos=masked_mimic_ref_pos,
        masked_mimic_ref_rot=masked_mimic_ref_rot,
        conditionable_body_ids=conditionable_body_ids,
        w_last=True,
        include_root_relative=include_root_relative,
    )
    
    # Reshape for masking: [envs, steps, bodies, types, features]
    # With include_root_relative=True: [envs, steps, bodies, 2, 12] -> 24 per body
    # With include_root_relative=False: [envs, steps, bodies, 2, 6] -> 12 per body
    features_per_body = 24 if include_root_relative else 12
    obs = obs.view(num_envs, num_future_steps_actual, num_conditionable_bodies, 2, features_per_body // 2)
    
    mask = masked_mimic_target_bodies_masks.view(
        num_envs, num_future_steps_actual, num_conditionable_bodies, 2, 1
    )
    masked_obs = obs * mask
    
    return masked_obs.view(num_envs, -1)


def compute_target_masks_only(
    masked_mimic_target_bodies_masks: Tensor,
    conditionable_body_ids: Tensor,
    future_steps: Union[int, List[int]] = None,
) -> Tensor:
    """Return binary target masks.
    
    Args:
        masked_mimic_target_bodies_masks: Binary visibility masks [num_envs, future_steps * num_bodies * 2].
        conditionable_body_ids: List/tensor of body indices that can be conditioned on.
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
    
    Returns:
        Binary visibility masks [num_envs, selected_steps * num_bodies * 2].
    """
    num_envs = masked_mimic_target_bodies_masks.shape[0]
    num_conditionable_bodies = len(conditionable_body_ids)
    masks_per_step = num_conditionable_bodies * 2
    full_num_steps = masked_mimic_target_bodies_masks.shape[1] // masks_per_step
    
    if future_steps is None:
        return masked_mimic_target_bodies_masks
    
    masks_reshaped = masked_mimic_target_bodies_masks.view(
        num_envs, full_num_steps, masks_per_step
    )
    masks_selected = select_step_indices(masks_reshaped, future_steps)
    return masks_selected.reshape(num_envs, -1)


def compute_target_time_offsets(
    masked_mimic_time_offsets: Tensor,
    future_steps: Union[int, List[int]] = None,
) -> Tensor:
    """Return target time offsets.
    
    Args:
        masked_mimic_time_offsets: Time offsets for each future step [num_envs, future_steps].
        future_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
    
    Returns:
        Time offsets to each target frame [num_envs, selected_steps].
    """
    if future_steps is None:
        return masked_mimic_time_offsets
    return select_step_indices(masked_mimic_time_offsets, future_steps)


__all__ = [
    "compute_target_poses_only",
    "compute_target_masks_only",
    "compute_target_time_offsets",
]
