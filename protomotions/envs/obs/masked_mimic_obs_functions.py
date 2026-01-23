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
"""Observation functions for masked mimic tasks."""

import torch
from torch import Tensor

from protomotions.envs.obs.target_poses import build_sparse_target_poses
from protomotions.envs.obs.observation_component import ObservationComponentConfig


def masked_mimic_target_poses_factory(
    conditionable_body_ids: list,
    num_future_steps: int = None,
    include_root_relative: bool = True,
) -> ObservationComponentConfig:
    """Factory for masked target poses only (hidden bodies zeroed out).
    
    Args:
        conditionable_body_ids: List of body indices that can be conditioned on.
            Computed from robot_cfg.trackable_bodies_subset.
        num_future_steps: Number of future steps to use (None = use all).
        include_root_relative: If True (default), include both body-relative AND
            root-relative poses (24 features per body: pos_delta, pos_from_root,
            rot_delta, rot_in_heading_frame).
            If False, only include body-relative poses (12 features per body:
            pos_delta from current body to target, rot_delta from current to target).
            This is simpler and focuses on "move body towards target" direction.
    """
    return ObservationComponentConfig(
        function=compute_target_poses_only,
        variables={
            "current_state_body_pos": "current_state_rigid_body_pos",
            "current_state_body_rot": "current_state_rigid_body_rot",
            "masked_mimic_ref_pos": "masked_mimic_ref_pos",
            "masked_mimic_ref_rot": "masked_mimic_ref_rot",
            "masked_mimic_target_bodies_masks": "masked_mimic_target_bodies_masks",
            "conditionable_body_ids": conditionable_body_ids,
            "num_future_steps": num_future_steps,
            "include_root_relative": include_root_relative,
        },
    )


def target_masks_factory(
    conditionable_body_ids: list,
    num_future_steps: int = None,
) -> ObservationComponentConfig:
    """Factory for binary visibility masks (should NOT be normalized).
    
    Args:
        conditionable_body_ids: List of body indices that can be conditioned on.
            Computed from robot_cfg.trackable_bodies_subset.
        num_future_steps: Number of future steps to use (None = use all).
    """
    return ObservationComponentConfig(
        function=compute_target_masks_only,
        variables={
            "masked_mimic_target_bodies_masks": "masked_mimic_target_bodies_masks",
            "conditionable_body_ids": conditionable_body_ids,
            "num_future_steps": num_future_steps,
        },
    )


def target_time_offsets_factory(
    num_future_steps: int = None,
) -> ObservationComponentConfig:
    """Factory for time offsets to each target frame."""
    return ObservationComponentConfig(
        function=compute_target_time_offsets,
        variables={
            "masked_mimic_time_offsets": "masked_mimic_time_offsets",
            "num_future_steps": num_future_steps,
        },
    )
    

def compute_target_poses_only(
    current_state_body_pos: Tensor,
    current_state_body_rot: Tensor,
    masked_mimic_ref_pos: Tensor,
    masked_mimic_ref_rot: Tensor,
    masked_mimic_target_bodies_masks: Tensor,
    conditionable_body_ids: Tensor,
    num_future_steps: int = None,
    include_root_relative: bool = True,
) -> Tensor:
    """Compute masked target poses (hidden bodies zeroed out).
    
    Args:
        include_root_relative: If True, output 24 features per body (body-relative + root-relative).
            If False, output 12 features per body (body-relative only: pos delta + rot delta).
    """
    num_envs = current_state_body_pos.shape[0]
    num_conditionable_bodies = len(conditionable_body_ids)
    
    if num_future_steps is not None:
        masked_mimic_ref_pos = masked_mimic_ref_pos[:, :num_future_steps]
        masked_mimic_ref_rot = masked_mimic_ref_rot[:, :num_future_steps]
        masks_per_step = num_conditionable_bodies * 2
        full_num_steps = masked_mimic_target_bodies_masks.shape[1] // masks_per_step
        masks_reshaped = masked_mimic_target_bodies_masks.view(
            num_envs, full_num_steps, masks_per_step
        )
        masked_mimic_target_bodies_masks = masks_reshaped[:, :num_future_steps, :].reshape(
            num_envs, -1
        )
    
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
    num_future_steps: int = None,
) -> Tensor:
    """Return binary target masks."""
    num_envs = masked_mimic_target_bodies_masks.shape[0]
    num_conditionable_bodies = len(conditionable_body_ids)
    masks_per_step = num_conditionable_bodies * 2
    
    if num_future_steps is not None:
        full_num_steps = masked_mimic_target_bodies_masks.shape[1] // masks_per_step
        masks_reshaped = masked_mimic_target_bodies_masks.view(
            num_envs, full_num_steps, masks_per_step
        )
        masks = masks_reshaped[:, :num_future_steps, :]
    else:
        full_num_steps = masked_mimic_target_bodies_masks.shape[1] // masks_per_step
        masks = masked_mimic_target_bodies_masks.view(num_envs, full_num_steps, masks_per_step)
    
    return masks.reshape(num_envs, -1).float()


def compute_target_time_offsets(
    masked_mimic_time_offsets: Tensor,
    num_future_steps: int = None,
) -> Tensor:
    """Return target time offsets."""
    if num_future_steps is not None:
        return masked_mimic_time_offsets[:, :num_future_steps]
    return masked_mimic_time_offsets

