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
"""Observation noise utilities for domain randomization.

Provides functions to apply configurable noise to robot state observations
for sim-to-real transfer training.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from protomotions.simulator.base_simulator.config import ObservationNoiseDomainRandomizationConfig
from protomotions.simulator.base_simulator.simulator_state import RobotState


@dataclass
class NoisyObservations:
    """Container for noisy observation tensors.
    
    When noise is not configured, these point to the same tensors as clean observations.
    """
    # Whole-body
    rigid_body_pos: Tensor
    rigid_body_rot: Tensor
    rigid_body_vel: Tensor
    rigid_body_ang_vel: Tensor
    
    # DOF
    dof_pos: Tensor
    dof_vel: Tensor
    
    # Root
    root_rot: Tensor
    root_local_ang_vel: Tensor
    
    # Anchor
    anchor_rot: Tensor
    anchor_local_ang_vel: Tensor
    
    # Ground height
    ground_heights: Tensor


def apply_observation_noise(
    obs_noise_cfg: Optional[ObservationNoiseDomainRandomizationConfig],
    robot_state: RobotState,
    anchor_idx: int,
    root_local_ang_vel: Tensor,
    anchor_rot: Tensor,
    anchor_local_ang_vel: Tensor,
    ground_heights: Tensor,
) -> NoisyObservations:
    """Apply observation noise to robot state.
    
    Noise is applied independently to each component from clean values,
    not cascaded (e.g., root noise is applied to clean root, not noisy whole-body).
    
    Args:
        obs_noise_cfg: Observation noise configuration. If None, returns clean tensors.
        robot_state: Clean robot state from simulator.
        anchor_idx: Index of the anchor body.
        root_local_ang_vel: Clean root local angular velocity [num_envs, 3].
        anchor_rot: Clean anchor rotation [num_envs, 4].
        anchor_local_ang_vel: Clean anchor local angular velocity [num_envs, 3].
        ground_heights: Clean ground heights [num_envs, 1].
    
    Returns:
        NoisyObservations containing potentially noisy tensors.
        When obs_noise_cfg is None, returns references to the clean tensors.
    """
    if obs_noise_cfg is None:
        # No noise - return clean tensors
        return NoisyObservations(
            rigid_body_pos=robot_state.rigid_body_pos,
            rigid_body_rot=robot_state.rigid_body_rot,
            rigid_body_vel=robot_state.rigid_body_vel,
            rigid_body_ang_vel=robot_state.rigid_body_ang_vel,
            dof_pos=robot_state.dof_pos,
            dof_vel=robot_state.dof_vel,
            root_rot=robot_state.root_rot,
            root_local_ang_vel=root_local_ang_vel,
            anchor_rot=anchor_rot,
            anchor_local_ang_vel=anchor_local_ang_vel,
            ground_heights=ground_heights,
        )
    
    noisy_rigid_body_pos = _add_noise(robot_state.rigid_body_pos, obs_noise_cfg.body_pos_noise)
    noisy_rigid_body_rot = _add_quaternion_noise(robot_state.rigid_body_rot, obs_noise_cfg.body_rot_noise)
    noisy_rigid_body_vel = _add_noise(robot_state.rigid_body_vel, obs_noise_cfg.body_vel_noise)
    noisy_rigid_body_ang_vel = _add_noise(robot_state.rigid_body_ang_vel, obs_noise_cfg.body_ang_vel_noise)
    noisy_dof_pos = _add_noise(robot_state.dof_pos, obs_noise_cfg.dof_pos_noise)
    noisy_dof_vel = _add_noise(robot_state.dof_vel, obs_noise_cfg.dof_vel_noise)
    noisy_root_rot = _add_quaternion_noise(robot_state.root_rot, obs_noise_cfg.root_rot_noise)
    noisy_root_local_ang_vel = _add_noise(root_local_ang_vel, obs_noise_cfg.root_ang_vel_noise)
    noisy_anchor_rot = _add_quaternion_noise(anchor_rot, obs_noise_cfg.anchor_rot_noise)
    noisy_anchor_local_ang_vel = _add_noise(anchor_local_ang_vel, obs_noise_cfg.anchor_ang_vel_noise)
    noisy_ground_heights = _add_noise(ground_heights, obs_noise_cfg.ground_height_noise)
    
    return NoisyObservations(
        rigid_body_pos=noisy_rigid_body_pos,
        rigid_body_rot=noisy_rigid_body_rot,
        rigid_body_vel=noisy_rigid_body_vel,
        rigid_body_ang_vel=noisy_rigid_body_ang_vel,
        dof_pos=noisy_dof_pos,
        dof_vel=noisy_dof_vel,
        root_rot=noisy_root_rot,
        root_local_ang_vel=noisy_root_local_ang_vel,
        anchor_rot=noisy_anchor_rot,
        anchor_local_ang_vel=noisy_anchor_local_ang_vel,
        ground_heights=noisy_ground_heights,
    )


def _add_noise(tensor: Tensor, noise_scale: float) -> Tensor:
    """Add uniform noise to a tensor."""
    if noise_scale > 0.0:
        sampled_noise = (torch.rand_like(tensor) * 2.0 - 1.0) * noise_scale
        return tensor + sampled_noise
    return tensor


def _add_quaternion_noise(quat: Tensor, noise_scale: float) -> Tensor:
    """Add uniform noise to quaternions and re-normalize."""
    if noise_scale > 0.0:
        sampled_noise = (torch.rand_like(quat) * 2.0 - 1.0) * noise_scale
        noisy_quat = quat + sampled_noise
        return noisy_quat / torch.norm(noisy_quat, dim=-1, keepdim=True)
    return quat
