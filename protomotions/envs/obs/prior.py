# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation functions for prior model training."""

import torch
from torch import Tensor

from protomotions.envs.obs.humanoid import compute_humanoid_max_coords_observations

def compute_prior_historical_max_coords(
    historical_rigid_body_pos: Tensor,
    historical_rigid_body_rot: Tensor,
    historical_rigid_body_vel: Tensor,
    historical_rigid_body_ang_vel: Tensor,
    local_obs: bool = True,
    root_height_obs: bool = True,
    w_last: bool = True,
) -> Tensor:
    """Compute historical max_coords observations for prior dataset."""
    num_envs = historical_rigid_body_pos.shape[0]
    num_steps = historical_rigid_body_pos.shape[1]
    num_bodies = historical_rigid_body_pos.shape[2]
    device = historical_rigid_body_pos.device
    
    batch_size = num_envs * num_steps
    flat_body_pos = historical_rigid_body_pos.reshape(batch_size, num_bodies, 3)
    flat_body_rot = historical_rigid_body_rot.reshape(batch_size, num_bodies, 4)
    flat_body_vel = historical_rigid_body_vel.reshape(batch_size, num_bodies, 3)
    flat_body_ang_vel = historical_rigid_body_ang_vel.reshape(batch_size, num_bodies, 3)
    
    flat_ground_height = torch.zeros(batch_size, 1, device=device)
    flat_contacts = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
    
    flat_obs = compute_humanoid_max_coords_observations(
        body_pos=flat_body_pos,
        body_rot=flat_body_rot,
        body_vel=flat_body_vel,
        body_ang_vel=flat_body_ang_vel,
        ground_height=flat_ground_height,
        body_contacts=flat_contacts,
        local_obs=local_obs,
        root_height_obs=root_height_obs,
        observe_contacts=False,
        w_last=w_last,
    )
    
    obs_dim = flat_obs.shape[-1]
    return flat_obs.view(num_envs, num_steps * obs_dim)

