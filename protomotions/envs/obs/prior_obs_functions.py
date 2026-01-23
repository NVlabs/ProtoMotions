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
"""Observation functions for prior model training."""

import torch
from torch import Tensor

from protomotions.envs.obs.humanoid import compute_humanoid_max_coords_observations
from protomotions.envs.obs.observation_component import ObservationComponentConfig


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


def prior_historical_obs_factory(
    local_obs: bool = True,
    root_height_obs: bool = True,
) -> ObservationComponentConfig:
    """Factory for prior historical max_coords observation."""
    return ObservationComponentConfig(
        function=compute_prior_historical_max_coords,
        variables={
            "historical_rigid_body_pos": "historical_rigid_body_pos",
            "historical_rigid_body_rot": "historical_rigid_body_rot",
            "historical_rigid_body_vel": "historical_rigid_body_vel",
            "historical_rigid_body_ang_vel": "historical_rigid_body_ang_vel",
            "local_obs": local_obs,
            "root_height_obs": root_height_obs,
            "w_last": True,
        },
    )

