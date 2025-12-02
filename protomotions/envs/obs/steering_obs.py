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
"""Steering task observation component.

This module handles:
- Target direction and speed state management
- Periodic task updates (heading/speed changes)
- Computation of steering observations in the robot's local frame

The steering task provides heading and speed targets that change periodically,
and this observation component manages that state and transforms targets to local frame.
"""

import numpy as np
import torch
from torch import Tensor

from protomotions.utils import rotations
from protomotions.envs.obs.config import SteeringObsConfig


class SteeringObs:
    """Handles steering task state and observation computation.

    Manages target direction and speed, updates them periodically, and computes
    observations in the robot's local frame.

    The observation is a 3D vector: [local_dir_x, local_dir_y, tar_speed].

    Args:
        config: Configuration for steering observations and task parameters.
        env: Parent environment instance.
    """

    def __init__(self, config: SteeringObsConfig, env):
        self.config = config
        self.env = env

        self.steering_obs = torch.zeros(
            (self.env.num_envs, 3),
            device=self.env.device,
            dtype=torch.float,
        )

        # Task state buffers
        self._heading_change_steps = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.int64
        )
        self._tar_dir_theta = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.float
        )
        self._tar_dir = torch.zeros(
            self.env.num_envs, 2, device=self.env.device, dtype=torch.float
        )
        self._tar_dir[..., 0] = 1.0  # Default: forward direction

        self._tar_speed = torch.ones(
            self.env.num_envs, device=self.env.device, dtype=torch.float
        )

    @property
    def tar_dir(self) -> Tensor:
        """Current target direction vectors [num_envs, 2]."""
        return self._tar_dir

    @property
    def tar_dir_theta(self) -> Tensor:
        """Current target direction angles [num_envs]."""
        return self._tar_dir_theta

    @property
    def tar_speed(self) -> Tensor:
        """Current target speeds [num_envs]."""
        return self._tar_speed

    def reset_task(self, env_ids: Tensor):
        """Reset steering task for given environments.

        Samples new random heading and speed targets.

        Args:
            env_ids: Environment indices to reset.
        """
        if len(env_ids) == 0:
            return

        n = len(env_ids)
        device = self.env.device

        if np.random.binomial(1, self.config.random_heading_probability):
            # Fully random heading and speed
            dir_theta = 2 * np.pi * torch.rand(n, device=device) - np.pi
            tar_speed = (
                self.config.tar_speed_max - self.config.tar_speed_min
            ) * torch.rand(n, device=device) + self.config.tar_speed_min
        else:
            # Incremental change from current heading/speed
            dir_delta_theta = (
                2 * self.config.standard_heading_change * torch.rand(n, device=device)
                - self.config.standard_heading_change
            )
            dir_theta = (dir_delta_theta + self._tar_dir_theta[env_ids] + np.pi) % (
                2 * np.pi
            ) - np.pi

            speed_delta = (
                2 * self.config.standard_speed_change * torch.rand(n, device=device)
                - self.config.standard_speed_change
            )
            tar_speed = torch.clamp(
                speed_delta + self._tar_speed[env_ids],
                min=self.config.tar_speed_min,
                max=self.config.tar_speed_max,
            )

        tar_dir = torch.stack([torch.cos(dir_theta), torch.sin(dir_theta)], dim=-1)

        # Sample when to change heading next
        change_steps = torch.randint(
            low=self.config.heading_change_steps_min,
            high=self.config.heading_change_steps_max,
            size=(n,),
            device=device,
            dtype=torch.int64,
        )

        # Randomly set some targets to stop (speed=0)
        stop_probs = torch.ones(n, device=device) * self.config.stop_probability
        should_stop = torch.bernoulli(stop_probs)

        self._tar_speed[env_ids] = tar_speed * (1.0 - should_stop)
        self._tar_dir_theta[env_ids] = dir_theta
        self._tar_dir[env_ids] = tar_dir
        self._heading_change_steps[env_ids] = (
            self.env.progress_buf[env_ids] + change_steps
        )

    def check_update_task(self):
        """Check if any environments need their heading task updated."""
        reset_task_mask = self.env.progress_buf >= self._heading_change_steps
        env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_task(env_ids)

    def compute_observations(self, env_ids: Tensor):
        """Compute steering observations for given environments.

        Args:
            env_ids: Environment indices to compute observations for.
        """
        # Get root rotation for the specified envs
        root_rot = self.env.simulator.get_root_state(env_ids).root_rot
        tar_dir = self._tar_dir[env_ids]
        tar_speed = self._tar_speed[env_ids]

        obs = compute_steering_observations(root_rot, tar_dir, tar_speed)

        self.steering_obs[env_ids] = obs

    def get_obs(self):
        """Get steering observations dict."""
        return {"steering": self.steering_obs.clone()}


@torch.jit.script
def compute_steering_observations(
    root_rot: Tensor, tar_dir: Tensor, tar_speed: Tensor
) -> Tensor:
    """Compute steering observations in the robot's local frame.

    Transforms the target direction from world frame to the robot's local frame
    and concatenates with the target speed.

    Args:
        root_rot: Root orientation quaternions [num_envs, 4] (w-last format).
        tar_dir: Target direction vectors [num_envs, 2] in world frame.
        tar_speed: Target speeds [num_envs].

    Returns:
        Steering observations [num_envs, 3]: [local_dir_x, local_dir_y, tar_speed].
    """
    # Extend 2D direction to 3D (z=0)
    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)

    # Get inverse heading rotation (to transform world -> local)
    heading_rot = rotations.calc_heading_quat_inv(root_rot, True)

    # Transform target direction to local frame
    local_tar_dir = rotations.quat_rotate(heading_rot, tar_dir3d, True)
    local_tar_dir = local_tar_dir[..., 0:2]

    # Concatenate with target speed
    tar_speed = tar_speed.unsqueeze(-1)
    obs = torch.cat([local_tar_dir, tar_speed], dim=-1)

    return obs
