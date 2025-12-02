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
"""Path following observation component.

This module handles:
- Path generator management
- Future waypoint sampling along the path
- Computation of path observations in the robot's local frame

The path following task provides future waypoints that the agent should follow,
and this observation component manages the path generator and transforms waypoints to local frame.
"""

import torch
from torch import Tensor

from protomotions.utils import rotations
from protomotions.envs.obs.config import PathObsConfig
from protomotions.envs.path_follower.path_generator import PathGenerator


class PathObs:
    """Handles path generation and observation computation.

    Manages the path generator, samples future waypoints, and computes
    observations in the robot's local frame.

    The observation is a flattened vector of local waypoint offsets:
    [dx1, dy1, (dz1), dx2, dy2, (dz2), ...] where dz is included if height_conditioned.

    Args:
        config: Configuration for path observations and path generator.
        env: Parent environment instance.
    """

    def __init__(self, config: PathObsConfig, env):
        self.config = config
        self.env = env

        # Observation buffer
        obs_per_sample = 3 if self.config.path_generator.height_conditioned else 2
        obs_size = self.config.num_traj_samples * obs_per_sample
        self.path_obs = torch.zeros(
            (self.env.num_envs, obs_size),
            device=self.env.device,
            dtype=torch.float,
        )

        # Path generator (built later when head_body_id is known)
        self.path_generator = None
        self._head_body_id = None

    @property
    def num_traj_samples(self) -> int:
        """Number of trajectory samples."""
        return self.config.num_traj_samples

    @property
    def traj_sample_timestep(self) -> float:
        """Time interval between trajectory samples."""
        return self.config.traj_sample_timestep

    @property
    def height_conditioned(self) -> bool:
        """Whether observations include height."""
        return self.config.path_generator.height_conditioned

    def build_path_generator(self, head_body_id: int):
        """Build the path generator.

        Args:
            head_body_id: Index of the head body for tracking.
        """
        self._head_body_id = head_body_id
        episode_dur = self.env.config.max_episode_length * self.env.dt
        self.path_generator = PathGenerator(
            self.config.path_generator,
            self.env.device,
            self.env.num_envs,
            episode_dur,
            self.config.path_generator.height_conditioned,
        )

    def reset_path(self, env_ids: Tensor, head_position: Tensor):
        """Reset path generator for given environments.

        Args:
            env_ids: Environment indices to reset.
            head_position: Head positions to start paths from [len(env_ids), 3].
        """
        if self.path_generator is not None:
            self.path_generator.reset(env_ids, head_position)

    def fetch_path_samples(self, env_ids: Tensor = None) -> Tensor:
        """Fetch future waypoint samples along the path.

        Args:
            env_ids: Environment indices to sample for. If None, samples for all envs.

        Returns:
            Trajectory samples [len(env_ids), num_traj_samples, 3].
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.env.num_envs, device=self.env.device, dtype=torch.long
            )

        timestep_beg = self.env.progress_buf[env_ids] * self.env.dt
        timesteps = torch.arange(
            self.config.num_traj_samples, device=self.env.device, dtype=torch.float
        )
        timesteps = timesteps * self.config.traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self.path_generator.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps.flatten()
        )
        traj_samples = torch.reshape(
            traj_samples_flat,
            shape=(
                env_ids.shape[0],
                self.config.num_traj_samples,
                traj_samples_flat.shape[-1],
            ),
        )

        return traj_samples

    def calc_target_pos(self, env_ids: Tensor = None) -> Tensor:
        """Calculate target position at current time.

        Args:
            env_ids: Environment indices. If None, calculates for all envs.

        Returns:
            Target positions [len(env_ids), 3].
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.env.num_envs, device=self.env.device, dtype=torch.long
            )

        time = self.env.progress_buf[env_ids] * self.env.dt
        return self.path_generator.calc_pos(env_ids, time)

    def compute_observations(self, env_ids: Tensor):
        """Compute path observations for given environments.

        Args:
            env_ids: Environment indices to compute observations for.
        """
        current_state = self.env.simulator.get_robot_state(env_ids)
        root_rot = self.env.simulator.get_root_state(env_ids).root_rot
        ground_below_head = torch.min(current_state.rigid_body_pos, dim=1).values[
            ..., 2
        ]
        head_position = current_state.rigid_body_pos[:, self._head_body_id, :].clone()

        traj_samples = self.fetch_path_samples(env_ids)

        # Make head position ground-relative
        head_position[..., 2] -= ground_below_head

        obs = compute_path_observations(
            root_rot,
            head_position,
            traj_samples,
            self.config.path_generator.height_conditioned,
        )

        # Lazy init obs buffer
        if self.path_obs is None:
            self.path_obs = torch.zeros(
                (self.env.num_envs, obs.shape[-1]),
                device=self.env.device,
                dtype=torch.float,
            )
        self.path_obs[env_ids] = obs

    def get_obs(self):
        """Get path observations dict."""
        return {"path": self.path_obs.clone()}


@torch.jit.script
def compute_path_observations(
    root_rot: Tensor,
    head_states: Tensor,
    traj_samples: Tensor,
    height_conditioned: bool,
) -> Tensor:
    """Compute path observations in the robot's local frame.

    Transforms the future waypoints from world frame to the robot's local frame.

    Args:
        root_rot: Root orientation quaternions [num_envs, 4] (w-last format).
        head_states: Head positions [num_envs, 3] in ground-relative frame.
        traj_samples: Future waypoint positions [num_envs, num_samples, 3].
        height_conditioned: Whether to include height in observations.

    Returns:
        Path observations [num_envs, num_samples * (2 or 3)].
    """
    heading_rot = rotations.calc_heading_quat_inv(root_rot, True)

    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(-2),
        (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]),
    )
    heading_rot_exp = torch.reshape(
        heading_rot_exp,
        (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]),
    )

    traj_samples_delta = traj_samples - head_states.unsqueeze(-2)

    traj_samples_delta_flat = torch.reshape(
        traj_samples_delta,
        (
            traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
            traj_samples_delta.shape[2],
        ),
    )

    local_traj_pos = rotations.quat_rotate(
        heading_rot_exp, traj_samples_delta_flat, True
    )
    if not height_conditioned:
        local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(
        local_traj_pos,
        (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]),
    )
    return obs
