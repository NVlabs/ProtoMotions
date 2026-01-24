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
"""Path follower control component for navigation tasks.

Manages path generation and provides target positions for path following tasks.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.utils.path_generator import PathGenerator, PathGeneratorConfig
from protomotions.components.pose_lib import build_body_ids_tensor
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class PathFollowerControlConfig(ControlComponentConfig):
    """Configuration for path follower control component.

    Attributes:
        num_traj_samples: Number of future waypoint samples to provide.
        traj_sample_timestep: Time interval between waypoint samples.
        path_generator: Path generation parameters.
        enable_path_termination: Whether to terminate on path deviation.
        fail_dist: Maximum horizontal distance from path before termination.
        fail_height_dist: Maximum height deviation from path before termination.
    """

    _target_: str = "protomotions.envs.control.path_follower_control.PathFollowerControl"

    num_traj_samples: int = 10
    traj_sample_timestep: float = 0.3
    path_generator: PathGeneratorConfig = field(default_factory=PathGeneratorConfig)
    enable_path_termination: bool = True
    fail_dist: float = 1.0
    fail_height_dist: float = 0.5


class PathFollowerControl(ControlComponent):
    """Path follower control component that manages path generation.

    Generates paths and provides target positions and trajectory samples
    for observations and rewards.

    Args:
        config: Path follower control configuration.
        env: Parent environment instance.
    """

    def __init__(self, config: PathFollowerControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self.config: PathFollowerControlConfig = config

        # Get head body ID for tracking
        self._head_body_id = build_body_ids_tensor(
            self.env.robot_config.kinematic_info.body_names,
            self.env.robot_config.common_naming_to_robot_body_names["head_body_name"],
            self.env.device,
        ).item()

        # Build path generator
        episode_dur = self.env.config.max_episode_length * self.env.dt
        self.path_generator = PathGenerator(
            self.config.path_generator,
            self.env.device,
            self.env.num_envs,
            episode_dur,
            self.config.path_generator.height_conditioned,
        )

    @property
    def height_conditioned(self) -> bool:
        """Whether paths include height conditioning."""
        return self.config.path_generator.height_conditioned

    @property
    def head_body_id(self) -> int:
        """Index of the head body for tracking."""
        return self._head_body_id

    def reset(self, env_ids: Tensor):
        """Reset path generator for given environments."""
        if len(env_ids) == 0:
            return

        # Get current root position
        root_pos = self.env.simulator.get_root_state(env_ids).root_pos

        # Build head position from root x,y and approximate head height
        head_position = root_pos.clone()
        height_below_head = self.env.terrain.get_ground_heights(head_position).squeeze(1)
        head_position[..., 2] -= height_below_head

        # Reset path starting from ground-relative head position
        self.path_generator.reset(env_ids, head_position)

    def step(self):
        """No per-step updates needed for path following."""
        pass

    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """Check for path deviation terminations."""
        reset_buf = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        terminate_buf = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)

        if not self.config.enable_path_termination:
            return reset_buf, terminate_buf

        # Get current head position
        bodies_positions = self.env.simulator.get_bodies_state().rigid_body_pos.clone()
        ground_min = torch.min(bodies_positions, dim=1).values[:, 2].view(-1, 1)
        bodies_positions[..., 2] -= ground_min

        head_pos = bodies_positions[:, self._head_body_id, :]

        # Get target position
        env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        tar_pos = self.calc_target_pos(env_ids)

        # Check horizontal distance from path
        from protomotions.envs.terminations import (
            check_path_distance_term,
            check_path_height_term,
        )

        tar_fail = check_path_distance_term(
            head_pos,
            tar_pos,
            self.config.fail_dist,
            self.env.progress_buf,
            min_progress=10,
        )
        terminate_buf = terminate_buf | tar_fail

        # Check height deviation (if height-conditioned)
        if self.config.path_generator.height_conditioned:
            tar_height_fail = check_path_height_term(
                head_pos,
                tar_pos,
                self.config.fail_height_dist,
                self.env.progress_buf,
                min_progress=10,
            )
            terminate_buf = terminate_buf | tar_height_fail

        return reset_buf, terminate_buf

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

    def get_head_position(self, env_ids: Tensor = None) -> Tensor:
        """Get ground-relative head position.

        Args:
            env_ids: Environment indices. If None, gets for all envs.

        Returns:
            Head positions [len(env_ids), 3].
        """
        if env_ids is None:
            env_ids = torch.arange(
                self.env.num_envs, device=self.env.device, dtype=torch.long
            )

        current_state = self.env.simulator.get_robot_state(env_ids)
        bodies_positions = current_state.rigid_body_pos
        ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]
        head_position = bodies_positions[:, self._head_body_id, :].clone()
        head_position[..., 2] -= ground_below_head

        return head_position

    def get_context(self) -> Dict[str, Any]:
        """Get path context for observations and rewards."""
        env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)

        return {
            "tar_pos": self.calc_target_pos(env_ids),
            "head_pos": self.get_head_position(env_ids),
            "traj_samples": self.fetch_path_samples(env_ids),
            "height_conditioned": self.height_conditioned,
            "head_body_id": self._head_body_id,
        }

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, VisualizationMarkerConfig]:
        """Create path waypoint markers."""
        if headless:
            return {}

        path_markers = [MarkerConfig(size="regular") for _ in range(self.config.num_traj_samples)]
        path_markers_cfg = VisualizationMarkerConfig(
            type="sphere", color=(1.0, 0.0, 0.0), markers=path_markers
        )
        return {"path_markers": path_markers_cfg}

    def get_markers_state(self) -> Dict[str, MarkerState]:
        """Get marker states for visualization."""
        if self.env.simulator.headless:
            return {}

        traj_samples = self.fetch_path_samples().clone()
        if not self.height_conditioned:
            traj_samples[..., 2] = 0.8  # Set height to 0.8m when not height conditioned

        ground_below_marker = self.env.terrain.get_ground_heights(
            traj_samples[..., :2].view(-1, 2)
        ).view(traj_samples.shape[:-1])
        traj_samples[..., 2] += ground_below_marker

        traj_samples = traj_samples.view(self.env.num_envs, -1, 3)
        return {
            "path_markers": MarkerState(
                translation=traj_samples,
                orientation=torch.zeros(
                    self.env.num_envs, traj_samples.shape[1], 4, device=self.env.device
                ),
            )
        }

