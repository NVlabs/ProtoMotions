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
"""Steering control component for locomotion tasks.

Manages target direction and speed state for steering tasks.
The target direction and speed change periodically to encourage versatile locomotion.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.utils import rotations
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class SteeringControlConfig(ControlComponentConfig):
    """Configuration for steering control component.

    Attributes:
        tar_speed_min: Minimum target speed.
        tar_speed_max: Maximum target speed.
        heading_change_steps_min: Minimum steps between heading changes.
        heading_change_steps_max: Maximum steps between heading changes.
        random_heading_probability: Probability of fully random heading vs incremental change.
        standard_heading_change: Maximum incremental heading change (radians).
        standard_speed_change: Maximum incremental speed change.
        stop_probability: Probability of setting speed to zero.
        enable_rand_facing: Enable independent random facing direction (for strafing, etc).
    """

    _target_: str = "protomotions.envs.control.steering_control.SteeringControl"

    tar_speed_min: float = 0.0
    tar_speed_max: float = 2.0
    heading_change_steps_min: int = 50
    heading_change_steps_max: int = 150
    random_heading_probability: float = 0.1
    standard_heading_change: float = 0.5  # radians
    standard_speed_change: float = 0.5
    stop_probability: float = 0.1
    enable_rand_facing: bool = True


class SteeringControl(ControlComponent):
    """Steering control component that manages target direction and speed.

    Provides target direction and speed that change periodically during training
    to encourage versatile locomotion. Exposes state via get_context() for
    observation and reward functions.

    Args:
        config: Steering control configuration.
        env: Parent environment instance.
    """

    def __init__(self, config: SteeringControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self.config: SteeringControlConfig = config

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

        # Target facing direction (2D) - can be different from tar_dir for strafing
        self._tar_face_dir = torch.zeros(
            self.env.num_envs, 2, device=self.env.device, dtype=torch.float
        )
        self._tar_face_dir[..., 0] = 1.0  # Default: forward direction

        self._tar_speed = torch.ones(
            self.env.num_envs, device=self.env.device, dtype=torch.float
        )

        # Previous root position for reward computation
        self._prev_root_pos = torch.zeros(
            self.env.num_envs, 3, device=self.env.device, dtype=torch.float
        )

    def reset(self, env_ids: Tensor):
        """Reset steering task for given environments."""
        if len(env_ids) == 0:
            return

        self._prev_root_pos[env_ids] = self.env.simulator.get_root_state().root_pos[env_ids]

        n = len(env_ids)
        device = self.env.device

        # Per-environment sampling: which envs get random heading vs incremental
        rand_probs = torch.ones(n, device=device) * self.config.random_heading_probability
        use_random = torch.bernoulli(rand_probs).bool()

        # Fully random heading and speed (for envs with use_random=True)
        rand_dir_theta = 2 * np.pi * torch.rand(n, device=device) - np.pi
        rand_tar_speed = (
            self.config.tar_speed_max - self.config.tar_speed_min
        ) * torch.rand(n, device=device) + self.config.tar_speed_min

        # Incremental change from current heading/speed (for envs with use_random=False)
        dir_delta_theta = (
            2 * self.config.standard_heading_change * torch.rand(n, device=device)
            - self.config.standard_heading_change
        )
        inc_dir_theta = (dir_delta_theta + self._tar_dir_theta[env_ids] + np.pi) % (
            2 * np.pi
        ) - np.pi

        speed_delta = (
            2 * self.config.standard_speed_change * torch.rand(n, device=device)
            - self.config.standard_speed_change
        )
        inc_tar_speed = torch.clamp(
            speed_delta + self._tar_speed[env_ids],
            min=self.config.tar_speed_min,
            max=self.config.tar_speed_max,
        )

        # Select per-environment based on use_random mask
        dir_theta = torch.where(use_random, rand_dir_theta, inc_dir_theta)
        tar_speed = torch.where(use_random, rand_tar_speed, inc_tar_speed)

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

        # Sample facing direction - independent from movement direction if enabled
        if self.config.enable_rand_facing:
            face_theta = 2 * np.pi * torch.rand(n, device=device) - np.pi
        else:
            face_theta = dir_theta  # Face same direction as movement
        tar_face_dir = torch.stack([torch.cos(face_theta), torch.sin(face_theta)], dim=-1)

        self._tar_speed[env_ids] = tar_speed * (1.0 - should_stop)
        self._tar_dir_theta[env_ids] = dir_theta
        self._tar_dir[env_ids] = tar_dir
        self._tar_face_dir[env_ids] = tar_face_dir
        self._heading_change_steps[env_ids] = (
            self.env.progress_buf[env_ids] + change_steps
        )

    def step(self):
        """Check if any environments need their heading task updated."""
        # Store previous root position before physics step
        self._prev_root_pos[:] = self.env.simulator.get_root_state().root_pos

        # Check for heading changes
        reset_task_mask = self.env.progress_buf >= self._heading_change_steps
        env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset(env_ids)

    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """No terminations from steering control."""
        reset_buf = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        terminate_buf = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        return reset_buf, terminate_buf

    def get_context(self) -> Dict[str, Any]:
        """Get steering context for observations and rewards."""
        return {
            "tar_dir": self._tar_dir,
            "tar_dir_theta": self._tar_dir_theta,
            "tar_speed": self._tar_speed,
            "tar_face_dir": self._tar_face_dir,
            "prev_root_pos": self._prev_root_pos,
        }

    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, VisualizationMarkerConfig]:
        """Create steering direction markers.
        
        Creates two arrow markers:
        - Red arrow: movement direction (tar_dir)
        - Blue arrow: facing direction (tar_face_dir)
        """
        if headless:
            return {}

        # Movement direction marker (red, like ASE)
        movement_markers = [MarkerConfig(size="regular")]
        movement_markers_cfg = VisualizationMarkerConfig(
            type="arrow", color=(0.8, 0.0, 0.0), markers=movement_markers
        )

        # Facing direction marker (blue, like ASE)
        facing_markers = [MarkerConfig(size="regular")]
        facing_markers_cfg = VisualizationMarkerConfig(
            type="arrow", color=(0.0, 0.0, 0.8), markers=facing_markers
        )

        return {
            "movement_markers": movement_markers_cfg,
            "facing_markers": facing_markers_cfg,
        }

    def get_markers_state(self) -> Dict[str, MarkerState]:
        """Get marker states for visualization."""
        if self.env.simulator.headless:
            return {}

        root_pos = self.env.simulator.get_root_state().root_pos
        heading_axis = torch.zeros_like(root_pos)
        heading_axis[..., -1] = 1.0

        # Movement direction marker position and rotation
        movement_marker_pos = root_pos.clone()
        movement_marker_pos[..., 0:2] += self._tar_dir

        movement_theta = torch.atan2(self._tar_dir[..., 1], self._tar_dir[..., 0])
        movement_rot = rotations.quat_from_angle_axis(
            movement_theta, heading_axis, True
        )

        # Facing direction marker position and rotation
        facing_marker_pos = root_pos.clone()
        facing_marker_pos[..., 0:2] += self._tar_face_dir

        facing_theta = torch.atan2(self._tar_face_dir[..., 1], self._tar_face_dir[..., 0])
        facing_rot = rotations.quat_from_angle_axis(
            facing_theta, heading_axis, True
        )

        return {
            "movement_markers": MarkerState(
                translation=movement_marker_pos.view(self.env.num_envs, -1, 3),
                orientation=movement_rot.view(self.env.num_envs, -1, 4),
            ),
            "facing_markers": MarkerState(
                translation=facing_marker_pos.view(self.env.num_envs, -1, 3),
                orientation=facing_rot.view(self.env.num_envs, -1, 4),
            ),
        }

