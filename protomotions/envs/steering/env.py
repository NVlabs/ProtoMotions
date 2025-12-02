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
"""Steering task environment for locomotion control.

This module implements a steering task where the agent must walk in a target direction
at a target speed. The target direction and speed change periodically to encourage
versatile locomotion capabilities.

Key Classes:
    - Steering: Steering task environment

Key Features:
    - Variable target speeds (including stopping)
    - Periodic heading changes with random variations
    - Visual markers for target direction
    - Rewards for velocity and heading matching
"""

import torch
from protomotions.utils import rotations
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.obs.steering_obs import SteeringObs
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)


class Steering(BaseEnv):
    """Steering task environment for humanoid locomotion.

    Trains agents to walk in a target direction at a target speed. The agent
    receives observations of the target heading and speed, and is rewarded for
    moving in the correct direction at the correct velocity. The target direction
    and speed change periodically during training to encourage versatile locomotion.

    Key features:
    - Variable target speeds (can include stopping)
    - Periodic heading changes with random variations
    - Visual markers showing target direction
    - Rewards for matching target velocity and heading

    Args:
        config: Steering environment configuration.
        device: PyTorch device for computations.
        *args: Additional arguments passed to BaseEnv.
        **kwargs: Additional keyword arguments passed to BaseEnv.

    Attributes:
        steering_obs_cb: Steering observation component that manages task state.

    Example:
        >>> config = SteeringEnvConfig()
        >>> env = Steering(config, robot_config, simulator_config, device)
        >>> obs, _ = env.reset()
        >>> next_obs, rewards, dones, info = env.step(actions)
    """

    def __init__(
        self,
        config,
        robot_config,
        device: torch.device,
        terrain,
        simulator,
        scene_lib,
        motion_lib,
        *args,
        **kwargs,
    ):
        super().__init__(
            config=config,
            robot_config=robot_config,
            device=device,
            terrain=terrain,
            simulator=simulator,
            scene_lib=scene_lib,
            motion_lib=motion_lib,
            *args,
            **kwargs,
        )

        # Initialize steering observation component (manages all task state)
        self.steering_obs_cb = SteeringObs(self.config.steering_obs, self)

        # Previous root position for reward computation
        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )

    def create_visualization_markers(self, headless: bool):
        visualization_markers = super().create_visualization_markers(headless)

        if headless:
            return visualization_markers

        steering_markers = [MarkerConfig(size="regular")]
        steering_markers_cfg = VisualizationMarkerConfig(
            type="arrow", color=(0.0, 1.0, 1.0), markers=steering_markers
        )
        visualization_markers["steering_markers"] = steering_markers_cfg

        return visualization_markers

    def get_markers_state(self):
        if self.simulator.headless:
            return {}

        markers_state = super().get_markers_state()

        marker_root_pos = self.simulator.get_root_state().root_pos.clone()
        marker_root_pos[..., 0:2] += self.steering_obs_cb.tar_dir

        heading_axis = torch.zeros_like(marker_root_pos)
        heading_axis[..., -1] = 1.0
        marker_rot = rotations.quat_from_angle_axis(
            self.steering_obs_cb.tar_dir_theta, heading_axis, True
        )
        markers_state["steering_markers"] = MarkerState(
            translation=marker_root_pos.view(self.num_envs, -1, 3),
            orientation=marker_rot.view(self.num_envs, -1, 4),
        )

        return markers_state

    def reset(self, env_ids=None, sample_flat=False, disable_motion_resample=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        obs, info = super().reset(
            env_ids, sample_flat, disable_motion_resample=disable_motion_resample
        )
        if len(env_ids) > 0:
            self.steering_obs_cb.reset_task(env_ids)
        return obs, info

    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # Compute steering observations (callback handles everything internally)
        self.steering_obs_cb.compute_observations(env_ids)

    def get_obs(self):
        obs = super().get_obs()
        obs.update(self.steering_obs_cb.get_obs())
        return obs

    def _get_reward_context(self):
        """Extend reward context with steering-specific variables."""
        context = super()._get_reward_context()

        # Add steering-specific context for reward computation
        context["tar_dir"] = self.steering_obs_cb.tar_dir
        context["tar_speed"] = self.steering_obs_cb.tar_speed
        context["prev_root_pos"] = self._prev_root_pos

        return context

    def post_physics_step(self):
        """Update environment state after physics simulation step."""
        # Check if heading task needs updating (before observations are computed)
        self.steering_obs_cb.check_update_task()
        super().post_physics_step()
        # Update prev_root_pos for next step's reward computation (after reward computed)
        self._prev_root_pos[:] = self.simulator.get_root_state().root_pos
