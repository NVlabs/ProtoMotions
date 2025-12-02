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
"""Path following task environment for navigation.

This module implements a path following task where agents must follow predefined
trajectories. The agent receives observations of future waypoints along the path
and is rewarded for staying close to the path and making forward progress.

Key Classes:
    - PathFollowing: Path following task environment

Key Features:
    - Multiple future waypoint observations (via PathObs component)
    - Path generation (straight lines, curves, complex trajectories)
    - Visual path markers
    - Configurable path complexity
    - Reward via reward component system
"""

import torch

from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.obs.path_obs import PathObs
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)
from protomotions.components.pose_lib import build_body_ids_tensor
from protomotions.envs.path_follower.config import PathFollowerEnvConfig
from protomotions.envs.utils.terminations import (
    combine_fall_termination,
    check_max_length_term,
    check_path_distance_term,
    check_path_height_term,
)


class PathFollowing(BaseEnv):
    """Path following task environment for humanoid locomotion.

    Trains agents to follow predefined paths in the environment. The agent receives
    observations of future waypoints along the path and is rewarded for staying close
    to the path and making forward progress. Paths can be straight lines, curves, or
    complex trajectories.

    The environment samples multiple future positions along the path and provides them
    as observations to the agent. Visual markers show the path waypoints during
    visualization. This task is useful for training navigation behaviors that can be
    combined with style rewards (e.g., via AMP).

    Args:
        config: Path following configuration including path generator parameters.
        device: PyTorch device for computations.
        *args: Additional arguments passed to BaseEnv.
        **kwargs: Additional keyword arguments passed to BaseEnv.

    Attributes:
        path_obs_cb: Path observation component (manages path generator and observations).
        head_body_id: Index of the head body for tracking agent position.

    Example:
        >>> config = PathFollowerEnvConfig()
        >>> env = PathFollowing(config, robot_config, simulator_config, device)
        >>> obs, _ = env.reset()
        >>> next_obs, rewards, dones, info = env.step(actions)
    """

    config: PathFollowerEnvConfig

    def __init__(
        self,
        config: PathFollowerEnvConfig,
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

        # Get head body ID for tracking
        self.head_body_id = build_body_ids_tensor(
            self.robot_config.kinematic_info.body_names,
            self.robot_config.common_naming_to_robot_body_names["head_body_name"],
            self.device,
        ).item()

        # Initialize path observation component
        self.path_obs_cb = PathObs(self.config.path_obs, self)
        self.path_obs_cb.build_path_generator(self.head_body_id)

    def create_visualization_markers(self, headless: bool):
        visualization_markers = super().create_visualization_markers(headless)

        if headless:
            return visualization_markers

        num_samples = self.config.path_obs.num_traj_samples
        path_markers = [MarkerConfig(size="regular") for _ in range(num_samples)]
        path_markers_cfg = VisualizationMarkerConfig(
            type="sphere", color=(1.0, 0.0, 0.0), markers=path_markers
        )
        visualization_markers["path_markers"] = path_markers_cfg

        return visualization_markers

    def get_markers_state(self):
        if self.simulator.headless:
            return {}

        markers_state = super().get_markers_state()

        traj_samples = self.path_obs_cb.fetch_path_samples().clone()
        if not self.path_obs_cb.height_conditioned:
            traj_samples[..., 2] = 0.8  # Set height to 0.8m when not height conditioned

        ground_below_marker = self.terrain.get_ground_heights(
            traj_samples[..., :2].view(-1, 2)
        ).view(traj_samples.shape[:-1])
        traj_samples[..., 2] += ground_below_marker

        traj_samples = traj_samples.view(self.num_envs, -1, 3)
        markers_state["path_markers"] = MarkerState(
            translation=traj_samples,
            orientation=torch.zeros(
                self.num_envs, traj_samples.shape[1], 4, device=self.device
            ),
        )

        return markers_state

    ###############################################################
    # Handle resets
    ###############################################################
    def _reset_path_generator(self, env_ids, root_pos: torch.Tensor):
        """Reset path generator with ground-relative head position.

        Args:
            env_ids: Environment indices being reset.
            root_pos: Root positions from reset state [len(env_ids), 3].
        """
        # Build head position from root x,y and approximate head height
        head_position = root_pos.clone()
        height_below_head = self.terrain.get_ground_heights(head_position).squeeze(1)
        head_position[..., 2] -= height_below_head

        # Reset path starting from ground-relative head position
        self.path_obs_cb.reset_path(env_ids, head_position)

    def compute_default_reset_state(self, env_ids, sample_flat: bool = False):
        """Reset environments to default state and initialize path."""
        new_states, new_object_states = super().compute_default_reset_state(
            env_ids, sample_flat
        )
        self._reset_path_generator(env_ids, new_states.root_pos)
        return new_states, new_object_states

    def compute_ref_reset_state(
        self, env_ids, motion_ids, motion_times, sample_flat: bool = False
    ):
        """Reset environments from reference motion and initialize path."""
        new_states, new_object_states = super().compute_ref_reset_state(
            env_ids, motion_ids, motion_times, sample_flat
        )
        self._reset_path_generator(env_ids, new_states.root_pos)
        return new_states, new_object_states

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # Compute path observations (callback handles everything internally)
        self.path_obs_cb.compute_observations(env_ids)

    def get_obs(self):
        obs = super().get_obs()
        obs.update(self.path_obs_cb.get_obs())
        return obs

    def _get_reward_context(self):
        """Extend reward context with path-specific variables."""
        context = super()._get_reward_context()

        # Get current robot state
        current_state = self.simulator.get_robot_state()
        bodies_positions = current_state.rigid_body_pos
        head_position = bodies_positions[:, self.head_body_id, :].clone()

        # Get ground-relative head position
        ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]
        head_position[..., 2] -= ground_below_head

        # Get target position from path
        tar_pos = self.path_obs_cb.calc_target_pos()

        # Add path-specific context for reward computation
        context["head_pos"] = head_position
        context["tar_pos"] = tar_pos
        context["height_conditioned"] = self.path_obs_cb.height_conditioned

        return context

    def check_resets_and_terminations(self):
        """Check reset and termination conditions for path following.

        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors
        """
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_obs_cb.calc_target_pos(env_ids)

        bodies_positions = self.simulator.get_bodies_state().rigid_body_pos

        # Adjust body positions relative to ground
        bodies_positions = bodies_positions.clone()
        bodies_positions[..., 2] -= (
            torch.min(bodies_positions, dim=1).values[:, 2].view(-1, 1)
        )

        # Get binary contact flags from simulator
        all_body_contacts = self.simulator.get_binary_body_contacts(
            threshold=0.01
        ).rigid_body_contacts

        terminated = torch.zeros_like(self.reset_buf, dtype=torch.bool)

        # Check for falls (contact + height conditions)
        if self.config.enable_height_termination:
            ground_heights = self.terrain.get_ground_heights(
                bodies_positions[:, self.head_body_id, :2]
            )
            adjusted_termination_heights = self.termination_heights + ground_heights

            has_fallen = combine_fall_termination(
                all_body_contacts,
                bodies_positions,
                adjusted_termination_heights,
                self.non_termination_contact_body_ids,
                self.progress_buf,
            )
            terminated = terminated | has_fallen

        # Check path-specific termination conditions
        path_cfg = self.config.path_obs
        if path_cfg.enable_path_termination:
            head_pos = bodies_positions[..., self.head_body_id, :]

            # Check distance from path
            tar_fail = check_path_distance_term(
                head_pos,
                tar_pos,
                path_cfg.fail_dist,
                self.progress_buf,
                min_progress=10,
            )
            terminated = terminated | tar_fail

        # Check height deviation from path (if height-conditioned and path termination enabled)
        if (
            path_cfg.enable_path_termination
            and path_cfg.path_generator.height_conditioned
        ):
            head_pos = bodies_positions[..., self.head_body_id, :]
            tar_height_fail = check_path_height_term(
                head_pos,
                tar_pos,
                path_cfg.fail_height_dist,
                self.progress_buf,
                min_progress=10,
            )
            terminated = terminated | tar_height_fail

        # Check max episode length
        max_length_reached = check_max_length_term(
            self.progress_buf, self.config.max_episode_length
        )
        reset_buf = max_length_reached | terminated

        return reset_buf, terminated
