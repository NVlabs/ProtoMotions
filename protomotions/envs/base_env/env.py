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
"""Base environment implementation for reinforcement learning.

This module provides the foundational environment class for all RL tasks. It integrates
the simulator, handles robot state management, computes observations and rewards, manages
episode resets, and coordinates with terrain and scene systems.

Key Classes:
    - BaseEnv: Core environment class that all tasks inherit from

Key Features:
    - Multi-simulator support (IsaacGym, IsaacLab, Genesis)
    - Terrain integration for complex ground surfaces
    - Scene management for object interaction
    - Motion library integration for reference motions
    - Modular observation components

## BaseEnv

| Member | Type | Why Kept |
|--------|------|----------|
| `config` | `EnvConfig` | Core config, used everywhere |
| `robot_config` | `RobotConfig` | Core config, used everywhere |
| `device` | `torch.device` | Required for tensor creation |
| `terrain` | `Terrain` | Core dependency for terrain queries |
| `scene_lib` | `SceneLib` | Core dependency for scene/object handling |
| `motion_lib` | `MotionLib` | Core dependency for reference motions |
| `simulator` | `Simulator` | Core dependency for physics |
| `num_envs` | `int` | Frequently accessed, avoiding repeated `simulator.num_envs` |
| `max_episode_length` | `int` | Mutable - modified by agent for curriculum learning |
| `dt` | `float` | Frequently accessed, avoiding repeated `simulator.dt` |
| `rew_buf` | `Tensor` | Mutable buffer - accumulates rewards each step |
| `reset_buf` | `Tensor` | Mutable buffer - tracks which envs need reset |
| `progress_buf` | `Tensor` | Mutable buffer - tracks episode progress |
| `terminate_buf` | `Tensor` | Mutable buffer - tracks terminations |
| `extras` | `dict` | Mutable - collects per-step logging data |
| `respawn_root_offset` | `Tensor` | Mutable state - tracks spawn position offsets |
| `skip_height_correction` | `bool` | Performance optimization flag (read-only after init) |
| `motion_manager` | `MotionManager` | Core component for motion sampling |
| `motion_manager_disable_resample` | `bool` | Mutable flag - controlled by evaluator |
| `self_obs_cb` | `HumanoidObs` | Observation component |
| `terrain_obs_cb` | `TerrainObs` | Observation component |
| `scene_obs_cb` | `SceneObs` | Observation component |

"""

from functools import cached_property
from typing import Optional, TYPE_CHECKING, Tuple, Dict, Any, Union, List

import numpy as np
import torch
from protomotions.utils.hydra_replacement import get_class
from protomotions.utils import torch_utils

from protomotions.simulator.base_simulator.simulator import Simulator
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    ObjectState,
    ResetState,
)
from protomotions.envs.utils.terminations import (
    combine_fall_termination,
    check_max_length_term,
)
from protomotions.components.terrains.terrain import Terrain
from protomotions.envs.obs.humanoid_obs import HumanoidObs
from protomotions.envs.obs.scene_obs import SceneObs
from protomotions.envs.obs.terrain_obs import TerrainObs
from protomotions.envs.base_env.config import EnvConfig

from protomotions.components.pose_lib import build_body_ids_tensor

from protomotions.robot_configs.base import RobotConfig

if TYPE_CHECKING:
    from protomotions.components.scene_lib import SceneLib
    from protomotions.components.motion_lib import MotionLib


class BaseEnv:
    """Base class for all reinforcement learning environments.

    Provides core functionality for robot simulation including:
    - Simulator integration (IsaacGym, IsaacLab, Genesis)
    - Terrain management
    - Scene and object handling
    - Motion library integration
    - Observation and reward computation
    - Episode management and resets

    Subclasses should implement task-specific reward functions and
    observation spaces by overriding compute_reward() and compute_observations().

    Attributes:
        simulator: The physics simulator instance.
        num_envs: Number of parallel environments.
        device: PyTorch device for computations.
        terrain: Terrain instance for complex ground surfaces.
        scene_lib: Library of object scenes for interaction tasks.
        motion_lib: Library of reference motions for imitation tasks.

    Example:
        >>> config = SteeringEnvConfig()
        >>> robot_config = G1Config()
        >>> env = Steering(config, robot_config, simulator_config, device)
        >>> obs, _ = env.reset()
        >>> next_obs, rewards, dones, info = env.step(actions)
    """

    def __init__(
        self,
        config: EnvConfig,
        robot_config: RobotConfig,
        device: torch.device,
        terrain: "Terrain",
        simulator: Simulator,
        scene_lib: "SceneLib",
        motion_lib: "MotionLib",
        *args,
        **kwargs,
    ):
        """Initialize BaseEnv.

        Args:
            config: Environment configuration
            robot_config: Robot configuration
            device: Device for computation
            terrain: Pre-created Terrain object (always provided, can be None for visualizers)
            simulator: Pre-created Simulator shell (not yet initialized, will be initialized by env)
            scene_lib: Pre-created SceneLib (always provided, empty if no scenes)
            motion_lib: Pre-created MotionLib (always provided, empty if no motions)
            *args: Additional arguments
            **kwargs: Additional keyword arguments
        """
        self.config = config
        self.robot_config = robot_config
        self.device = device
        self.terrain = terrain  # Always provided (required)
        self.scene_lib = scene_lib  # Always provided (empty if no scenes)
        self.motion_lib = motion_lib  # Always provided (empty if no motions)
        self.simulator = simulator  # Always provided (shell, will be initialized)
        self.num_envs = simulator.num_envs

        self.max_episode_length = self.config.max_episode_length

        # Buffers
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.terminate_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )

        self.respawn_root_offset = torch.zeros(
            self.num_envs, 3, dtype=torch.float, device=self.device
        )

        # Cache terrain height correction flag (optimization for flat terrain)
        self.skip_height_correction = (
            self.config.skip_correct_terrain_height_on_flat and self.terrain.is_flat()
        )

        # Initialize simulator with visualization markers and complete setup
        self.initialize_simulator()

    def initialize_simulator(self):
        """Initialize simulator with task-specific visualization markers.

        Called at the end of __init__ to finalize simulator setup after visualization
        markers have been created (potentially by child env class override).
        """
        # Apply contact smoothing to motion_lib if configured (Env's responsibility)
        if (
            self.motion_lib.num_motions() > 0
            and self.config.ref_contact_smooth_window > 0
        ):
            self.motion_lib.smooth_contacts(self.config.ref_contact_smooth_window)

        # Pass visualization markers to simulator and trigger simulation creation
        # (inline creation instead of storing as member)
        visualization_markers = self.create_visualization_markers(
            self.simulator.headless
        )
        self.simulator._initialize_with_markers(visualization_markers)

        self.dt = self.simulator.dt

        # Create motion manager if motion_lib has motions
        if self.motion_lib.num_motions() > 0:
            self.create_motion_manager()
        else:
            self.motion_manager = None

        # Create observation components (need simulator.dt)
        self.self_obs_cb = HumanoidObs(self.config.humanoid_obs, self)
        self.terrain_obs_cb = TerrainObs(self.terrain.config, self)
        # Always create scene_obs_cb (it checks if scenes exist internally)
        self.scene_obs_cb = SceneObs(self.config.scene_obs, self)

    ###############################################################
    # Getters
    ###############################################################
    def is_simulation_running(self):
        """Check if the physics simulation is running.

        Returns:
            Boolean indicating simulation state
        """
        return self.simulator.is_simulation_running()

    def get_obs(self):
        """Gather observations from all components.

        Returns:
            Dictionary of observation tensors from humanoid, terrain, and scene components
        """
        obs = self.self_obs_cb.get_obs()
        terrain_obs = self.terrain_obs_cb.get_obs()
        obs.update(terrain_obs)
        if self.scene_lib.num_scenes() > 0 and self.config.scene_obs.enabled:
            scene_obs = self.scene_obs_cb.get_obs()
            obs.update(scene_obs)
        return obs

    def get_action_size(self):
        """Get the dimensionality of the action space.

        Returns:
            Number of action dimensions
        """
        return self.simulator.num_act

    ###############################################################
    # Cached Properties (derived values computed once on first access)
    ###############################################################
    @cached_property
    def contact_body_ids(self) -> torch.Tensor:
        """Body indices for contact sensing."""
        return build_body_ids_tensor(
            self.robot_config.kinematic_info.body_names,
            self.robot_config.contact_bodies,
            self.device,
        )

    @cached_property
    def non_termination_contact_body_ids(self) -> torch.Tensor:
        """Body indices that don't trigger termination on contact."""
        body_names = self.robot_config.kinematic_info.body_names
        if self.robot_config.non_termination_contact_bodies == "all":
            return build_body_ids_tensor(body_names, body_names, self.device)
        else:
            return build_body_ids_tensor(
                body_names,
                self.robot_config.non_termination_contact_bodies,
                self.device,
            )

    @cached_property
    def termination_heights(self) -> torch.Tensor:
        """Per-body termination height thresholds."""
        num_bodies = self.robot_config.kinematic_info.num_bodies
        termination_heights = np.array([self.config.termination_height] * num_bodies)
        return torch_utils.to_torch(termination_heights, device=self.device)

    @cached_property
    def default_reset_state(self) -> ResetState:
        """Default robot reset state from simulator."""
        return self.simulator.get_default_robot_reset_state()

    @cached_property
    def default_object_state(self) -> ObjectState:
        """Default object state (empty if no scenes)."""
        return self.scene_lib.get_default_object_state(self.device)

    def update_respawn_root_offset_by_env_ids(
        self,
        env_ids,
        ref_state: Optional[RobotState] = None,
        sample_flat: bool = False,
    ) -> torch.Tensor:
        """
        Samples a new starting position for the environment.
        And obtains the root translation offset relative to the reference state.

        This method considers both scene and terrain requirements.

        When a scene is required for obj interaction,
        the character is spawned relative to the scene's position.

        For environments without a scene, a random valid coordinate is sampled,
        and non-negative vertical offset is added based on terrain height.

        ref terrain.py, when motion require scene terrain is always flat.

        """

        respawn_offset = torch.zeros((len(env_ids), 3), device=self.device)

        # Get boolean masks for scene vs non-scene envs
        scene_mask, non_scene_mask = self.get_scene_non_scene_mask(env_ids)

        if scene_mask.any():
            scene_pos = self.scene_lib.get_scene_positions(self.terrain, self.device)
            # Use mask for respawn_offset (local), env_ids[scene_mask] for scene_pos (global)
            respawn_offset[scene_mask, :2] = scene_pos[env_ids[scene_mask], :2]

        if non_scene_mask.any():
            num_non_scene = non_scene_mask.sum().item()
            respawn_position_xy = self.terrain.sample_valid_locations(
                num_envs=num_non_scene, sample_flat=sample_flat
            )

            if ref_state is None:
                ref_root = torch.zeros((num_non_scene, 2), device=self.device)
            else:
                ref_root = ref_state.root_pos[non_scene_mask, :2]
            respawn_offset[non_scene_mask, :2] = respawn_position_xy - ref_root

            # Calculate terrain height for non-scene envs only (skip if terrain is flat and config skip flag is enabled)
            if not self.skip_height_correction:
                if ref_state is not None:
                    # Get rigid body positions for non-scene envs
                    rigid_body_pos = ref_state.rigid_body_pos[non_scene_mask].clone()
                    # Apply respawn offset to get spawned positions
                    rigid_body_pos_spawned = rigid_body_pos + respawn_offset[
                        non_scene_mask
                    ].unsqueeze(1)
                    # (num_non_scene_envs, num_bodies, 3)
                else:
                    # Rigid body positions are just the respawn offset (just root body loc)
                    rigid_body_pos_spawned = respawn_offset[non_scene_mask].unsqueeze(1)
                    # (num_non_scene_envs, 1, 3)

                terrain_heights = self.terrain.find_terrain_height_for_max_below_body(
                    rigid_body_pos_spawned
                )
                respawn_offset[non_scene_mask, 2] = terrain_heights

        respawn_offset[:, 2] += self.config.ref_respawn_offset

        self.respawn_root_offset[env_ids] = respawn_offset

    def align_motion_with_humanoid(self, env_ids, root_pos):
        """Compute XY offset between humanoid spawn position and reference motion data.

        Args:
            env_ids: Environment indices to align
            root_pos: Desired root positions [len(env_ids), 3]
        """
        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids[env_ids],
            self.motion_manager.motion_times[env_ids],
        )

        self.respawn_root_offset[env_ids, :2] = (
            root_pos[:, :2] - ref_state.rigid_body_pos[:, 0, :2]
        )

    def get_scene_non_scene_mask(self, env_ids):
        """
        Returns boolean masks indicating which envs require a scene and which don't.

        Args:
            env_ids: Environment IDs to check

        Returns:
            scene_mask: Boolean tensor (len(env_ids),) - True for scene envs
            non_scene_mask: Boolean tensor (len(env_ids),) - True for non-scene envs

        Note: For now assumes either all or none require a scene
        """
        num_envs = len(env_ids)
        if self.scene_lib.num_scenes() > 0:
            scene_mask = torch.ones(num_envs, device=self.device, dtype=torch.bool)
            non_scene_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        else:
            scene_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
            non_scene_mask = torch.ones(num_envs, device=self.device, dtype=torch.bool)
        return scene_mask, non_scene_mask

    def get_markers_state(self):
        """Compute visualization marker positions for rendering.

        Returns:
            Dictionary mapping marker names to MarkerState objects
        """
        if self.simulator.headless:
            return {}

        markers_state = {}

        # Update terrain markers
        if self.config.show_terrain_markers:
            height_maps = self.terrain.get_height_maps(
                self.simulator.get_root_state(), None, return_all_dims=True
            ).view(self.num_envs, -1, 3)
            markers_state["terrain_markers"] = MarkerState(
                translation=height_maps,
                orientation=torch.zeros(
                    self.num_envs, height_maps.shape[1], 4, device=self.device
                ),
            )

        return markers_state

    ###############################################################
    # Environment step logic
    ###############################################################
    def step(self, actions):
        """Step the environment forward one timestep.

        Args:
            actions: Action tensor [num_envs, action_dim]

        Returns:
            obs: Dictionary of observation tensors
            rewards: Reward tensor [num_envs]
            dones: Reset flags [num_envs]
            terminated: Termination flags [num_envs]
            info: Dictionary containing step metadata (extras)
        """

        self.extras = {}

        actions = self.process_actions(actions)

        # Pass callback - simulator will call it after physics step
        self.simulator.step(actions, markers_callback=self.get_markers_state)

        self.post_physics_step()

        if self.simulator.user_requested_reset:
            self.user_reset()

        obs = self.get_obs()
        return obs, self.rew_buf, self.reset_buf, self.terminate_buf, self.extras

    def process_actions(self, actions):
        """Process and clamp actions before passing to simulator.

        Args:
            actions: Raw action tensor [num_envs, action_dim]

        Returns:
            Processed action tensor [num_envs, action_dim]
        """
        clamp_actions = self.robot_config.control.clamp_actions
        if clamp_actions is not None:
            actions = torch.clamp(actions, -clamp_actions, clamp_actions)
            self.extras["action_clamp"] = (actions.abs() == clamp_actions).float()

        return actions

    def on_epoch_end(self, current_epoch: int):
        """Hook called at end of each training epoch. Override in subclasses if needed.

        Args:
            current_epoch: Current epoch number
        """
        pass

    def post_physics_step(self):
        """Update environment state after physics simulation step.

        Increments progress counter, computes observations and rewards, checks for resets,
        and stores raw robot state in extras for logging.
        """
        self.progress_buf += 1
        self.self_obs_cb.post_physics_step()

        self.compute_observations()
        self.compute_reward()
        self.reset_buf[:], self.terminate_buf[:] = self.check_resets_and_terminations()

        if (
            self.motion_manager is not None
            and self.motion_manager.config.realign_motion_with_humanoid_on_each_step
        ):
            self.align_motion_with_humanoid(
                torch.arange(self.num_envs, device=self.device, dtype=torch.long),
                self.simulator.get_root_state().root_pos,
            )

        self.extras["terminate"] = self.terminate_buf

        rbs: RobotState = self.simulator.get_robot_state()
        rbs.translate(-self.respawn_root_offset.clone())
        for k, _ in rbs.get_shape_mapping(flattened=True).items():
            self.extras[f"raw/{k}"] = rbs.flatten_bodies(k)

    def user_reset(self):
        """Force environments to reset on next check (triggered by user input)."""
        self.progress_buf[:] = 100000000000

    def compute_observations(self, env_ids=None):
        """Compute observations for specified environments.

        Args:
            env_ids: Environment indices to update (None = all environments)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        self.self_obs_cb.compute_observations(env_ids)
        self.terrain_obs_cb.compute_observations(env_ids)
        if self.scene_lib.num_scenes() > 0:
            self.scene_obs_cb.compute_observations(env_ids)

    def check_resets_and_terminations(self):
        """Check reset and termination conditions.

        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors
        """
        current_state: RobotState = self.simulator.get_robot_state()

        terminated = torch.zeros_like(self.reset_buf, dtype=torch.bool)

        if self.config.enable_height_termination:
            # Combine fall contact and height checks
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[:, 0]
            )
            adjusted_termination_heights = self.termination_heights + ground_heights

            has_fallen = combine_fall_termination(
                current_state.rigid_body_contacts,
                current_state.rigid_body_pos,
                adjusted_termination_heights,
                self.non_termination_contact_body_ids,
                self.progress_buf,
            )
            terminated = terminated | has_fallen

        # Check max episode length
        max_length_reached = check_max_length_term(
            self.progress_buf, self.max_episode_length
        )
        reset_buf = max_length_reached | terminated

        return reset_buf, terminated

    ###############################################################
    # Dynamic Reward System
    ###############################################################
    def _resolve_body_indices(
        self, names_or_indices: Optional[Union[List[int], List[str]]]
    ) -> Optional[torch.Tensor]:
        """Convert body names or indices to a tensor of body indices.

        Args:
            names_or_indices: Either:
                - List of body names (strings) - can include common names like "all_left_foot_bodies"
                - List of indices (ints), do nothing just convert to tensor
                - None, return None

        Returns:
            Tensor of body indices on self.device, or None if input is None/empty.
        """
        if not names_or_indices:  # Handles None and empty list
            return None

        # If already indices, just convert to tensor
        if isinstance(names_or_indices[0], int):
            return torch.tensor(names_or_indices, dtype=torch.long, device=self.device)

        # Convert names to indices
        body_names = self.robot_config.kinematic_info.body_names
        common_names = self.robot_config.common_naming_to_robot_body_names

        indices = []
        for name in names_or_indices:
            # Expand common names to actual body names, or use name directly
            actual_names = common_names.get(name, [name])

            for actual_name in actual_names:
                if actual_name not in body_names:
                    source = (
                        f" (from common name '{name}')" if name in common_names else ""
                    )
                    raise ValueError(
                        f"Body name '{actual_name}'{source} not found in: {body_names}"
                    )
                indices.append(body_names.index(actual_name))

        return torch.tensor(indices, dtype=torch.long, device=self.device)

    @cached_property
    def _reward_indices_cache(self) -> Dict[str, Optional[torch.Tensor]]:
        """Pre-resolve indices for all reward components.

        Computed once on first access, caching resolved indices to avoid
        repeated computation during reward evaluation.
        """
        cache: Dict[str, torch.Tensor] = {}

        # Cache indices for reward components
        for reward_name, component in self.config.reward_config.items():
            if component.indices_subset is not None:
                resolved = self._resolve_body_indices(component.indices_subset)
                assert (
                    resolved is not None
                ), f"indices_subset for '{reward_name}' resolved to None"
                cache[reward_name] = resolved

        return cache

    def _compute_dynamic_rewards(
        self,
        reward_components: Dict[str, Any],
        context: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute rewards from dynamic reward component configurations.

        Args:
            reward_components: Dict mapping reward names to RewardComponentConfig
            context: Dict of variables available for eval (e.g., current_state, ref_state)

        Returns:
            Combined reward tensor [num_envs]
        """
        multiplicative_reward = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float
        )
        additive_reward = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        any_multiplicative = False

        # Get grace period mask (None if not applicable)
        grace_mask = self.get_has_reset_grace()

        for reward_name, component in reward_components.items():
            if component.function is None:
                continue

            # Evaluate variable strings to get actual tensors
            func_kwargs = {}
            for arg_name, eval_string in component.variables.items():
                try:
                    func_kwargs[arg_name] = eval(
                        eval_string, {"__builtins__": {}}, context
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to evaluate '{eval_string}' for reward '{reward_name}': {e}"
                    )

            # Get cached indices if specified (pre-resolved at init time)
            if reward_name in self._reward_indices_cache:
                func_kwargs["indices"] = self._reward_indices_cache[reward_name]

            # Call the reward function
            reward_value = component.function(**func_kwargs)

            # Zero out reward during grace period if configured
            if component.zero_during_grace_period and grace_mask is not None:
                reward_value = reward_value.clone()
                reward_value[grace_mask] = 0.0

            assert torch.all(
                torch.isfinite(reward_value)
            ), f"Reward '{reward_name}' is not finite: {reward_value}"

            self.extras[f"raw_r/{reward_name}"] = reward_value

            # Apply weight/multiplicative logic
            if component.multiplicative:
                multiplicative_reward *= reward_value
                any_multiplicative = True
            elif component.weight != 0:
                scaled_reward = reward_value * component.weight
                if component.min_value is not None:
                    scaled_reward = torch.clamp(scaled_reward, min=component.min_value)
                if component.max_value is not None:
                    scaled_reward = torch.clamp(scaled_reward, max=component.max_value)
                self.extras[f"scaled_r/{reward_name}"] = scaled_reward
                additive_reward += scaled_reward

        # Combine rewards
        if any_multiplicative:
            self.extras["multiplicative_reward"] = multiplicative_reward
            self.extras["additive_reward"] = additive_reward
            combined_reward = additive_reward + multiplicative_reward
        else:
            combined_reward = additive_reward

        return combined_reward

    def _get_reward_context(self):
        """Get the context for reward computation.

        Returns:
            Dict of variables available for eval (e.g., current_state, ref_state)
        """
        return {
            "current_state": self.simulator.get_robot_state(),
            "current_actions": self.simulator.get_current_actions(),
            "previous_actions": self.simulator.get_previous_actions(),
            "soft_dof_limits_lower": self.robot_config.kinematic_info.dof_limits_lower.to(
                self.device
            )
            * self.robot_config.control.soft_pos_limit,
            "soft_dof_limits_upper": self.robot_config.kinematic_info.dof_limits_upper.to(
                self.device
            )
            * self.robot_config.control.soft_pos_limit,
            "dt": self.dt,
        }

    def get_has_reset_grace(self):
        """Check if environments are in the grace period after reset.

        Override in subclasses that support grace periods (e.g., Mimic).

        Returns:
            Boolean tensor indicating which environments are within grace period,
            or None if no grace period is configured.
        """
        return None

    def compute_reward(self):
        """Compute base rewards using the dynamic reward component system.

        Subclasses should override this to add task-specific rewards, calling super().compute_reward() first.
        """

        # Build context for variable evaluation
        context = self._get_reward_context()

        combined_reward = self._compute_dynamic_rewards(
            self.config.reward_config, context
        )

        self.rew_buf[:] = combined_reward

        self.extras["total_env_reward"] = combined_reward

    ###############################################################
    # Handle Resets
    ###############################################################
    def move_reset_robot_obj_states_to_respawn_position(
        self,
        env_ids,
        new_states: ResetState,
        new_object_states: ObjectState,
    ) -> Tuple[ResetState, ObjectState]:
        new_states.root_pos += self.respawn_root_offset[env_ids]
        # if obj state is not empty, translate all objs also.
        if self.scene_lib.num_scenes() > 0:
            # (num_envs, num_objects_per_scene, 3)
            new_object_states.root_pos += self.respawn_root_offset[env_ids].unsqueeze(1)

        return new_states, new_object_states

    def compute_default_reset_state(
        self, env_ids, sample_flat: bool = False
    ) -> Tuple[ResetState, ObjectState]:
        """Reset environments to default state."""

        new_states = self.default_reset_state[env_ids].clone()
        # Object states (empty if no scenes)
        new_object_states = self.default_object_state[env_ids].clone()

        self.update_respawn_root_offset_by_env_ids(
            env_ids,
            ref_state=None,
            sample_flat=sample_flat,
        )

        return self.move_reset_robot_obj_states_to_respawn_position(
            env_ids, new_states, new_object_states
        )

    def compute_ref_reset_state(
        self,
        env_ids,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        sample_flat: bool = False,
    ) -> Tuple[ResetState, ObjectState]:
        """Compute reset state from reference motion data.

        Args:
            env_ids: Environment indices to reset
            motion_ids: Motion IDs to use [len(env_ids)]
            motion_times: Start times for each motion [len(env_ids)]
            sample_flat: If True, spawn on flat terrain

        Returns:
            Tuple of (reset_state, object_reset_state)
        """

        ref_state = self.motion_lib.get_motion_state(motion_ids, motion_times)
        # Convert to ResetState (extract only what's needed for reset)
        new_states = ResetState.from_robot_state(ref_state)

        # Get object reference state (empty if no scenes)
        new_object_states = self.scene_lib.get_scene_pose(
            env_ids, motion_times, respawn_offset=self.config.ref_object_respawn_offset
        )
        new_object_states.root_vel = torch.zeros_like(new_object_states.root_pos)
        new_object_states.root_ang_vel = torch.zeros_like(new_object_states.root_pos)

        self.update_respawn_root_offset_by_env_ids(
            env_ids,
            ref_state=ref_state,
            sample_flat=sample_flat,
        )  # (num_envs, 3)

        return self.move_reset_robot_obj_states_to_respawn_position(
            env_ids, new_states, new_object_states
        )

    def reset(
        self,
        env_ids=None,
        sample_flat=False,
        force_default_mask=None,
        disable_motion_resample=False,
    ):
        """Reset environments and return observations.

        - auto if no motion_lib: reset from default state
        - auto if motion_lib exists: reset from reference motion
        - force_default_mask: optional boolean mask [len(env_ids)] to force specific envs
            ref_prob = 0.5
            mask = torch.bernoulli(torch.full((len(env_ids),), 1-ref_prob)).bool()
            env.reset(env_ids, force_default_mask=mask)

        Args:
            env_ids: Environment IDs to reset, or None to reset all
            sample_flat: If True, spawn on flat terrain (useful for evaluation)
            force_default_mask: Optional boolean mask [len(env_ids)] to force specific envs
                               to use default reset instead of reference motion reset.
                               Only used if motion_lib exists.
            disable_motion_resample: If True, skip resampling motions (use existing motion_ids/times).
                               Useful for evaluation when you want to replay specific motions.

        Returns:
            obs: Dictionary of observation tensors
            info: Dictionary containing reset metadata (currently empty)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if len(env_ids) == 0:
            return self.get_obs(), {}

        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)

        # Start with default reset for all envs
        new_states, new_object_states = self.compute_default_reset_state(
            env_ids, sample_flat
        )

        # Determine which envs need reference motion reset
        ref_env_ids, motion_ids, motion_times = self._get_ref_reset_envs(
            env_ids, force_default_mask, disable_motion_resample
        )

        # Overwrite ref envs with reference motion reset
        if len(ref_env_ids) > 0:
            ref_states, ref_object_states = self.compute_ref_reset_state(
                ref_env_ids, motion_ids, motion_times, sample_flat
            )

            # Find indices of ref_env_ids within env_ids
            ref_indices = torch.isin(env_ids, ref_env_ids).nonzero(as_tuple=True)[0]

            # Overwrite robot state
            new_states[ref_indices] = ref_states
            # Overwrite object state
            new_object_states[ref_indices] = ref_object_states

        # Apply to simulator
        self.simulator.reset_envs(new_states, new_object_states, env_ids)

        # Reset observation history buffers
        default_mask = ~torch.isin(env_ids, ref_env_ids)
        self.self_obs_cb.reset_hist_buf(env_ids, default_mask, motion_ids, motion_times)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.terminate_buf[env_ids] = False

        return self.get_obs(), {}

    def _get_ref_reset_envs(
        self, env_ids, force_default_mask, disable_motion_resample=False
    ):
        """Determine which envs should use reference motion reset.

        Args:
            env_ids: Environment IDs to check
            force_default_mask: Boolean mask to force default reset
            disable_motion_resample: If True, use existing motion_ids/times instead of resampling

        Returns:
            ref_env_ids: Environments to reset with reference motion
            motion_ids: Motion IDs for ref resets (or None)
            motion_times: Motion times for ref resets (or None)
        """
        # No motions - no ref resets
        if self.motion_lib.num_motions() == 0:
            empty_ids = torch.tensor([], device=self.device, dtype=torch.long)
            return empty_ids, None, None

        # Determine which envs use reference motion
        if force_default_mask is not None:
            assert (
                len(force_default_mask) == len(env_ids)
            ), f"force_default_mask length {len(force_default_mask)} != env_ids length {len(env_ids)}"
            ref_env_ids = env_ids[~force_default_mask]
        else:
            # All use reference by default when motion_lib exists
            ref_env_ids = env_ids

        # Sample motions for ref resets
        if len(ref_env_ids) > 0:
            if not disable_motion_resample:
                self.motion_manager.sample_motions(ref_env_ids)
            motion_ids = self.motion_manager.motion_ids[ref_env_ids]
            motion_times = self.motion_manager.motion_times[ref_env_ids]
        else:
            motion_ids = None
            motion_times = None

        return ref_env_ids, motion_ids, motion_times

    ###############################################################
    # Motion and Visualization Helpers
    ###############################################################
    def create_motion_manager(self):
        """Instantiate motion manager from configuration.

        Creates the motion manager for sampling and tracking reference motions.
        """
        MotionManagerClass = get_class(self.config.motion_manager._target_)
        self.motion_manager = MotionManagerClass(
            config=self.config.motion_manager,
            num_envs=self.num_envs,
            env_dt=self.dt,
            device=self.device,
            motion_lib=self.motion_lib,
        )

    def create_visualization_markers(self, headless: bool):
        """Create visualization markers based on headless flag.

        Args:
            headless: If True, no markers are created (empty dict).
                      If False, creates markers according to config.

        Returns:
            Dict of visualization markers.
        """
        if headless:
            return {}

        visualization_markers = {}

        if self.config.show_terrain_markers:
            terrain_markers = []
            for _ in range(self.terrain.num_height_points):
                terrain_markers.append(MarkerConfig(size="small"))
            terrain_markers_cfg = VisualizationMarkerConfig(
                type="sphere", color=(0.008, 0.345, 0.224), markers=terrain_markers
            )
            visualization_markers["terrain_markers"] = terrain_markers_cfg

        return visualization_markers

    def get_state_dict(self):
        """Get environment state for checkpointing.

        Returns:
            Dictionary containing motion manager state
        """
        if self.motion_manager is not None:
            return {"motion_manager": self.motion_manager.get_state_dict()}
        return {}

    def load_state_dict(self, state_dict):
        """Load environment state from checkpoint.

        Args:
            state_dict: State dictionary from checkpoint
        """
        if self.motion_manager is not None:
            self.motion_manager.load_state_dict(state_dict["motion_manager"])

    def get_task_id(self):
        """Get task identifier for logging and checkpointing.

        Returns:
            String identifier (motion file name or 'null')
        """
        if self.motion_manager is not None:
            return self.motion_lib.motion_file.split("/")[-1]
        return "null"

    @staticmethod
    def apply_motion_weights_to_scene_weights(
        save_dir: Optional[str], motion_file: Optional[str], device: torch.device
    ) -> Optional[list]:
        """Apply motion weights from checkpoint as scene weights for curriculum learning.

        Loads motion weights from a previous training checkpoint and uses them as
        scene replication weights, allowing over-sampling of scenes corresponding to
        failed motions in curriculum learning.

        IMPORTANT: Assumes 1:1 correspondence between scenes and motions,
        where scene[i].humanoid_motion_id == i.

        Args:
            save_dir: Directory where checkpoints are saved (or None)
            motion_file: Motion file path to identify checkpoint (or None)
            device: PyTorch device

        Returns:
            List of scene weights from motion training or None if not available
        """
        from pathlib import Path

        if not save_dir or not motion_file:
            return None

        try:
            evaluated_motions = motion_file.split("/")[-1]
            checkpoint_path = Path(save_dir) / f"env_{evaluated_motions}.ckpt"

            if not checkpoint_path.exists():
                return None

            print(f"Loading motion weights from checkpoint: {checkpoint_path}")
            checkpoint_data = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )

            if "motion_manager" not in checkpoint_data:
                print(
                    "No motion_manager found in checkpoint, using uniform scene weights."
                )
                return None

            motion_weights = checkpoint_data["motion_manager"]["motion_weights"]
            print(f"Applying {len(motion_weights)} motion weights as scene weights")
            print(
                "WARNING: Assumes 1:1 scene-to-motion correspondence (scene[i].humanoid_motion_id == i)"
            )
            return motion_weights.cpu().tolist()

        except Exception as e:
            print(f"Error applying motion weights to scene weights: {e}")
            return None

    def close(self):
        """
        Clean up environment resources.
        This method should be called when the environment is no longer needed.
        """
        if hasattr(self, "simulator") and self.simulator is not None:
            self.simulator.close()
