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
| `terrain_obs_cb` | `TerrainObs` | Observation component |
| `scene_obs_cb` | `SceneObs` | Observation component |

"""

from functools import cached_property
from typing import Optional, TYPE_CHECKING, Tuple

import numpy as np
import torch
from torch import Tensor
from protomotions.utils.hydra_replacement import get_class

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
from protomotions.envs.terminations import check_max_length_term
from protomotions.envs.obs.humanoid import compute_local_ang_vel
from protomotions.envs.obs.observation_noise import apply_observation_noise
from protomotions.components.terrains.terrain import Terrain
from protomotions.envs.obs.scene_obs import SceneObs
from protomotions.envs.obs.terrain_obs import TerrainObs
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.managers.control_manager import ControlManager
from protomotions.envs.managers.observation_manager import ObservationManager
from protomotions.envs.managers.reward_manager import RewardManager
from protomotions.envs.managers.termination_manager import TerminationManager


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
        self.terrain = terrain
        self.scene_lib = scene_lib
        self.motion_lib = motion_lib
        self.simulator = simulator
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
        
        # Contact force tracking for impact penalty rewards
        # Initialized properly after simulator init when we know num_bodies
        self.prev_contact_force_magnitudes = None

        self.skip_height_correction = (
            self.config.skip_correct_terrain_height_on_flat and self.terrain.is_flat()
        )

        self.initialize_simulator()

    def initialize_simulator(self):
        """Initialize simulator with task-specific visualization markers.

        Called at the end of __init__ to finalize simulator setup after visualization
        markers have been created (potentially by child env class override).
        """
        if hasattr(self.robot_config, 'kinematic_info') and self.robot_config.kinematic_info is not None:
            self.robot_config.kinematic_info.to(self.device)
        
        # Initialize contact force buffer now that we know num_bodies
        num_bodies = self.robot_config.kinematic_info.num_bodies
        self.prev_contact_force_magnitudes = torch.zeros(
            self.num_envs, num_bodies, dtype=torch.float, device=self.device
        )
        
        if self.config.num_state_history_steps > 0:
            # Check if observation noise is configured - if so, allocate noisy buffers
            store_noisy = (
                self.simulator.config.domain_randomization is not None
                and self.simulator.config.domain_randomization.observation_noise is not None
                and self.simulator.config.domain_randomization.observation_noise.has_noise()
            )
            self.state_history = StateHistoryBuffer(
                num_envs=self.num_envs,
                num_history_steps=self.config.num_state_history_steps,
                num_bodies=num_bodies,
                num_dofs=self.robot_config.kinematic_info.num_dofs,
                action_dim=self.robot_config.number_of_actions,
                num_contact_bodies=len(self.contact_body_ids),
                anchor_body_index=self.robot_config.anchor_body_index,
                device=self.device,
                store_noisy=store_noisy,
            )
        else:
            self.state_history = None
        
        if (
            self.motion_lib.num_motions() > 0
            and self.config.ref_contact_smooth_window > 0
        ):
            self.motion_lib.smooth_contacts(self.config.ref_contact_smooth_window)

        self.dt = self.simulator.dt

        if self.motion_lib.num_motions() > 0:
            self.create_motion_manager()
        else:
            self.motion_manager = None

        self.terrain_obs_cb = TerrainObs(self.terrain.config, self)
        self.scene_obs_cb = SceneObs(self.config.scene_obs, self)
        
        self.control_manager = ControlManager(self.config.control_components, self)

        visualization_markers = self.create_visualization_markers(
            self.simulator.headless
        )
        self.simulator._initialize_with_markers(visualization_markers)
        
        self.observation_manager = ObservationManager(
            self.config.observation_components,
            self.robot_config,
            self.device,
            self.num_envs,
            self.dt,
        )
        self.reward_manager = RewardManager(
            self.config.reward_components,
            self.robot_config,
            self.device,
            self.num_envs,
        )
        self.termination_manager = TerminationManager(
            self.config.termination_components,
            self.robot_config,
            self.device,
            self.num_envs,
        )
        
        context = self._get_global_context()
        self.observation_manager.initialize(context)

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
            Dictionary of observation tensors from humanoid, terrain, scene,
            and dynamic observation components
        """
        obs = {}
        terrain_obs = self.terrain_obs_cb.get_obs()
        obs.update(terrain_obs)
        if self.scene_lib.num_scenes() > 0 and self.config.scene_obs.enabled:
            scene_obs = self.scene_obs_cb.get_obs()
            obs.update(scene_obs)
        
        dynamic_obs = self.observation_manager.get_observations()
        obs.update(dynamic_obs)
        
        return obs

    def get_action_size(self):
        """Get the dimensionality of the action space.

        Returns:
            Number of action dimensions
        """
        return self.simulator.num_act

    ###############################################################
    # Cached Properties
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

            if not self.skip_height_correction:
                if ref_state is not None:
                    rigid_body_pos = ref_state.rigid_body_pos[non_scene_mask].clone()
                    rigid_body_pos_spawned = rigid_body_pos + respawn_offset[
                        non_scene_mask
                    ].unsqueeze(1)
                else:
                    rigid_body_pos_spawned = respawn_offset[non_scene_mask].unsqueeze(1)

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
    
    def get_spawn_to_ref_pose_offset_with_terrain_height_correction(
        self, target_pos: Tensor, env_ids: Optional[Tensor] = None
    ) -> Tensor:
        """Compute spawn offset with terrain height correction for reference poses.
        
        Used by motion tracking tasks to correctly position reference poses in the environment,
        accounting for both XY spawn offset and terrain height.
        
        Args:
            target_pos: Reference body positions [num_envs, num_bodies, 3] 
                       without spawning offset applied.
            env_ids: Environment indices [num_envs]. If None, uses all envs.
        
        Returns:
            Offset to add to target_pos [num_envs, num_bodies, 3].
            
        Note:
            - For XY offset: all bodies share the same respawn_root_offset
            - For Z offset: all bodies share the same offset computed from 
              the body furthest below terrain
            - This preserves the rigid body structure during spawning
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        new_offset = torch.zeros_like(target_pos)
        new_offset[:, :, :2] = self.respawn_root_offset[env_ids, :2][:, None, :]

        if not self.skip_height_correction:
            target_pos_spawned = target_pos.clone() + new_offset
            z_offset = self.terrain.find_terrain_height_for_max_below_body(
                target_pos_spawned
            )
            new_offset[:, :, 2] = z_offset.unsqueeze(1)

        return new_offset

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
        
        # Merge markers from control components
        control_markers_state = self.control_manager.get_markers_state()
        markers_state.update(control_markers_state)

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

        Increments progress counter, updates motion manager, computes observations and rewards,
        checks for resets, and stores raw robot state in extras for logging.
        """
        self.progress_buf += 1
        
        if self.state_history is not None:
            current_state = self.simulator.get_robot_state()
            current_actions = self.simulator.get_current_actions()
            ground_heights = self.terrain.get_ground_heights(current_state.rigid_body_pos[:, 0]).squeeze(-1)
            body_contacts = current_state.rigid_body_contacts[:, self.contact_body_ids].bool()
            
            # Compute noisy versions if observation noise is configured and history stores noisy data
            noisy_kwargs = {}
            if self.state_history.store_noisy:
                obs_noise_cfg = self.simulator.config.domain_randomization.observation_noise
                
                # Apply whole-body noise
                if obs_noise_cfg.body_pos_noise > 0.0:
                    noisy_kwargs["noisy_rigid_body_pos"] = current_state.rigid_body_pos + torch.randn_like(current_state.rigid_body_pos) * obs_noise_cfg.body_pos_noise
                if obs_noise_cfg.body_rot_noise > 0.0:
                    noisy_rot = current_state.rigid_body_rot + torch.randn_like(current_state.rigid_body_rot) * obs_noise_cfg.body_rot_noise
                    noisy_kwargs["noisy_rigid_body_rot"] = noisy_rot / torch.norm(noisy_rot, dim=-1, keepdim=True)
                if obs_noise_cfg.body_vel_noise > 0.0:
                    noisy_kwargs["noisy_rigid_body_vel"] = current_state.rigid_body_vel + torch.randn_like(current_state.rigid_body_vel) * obs_noise_cfg.body_vel_noise
                if obs_noise_cfg.body_ang_vel_noise > 0.0:
                    noisy_kwargs["noisy_rigid_body_ang_vel"] = current_state.rigid_body_ang_vel + torch.randn_like(current_state.rigid_body_ang_vel) * obs_noise_cfg.body_ang_vel_noise
                
                # Apply DOF noise
                if obs_noise_cfg.dof_pos_noise > 0.0:
                    noisy_kwargs["noisy_dof_pos"] = current_state.dof_pos + torch.randn_like(current_state.dof_pos) * obs_noise_cfg.dof_pos_noise
                if obs_noise_cfg.dof_vel_noise > 0.0:
                    noisy_kwargs["noisy_dof_vel"] = current_state.dof_vel + torch.randn_like(current_state.dof_vel) * obs_noise_cfg.dof_vel_noise
                
                # Apply ground height noise
                if obs_noise_cfg.ground_height_noise > 0.0:
                    noisy_kwargs["noisy_ground_heights"] = ground_heights + torch.randn_like(ground_heights) * obs_noise_cfg.ground_height_noise
            
            self.state_history.rotate_and_update(
                rigid_body_pos=current_state.rigid_body_pos,
                rigid_body_rot=current_state.rigid_body_rot,
                rigid_body_vel=current_state.rigid_body_vel,
                rigid_body_ang_vel=current_state.rigid_body_ang_vel,
                dof_pos=current_state.dof_pos,
                dof_vel=current_state.dof_vel,
                actions=current_actions,
                ground_heights=ground_heights,
                body_contacts=body_contacts,
                **noisy_kwargs,
            )
        
        if self.motion_manager is not None and hasattr(self.motion_manager, 'post_physics_step'):
            self.motion_manager.post_physics_step()
        
        self.control_manager.step()

        if (
            self.motion_manager is not None
            and self.motion_manager.config.realign_motion_with_humanoid_on_each_step
        ):
            # When realign_motion_with_humanoid_on_each_step is True, we re-align before computing observations and rewards.
            # This ensures the robot only matches the local-pose with global orientation.
            self.align_motion_with_humanoid(
                torch.arange(self.num_envs, device=self.device, dtype=torch.long),
                self.simulator.get_root_state().root_pos,
            )

        self.compute_observations()
        self.compute_reward()
        self.reset_buf[:], self.terminate_buf[:] = self.check_resets_and_terminations()

        self.extras["terminate"] = self.terminate_buf

        rbs: RobotState = self.simulator.get_robot_state()
        for k, _ in rbs.get_shape_mapping(flattened=True).items():
            self.extras[f"raw/{k}"] = rbs.flatten_bodies(k)
        
        # Update previous contact forces for next step's impact penalty
        self.prev_contact_force_magnitudes[:] = torch.norm(
            rbs.rigid_body_contact_forces, dim=-1
        )

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

        context = self._get_global_context()
        self.observation_manager.compute_observations(context, env_ids)
        
        self.terrain_obs_cb.compute_observations(env_ids)
        if self.scene_lib.num_scenes() > 0:
            self.scene_obs_cb.compute_observations(env_ids)

    def check_resets_and_terminations(self):
        """Check reset and termination conditions.
        
        Only handles max episode length directly. All other terminations
        (including height/fall termination) should be configured via:
        - termination_components (dynamic termination system)
        - control_components (task-specific terminations)

        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors
        """
        max_length_reached = check_max_length_term(
            self.progress_buf, self.max_episode_length
        )
        reset_buf = max_length_reached.clone()
        terminated = torch.zeros_like(self.reset_buf, dtype=torch.bool)
        
        comp_reset, comp_terminate = self.control_manager.check_resets_and_terminations()
        reset_buf = reset_buf | comp_reset
        terminated = terminated | comp_terminate
        
        context = self._get_global_context()
        comp_reset, comp_terminate, term_logging = self.termination_manager.check_terminations(context)
        reset_buf = reset_buf | comp_reset
        terminated = terminated | comp_terminate
        self.extras.update(term_logging)

        return reset_buf, terminated

    ###############################################################
    # Dynamic Reward System
    ###############################################################
    def _get_global_context(self):
        """Get the global context for observations, rewards, and terminations.
        
        Merges base environment state with control component context.
        
        Naming convention:
        - Clean variables (no prefix): e.g., current_state_dof_pos - always ground truth
        - Noisy variables (noisy_* prefix): e.g., noisy_current_state_dof_pos - potentially noisy
        
        When observation noise is configured in domain randomization:
        - noisy_* variables have noise applied
        - Regular variables contain clean data
        
        When no observation noise is configured:
        - Both point to the same tensors (memory efficient)

        Returns:
            Dict of variables available for eval in observation/reward/termination functions.
        """
        current_state = self.simulator.get_robot_state()
        
        ground_heights = self.terrain.get_ground_heights(
            current_state.rigid_body_pos[:, 0]
        )
        
        body_contacts = current_state.rigid_body_contacts[
            :, self.contact_body_ids
        ].bool()
        
        # Contact force magnitudes for impact penalty rewards
        current_contact_force_magnitudes = torch.norm(
            current_state.rigid_body_contact_forces, dim=-1
        )
        
        # Check if observation noise is configured
        obs_noise_cfg = None
        if (
            self.simulator.config.domain_randomization is not None
            and self.simulator.config.domain_randomization.observation_noise is not None
            and self.simulator.config.domain_randomization.observation_noise.has_noise()
        ):
            obs_noise_cfg = self.simulator.config.domain_randomization.observation_noise
        
        # =====================================================================
        # Compute derived clean values from robot state
        # =====================================================================
        anchor_idx = self.robot_config.anchor_body_index
        
        # Root local angular velocity (derived from clean whole-body)
        root_local_ang_vel = compute_local_ang_vel(
            current_state.root_rot,
            current_state.rigid_body_ang_vel[:, 0, :],
        )
        
        # Anchor values (derived from clean whole-body)
        anchor_rot = current_state.rigid_body_rot[:, anchor_idx, :]
        anchor_local_ang_vel = compute_local_ang_vel(
            anchor_rot,
            current_state.rigid_body_ang_vel[:, anchor_idx, :],
        )
        
        # =====================================================================
        # Apply observation noise if configured
        # =====================================================================
        noisy = apply_observation_noise(
            obs_noise_cfg=obs_noise_cfg,
            robot_state=current_state,
            anchor_idx=anchor_idx,
            root_local_ang_vel=root_local_ang_vel,
            anchor_rot=anchor_rot,
            anchor_local_ang_vel=anchor_local_ang_vel,
            ground_heights=ground_heights,
        )
        
        context = {
            # Clean state variables - always ground truth (no prefix)
            "current_state_rigid_body_pos": current_state.rigid_body_pos,
            "current_state_rigid_body_rot": current_state.rigid_body_rot,
            "current_state_rigid_body_vel": current_state.rigid_body_vel,
            "current_state_rigid_body_ang_vel": current_state.rigid_body_ang_vel,
            "current_state_rigid_body_contacts": current_state.rigid_body_contacts,
            "current_state_dof_pos": current_state.dof_pos,
            "current_state_dof_vel": current_state.dof_vel,
            "current_state_dof_forces": current_state.dof_forces,
            "current_state_root_rot": current_state.root_rot,
            
            # Derived clean root values
            "current_state_root_pos": current_state.rigid_body_pos[:, 0, :],
            "current_state_root_ang_vel": current_state.rigid_body_ang_vel[:, 0, :],
            "current_state_root_local_ang_vel": root_local_ang_vel,
            "current_state_root_height": current_state.rigid_body_pos[:, 0, 2],

            # Derived clean anchor values
            "current_state_anchor_pos": current_state.rigid_body_pos[:, anchor_idx, :],
            "current_state_anchor_rot": anchor_rot,
            "current_state_anchor_vel": current_state.rigid_body_vel[:, anchor_idx, :],
            "current_state_anchor_ang_vel": current_state.rigid_body_ang_vel[:, anchor_idx, :],
            "current_state_anchor_local_ang_vel": anchor_local_ang_vel,
            
            # Noisy state variables (noisy_* prefix) - potentially noisy if obs noise is configured
            "noisy_current_state_rigid_body_pos": noisy.rigid_body_pos,
            "noisy_current_state_rigid_body_rot": noisy.rigid_body_rot,
            "noisy_current_state_rigid_body_vel": noisy.rigid_body_vel,
            "noisy_current_state_rigid_body_ang_vel": noisy.rigid_body_ang_vel,
            "noisy_current_state_dof_pos": noisy.dof_pos,
            "noisy_current_state_dof_vel": noisy.dof_vel,
            "noisy_current_state_root_rot": noisy.root_rot,
            "noisy_current_state_root_local_ang_vel": noisy.root_local_ang_vel,
            "noisy_current_state_anchor_rot": noisy.anchor_rot,
            "noisy_current_state_anchor_local_ang_vel": noisy.anchor_local_ang_vel,
            
            # Derived noisy root values
            "noisy_current_state_root_pos": noisy.rigid_body_pos[:, 0, :],
            "noisy_current_state_root_ang_vel": noisy.rigid_body_ang_vel[:, 0, :],
            "noisy_current_state_root_height": noisy.rigid_body_pos[:, 0, 2],
            
            # Derived noisy anchor values
            "noisy_current_state_anchor_pos": noisy.rigid_body_pos[:, anchor_idx, :],
            "noisy_current_state_anchor_vel": noisy.rigid_body_vel[:, anchor_idx, :],
            "noisy_current_state_anchor_ang_vel": noisy.rigid_body_ang_vel[:, anchor_idx, :],

            "current_actions": self.simulator.get_current_actions(),
            # Previous actions (t-1) - from history buffer index 1 (index 0 is current after rotate_and_update)
            "previous_actions": self.state_history.actions[:, 1] if (self.state_history and self.state_history.num_history_steps > 1) else self.simulator.get_previous_actions(),
            # PD action scale for converting normalized actions to radians
            "pd_action_scale": self.simulator._common_pd_action_scale,
            
            # Clean historical state tensors [envs, history_steps-1, ...] (past only, excludes current)
            "historical_rigid_body_pos": self.state_history.historical_rigid_body_pos if self.state_history else None,
            "historical_rigid_body_rot": self.state_history.historical_rigid_body_rot if self.state_history else None,
            "historical_rigid_body_vel": self.state_history.historical_rigid_body_vel if self.state_history else None,
            "historical_rigid_body_ang_vel": self.state_history.historical_rigid_body_ang_vel if self.state_history else None,
            "historical_dof_pos": self.state_history.historical_dof_pos if self.state_history else None,
            "historical_dof_vel": self.state_history.historical_dof_vel if self.state_history else None,
            "historical_ground_heights": self.state_history.historical_ground_heights if self.state_history else None,
            # These terms are not affected by observation noise
            "historical_actions": self.state_history.historical_actions if self.state_history else None,
            "historical_body_contacts": self.state_history.historical_body_contacts if self.state_history else None,
            # Derived clean historical root values (past only)
            "historical_root_pos": self.state_history.historical_root_pos if self.state_history else None,
            "historical_root_rot": self.state_history.historical_root_rot if self.state_history else None,
            "historical_root_ang_vel": self.state_history.historical_root_ang_vel if self.state_history else None,
            "historical_root_local_ang_vel": compute_local_ang_vel(self.state_history.historical_root_rot, self.state_history.historical_root_ang_vel) if self.state_history else None,
            # Derived clean historical anchor values (past only)
            "historical_anchor_pos": self.state_history.historical_anchor_pos if self.state_history else None,
            "historical_anchor_rot": self.state_history.historical_anchor_rot if self.state_history else None,
            "historical_anchor_vel": self.state_history.historical_anchor_vel if self.state_history else None,
            "historical_anchor_ang_vel": self.state_history.historical_anchor_ang_vel if self.state_history else None,
            
            # Noisy historical state tensors (for actor with observation noise)
            # These point to same data as clean versions when store_noisy=False
            "noisy_historical_rigid_body_pos": self.state_history.noisy_historical_rigid_body_pos if self.state_history else None,
            "noisy_historical_rigid_body_rot": self.state_history.noisy_historical_rigid_body_rot if self.state_history else None,
            "noisy_historical_rigid_body_vel": self.state_history.noisy_historical_rigid_body_vel if self.state_history else None,
            "noisy_historical_rigid_body_ang_vel": self.state_history.noisy_historical_rigid_body_ang_vel if self.state_history else None,
            "noisy_historical_dof_pos": self.state_history.noisy_historical_dof_pos if self.state_history else None,
            "noisy_historical_dof_vel": self.state_history.noisy_historical_dof_vel if self.state_history else None,
            # Derived noisy historical root values
            "noisy_historical_root_pos": self.state_history.noisy_historical_root_pos if self.state_history else None,
            "noisy_historical_root_rot": self.state_history.noisy_historical_root_rot if self.state_history else None,
            "noisy_historical_root_ang_vel": self.state_history.noisy_historical_root_ang_vel if self.state_history else None,
            "noisy_historical_root_local_ang_vel": compute_local_ang_vel(self.state_history.noisy_historical_root_rot, self.state_history.noisy_historical_root_ang_vel) if self.state_history else None,
            # Derived noisy historical anchor values
            "noisy_historical_anchor_pos": self.state_history.noisy_historical_anchor_pos if self.state_history else None,
            "noisy_historical_anchor_rot": self.state_history.noisy_historical_anchor_rot if self.state_history else None,
            # Noisy historical ground heights
            "noisy_historical_ground_heights": self.state_history.noisy_historical_ground_heights if self.state_history else None,
            
            # Environment state (clean ground heights and noisy version)
            "ground_heights_beneath_root": ground_heights,
            "noisy_ground_heights_beneath_root": noisy.ground_heights,
            "body_contacts": body_contacts,
            "current_contact_force_magnitudes": current_contact_force_magnitudes,
            "prev_contact_force_magnitudes": self.prev_contact_force_magnitudes,
            
            # Control parameters (tensors)
            "soft_dof_limits_lower": self.robot_config.kinematic_info.dof_limits_lower.to(
                self.device
            )
            * self.robot_config.control.soft_pos_limit,
            "soft_dof_limits_upper": self.robot_config.kinematic_info.dof_limits_upper.to(
                self.device
            )
            * self.robot_config.control.soft_pos_limit,
            "contact_body_ids": self.contact_body_ids,
            "non_termination_contact_body_ids": self.non_termination_contact_body_ids,
            
            # Constants
            "dt": self.dt,
            "hinge_axes_map": self.robot_config.kinematic_info.hinge_axes_map,
        }
        
        control_context = self.control_manager.get_context()
        context.update(control_context)
        
        return context

    def get_has_reset_grace(self):
        """Check if environments are in the grace period after reset.

        Grace period is useful for zeroing rewards that are unreliable immediately
        after reset (e.g., power consumption, contact changes).

        Returns:
            Boolean tensor indicating which environments are within reset_grace_period steps of last reset.
            Returns None if reset_grace_period is 0 or negative.
        """
        if self.config.reset_grace_period <= 0:
            return None
        return self.progress_buf <= self.config.reset_grace_period

    def compute_reward(self):
        """Compute base rewards using the dynamic reward component system.

        Subclasses should override this to add task-specific rewards, calling super().compute_reward() first.
        """

        context = self._get_global_context()
        grace_mask = self.get_has_reset_grace()

        combined_reward, reward_logging = self.reward_manager.compute_rewards(context, grace_mask)

        self.rew_buf[:] = combined_reward
        self.extras.update(reward_logging)
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
        if self.scene_lib.num_scenes() > 0:
            new_object_states.root_pos += self.respawn_root_offset[env_ids].unsqueeze(1)

        return new_states, new_object_states

    def compute_default_reset_state(
        self, env_ids, sample_flat: bool = False
    ) -> Tuple[ResetState, ObjectState]:
        """Reset environments to default state."""

        new_states = self.default_reset_state[env_ids].clone()
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
        new_states = ResetState.from_robot_state(ref_state)

        new_object_states = self.scene_lib.get_scene_pose(
            env_ids, motion_times, respawn_offset=self.config.ref_object_respawn_offset
        )
        new_object_states.root_vel = torch.zeros_like(new_object_states.root_pos)
        new_object_states.root_ang_vel = torch.zeros_like(new_object_states.root_pos)

        self.update_respawn_root_offset_by_env_ids(
            env_ids,
            ref_state=ref_state,
            sample_flat=sample_flat,
        )

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

        # STEP 1: Reset motion manager and determine which envs need reference motion reset
        # This calls motion_manager.sample_motions() internally
        ref_env_ids, motion_ids, motion_times = self._get_ref_reset_envs(
            env_ids, force_default_mask, disable_motion_resample
        )

        # Overwrite ref envs with reference motion reset
        if len(ref_env_ids) > 0:
            ref_states, ref_object_states = self.compute_ref_reset_state(
                ref_env_ids, motion_ids, motion_times, sample_flat
            )

            ref_indices = torch.isin(env_ids, ref_env_ids).nonzero(as_tuple=True)[0]

            new_states[ref_indices] = ref_states
            new_object_states[ref_indices] = ref_object_states

        self.simulator.reset_envs(new_states, new_object_states, env_ids)

        default_mask = ~torch.isin(env_ids, ref_env_ids)
        if self.state_history is not None:
            self._reset_state_history(env_ids, default_mask, ref_env_ids, motion_ids, motion_times)
        
        # Reset control components after motion_manager has been reset
        self.control_manager.reset(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.terminate_buf[env_ids] = False
        self.prev_contact_force_magnitudes[env_ids] = 0.0

        # Recompute observations after reset to reflect new control component state
        self.compute_observations(env_ids)

        return self.get_obs(), {}

    def _get_ref_reset_envs(
        self, env_ids, force_default_mask, disable_motion_resample=False
    ):
        """Determine which envs should use reference motion reset and reset motion manager.
        
        This method is responsible for resetting the motion_manager by calling
        motion_manager.sample_motions(). Control components should be reset AFTER
        this method is called so they have access to fresh motion_ids and motion_times.

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

        if force_default_mask is not None:
            assert (
                len(force_default_mask) == len(env_ids)
            ), f"force_default_mask length {len(force_default_mask)} != env_ids length {len(env_ids)}"
            ref_env_ids = env_ids[~force_default_mask]
        else:
            ref_env_ids = env_ids

        if len(ref_env_ids) > 0:
            if not disable_motion_resample:
                self.motion_manager.sample_motions(ref_env_ids)
            motion_ids = self.motion_manager.motion_ids[ref_env_ids]
            motion_times = self.motion_manager.motion_times[ref_env_ids]
        else:
            motion_ids = None
            motion_times = None

        return ref_env_ids, motion_ids, motion_times

    def _reset_state_history(
        self,
        env_ids: Tensor,
        default_mask: Tensor,
        ref_env_ids: Tensor,
        motion_ids: Optional[Tensor],
        motion_times: Optional[Tensor],
    ):
        """Reset state history buffer for specified environments.
        
        For default reset: repeat current state across all history slots.
        For ref reset: query motion_lib at t-dt, t-2*dt, ... to get historical states.
        
        Args:
            env_ids: All environment indices being reset.
            default_mask: Boolean mask indicating which envs use default reset.
            ref_env_ids: Environment indices using reference motion reset.
            motion_ids: Motion IDs for ref envs (or None).
            motion_times: Motion times for ref envs (or None).
        """
        default_env_ids = env_ids[default_mask]
        num_history_steps = self.state_history.num_history_steps
        # Buffer stores current + history, so total slots = num_history_steps + 1
        buffer_size = num_history_steps + 1
        
        # Default reset: repeat current simulator state to all buffer slots
        if len(default_env_ids) > 0:
            current_state = self.simulator.get_robot_state()
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[default_env_ids, 0]
            ).squeeze(-1)
            body_contacts = current_state.rigid_body_contacts[default_env_ids][:, self.contact_body_ids].bool()
            self.state_history.reset_from_single_state(
                env_ids=default_env_ids,
                rigid_body_pos=current_state.rigid_body_pos[default_env_ids],
                rigid_body_rot=current_state.rigid_body_rot[default_env_ids],
                rigid_body_vel=current_state.rigid_body_vel[default_env_ids],
                rigid_body_ang_vel=current_state.rigid_body_ang_vel[default_env_ids],
                dof_pos=current_state.dof_pos[default_env_ids],
                dof_vel=current_state.dof_vel[default_env_ids],
                ground_heights=ground_heights,
                body_contacts=body_contacts,
            )
        
        # Reference reset: fill buffer with current state at index 0 and historical states at index 1+
        # This ensures historical_* properties (which return [:, 1:]) give exactly num_history_steps elements
        if len(ref_env_ids) > 0 and motion_ids is not None and motion_times is not None:
            # motion_ids shape: [len(ref_env_ids)]
            # motion_times shape: [len(ref_env_ids)]
            num_ref_envs = len(ref_env_ids)
            
            # Create time offsets: [0, -dt, -2*dt, ..., -N*dt] for buffer_size slots
            # Index 0 = current (t), Index 1..N = historical (t-dt, t-2*dt, ..., t-N*dt)
            time_offsets = -self.dt * torch.arange(buffer_size, device=self.device)
            
            # Expand for batch query: [num_ref_envs, buffer_size]
            expanded_motion_ids = motion_ids.unsqueeze(1).expand(-1, buffer_size)
            expanded_motion_times = motion_times.unsqueeze(1) + time_offsets.unsqueeze(0)
            
            # Clamp times to valid range
            motion_lengths = self.motion_lib.motion_lengths[motion_ids]
            expanded_motion_times = expanded_motion_times.clamp(min=0.0)
            expanded_motion_times = torch.min(
                expanded_motion_times,
                motion_lengths.unsqueeze(1).expand(-1, buffer_size)
            )
            
            # Flatten for motion_lib query
            flat_motion_ids = expanded_motion_ids.reshape(-1)
            flat_motion_times = expanded_motion_times.reshape(-1)
            
            # Query motion library
            historical_state = self.motion_lib.get_motion_state(flat_motion_ids, flat_motion_times)
            
            # Motion library data is recorded on flat terrain (height = 0)
            # Only simulator-based states need terrain height queries
            historical_ground_heights = torch.zeros(
                num_ref_envs, buffer_size, device=self.device
            )
            
            # Get contacts from motion library if available, otherwise zeros
            if historical_state.rigid_body_contacts is not None:
                flat_contacts = historical_state.rigid_body_contacts[:, self.contact_body_ids].bool()
                historical_body_contacts = flat_contacts.view(
                    num_ref_envs, buffer_size, -1
                )
            else:
                historical_body_contacts = torch.zeros(
                    num_ref_envs, buffer_size, len(self.contact_body_ids),
                    dtype=torch.bool, device=self.device
                )
            
            # Reshape back to [num_ref_envs, buffer_size, ...]
            self.state_history.reset_from_states(
                env_ids=ref_env_ids,
                rigid_body_pos=historical_state.rigid_body_pos.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                rigid_body_rot=historical_state.rigid_body_rot.view(
                    num_ref_envs, buffer_size, -1, 4
                ),
                rigid_body_vel=historical_state.rigid_body_vel.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                rigid_body_ang_vel=historical_state.rigid_body_ang_vel.view(
                    num_ref_envs, buffer_size, -1, 3
                ),
                dof_pos=historical_state.dof_pos.view(
                    num_ref_envs, buffer_size, -1
                ),
                dof_vel=historical_state.dof_vel.view(
                    num_ref_envs, buffer_size, -1
                ),
                ground_heights=historical_ground_heights,
                body_contacts=historical_body_contacts,
                actions=None,  # Zero actions for historical reset
            )

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
        
        # Merge markers from control components
        control_markers = self.control_manager.create_visualization_markers(headless)
        visualization_markers.update(control_markers)

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
