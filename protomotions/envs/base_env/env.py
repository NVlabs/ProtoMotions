# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from typing import Any, Dict, Optional, TYPE_CHECKING, Tuple

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
from protomotions.envs.context_views import (
    EnvContext,
    CurrentStateView,
    HistoricalView,
    TerrainContext,
    SceneSurfaceContext,
)
from protomotions.envs.obs.observation_noise import (
    NoisyObservations,
    apply_observation_noise,
    apply_reset_noise,
)
from protomotions.components.terrains.terrain import Terrain
from protomotions.envs.obs.scene_obs import SceneObs
from protomotions.envs.obs.terrain_obs import TerrainObs
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer
from protomotions.envs.base_env.config import EnvConfig
from protomotions.envs.control.manager import ControlManager

# Component infrastructure for MdpComponent-based configs
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.base_env.utils import (
    combine_rewards,
    combine_terminations,
)
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
        >>> next_obs, rewards, dones, info = env.step(action_dict)
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

        # Per-episode odometer corruption parameters.
        # Sampled once at episode reset; held constant within the episode.
        # Identity values (scale=1, yaw_bias=0) until first reset.
        self.odom_scale = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.odom_yaw_cos_sin = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )
        self.odom_yaw_cos_sin[:, 0] = 1.0  # cos(0) = 1

        # Contact force tracking for impact penalty rewards
        # Initialized properly after simulator init when we know num_bodies
        self.prev_contact_force_magnitudes = None

        # Action buffers (current step only; previous actions come from state_history)
        num_actions = robot_config.number_of_actions
        self._current_raw_action = torch.zeros(
            self.num_envs, num_actions, dtype=torch.float, device=self.device
        )
        self._current_processed_action = torch.zeros(
            self.num_envs, num_actions, dtype=torch.float, device=self.device
        )

        # Global context cache - built once per step in post_physics_step
        # and reused by observations, rewards, and terminations
        self._current_context: Dict[str, Any] = None

        # Noisy observation cache - computed once in post_physics_step,
        # reused by both state_history and _build_global_context
        self._current_noisy_obs = None

        self.skip_height_correction = (
            self.config.skip_correct_terrain_height_on_flat and self.terrain.is_flat()
        )

        self.initialize_simulator()

    def initialize_simulator(self):
        """Initialize simulator with task-specific visualization markers.

        Called at the end of __init__ to finalize simulator setup after visualization
        markers have been created (potentially by child env class override).
        """
        if (
            hasattr(self.robot_config, "kinematic_info")
            and self.robot_config.kinematic_info is not None
        ):
            self.robot_config.kinematic_info.to(self.device)

        # Initialize contact force buffer now that we know num_bodies
        num_bodies = self.robot_config.kinematic_info.num_bodies
        self.prev_contact_force_magnitudes = torch.zeros(
            self.num_envs, num_bodies, dtype=torch.float, device=self.device
        )

        # Initialize actuation/observation delay DR state (no-op unless configured).
        self._init_delay_state()

        if self.config.num_state_history_steps > 0:
            # Check if observation noise is configured - if so, allocate noisy buffers
            store_noisy = (
                self.simulator.config.domain_randomization is not None
                and self.simulator.config.domain_randomization.observation_noise
                is not None
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
            self._validate_motion_lib_compatibility()
            self.create_motion_manager()
        else:
            self.motion_manager = None

        self.terrain_obs_cb = TerrainObs(self.terrain.config, self)
        self.scene_obs_cb = SceneObs(self.config.scene_obs, self)

        self._key_bindings = self.simulator.user_interface.scope("env")
        self._key_bindings.register("R", "reset", "Reset all environments")
        self.control_manager = ControlManager(self.config.control_components, self)

        visualization_markers = self.create_visualization_markers(
            self.simulator.headless
        )
        self.simulator._initialize_with_markers(visualization_markers)

        # Component infrastructure for MdpComponent
        self._component_manager = ComponentManager(self.device)
        self._observation_buffer: Dict[str, Tensor] = {}

        # Initialize observations
        self._initialize_observations()

    def _validate_motion_lib_compatibility(self):
        """Validate that the motion file is compatible with the robot config."""
        ki = self.robot_config.kinematic_info
        expected_dofs = ki.num_dofs
        expected_bodies = ki.num_bodies

        sample_state = self.motion_lib.get_motion_state(
            torch.zeros(1, dtype=torch.long, device=self.device),
            torch.zeros(1, device=self.device),
        )
        motion_dofs = sample_state.dof_pos.shape[1]
        motion_bodies = sample_state.rigid_body_pos.shape[1]

        if motion_dofs != expected_dofs or motion_bodies != expected_bodies:
            raise ValueError(
                f"\n{'=' * 70}\n"
                f"MOTION FILE / ROBOT MISMATCH\n"
                f"{'=' * 70}\n"
                f"Motion file has {motion_dofs} DOFs and {motion_bodies} bodies,\n"
                f"but robot '{type(self.robot_config).__name__}' expects "
                f"{expected_dofs} DOFs and {expected_bodies} bodies.\n\n"
                f"The motion file was likely generated for a different robot.\n"
                f"Make sure --motion-file matches the robot in your "
                f"checkpoint/config.\n"
                f"{'=' * 70}"
            )

    ###############################################################
    # Getters
    ###############################################################
    def is_simulation_running(self):
        """Check if the physics simulation is running.

        Returns:
            Boolean indicating simulation state
        """
        return self.simulator.is_simulation_running()

    ###############################################################
    # Actuation / Observation delay DR
    ###############################################################
    def _get_delay_cfg(self):
        """Return the DelayDomainRandomizationConfig or None (getattr-guarded)."""
        dr = getattr(self.simulator.config, "domain_randomization", None)
        if dr is None:
            return None
        return getattr(dr, "delay", None)

    def _init_delay_state(self):
        """Allocate per-env delay buffers/state. Pure no-op when unconfigured.

        Actuation delay: the PD target sent to the sim is the commanded target from
        d_i control steps ago (per-env d_i sampled at reset). Observation delay: the
        obs returned to the policy is the obs from o_i control steps ago. Effective
        delay is clamped to the number of steps elapsed since the env's last reset, so
        freshly reset envs never read stale cross-episode data (no explicit buffer
        clearing needed).
        """
        cfg = self._get_delay_cfg()
        self._delay_cfg = cfg
        self._has_action_delay = bool(cfg is not None and cfg.has_action_delay())
        self._has_obs_delay = bool(cfg is not None and cfg.has_observation_delay())

        # Per-env sampled delays (in control steps).
        self._action_delay = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._obs_delay = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # Global lockstep step counters (all envs advance together).
        self._action_delay_step = 0
        self._obs_delay_step = 0
        # Current training epoch, used only by the ramp (see DelayDomainRandomizationConfig
        # .ramp_epochs). Updated once per epoch by on_epoch_end(), called identically on
        # every rank by the agent's lockstep training loop -> rank-consistent by construction.
        self._current_epoch = 0

        if self._has_action_delay:
            num_actions = self.robot_config.number_of_actions
            length = cfg.max_action_delay() + 1
            self._action_delay_buf = torch.zeros(
                self.num_envs, length, num_actions, dtype=torch.float, device=self.device
            )
        else:
            self._action_delay_buf = None

        # Observation buffers are dict-of-tensors, allocated lazily on first use
        # (keys/shapes are unknown until the first get_obs()).
        self._obs_delay_buf = None

        if self._delay_cfg is not None and self._delay_cfg.has_delay():
            self._sample_delays(
                torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            )

    def _sample_delays(self, env_ids: Tensor):
        """(Re)sample per-env integer delays for the given envs (called at reset)."""
        cfg = self._delay_cfg
        if cfg is None:
            return
        n = len(env_ids)
        if n == 0:
            return
        if self._has_action_delay:
            lo, _ = cfg.action_delay_steps
            hi = cfg.effective_max_action_delay(self._current_epoch)
            self._action_delay[env_ids] = torch.randint(
                int(lo), int(hi) + 1, (n,), dtype=torch.long, device=self.device
            )
        if self._has_obs_delay:
            lo, _ = cfg.observation_delay_steps
            hi = cfg.effective_max_observation_delay(self._current_epoch)
            self._obs_delay[env_ids] = torch.randint(
                int(lo), int(hi) + 1, (n,), dtype=torch.long, device=self.device
            )

    def _apply_action_delay(self, processed_action: Tensor) -> Tensor:
        """Return the PD target to actually send to the sim, applying per-env delay.

        Writes the current commanded target into the ring buffer, then reads back, per
        env, the entry effective_delay steps old (clamped to steps-since-reset).
        """
        if not self._has_action_delay:
            return processed_action
        buf = self._action_delay_buf
        length = buf.shape[1]
        write_idx = self._action_delay_step % length
        buf[:, write_idx, :] = processed_action
        # Steps elapsed this episode BEFORE the current step (progress not yet bumped).
        steps_since_reset = self.progress_buf
        eff_delay = torch.minimum(self._action_delay, steps_since_reset)
        read_idx = (self._action_delay_step - eff_delay) % length
        env_ids = torch.arange(self.num_envs, device=self.device)
        delayed = buf[env_ids, read_idx]
        self._action_delay_step += 1
        return delayed

    def _apply_observation_delay(self, obs: dict) -> dict:
        """Return a per-env time-delayed version of the observation dict."""
        if not self._has_obs_delay:
            return obs
        length = self._delay_cfg.max_observation_delay() + 1
        if self._obs_delay_buf is None:
            # Lazily allocate a ring buffer per obs key now that shapes are known.
            self._obs_delay_buf = {
                key: torch.zeros(
                    (self.num_envs, length) + tuple(t.shape[1:]),
                    dtype=t.dtype,
                    device=t.device,
                )
                for key, t in obs.items()
            }
        write_idx = self._obs_delay_step % length
        # progress_buf has already been incremented for the current step in
        # post_physics_step, so history available before this step is progress_buf - 1.
        steps_hist = torch.clamp(self.progress_buf - 1, min=0)
        eff_delay = torch.minimum(self._obs_delay, steps_hist)
        read_idx = (self._obs_delay_step - eff_delay) % length
        env_ids = torch.arange(self.num_envs, device=self.device)
        delayed = {}
        for key, t in obs.items():
            buf = self._obs_delay_buf.get(key)
            if buf is None or buf.shape[2:] != t.shape[1:]:
                # Shape changed / new key — (re)allocate this key's buffer.
                buf = torch.zeros(
                    (self.num_envs, length) + tuple(t.shape[1:]),
                    dtype=t.dtype,
                    device=t.device,
                )
                self._obs_delay_buf[key] = buf
            buf[:, write_idx] = t
            delayed[key] = buf[env_ids, read_idx]
        self._obs_delay_step += 1
        return delayed

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

        # Get dynamic observations
        dynamic_obs = {
            name: tensor.clone() for name, tensor in self._observation_buffer.items()
        }
        obs.update(dynamic_obs)

        return obs

    def get_action_size(self):
        """Get the dimensionality of the action space.

        Returns:
            Number of action dimensions
        """
        return self.simulator.num_act

    def consume_reset_request(self) -> bool:
        """Return and consume a user-interface reset request."""
        return self._key_bindings.reset.consume()

    ###############################################################
    # Component Processing
    ###############################################################
    def _initialize_observations(self):
        """Initialize observation buffers."""
        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._process_observations(self.context, all_env_ids)

    def _process_observations(self, context: EnvContext, env_ids: Tensor):
        """Process observations using MdpComponent."""
        raw_obs = self._component_manager.execute_all(
            components=self.config.observation_components,
            ctx=context,
        )

        # Update observation buffer with results
        for name, obs_value in raw_obs.items():
            if name not in self._observation_buffer:
                self._observation_buffer[name] = torch.zeros(
                    self.num_envs,
                    obs_value.shape[-1],
                    dtype=obs_value.dtype,
                    device=self.device,
                )
            # MdpComponent always computes for all envs, update specified subset
            self._observation_buffer[name][env_ids] = obs_value[env_ids]

    def _process_rewards(
        self, context: EnvContext, grace_mask: Optional[Tensor] = None
    ):
        """Process rewards using MdpComponent."""
        raw_rewards = self._component_manager.execute_all(
            components=self.config.reward_components,
            ctx=context,
        )

        return combine_rewards(
            raw_rewards=raw_rewards,
            configs=self.config.reward_components,
            grace_mask=grace_mask,
            num_envs=self.num_envs,
            device=self.device,
        )

    def _process_terminations(self, context: EnvContext):
        """Process terminations using MdpComponent."""
        raw_terms = self._component_manager.execute_all(
            components=self.config.termination_components,
            ctx=context,
        )

        return combine_terminations(
            raw_terms=raw_terms,
            configs=self.config.termination_components,
            num_envs=self.num_envs,
            device=self.device,
        )

    _action_config_device_ready: bool = False

    def _process_action(self, action: Tensor, context: EnvContext) -> Dict[str, Tensor]:
        """Process action using single action config dict.

        action_config is a single dict with "fn" key and parameters.
        """
        if self.config.action_config is None:
            return {"processed_action": action}

        # Lazy device migration on first call
        if not self._action_config_device_ready:
            for key, val in self.config.action_config.items():
                if isinstance(val, torch.Tensor):
                    self.config.action_config[key] = val.to(action.device)
            self._action_config_device_ready = True

        fn = self.config.action_config["fn"]
        # Extract all params except "fn"
        params = {k: v for k, v in self.config.action_config.items() if k != "fn"}
        params["action"] = action
        return fn(**params)

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

        During co-training, scene groups use flat terrain, but during
        inference the resolved terrain may be complex (with negative heights
        that get normalised).  Height correction is applied to both scene
        and non-scene envs unless the terrain is entirely flat.

        """

        respawn_offset = torch.zeros((len(env_ids), 3), device=self.device)

        # Get boolean masks for scene vs non-scene envs
        scene_mask, non_scene_mask = self.get_scene_non_scene_mask(env_ids)

        if scene_mask.any():
            scene_pos = self.scene_lib.get_scene_positions(self.terrain, self.device)
            respawn_offset[scene_mask, :2] = scene_pos[env_ids[scene_mask], :2]

            # Scene envs also need terrain height correction — the object
            # playground is flat at height-field 0, but terrain normalisation
            # (shifting min height to z=0) can raise the playground above
            # world z=0.  Without correction the agent spawns underground.
            if not self.skip_height_correction:
                if ref_state is not None:
                    rigid_body_pos = ref_state.rigid_body_pos[scene_mask].clone()
                    rigid_body_pos_spawned = rigid_body_pos + respawn_offset[
                        scene_mask
                    ].unsqueeze(1)
                else:
                    rigid_body_pos_spawned = respawn_offset[scene_mask].unsqueeze(1)

                terrain_heights = self.terrain.find_terrain_height_for_max_below_body(
                    rigid_body_pos_spawned
                )
                respawn_offset[scene_mask, 2] = terrain_heights

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
    def step(self, action: Tensor):
        """Step the environment forward one timestep.

        Args:
            action: Raw action tensor from the policy [num_envs, num_actions]

        Returns:
            obs, rewards, dones, terminated, extras
        """
        self.extras = {}

        # Invalidate cached context - will be rebuilt after physics in post_physics_step
        self._current_context = None
        self._current_noisy_obs = None

        # Store current actions
        self._current_raw_action[:] = action

        # Process action
        action_dict = self._process_action(action, self.context)
        processed_action = action_dict["processed_action"]
        # Commanded target stays UNDELAYED for reward/obs consistency; only the target
        # actually sent to the sim is delayed (actuation-delay DR, no-op if disabled).
        self._current_processed_action[:] = processed_action
        applied_action = self._apply_action_delay(processed_action)

        self.simulator.step(applied_action, markers_callback=self.get_markers_state)

        self.post_physics_step()

        if self.consume_reset_request():
            self.user_reset()

        obs = self.get_obs()
        # Observation-delay DR (no-op if disabled).
        obs = self._apply_observation_delay(obs)
        return obs, self.rew_buf, self.reset_buf, self.terminate_buf, self.extras

    def on_epoch_end(self, current_epoch: int):
        """Hook called at end of each training epoch. Override in subclasses if needed.

        Args:
            current_epoch: Current epoch number

        Also feeds the delay-DR ramp (DelayDomainRandomizationConfig.ramp_epochs), if
        configured: self._current_epoch is read by _sample_delays() at each env reset
        to compute the effective (ramped) max delay. Rank-consistency: current_epoch is
        the agent's plain-Python lockstep counter (protomotions/agents/base_agent/agent.py,
        self.current_epoch), incremented once per epoch and identical across all DDP
        ranks — every rank calls env.on_epoch_end(self.current_epoch) with the same value
        in the same training-loop iteration (no per-rank randomness or wall-clock input),
        so every rank computes the identical ramp fraction/effective max and the delay-DR
        RNG draws stay independent-but-parameterized-identically (no collective divergence).
        """
        self._current_epoch = current_epoch

    def post_physics_step(self):
        """Update environment state after physics simulation step.

        Increments progress counter, updates motion manager, computes observations and rewards,
        checks for resets, and stores raw robot state in extras for logging.
        """
        self.progress_buf += 1

        if self.state_history is not None:
            current_state = self.simulator.get_robot_state()
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[:, 0]
            ).squeeze(-1)
            body_contacts = current_state.rigid_body_contacts[
                :, self.contact_body_ids
            ].bool()

            # Compute noisy versions if observation noise is configured and history stores noisy data
            noisy_kwargs = {}
            if self.state_history.store_noisy:
                obs_noise_cfg = (
                    self.simulator.config.domain_randomization.observation_noise
                )

                # Single source of truth: uniform noise via apply_observation_noise
                noisy = apply_observation_noise(
                    obs_noise_cfg=obs_noise_cfg,
                    robot_state=current_state,
                    anchor_idx=self.robot_config.anchor_body_index,
                    ground_heights=ground_heights,
                )
                self._current_noisy_obs = noisy

                # Extract noisy tensors for history buffer
                noisy_kwargs["noisy_rigid_body_pos"] = noisy.rigid_body_pos
                noisy_kwargs["noisy_rigid_body_rot"] = noisy.rigid_body_rot
                noisy_kwargs["noisy_rigid_body_vel"] = noisy.rigid_body_vel
                noisy_kwargs["noisy_rigid_body_ang_vel"] = noisy.rigid_body_ang_vel
                noisy_kwargs["noisy_dof_pos"] = noisy.dof_pos
                noisy_kwargs["noisy_dof_vel"] = noisy.dof_vel
                noisy_kwargs["noisy_ground_heights"] = noisy.ground_heights

            self.state_history.rotate_and_update(
                rigid_body_pos=current_state.rigid_body_pos,
                rigid_body_rot=current_state.rigid_body_rot,
                rigid_body_vel=current_state.rigid_body_vel,
                rigid_body_ang_vel=current_state.rigid_body_ang_vel,
                dof_pos=current_state.dof_pos,
                dof_vel=current_state.dof_vel,
                actions=self._current_raw_action,
                ground_heights=ground_heights,
                body_contacts=body_contacts,
                processed_actions=self._current_processed_action,
                **noisy_kwargs,
            )

        if self.motion_manager is not None and hasattr(
            self.motion_manager, "post_physics_step"
        ):
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

        # Build context once and reuse for observations, rewards, and terminations
        self._current_context = self._build_global_context()

        self.compute_observations(context=self._current_context)
        self.compute_reward(context=self._current_context)
        self.reset_buf[:], self.terminate_buf[:] = self.check_resets_and_terminations(
            context=self._current_context
        )

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

    def compute_observations(self, env_ids=None, context: EnvContext = None):
        """Compute observations for specified environments.

        Args:
            env_ids: Environment indices to update (None = all environments)
            context: Pre-built EnvContext from self.context property.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        if context is None:
            raise ValueError("context is required - use self.context to build it")

        # Process dynamic observations
        self._process_observations(context, env_ids)

        self.terrain_obs_cb.compute_observations(env_ids)
        if self.scene_lib.num_scenes() > 0:
            self.scene_obs_cb.compute_observations(env_ids)

    def check_resets_and_terminations(self, context: EnvContext):
        """Check reset and termination conditions.

        Only handles max episode length directly. All other terminations
        (including height/fall termination) should be configured via:
        - termination_components (dynamic termination system)
        - control_components (task-specific terminations)

        Args:
            context: Pre-built context from self.context property.

        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors
        """
        max_length_reached = check_max_length_term(
            self.progress_buf, self.max_episode_length
        )
        reset_buf = max_length_reached.clone()
        terminated = torch.zeros_like(self.reset_buf, dtype=torch.bool)

        comp_reset, comp_terminate = (
            self.control_manager.check_resets_and_terminations()
        )
        reset_buf = reset_buf | comp_reset
        terminated = terminated | comp_terminate

        # Process terminations
        comp_reset, comp_terminate, term_logging = self._process_terminations(context)
        reset_buf = reset_buf | comp_reset
        terminated = terminated | comp_terminate
        self.extras.update(term_logging)

        return reset_buf, terminated

    ###############################################################
    # Dynamic Reward System
    ###############################################################
    @property
    def context(self) -> EnvContext:
        """Get global context for observation/reward/termination evaluation.

        Returns cached context from _current_context if set (after post_physics_step),
        otherwise builds a fresh context.

        Returns:
            Typed EnvContext for observation/reward/termination functions.
        """
        if self._current_context is None:
            self._current_context = self._build_global_context()
        return self._current_context

    def _build_global_context(self) -> EnvContext:
        """Build a fresh global context for observations, rewards, and terminations.

        Creates typed EnvContext with view wrappers around existing data structures.
        Controllers populate their task-specific views via populate_context().

        When observation noise is configured:
        - noisy views have noise applied
        - current views contain clean data

        When no observation noise is configured:
        - Both point to the same tensors (memory efficient)

        Returns:
            Typed EnvContext for observation/reward/termination functions.
        """
        current_state = self.simulator.get_robot_state()
        anchor_idx = self.robot_config.anchor_body_index

        ground_heights = self.terrain.get_ground_heights(
            current_state.rigid_body_pos[:, 0]
        ).squeeze(-1)

        body_contacts = current_state.rigid_body_contacts[
            :, self.contact_body_ids
        ].bool()

        # Contact force magnitudes for impact penalty rewards
        current_contact_force_magnitudes = torch.norm(
            current_state.rigid_body_contact_forces, dim=-1
        )

        # Use cached noisy obs from post_physics_step when available.
        # During init/reset the cache is None — use clean (no-noise) fallback.
        if self._current_noisy_obs is not None:
            noisy = self._current_noisy_obs
        else:
            noisy = apply_observation_noise(
                obs_noise_cfg=None,
                robot_state=current_state,
                anchor_idx=anchor_idx,
                ground_heights=ground_heights,
            )

        scene_surface_context = self._build_scene_surface_context()

        # Build context with view wrappers
        ctx = EnvContext(
            # Core state views (wrap RobotState without copying)
            current=CurrentStateView(current_state, anchor_idx),
            noisy=CurrentStateView(noisy, anchor_idx),
            # Historical views (wrap StateHistoryBuffer without copying)
            historical=HistoricalView(self.state_history, use_noisy=False)
            if self.state_history
            else None,
            noisy_historical=HistoricalView(self.state_history, use_noisy=True)
            if self.state_history
            else None,
            # Actions (historical)
            current_processed_action=self._current_processed_action,
            previous_action=self.state_history.actions[:, 1]
            if (self.state_history and self.state_history.num_history_steps >= 1)
            else None,
            previous_processed_action=self.state_history.processed_actions[:, 1]
            if (self.state_history and self.state_history.num_history_steps >= 1)
            else None,
            # Environment state
            ground_heights=ground_heights,
            noisy_ground_heights=noisy.ground_heights,
            terrain=TerrainContext(
                self.terrain.height_points,
                self.terrain.height_samples,
            ),
            scene=scene_surface_context,
            body_contacts=body_contacts,
            current_contact_force_magnitudes=current_contact_force_magnitudes,
            prev_contact_force_magnitudes=self.prev_contact_force_magnitudes,
            dt=self.dt,
            progress_buf=self.progress_buf,
            # Contact tracking
            contact_body_ids=self.contact_body_ids,
            non_termination_contact_body_ids=self.non_termination_contact_body_ids,
            # Per-episode odometer corruption parameters
            odom_scale=self.odom_scale,
            odom_yaw_cos_sin=self.odom_yaw_cos_sin,
        )

        # Controllers populate their task-specific views
        self.control_manager.populate_context(ctx)

        return ctx

    def _build_scene_surface_context(self) -> SceneSurfaceContext:
        """Build scene-object surface tensors for component observations.

        Nearest-surface observations bind these fields unconditionally. Envs
        without object pointclouds receive empty tensors, which lets the compute
        kernel naturally fall back to terrain-only behavior.
        """
        has_object_pointclouds = (
            getattr(self.scene_lib, "_object_pointclouds", None) is not None
        )
        if self.scene_lib.num_objects_per_scene <= 0 or not has_object_pointclouds:
            object_pos = torch.zeros(self.num_envs, 0, 3, device=self.device)
            object_rot = torch.zeros(self.num_envs, 0, 4, device=self.device)
            neutral_pointclouds = torch.zeros(
                self.num_envs, 0, 0, 3, device=self.device
            )
            object_valid_mask = torch.zeros(
                self.num_envs, 0, dtype=torch.bool, device=self.device
            )
            return SceneSurfaceContext(
                object_pos=object_pos,
                object_rot=object_rot,
                neutral_pointclouds=neutral_pointclouds,
                object_valid_mask=object_valid_mask,
            )

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        object_state = self.simulator.get_object_root_state()
        return SceneSurfaceContext(
            object_pos=object_state.root_pos,
            object_rot=object_state.root_rot,
            neutral_pointclouds=self.scene_lib.get_scene_neutral_pointcloud(env_ids),
            object_valid_mask=self.scene_lib.get_per_object_valid_mask(env_ids),
        )

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

    def compute_reward(self, context: EnvContext):
        """Compute base rewards using the dynamic reward component system.

        Args:
            context: Pre-built EnvContext from self.context property.

        Subclasses should override this to add task-specific rewards, calling super().compute_reward() first.
        """
        grace_mask = self.get_has_reset_grace()

        # Process rewards
        combined_reward, reward_logging = self._process_rewards(context, grace_mask)

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

        if self.robot_config.reset_noise is not None:
            apply_reset_noise(
                reset_state=new_states,
                config=self.robot_config.reset_noise,
                dof_limits_lower=self.robot_config.kinematic_info.dof_limits_lower,
                dof_limits_upper=self.robot_config.kinematic_info.dof_limits_upper,
            )

        self.simulator.reset_envs(new_states, new_object_states, env_ids)

        default_mask = ~torch.isin(env_ids, ref_env_ids)
        if self.state_history is not None:
            self._reset_state_history(
                env_ids, default_mask, ref_env_ids, motion_ids, motion_times
            )

        # Reset control components after motion_manager has been reset
        self.control_manager.reset(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = False
        self.terminate_buf[env_ids] = False
        self.prev_contact_force_magnitudes[env_ids] = 0.0
        self._current_raw_action[env_ids] = 0.0
        self._current_processed_action[env_ids] = 0.0

        # Resample per-episode odometer corruption parameters.
        # These remain constant within an episode and are used by
        # corrupted_xy_offset_factory when present in observation components.
        n = len(env_ids)
        self.odom_scale[env_ids] = torch.empty(n, device=self.device).uniform_(
            self.config.odom_scale_range[0], self.config.odom_scale_range[1]
        )
        yaw_bias = torch.empty(n, device=self.device).uniform_(
            -self.config.odom_yaw_range_deg, self.config.odom_yaw_range_deg
        ) * (3.14159265358979 / 180.0)
        self.odom_yaw_cos_sin[env_ids, 0] = torch.cos(yaw_bias)
        self.odom_yaw_cos_sin[env_ids, 1] = torch.sin(yaw_bias)

        # Resample per-episode actuation/observation delays (no-op if unconfigured).
        if getattr(self, "_delay_cfg", None) is not None and self._delay_cfg.has_delay():
            self._sample_delays(env_ids)

        # Update cached noisy obs for the reset envs with fresh noise
        if self._current_noisy_obs is not None:
            current_state = self.simulator.get_robot_state()
            ground_heights = self.terrain.get_ground_heights(
                current_state.rigid_body_pos[env_ids, 0]
            ).squeeze(-1)
            obs_noise_cfg = self.simulator.config.domain_randomization.observation_noise
            noisy_subset = apply_observation_noise(
                obs_noise_cfg=obs_noise_cfg,
                robot_state=current_state,
                env_ids=env_ids,
                anchor_idx=self.robot_config.anchor_body_index,
                ground_heights=ground_heights,
            )
            self._current_noisy_obs.update_subset(env_ids, noisy_subset)

        # Recompute observations after reset to reflect new control component state
        # Invalidate and rebuild context since state changed
        self._current_context = None
        self.compute_observations(env_ids, context=self.context)

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
            body_contacts = current_state.rigid_body_contacts[default_env_ids][
                :, self.contact_body_ids
            ].bool()
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
            expanded_motion_times = motion_times.unsqueeze(1) + time_offsets.unsqueeze(
                0
            )

            # Clamp times to valid range
            motion_lengths = self.motion_lib.motion_lengths[motion_ids]
            expanded_motion_times = expanded_motion_times.clamp(min=0.0)
            expanded_motion_times = torch.min(
                expanded_motion_times,
                motion_lengths.unsqueeze(1).expand(-1, buffer_size),
            )

            # Flatten for motion_lib query
            flat_motion_ids = expanded_motion_ids.reshape(-1)
            flat_motion_times = expanded_motion_times.reshape(-1)

            # Query motion library
            historical_state = self.motion_lib.get_motion_state(
                flat_motion_ids, flat_motion_times
            )

            # Motion library data is recorded on flat terrain (height = 0)
            # Only simulator-based states need terrain height queries
            historical_ground_heights = torch.zeros(
                num_ref_envs, buffer_size, device=self.device
            )

            # Get contacts from motion library if available, otherwise zeros
            if historical_state.rigid_body_contacts is not None:
                flat_contacts = historical_state.rigid_body_contacts[
                    :, self.contact_body_ids
                ].bool()
                historical_body_contacts = flat_contacts.view(
                    num_ref_envs, buffer_size, -1
                )
            else:
                historical_body_contacts = torch.zeros(
                    num_ref_envs,
                    buffer_size,
                    len(self.contact_body_ids),
                    dtype=torch.bool,
                    device=self.device,
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
                dof_pos=historical_state.dof_pos.view(num_ref_envs, buffer_size, -1),
                dof_vel=historical_state.dof_vel.view(num_ref_envs, buffer_size, -1),
                ground_heights=historical_ground_heights,
                body_contacts=historical_body_contacts,
                actions=None,  # Zero actions for historical reset
            )

    ###############################################################
    # Motion and Visualization Helpers
    ###############################################################
    def create_motion_manager(self):
        """Instantiate motion manager from configuration."""
        MotionManagerClass = get_class(self.config.motion_manager._target_)

        fixed_motion_ids = None
        if self.scene_lib.num_scenes() > 0:
            humanoid_motion_ids = self.scene_lib.get_humanoid_motion_ids()
            if humanoid_motion_ids is not None:
                fixed_motion_ids = torch.tensor(
                    humanoid_motion_ids, dtype=torch.long, device=self.device
                )

        self.motion_manager = MotionManagerClass(
            config=self.config.motion_manager,
            num_envs=self.num_envs,
            env_dt=self.dt,
            device=self.device,
            motion_lib=self.motion_lib,
            fixed_motion_ids_per_env=fixed_motion_ids,
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

    def save_state(self) -> dict:
        """Save all mutable env state for later restoration.

        Snapshots the current state of the environment including robot state,
        simulator state, progress/reset/terminate buffers, and state history.
        This is useful for temporarily interrupting normal training to run
        evaluation episodes, then restoring to continue training from where
        it left off.

        Returns:
            Dictionary containing cloned copies of all mutable state tensors
        """
        snapshot = {
            "robot_state": self.simulator.get_robot_state(),
            "markers_state": self.get_markers_state(),
            "actions": self.simulator.get_current_actions(),
            "progress_buf": self.progress_buf.clone(),
            "reset_buf": self.reset_buf.clone(),
            "terminate_buf": self.terminate_buf.clone(),
            "respawn_root_offset": self.respawn_root_offset.clone(),
            "odom_scale": self.odom_scale.clone(),
            "odom_yaw_cos_sin": self.odom_yaw_cos_sin.clone(),
        }
        if getattr(self, "_delay_cfg", None) is not None and self._delay_cfg.has_delay():
            snapshot["_delay_state"] = {
                "action_delay": self._action_delay.clone(),
                "obs_delay": self._obs_delay.clone(),
                "action_delay_step": self._action_delay_step,
                "obs_delay_step": self._obs_delay_step,
                "action_delay_buf": (
                    self._action_delay_buf.clone()
                    if self._action_delay_buf is not None
                    else None
                ),
                "obs_delay_buf": (
                    {k: v.clone() for k, v in self._obs_delay_buf.items()}
                    if self._obs_delay_buf is not None
                    else None
                ),
            }
        if self.state_history is not None:
            snapshot["state_history"] = self.state_history.save_state()
        if self._current_noisy_obs is not None:
            from dataclasses import fields as dc_fields

            noisy = self._current_noisy_obs
            snapshot["_current_noisy_obs"] = NoisyObservations(
                **{f.name: getattr(noisy, f.name).clone() for f in dc_fields(noisy)}
            )
        if self.scene_lib.num_objects_per_scene > 0:
            snapshot["object_state"] = self.simulator.get_object_root_state()
        return snapshot

    def restore_state(self, snapshot: dict) -> None:
        """Restore env state from a previous save_state() snapshot.

        Restores all mutable state that was captured by save_state(),
        including robot positions/velocities, buffers, and state history.

        Args:
            snapshot: Dictionary from save_state() containing state tensors
        """
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.reset_envs(
            snapshot["robot_state"], snapshot.get("object_state"), env_ids
        )

        if "state_history" in snapshot and self.state_history is not None:
            self.state_history.load_state(snapshot["state_history"])

        self.progress_buf.copy_(snapshot["progress_buf"])
        self.reset_buf.copy_(snapshot["reset_buf"])
        self.terminate_buf.copy_(snapshot["terminate_buf"])
        self.respawn_root_offset.copy_(snapshot["respawn_root_offset"])
        if "odom_scale" in snapshot:
            self.odom_scale.copy_(snapshot["odom_scale"])
            self.odom_yaw_cos_sin.copy_(snapshot["odom_yaw_cos_sin"])
        self._current_noisy_obs = snapshot.get("_current_noisy_obs")
        self._current_context = None

        delay_state = snapshot.get("_delay_state")
        if delay_state is not None and getattr(self, "_delay_cfg", None) is not None:
            self._action_delay.copy_(delay_state["action_delay"])
            self._obs_delay.copy_(delay_state["obs_delay"])
            self._action_delay_step = delay_state["action_delay_step"]
            self._obs_delay_step = delay_state["obs_delay_step"]
            if (
                delay_state["action_delay_buf"] is not None
                and self._action_delay_buf is not None
            ):
                self._action_delay_buf.copy_(delay_state["action_delay_buf"])
            if delay_state["obs_delay_buf"] is not None:
                self._obs_delay_buf = {
                    k: v.clone() for k, v in delay_state["obs_delay_buf"].items()
                }

        # IsaacGym needs an extra step after state restore to sync internal state
        if "isaacgym" in self.simulator.config._target_.lower():
            self.simulator.step(snapshot["actions"], markers_callback=None)

    def close(self) -> None:
        """Release control-component and env-owned UI handles, then close
        the simulator. Safe to call multiple times."""
        control_manager = getattr(self, "control_manager", None)
        if control_manager is not None:
            for component in control_manager.components.values():
                component.close()

        ui = getattr(self, "_key_bindings", None)
        if ui is not None:
            ui.unregister_all()
            self._key_bindings = None

        simulator = getattr(self, "simulator", None)
        if simulator is not None:
            close = getattr(simulator, "close", None)
            if callable(close):
                close()
