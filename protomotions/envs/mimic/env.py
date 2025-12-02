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
"""Mimic environment for motion tracking tasks.

This module implements the Mimic environment, where an agent learns to reproduce
reference motions from a dataset (e.g., AMASS). It supports full-body tracking,
masked mimicry (partial observation), and various reward formulations.

Key Classes:
    - Mimic: The core environment class.

Key Features:
    - Reference motion management via MotionManager.
    - Comprehensive reward system (position, rotation, velocity, etc.).
    - Support for "Masked Mimic" with sparse observations.
    - Terrain integration.

## Mimic (additional members)

| Member | Type | Why Kept |
|--------|------|----------|
| `mimic_obs_cb` | `MimicObs` | Observation component |
| `masked_mimic_obs_cb` | `MaskedMimicObs` | Observation component |
| `prev_contact_force_magnitudes` | `Tensor` | Mutable state - persists across steps for force change reward |

"""

from typing import Optional

import torch
from torch import Tensor
from protomotions.utils.rotations import quat_diff_norm

from protomotions.envs.utils.humanoid import dof_to_local
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)
from protomotions.envs.base_env.env import BaseEnv
from protomotions.envs.obs.mimic_obs import MimicObs
from protomotions.envs.motion_manager.mimic_motion_manager import MimicMotionManager
from protomotions.envs.obs.masked_mimic_obs import MaskedMimicObs
from protomotions.envs.mimic.config import MimicEnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.simulator.base_simulator.simulator_state import ResetState


class Mimic(BaseEnv):
    """Mimic environment for reference motion tracking.

    Trains agents to imitate reference motions loaded from a library. The agent
    observes the target pose (and optionally future poses) and is rewarded for
    matching the reference motion's joint positions, rotations, and velocities.

    This environment serves as the foundation for:
    1. Standard Mimic (full-body tracking).
    2. Masked Mimic (tracking from partial/sparse observations).

    Attributes:
        motion_manager: Manages reference motion playback and sampling.
        mimic_obs_cb: Handles standard mimic observations (target poses).
        masked_mimic_obs_cb: Handles masked mimic observations (sparse inputs).
    """

    config: MimicEnvConfig
    motion_manager: MimicMotionManager

    def _init_config_with_robot(
        self, config: MimicEnvConfig, robot_config: RobotConfig
    ):
        """Initialize config fields that depend on robot configuration.

        Converts abstract body names to robot-specific names for observations.
        Reward component body names are resolved at runtime by the dynamic reward system.
        Must be called before super().__init__().

        Args:
            config: Environment configuration to modify in-place.
            robot_config: Robot morphology configuration.
        """

    def __init__(
        self,
        config: MimicEnvConfig,
        robot_config: RobotConfig,
        device: torch.device,
        terrain,
        simulator,
        scene_lib,
        motion_lib,
        *args,
        **kwargs,
    ):
        """Initialize the Mimic environment.

        Args:
            config: Environment configuration.
            robot_config: Robot morphology and control configuration.
            simulator_config: Simulator settings (physics, rendering).
            device: Device for tensor computations.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        # Initialize config fields that depend on robot configuration
        self._init_config_with_robot(config, robot_config)

        if config.sync_motion:
            config.ref_respawn_offset = 0
            print("Sync Motion: decimation was modified during simulator creation")

        super().__init__(
            config,
            robot_config,
            device,
            terrain,
            simulator,
            scene_lib,
            motion_lib,
            *args,
            **kwargs,
        )

        if self.config.sync_motion:
            assert (
                self.motion_lib is not None
            ), "Motion lib must be set if sync_motion is True"

        if self.config.mimic_obs.enabled:
            self.mimic_obs_cb = MimicObs(self.config.mimic_obs, self)

        if self.config.masked_mimic_obs.enabled:
            self.masked_mimic_obs_cb = MaskedMimicObs(
                self.config.masked_mimic_obs, self
            )

        # Initialize previous contact force magnitudes for impact penalty reward
        # Shape: [num_envs, num_bodies] - stores magnitude of contact forces from previous step
        num_bodies = robot_config.kinematic_info.num_bodies
        self.prev_contact_force_magnitudes = torch.zeros(
            self.num_envs, num_bodies, dtype=torch.float, device=self.device
        )

    def create_motion_manager(self):
        """Create Mimic-specific motion manager with optional scene-based motion assignment.

        If scenes are present, assigns fixed motion IDs to environments based on scene requirements.
        """

        common_args = {
            "config": self.config.motion_manager,
            "num_envs": self.num_envs,
            "env_dt": self.dt,
            "device": self.device,
            "motion_lib": self.motion_lib,
        }
        if self.scene_lib.num_scenes() > 0:
            fixed_motion_ids_per_env = self.scene_lib.get_humanoid_motion_ids()
            if fixed_motion_ids_per_env is not None:
                fixed_motion_ids_per_env = torch.tensor(
                    fixed_motion_ids_per_env, dtype=torch.long, device=self.device
                )
                common_args["fixed_motion_ids_per_env"] = fixed_motion_ids_per_env

        self.motion_manager = MimicMotionManager(**common_args)

    def get_has_reset_grace(self):
        """Check if environments are in the grace period after reset.

        Returns:
            Boolean tensor indicating which environments are within reset_grace_period steps of last reset.
        """
        return self.progress_buf <= self.config.reset_grace_period

    def create_visualization_markers(self, headless: bool):
        """Create reference motion visualization markers.

        For standard Mimic: red markers show current target pose.
        For MaskedMimic: color-coded markers (blue/yellow/red) show time-to-target.

        Args:
            headless: If True, returns empty dict (no visualization).

        Returns:
            Dictionary of visualization marker configurations
        """
        visualization_markers = super().create_visualization_markers(headless)

        # If headless, super() returns {} and we should not create markers
        if headless:
            return visualization_markers

        body_markers = []
        if self.config.masked_mimic_obs.enabled:
            body_names = self.robot_config.trackable_bodies_subset
        else:
            body_names = self.robot_config.kinematic_info.body_names

        for body_name in body_names:
            if (
                self.robot_config.mimic_small_marker_bodies is not None
                and body_name in self.robot_config.mimic_small_marker_bodies
            ):
                body_markers.append(MarkerConfig(size="small"))
            else:
                body_markers.append(MarkerConfig(size="regular"))
            # Red markers: in maskedmimic used for time to target <= 0.1 seconds
            body_markers_red_cfg = VisualizationMarkerConfig(
                type="sphere", color=(1.0, 0.0, 0.0), markers=body_markers
            )
            visualization_markers["body_markers_red"] = body_markers_red_cfg

        # For masked mimic, create two additional separate marker groups for different time ranges
        if self.config.masked_mimic_obs.enabled:
            # Blue markers: time to target > 1 second
            body_markers_blue_cfg = VisualizationMarkerConfig(
                type="sphere", color=(0.0, 0.0, 1.0), markers=body_markers
            )
            visualization_markers["body_markers_blue"] = body_markers_blue_cfg

            # Yellow markers: 0.1 < time to target <= 1 second
            body_markers_yellow_cfg = VisualizationMarkerConfig(
                type="sphere", color=(1.0, 1.0, 0.0), markers=body_markers
            )
            visualization_markers["body_markers_yellow"] = body_markers_yellow_cfg

        return visualization_markers

    def get_markers_state(self):
        """Compute visualization marker positions for reference motion targets.

        Returns:
            Dictionary mapping marker names to MarkerState with target body positions
        """
        if self.simulator.headless:
            return {}

        markers_state = super().get_markers_state()

        # Update mimic markers
        if self.config.masked_mimic_obs.enabled:
            # For masked mimic, use the time of the first visible target pose
            pose_masks = (
                self.masked_mimic_obs_cb.masked_mimic_target_poses_masks
            )  # [num_envs, num_future_steps]
            first_valid_indices = torch.argmax(pose_masks.float(), dim=1)  # [num_envs]

            # Get the target times for the first visible pose
            env_indices = torch.arange(self.num_envs, device=self.device)
            target_motion_times = self.masked_mimic_obs_cb.target_times[
                env_indices, first_valid_indices
            ]

            # Compute time to target (time difference between target and current)
            current_motion_times = self.motion_manager.motion_times
            time_to_target = target_motion_times - current_motion_times  # [num_envs]

            ref_state = self.motion_lib.get_motion_state(
                self.motion_manager.motion_ids, target_motion_times
            )

            target_pos = ref_state.rigid_body_pos.clone()
            target_pos += (
                self.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                    target_pos
                )
            )

            num_conditionable_bodies = len(
                self.masked_mimic_obs_cb.conditionable_body_ids
            )
            target_pos = target_pos[
                :, self.masked_mimic_obs_cb.conditionable_body_ids, :
            ]

            # Get the translation view using the first valid pose index for each environment
            bodies_masks_reshaped = self.masked_mimic_obs_cb.masked_mimic_target_bodies_masks.view(
                self.num_envs,
                self.config.masked_mimic_obs.masked_mimic_target_pose.num_future_steps,
                num_conditionable_bodies,
                2,
            )

            # Use advanced indexing to get the correct time step for each environment
            translation_view = bodies_masks_reshaped[
                env_indices, first_valid_indices, :, 0
            ]
            active_translations = (
                translation_view == 1
            )  # [num_envs, num_conditionable_bodies]

            # Create masks for each time range
            # Blue: > 1 second
            blue_time_mask = (
                (time_to_target > 1.0).unsqueeze(1).expand(-1, num_conditionable_bodies)
            )  # [num_envs, num_conditionable_bodies]
            # Yellow: 0.1 to 1 second
            yellow_time_mask = (
                ((time_to_target > 0.1) & (time_to_target <= 1.0))
                .unsqueeze(1)
                .expand(-1, num_conditionable_bodies)
            )
            # Red: <= 0.1 seconds
            red_time_mask = (
                (time_to_target <= 0.1)
                .unsqueeze(1)
                .expand(-1, num_conditionable_bodies)
            )

            # Combine time masks with active translations
            blue_markers = active_translations & blue_time_mask
            yellow_markers = active_translations & yellow_time_mask
            red_markers = active_translations & red_time_mask

            # Create inactive marker masks for each color group
            inactive_blue = ~blue_markers
            inactive_yellow = ~yellow_markers
            inactive_red = ~red_markers

            # Create separate target positions for each color group
            target_pos_blue = target_pos.clone()
            target_pos_yellow = target_pos.clone()
            target_pos_red = target_pos.clone()

            # Move inactive markers off screen (add large offset)
            target_pos_blue[inactive_blue] += 100
            target_pos_yellow[inactive_yellow] += 100
            target_pos_red[inactive_red] += 100

            # Add marker states for each color group
            markers_state["body_markers_blue"] = MarkerState(
                translation=target_pos_blue.view(self.num_envs, -1, 3),
                orientation=torch.zeros(
                    self.num_envs, num_conditionable_bodies, 4, device=self.device
                ),
            )
            markers_state["body_markers_yellow"] = MarkerState(
                translation=target_pos_yellow.view(self.num_envs, -1, 3),
                orientation=torch.zeros(
                    self.num_envs, num_conditionable_bodies, 4, device=self.device
                ),
            )
            markers_state["body_markers_red"] = MarkerState(
                translation=target_pos_red.view(self.num_envs, -1, 3),
                orientation=torch.zeros(
                    self.num_envs, num_conditionable_bodies, 4, device=self.device
                ),
            )
        else:
            # For regular mimic, use the current motion time
            ref_state = self.motion_lib.get_motion_state(
                self.motion_manager.motion_ids, self.motion_manager.motion_times
            )

            target_pos = ref_state.rigid_body_pos.clone()
            target_pos += (
                self.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                    target_pos
                )
            )

            target_pos = target_pos.view(self.num_envs, -1, 3)
            markers_state["body_markers_red"] = MarkerState(
                translation=target_pos,
                orientation=torch.zeros(
                    self.num_envs, target_pos.shape[1], 4, device=self.device
                ),
            )

        return markers_state

    def get_obs(self):
        """Get observations including mimic-specific target pose information.

        Returns:
            Dictionary of observations with base obs + mimic target poses
        """
        obs = super().get_obs()
        if self.config.mimic_obs.enabled:
            mimic_obs = self.mimic_obs_cb.get_obs()
            obs.update(mimic_obs)
        if self.config.masked_mimic_obs.enabled:
            masked_mimic_obs = self.masked_mimic_obs_cb.get_obs()
            obs.update(masked_mimic_obs)
        return obs

    def check_resets_and_terminations(self):
        """Check reset and termination conditions for mimic environment.

        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors
        """
        # Get base reset and termination conditions
        reset_buf, terminate_buf = super().check_resets_and_terminations()

        # Check for early termination based on mimic-specific metrics
        if self.config.mimic_early_termination:
            early_term_mask = torch.zeros_like(self.reset_buf, dtype=torch.bool)
            for entry in self.config.mimic_early_termination:
                key = entry.mimic_early_termination_key
                thresh = entry.mimic_early_termination_thresh
                value = self.extras[f"mimic_other/{key}"]

                if entry.less_than:
                    entry_should_terminate = value < thresh
                else:
                    entry_should_terminate = value > thresh

                early_term_mask |= entry_should_terminate

            # Don't apply early termination during grace period
            has_reset_grace = self.get_has_reset_grace()
            early_term_mask &= ~has_reset_grace

            reset_buf = reset_buf | early_term_mask
            terminate_buf = terminate_buf | early_term_mask

        # Check if motion clip has finished
        done_clip = self.motion_manager.get_done_tracks()
        reset_buf = reset_buf | done_clip
        if not self.config.mimic_bootstrap_on_episode_end:
            terminate_buf = terminate_buf | done_clip

        return reset_buf, terminate_buf

    def get_spawn_to_ref_pose_offset_with_terrain_height_correction(
        self, target_pos: Tensor, env_ids: Optional[Tensor] = None
    ) -> Tensor:
        """
        target_pos: (num_envs, num_bodies, 3), in reference motion frame without spawning offset
        env_ids: (num_envs,)
        returns: (num_envs, num_bodies, 3)

        For XY offset, all bodies share the same self.respawn_root_offset.
        For Z offset, all bodies share the same offset computed from the body furthest below terrain.
        This ensures the rigid body structure is preserved during spawning.
        """

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        new_offset = torch.zeros_like(target_pos)
        new_offset[:, :, :2] = self.respawn_root_offset[env_ids, :2][:, None, :]

        # Calculate Z offset (skip if terrain is flat and optimization is enabled)
        if not self.skip_height_correction:
            target_pos_spawned = target_pos.clone() + new_offset
            z_offset = self.terrain.find_terrain_height_for_max_below_body(
                target_pos_spawned
            )  # (num_envs,)
            new_offset[:, :, 2] = z_offset.unsqueeze(
                1
            )  # Broadcast to (num_envs, num_bodies)

        # new_offset[:, :, 2] += self.config.ref_respawn_offset

        return new_offset  # (num_envs, num_bodies, 3)

    def _get_reward_context(self):
        """Get the context for reward computation.

        Returns:
            Dict of variables available for eval (e.g., current_state, ref_state)
        """
        reward_context = super()._get_reward_context()

        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids, self.motion_manager.motion_times
        )

        ref_gt = ref_state.rigid_body_pos.clone()
        ref_gt += self.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_gt
        )
        ref_state.rigid_body_pos = ref_gt

        hinge_axes_map = self.robot_config.kinematic_info.hinge_axes_map
        ref_lr = dof_to_local(ref_state.dof_pos, hinge_axes_map, True)

        # Reuse current_state from parent context
        current_state = reward_context["current_state"]
        lr = dof_to_local(current_state.dof_pos, hinge_axes_map, True)

        reward_context.update(
            {
                "ref_state": ref_state,
                "ref_lr": ref_lr,
                "lr": lr,
                "current_contact_force_magnitudes": torch.norm(
                    current_state.rigid_body_contact_forces, dim=-1
                ),
                "prev_contact_force_magnitudes": self.prev_contact_force_magnitudes,
            }
        )

        return reward_context

    def compute_reward(self):
        """Compute motion tracking rewards using the dynamic reward component system.

        Builds a context with current and reference states, then calls the dynamic
        reward computation system. Logs tracking errors and contact metrics to extras.
        """
        super().compute_reward()

        # From here we use the reward context to log additional metrics

        # Get reference state from motion library
        ref_state = self.motion_lib.get_motion_state(
            self.motion_manager.motion_ids, self.motion_manager.motion_times
        )

        ref_gt = ref_state.rigid_body_pos.clone()
        ref_gt += self.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_gt
        )
        ref_state.rigid_body_pos = ref_gt

        hinge_axes_map = self.robot_config.kinematic_info.hinge_axes_map
        ref_lr = dof_to_local(ref_state.dof_pos, hinge_axes_map, True)

        # Get current state from simulator
        current_state = self.simulator.get_robot_state()
        lr = dof_to_local(current_state.dof_pos, hinge_axes_map, True)

        # Compute logging metrics
        gt = current_state.rigid_body_pos
        gr = current_state.rigid_body_rot
        ref_gr = ref_state.rigid_body_rot

        left_foot_indices = self._resolve_body_indices(["all_left_foot_bodies"])
        right_foot_indices = self._resolve_body_indices(["all_right_foot_bodies"])

        # Use simulator's canonical method to compute binary contacts
        sim_contacts = current_state.rigid_body_contacts

        # Get reference contact states (smoothed: float in [0, 1] if smoothing enabled, else binary)
        ref_contacts = ref_state.rigid_body_contacts.float()

        # Compute L1 distance between predicted binary contacts and smoothed GT contacts
        contact_mismatches = torch.abs(
            sim_contacts - ref_contacts
        )  # [num_envs, num_bodies]

        gt_per_joint_err = (ref_gt - gt).pow(2).sum(-1).sqrt()
        gt_err = gt_per_joint_err.mean(-1)
        max_joint_err = gt_per_joint_err.max(-1)[0]
        rh_err = (ref_gt - gt)[:, 0, -1].abs()

        gr_diff = quat_diff_norm(gr, ref_gr, True)
        gr_err = gr_diff.mean(-1)
        gr_err_degrees = gr_err * 180 / torch.pi
        max_gr_err = gr_diff.max(-1)[0]
        max_gr_err_degrees = max_gr_err * 180 / torch.pi

        lr_diff = quat_diff_norm(lr, ref_lr, True)
        lr_err = lr_diff.mean(-1)
        lr_err_degrees = lr_err * 180 / torch.pi
        max_lr_err = lr_diff.max(-1)[0]
        max_lr_err_degrees = max_lr_err * 180 / torch.pi

        gt_left_foot_contact = (
            ref_state.rigid_body_contacts[:, left_foot_indices].float().mean(dim=-1)
        )
        gt_right_foot_contact = (
            ref_state.rigid_body_contacts[:, right_foot_indices].float().mean(dim=-1)
        )
        pred_left_foot_contact = (
            current_state.rigid_body_contacts[:, left_foot_indices].float().mean(dim=-1)
        )
        pred_right_foot_contact = (
            current_state.rigid_body_contacts[:, right_foot_indices]
            .float()
            .mean(dim=-1)
        )

        other_log_terms = {
            "gt_err": gt_err,
            "gr_err": gr_err,
            "gr_err_degrees": gr_err_degrees,
            "lr_err_degrees": lr_err_degrees,
            "max_joint_err": max_joint_err,
            "max_lr_err_degrees": max_lr_err_degrees,
            "max_gr_err_degrees": max_gr_err_degrees,
            "root_height_error": rh_err,
            "gt_left_foot_contact": gt_left_foot_contact,
            "gt_right_foot_contact": gt_right_foot_contact,
            "pred_left_foot_contact": pred_left_foot_contact,
            "pred_right_foot_contact": pred_right_foot_contact,
            # Contact matching metrics
            "contact_mismatch_mean": contact_mismatches.mean(dim=-1),
            "contact_match_accuracy": 1.0 - contact_mismatches.mean(dim=-1),
        }

        if self.config.masked_mimic_obs.enabled:
            mask = self.masked_mimic_obs_cb.masked_mimic_target_bodies_masks.view(
                self.num_envs,
                self.masked_mimic_obs_cb.config.masked_mimic_target_pose.num_future_steps,
                self.masked_mimic_obs_cb.num_conditionable_bodies,
                2,
            )[:, 0]
            translation_mask = mask[..., 0].unsqueeze(-1)
            rotation_mask = mask[..., 1]

            translation_mask_coeff = translation_mask.float().sum(1).view(-1) + 1e-6
            rotation_mask_coeff = rotation_mask.float().sum(1) + 1e-6

            active_bodies_ids = self.masked_mimic_obs_cb.conditionable_body_ids

            masked_gt_err = (
                (ref_gt - gt)[:, active_bodies_ids]
                .mul(translation_mask)
                .pow(2)
                .sum(-1)
                .sqrt()
                .sum(-1)
                .div(translation_mask_coeff)
            )
            masked_max_joint_err = (
                (ref_gt - gt)[:, active_bodies_ids]
                .mul(translation_mask)
                .pow(2)
                .sum(-1)
                .sqrt()
                .max(-1)[0]
            )
            masked_gr_err = (
                quat_diff_norm(gr, ref_gr, True)[:, active_bodies_ids]
                .mul(rotation_mask)
                .sum(-1)
                .div(rotation_mask_coeff)
            )
            masked_gr_err_degrees = masked_gr_err * 180 / torch.pi

            other_log_terms["masked_gt_err"] = masked_gt_err
            other_log_terms["masked_max_joint_err"] = masked_max_joint_err
            other_log_terms["masked_gr_err"] = masked_gr_err
            other_log_terms["masked_gr_err_degrees"] = masked_gr_err_degrees

        for name, value in other_log_terms.items():
            self.extras[f"mimic_other/{name}"] = value

        # Update previous contact force magnitudes for next timestep
        self.prev_contact_force_magnitudes = torch.norm(
            current_state.rigid_body_contact_forces, dim=-1
        ).clone()

    def compute_observations(self, env_ids=None):
        """Compute observations including mimic-specific target poses.

        Args:
            env_ids: Environment indices to update (None = all)
        """
        super().compute_observations(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device).long()

        if self.config.mimic_obs.enabled:
            self.mimic_obs_cb.compute_observations(env_ids)
        if self.config.masked_mimic_obs.enabled:
            self.masked_mimic_obs_cb.compute_observations(env_ids)

    def process_actions(self, actions):
        """Process actions (zeros them out for kinematic playback mode).

        Args:
            actions: Raw actions [batch, action_dim]

        Returns:
            Processed actions
        """
        if self.config.sync_motion:
            actions *= 0

        return super().process_actions(actions)

    def post_physics_step(self):
        """Update state after physics step, handling motion tracking and sync mode.

        In sync_motion mode, plays back reference motions kinematically.
        In training mode, advances motion manager and updates masked mimic observations.
        """
        if self.config.sync_motion:
            # specialized logic for play back motion in env_kinematic_playback.py, not used in training or testing.
            sync_motion_dt = (
                self.simulator.decimation * 1.0 / self.simulator.config.sim.fps
            )
            self.motion_manager.motion_times += sync_motion_dt

            # since not calling reset(), we need to handle done tracks ourselves
            done_clip = self.motion_manager.get_done_tracks()
            if any(done_clip):
                # During motion sync, re-sample new motion ids and the respawn positions.
                done_env_ids = torch.where(done_clip)[0]
                self.motion_manager.sample_motions(done_env_ids)

            ref_state = self.motion_lib.get_motion_state(
                self.motion_manager.motion_ids,
                self.motion_manager.motion_times,
            )
            ref_state.dof_vel *= 0
            ref_state.rigid_body_vel *= 0
            ref_state.rigid_body_ang_vel *= 0
            ref_reset_state = ResetState.from_robot_state(ref_state)

            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

            # Get object reference state (empty if no scenes)
            ref_object_state = self.scene_lib.get_scene_pose(
                env_ids,
                self.motion_manager.motion_times,
                self.config.ref_object_respawn_offset,
            )
            ref_object_state.root_vel = torch.zeros_like(ref_object_state.root_pos)
            ref_object_state.root_ang_vel = torch.zeros_like(ref_object_state.root_pos)

            offset = self.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                ref_reset_state.root_pos[:, None, :], env_ids
            ).squeeze(1)  # (num_envs, 3)
            ref_reset_state.root_pos += offset  # (num_envs, 3)
            if self.scene_lib.num_scenes() > 0:
                # (num_envs, num_objects_per_scene, 3)
                ref_object_state.root_pos += offset.unsqueeze(1)

            self.simulator.reset_envs(ref_reset_state, ref_object_state, env_ids)

            # prevent calling simulator.reset_envs twice (reset() will also call it otherwise)
            self.progress_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0
            self.terminate_buf[env_ids] = 0

            super().post_physics_step()

        else:
            self.motion_manager.post_physics_step()
            super().post_physics_step()

            if self.config.masked_mimic_obs.enabled:
                self.masked_mimic_obs_cb.post_physics_step()

    def user_reset(self):
        """Force reset on user input, invalidating motion times."""
        super().user_reset()
        self.motion_manager.motion_times[:] = 100000000000

    def reset(self, env_ids=None, sample_flat=False, disable_motion_resample=False):
        """Reset environments and prepare masked mimic conditioning.

        Args:
            env_ids: Environment indices to reset (None = all)
            sample_flat: If True, spawn on flat terrain
            disable_motion_resample: If True, skip resampling motions (use existing motion_ids/times)

        Returns:
            Tuple of (observations, info_dict)
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if len(env_ids) > 0:
            if self.config.masked_mimic_obs.enabled:
                # Mark these envs as requiring reset. Will actually reset the conditioning when compute_obs is called.
                # This ensures the motion manager has been updated and has re-sampled a new motion and time.
                self.masked_mimic_obs_cb.envs_requiring_reset = env_ids
            # Reset previous contact force magnitudes for all contact bodies
            self.prev_contact_force_magnitudes[env_ids] = 0.0
        return super().reset(
            env_ids, sample_flat, disable_motion_resample=disable_motion_resample
        )
