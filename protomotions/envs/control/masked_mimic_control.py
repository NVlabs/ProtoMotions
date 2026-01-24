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
"""Masked mimic control component for sparse conditioning tasks.

This component extends MimicControl with:
- Target pose masks (which poses are visible)
- Target body masks (which bodies are conditioned)
- Time-based pose visibility with beta distribution sampling
- Color-coded visualization markers (blue/yellow/red by time-to-target)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class FixedBodyCondition:
    """Fixed conditioning for a specific body."""
    body_name: str
    constraint_state: int  # 0=translation only, 1=both, 2=rotation only


@dataclass
class MaskedMimicControlConfig(MimicControlConfig):
    """Configuration for masked mimic control component.
    
    Extends MimicControlConfig with masking parameters.
    
    Attributes:
        num_masked_future_steps: Number of future pose samples for masked mimic conditioning.
        time_alpha: Alpha parameter for beta distribution time sampling.
        time_beta: Beta parameter for beta distribution time sampling.
        repeat_mask_probability: Probability of repeating previous body mask.
        force_max_conditioned_bodies_prob: Probability of conditioning on all bodies.
        force_small_num_conditioned_bodies_prob: Probability of using small number of bodies.
        visible_target_pose_prob: Probability of making a target pose visible.
        fixed_conditioning: Optional list of fixed body conditioning entries.
        fully_hidden_pose_prob: Probability that a pose is fully hidden (all bodies masked out).
    
    Note: The inherited num_future_steps from MimicControlConfig controls the base mimic
    reference data (mimic_ref_pos, etc.). num_masked_future_steps is for masked mimic specific obs.
    """
    _target_: str = "protomotions.envs.control.masked_mimic_control.MaskedMimicControl"
    
    num_masked_future_steps: int = 5
    
    # Time sampling (beta distribution)
    time_alpha: float = 2.0
    time_beta: float = 5.0
    
    # Joint masking
    repeat_mask_probability: float = 0.8
    force_max_conditioned_bodies_prob: float = 0.1
    force_small_num_conditioned_bodies_prob: float = 0.1
    visible_target_pose_prob: float = 0.8
    
    # Fixed conditioning (optional)
    fixed_conditioning: Optional[List[FixedBodyCondition]] = None
    
    # Probability that a pose is fully hidden (all bodies masked out)
    fully_hidden_pose_prob: float = 0.1


class MaskedMimicControl(MimicControl):
    """Control component for masked mimic tasks.
    
    Extends MimicControl with sparse conditioning state where only a subset of 
    bodies and poses are visible to the agent at each timestep. Handles time 
    sampling with beta distribution and body mask sampling with configurable 
    probabilities.
    
    Attributes:
        masked_mimic_target_poses_masks: Which future poses are visible [num_envs, num_future_steps].
        masked_mimic_target_bodies_masks: Which bodies are visible [num_envs, num_future_steps * num_bodies * 2].
        target_times: Target times for future poses [num_envs, num_future_steps].
        conditionable_body_ids: IDs of bodies that can be conditioned on.
    """
    
    config: MaskedMimicControlConfig
    
    def __init__(self, config: MaskedMimicControlConfig, env: "BaseEnv"):
        """Initialize masked mimic control component.
        
        Args:
            config: Component configuration.
            env: Parent environment instance.
        """
        super().__init__(config, env)
        
        # Get conditionable body IDs from robot config
        self._all_body_names = self.env.robot_config.kinematic_info.body_names
        self.conditionable_body_ids = torch.tensor(
            [
                self._all_body_names.index(name)
                for name in self.env.robot_config.trackable_bodies_subset
            ],
            device=self.env.device,
            dtype=torch.long,
        )
        self.num_conditionable_bodies = len(self.conditionable_body_ids)
        
        # Initialize masked mimic state buffers
        self.masked_mimic_target_poses_masks = torch.zeros(
            self.env.num_envs,
            self.config.num_masked_future_steps,
            dtype=torch.bool,
            device=self.env.device,
        )
        
        # Format: [num_envs, num_future_steps * num_conditionable_bodies * 2]
        # The *2 is for translation and rotation masks
        self.masked_mimic_target_bodies_masks = torch.zeros(
            self.env.num_envs,
            self.config.num_masked_future_steps * self.num_conditionable_bodies * 2,
            dtype=torch.bool,
            device=self.env.device,
        )
        
        # Target times for each future step
        self.target_times = torch.zeros(
            self.env.num_envs,
            self.config.num_masked_future_steps,
            dtype=torch.float,
            device=self.env.device,
        )
        
        self._initialized = False
    
    def reset(self, env_ids: Tensor):
        """Reset masked mimic state for given environments.
        
        Initializes target times from current motion time and samples all future
        time steps and body masks.
        
        Args:
            env_ids: Environment indices to reset.
        """
        # Call parent reset (resets contact forces)
        super().reset(env_ids)
        
        if len(env_ids) == 0:
            return
        
        # Initialize time steps from current time
        new_times = self.env.motion_manager.motion_times[env_ids]
        self.target_times[env_ids] = new_times.unsqueeze(-1)
        
        # Reset masks
        self.masked_mimic_target_poses_masks[env_ids] = False
        self.masked_mimic_target_bodies_masks[env_ids] = False
        
        # Sample all future steps
        for _ in range(self.config.num_masked_future_steps):
            self._shift_and_sample_time_steps(env_ids)
            self._shift_and_sample_body_masks(env_ids)
        
        self._initialized = True
    
    def step(self):
        """Update masked mimic state after physics step.
        
        Identifies environments where the current time has passed the first target
        time and triggers resampling of time steps and body masks for those environments.
        """
        # Call parent step (currently no-op)
        super().step()
        
        if not self._initialized:
            return
        
        current_time = self.env.motion_manager.motion_times
        
        # Check which envs have outdated target times
        outdated_target_times = current_time >= self.target_times[:, 0]
        resample_env_ids = torch.nonzero(outdated_target_times).squeeze(-1)
        
        if len(resample_env_ids) > 0:
            self._shift_and_sample_time_steps(resample_env_ids)
            self._shift_and_sample_body_masks(resample_env_ids)
    
    def _shift_and_sample_time_steps(self, env_ids: Tensor):
        """Shift target time steps forward and sample a new future time step.
        
        Samples new time steps using a beta distribution to determine the offset from
        the last conditioned time. The new time is added to the end after shifting
        existing times forward.
        
        Args:
            env_ids: Tensor of environment indices to update.
        """
        num_envs = len(env_ids)
        
        current_times = self.env.motion_manager.motion_times[env_ids]
        motion_ids = self.env.motion_manager.motion_ids[env_ids]
        motion_lengths = self.env.motion_lib.motion_lengths[motion_ids]
        
        # Get last conditioned time for each environment
        last_conditioned_times = torch.max(self.target_times[env_ids], dim=1)[0]
        remaining_times = motion_lengths - last_conditioned_times
        
        # Sample beta values for all environments at once
        beta_dist = torch.distributions.Beta(
            self.config.time_alpha, self.config.time_beta
        )
        beta_samples = beta_dist.sample((num_envs,)).to(self.env.device)
        
        # Calculate new times
        time_offsets = beta_samples * remaining_times
        absolute_times = last_conditioned_times + time_offsets
        
        # Clip absolute times to be within valid bounds
        min_times = current_times + self.env.dt
        absolute_times = torch.clamp(absolute_times, min_times, motion_lengths)
        
        # Shift existing frames and add new ones
        self.target_times[env_ids, :-1] = self.target_times[env_ids, 1:]
        self.target_times[env_ids, -1] = absolute_times
    
    def _shift_and_sample_body_masks(self, env_ids: Tensor):
        """Shift body masks forward and sample new masks for the last time step.
        
        Moves existing body masks one position forward (removing the first) and
        samples fresh masks for the newly added future time step.
        
        Args:
            env_ids: Tensor of environment indices to update masks for.
        """
        single_step_mask_size = self.num_conditionable_bodies * 2
        num_future_steps = self.config.num_masked_future_steps
        
        # Reshape masks to (num_envs, num_future_steps, single_step_mask_size)
        masks = self.masked_mimic_target_bodies_masks[env_ids].view(
            len(env_ids), num_future_steps, single_step_mask_size
        )
        
        shifted_masks = masks.roll(shifts=-1, dims=1)
        
        # Sample new masks for the last position
        new_body_masks = self._sample_body_masks(env_ids)
        shifted_masks[:, -1] = new_body_masks.view(len(env_ids), single_step_mask_size)
        
        # With probability (1 - visible_target_pose_prob), fully hide the pose
        # This replaces the old fully_hidden_pose_prob logic
        hide_prob = 1.0 - self.config.visible_target_pose_prob
        if hide_prob > 0:
            fully_hidden = torch.rand(len(env_ids), device=self.env.device) < hide_prob
            shifted_masks[fully_hidden, -1] = False
        
        # Update the masks
        self.masked_mimic_target_bodies_masks[env_ids] = shifted_masks.view(
            len(env_ids), -1
        )
        
        # Update pose visibility masks
        reshaped_masked_bodies_masks = self.masked_mimic_target_bodies_masks[
            env_ids
        ].view(len(env_ids), self.config.num_masked_future_steps, -1)
        
        for i in range(self.config.num_masked_future_steps):
            any_visible_joint = reshaped_masked_bodies_masks[:, i].any(dim=-1)
            self.masked_mimic_target_poses_masks[env_ids, i] = any_visible_joint
    
    def _sample_body_masks(self, env_ids: Tensor) -> Tensor:
        """Sample body visibility masks for conditioning.
        
        With a certain probability, repeats the previous mask; otherwise samples
        new masks to determine which bodies and their properties (translation/rotation)
        are visible for conditioning.
        
        Note: Never repeats an empty (all-False) mask - always samples fresh in that case.
        
        Args:
            env_ids: Tensor of environment indices to sample masks for.
            
        Returns:
            Tensor of shape (num_envs, num_conditionable_bodies * 2) with boolean masks.
        """
        num_envs = len(env_ids)
        
        # Get previous masks (last step)
        single_step_mask_size = self.num_conditionable_bodies * 2
        previous_masks = self.masked_mimic_target_bodies_masks[
            env_ids, -single_step_mask_size:
        ].view(num_envs, self.num_conditionable_bodies, 2)
        
        # Check which previous masks are empty (all False) - never repeat empty masks
        previous_is_empty = ~previous_masks.any(dim=-1).any(dim=-1)  # [num_envs]
        
        # Check if we should repeat previous masks (but not if previous is empty)
        repeat_mask_prob = self.config.repeat_mask_probability
        repeat_mask = torch.rand(num_envs, device=self.env.device) < repeat_mask_prob
        repeat_mask = repeat_mask & ~previous_is_empty  # Don't repeat empty masks
        
        if repeat_mask.any():
            # For environments that should repeat, return previous mask
            new_mask = torch.zeros(
                num_envs,
                self.num_conditionable_bodies,
                2,
                dtype=torch.bool,
                device=self.env.device,
            )
            new_mask[repeat_mask] = previous_masks[repeat_mask]
            
            # For environments that shouldn't repeat, sample new masks
            sample_new_mask = ~repeat_mask
            if sample_new_mask.any():
                num_new_masks = sample_new_mask.sum().item()
                new_sampled_masks = self._sample_new_body_masks(num_new_masks)
                new_mask[sample_new_mask] = new_sampled_masks.view(
                    num_new_masks, self.num_conditionable_bodies, 2
                )
            
            return new_mask.view(num_envs, -1)
        
        return self._sample_new_body_masks(num_envs).view(num_envs, -1)
    
    def _sample_new_body_masks(self, num_envs: int) -> Tensor:
        """Sample fresh body masks for multiple environments.
        
        Determines the number of active bodies and their constraint states
        (translation only, rotation only, or both). Supports fixed conditioning
        or random sampling strategies.
        
        Args:
            num_envs: Number of environments to sample masks for.
            
        Returns:
            Tensor of shape (num_envs, num_conditionable_bodies * 2) with boolean masks.
        """
        # Sample number of active bodies
        num_bodies_mask = (
            torch.rand(num_envs, device=self.env.device)
            >= self.config.force_small_num_conditioned_bodies_prob
        )
        max_num_bodies = torch.where(num_bodies_mask, self.num_conditionable_bodies, 3)
        num_active_bodies = torch.round(
            torch.rand(num_envs, device=self.env.device) * (max_num_bodies - 1) + 1
        ).long()
        
        max_bodies_mask = (
            torch.rand(num_envs, device=self.env.device)
            <= self.config.force_max_conditioned_bodies_prob
        )
        num_active_bodies[max_bodies_mask] = self.num_conditionable_bodies
        
        # Sample which bodies are active
        active_body_ids = torch.zeros(
            num_envs,
            self.num_conditionable_bodies,
            device=self.env.device,
            dtype=torch.bool,
        )
        constraint_states = torch.randint(
            0, 3, (num_envs, self.num_conditionable_bodies), device=self.env.device
        )
        
        if self.config.fixed_conditioning is None:
            # Parallelized sampling without replacement
            rand_values = torch.rand(
                num_envs, self.num_conditionable_bodies, device=self.env.device
            )
            # Get ranks: rank[i, j] is the rank of body j in env i among the random values
            ranks = torch.argsort(torch.argsort(rand_values, dim=1), dim=1)
            active_body_ids[:] = ranks < num_active_bodies.unsqueeze(1)
        else:
            # Use fixed conditioning
            fixed_conditioning = self.config.fixed_conditioning
            body_names = [entry.body_name for entry in fixed_conditioning]
            
            # Get indices of conditioned bodies
            conditioned_body_ids = [
                self._all_body_names.index(name) for name in body_names
            ]
            fixed_body_indices = [
                self.conditionable_body_ids.tolist().index(body_id)
                for body_id in conditioned_body_ids
            ]
            
            for i, body_index in enumerate(fixed_body_indices):
                active_body_ids[:, body_index] = True
                constraint_states[:, body_index] = fixed_conditioning[i].constraint_state
        
        # Create masks for translation and rotation
        translation_mask = (constraint_states <= 1) & active_body_ids
        rotation_mask = (constraint_states >= 1) & active_body_ids
        
        new_mask = torch.zeros(
            num_envs,
            self.num_conditionable_bodies,
            2,
            dtype=torch.bool,
            device=self.env.device,
        )
        new_mask[:, :, 0] = translation_mask
        new_mask[:, :, 1] = rotation_mask
        
        return new_mask.view(num_envs, -1)
    
    def get_context(self) -> Dict[str, any]:
        """Get masked mimic-specific context for observations and rewards.
        
        Extends parent context with masked mimic state variables.
        
        Returns:
            Dictionary with reference state, motion times, tracking variables,
            and masked mimic state.
            
            Key masked mimic context variables:
            - masked_mimic_ref_pos: Target body positions [envs, future_steps, bodies, 3]
            - masked_mimic_ref_rot: Target body rotations [envs, future_steps, bodies, 4]
            - masked_mimic_target_poses_masks: Which poses are visible [envs, future_steps]
            - masked_mimic_target_bodies_masks: Which bodies are visible [envs, future_steps * bodies * 2]
            - masked_mimic_target_times: Target times [envs, future_steps]
        """
        # Get parent context (ref_state, mimic_ref_*, etc.)
        context = super().get_context()
        
        # Query motion library for poses at target times
        num_envs = self.env.num_envs
        num_future_steps = self.config.num_masked_future_steps
        motion_ids = self.env.motion_manager.motion_ids
        motion_times = self.env.motion_manager.motion_times
        
        # Expand motion_ids for all future steps
        flat_motion_ids = motion_ids.unsqueeze(-1).expand(
            num_envs, num_future_steps
        ).reshape(-1)
        
        # Clip target times to motion lengths
        motion_lengths = self.env.motion_lib.get_motion_length(flat_motion_ids)
        flat_target_times = torch.minimum(self.target_times.reshape(-1), motion_lengths)
        
        # Query motion lib for poses at target times
        target_state = self.env.motion_lib.get_motion_state(flat_motion_ids, flat_target_times)
        
        # Reshape to [envs, future_steps, bodies, dim]
        num_bodies = target_state.rigid_body_pos.shape[1]
        
        masked_mimic_ref_pos = target_state.rigid_body_pos.view(
            num_envs, num_future_steps, num_bodies, 3
        ).clone()
        offset = self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            masked_mimic_ref_pos[:, 0, :, :]  # Use first step for offset
        )
        masked_mimic_ref_pos += offset.unsqueeze(1)
        
        masked_mimic_ref_rot = target_state.rigid_body_rot.view(
            num_envs, num_future_steps, num_bodies, 4
        )
        
        # Compute time offsets from current time
        masked_mimic_time_offsets = self.target_times - motion_times.unsqueeze(-1)
        
        # Add masked mimic specific context
        context.update({
            # Reference poses at target times [envs, future_steps, bodies, dim]
            "masked_mimic_ref_pos": masked_mimic_ref_pos,
            "masked_mimic_ref_rot": masked_mimic_ref_rot,
            # Time information
            "masked_mimic_target_times": self.target_times,
            "masked_mimic_time_offsets": masked_mimic_time_offsets,
            # Masks
            "masked_mimic_target_poses_masks": self.masked_mimic_target_poses_masks,
            "masked_mimic_target_bodies_masks": self.masked_mimic_target_bodies_masks,
        })
        
        return context
    
    def create_visualization_markers(self, headless: bool) -> Dict[str, VisualizationMarkerConfig]:
        """Create visualization markers for masked mimic targets.
        
        Overrides parent to create three sets of color-coded markers based on 
        time-to-target instead of single red markers:
        - Blue: time to target > 1 second
        - Yellow: 0.1 < time to target <= 1 second
        - Red: time to target <= 0.1 seconds
        
        Args:
            headless: If True, returns empty dict.
            
        Returns:
            Dictionary of marker configurations.
        """
        if headless:
            return {}
        
        visualization_markers = {}
        
        # Use trackable_bodies_subset for masked mimic (not all bodies)
        body_names = self.env.robot_config.trackable_bodies_subset
        
        body_markers = []
        for body_name in body_names:
            if (
                self.env.robot_config.mimic_small_marker_bodies is not None
                and body_name in self.env.robot_config.mimic_small_marker_bodies
            ):
                body_markers.append(MarkerConfig(size="small"))
            else:
                body_markers.append(MarkerConfig(size="regular"))
        
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
        
        # Red markers: time to target <= 0.1 seconds
        body_markers_red_cfg = VisualizationMarkerConfig(
            type="sphere", color=(1.0, 0.0, 0.0), markers=body_markers
        )
        visualization_markers["body_markers_red"] = body_markers_red_cfg
        
        return visualization_markers
    
    def get_markers_state(self) -> Dict[str, MarkerState]:
        """Compute marker positions colored by time-to-target.
        
        Overrides parent to show only conditionable bodies with color coding.
        
        Returns:
            Dictionary mapping marker names to MarkerState.
        """
        if self.env.simulator.headless:
            return {}
        
        if not self._initialized:
            return {}
        
        markers_state = {}
        
        # Get the first visible target pose for each env
        pose_masks = self.masked_mimic_target_poses_masks  # [num_envs, num_future_steps]
        first_valid_indices = torch.argmax(pose_masks.float(), dim=1)  # [num_envs]
        
        # Get the target times for the first visible pose
        env_indices = torch.arange(self.env.num_envs, device=self.env.device)
        target_motion_times = self.target_times[env_indices, first_valid_indices]
        
        # Compute time to target
        current_motion_times = self.env.motion_manager.motion_times
        time_to_target = target_motion_times - current_motion_times  # [num_envs]
        
        # Get reference state at target time
        ref_state = self.env.motion_lib.get_motion_state(
            self.env.motion_manager.motion_ids, target_motion_times
        )
        
        target_pos = ref_state.rigid_body_pos.clone()
        target_pos += (
            self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                target_pos
            )
        )
        
        # Extract only conditionable bodies
        target_pos = target_pos[:, self.conditionable_body_ids, :]
        
        # Get translation mask for first visible pose
        bodies_masks_reshaped = self.masked_mimic_target_bodies_masks.view(
            self.env.num_envs,
            self.config.num_masked_future_steps,
            self.num_conditionable_bodies,
            2,
        )
        translation_view = bodies_masks_reshaped[env_indices, first_valid_indices, :, 0]
        active_translations = (translation_view == 1)  # [num_envs, num_conditionable_bodies]
        
        # Create masks for each time range
        blue_time_mask = (
            (time_to_target > 1.0).unsqueeze(1).expand(-1, self.num_conditionable_bodies)
        )
        yellow_time_mask = (
            ((time_to_target > 0.1) & (time_to_target <= 1.0))
            .unsqueeze(1)
            .expand(-1, self.num_conditionable_bodies)
        )
        red_time_mask = (
            (time_to_target <= 0.1).unsqueeze(1).expand(-1, self.num_conditionable_bodies)
        )
        
        # Combine time masks with active translations
        blue_markers = active_translations & blue_time_mask
        yellow_markers = active_translations & yellow_time_mask
        red_markers = active_translations & red_time_mask
        
        # Create separate target positions for each color group
        target_pos_blue = target_pos.clone()
        target_pos_yellow = target_pos.clone()
        target_pos_red = target_pos.clone()
        
        # Move inactive markers off screen (add large offset)
        target_pos_blue[~blue_markers] += 100
        target_pos_yellow[~yellow_markers] += 100
        target_pos_red[~red_markers] += 100
        
        # Add marker states for each color group
        markers_state["body_markers_blue"] = MarkerState(
            translation=target_pos_blue.view(self.env.num_envs, -1, 3),
            orientation=torch.zeros(
                self.env.num_envs, self.num_conditionable_bodies, 4, device=self.env.device
            ),
        )
        markers_state["body_markers_yellow"] = MarkerState(
            translation=target_pos_yellow.view(self.env.num_envs, -1, 3),
            orientation=torch.zeros(
                self.env.num_envs, self.num_conditionable_bodies, 4, device=self.env.device
            ),
        )
        markers_state["body_markers_red"] = MarkerState(
            translation=target_pos_red.view(self.env.num_envs, -1, 3),
            orientation=torch.zeros(
                self.env.num_envs, self.num_conditionable_bodies, 4, device=self.env.device
            ),
        )
        
        return markers_state
