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
import torch

from protomotions.envs.obs.config import MaskedMimicObsConfig
from protomotions.envs.utils.target_poses import build_sparse_target_poses
from protomotions.components.pose_lib import build_body_ids_tensor
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from protomotions.envs.mimic.env import Mimic
else:
    Mimic = object


class MaskedMimicObs:
    """Manages masked motion imitation observations with dynamic body and time masking.

    This class handles the generation of sparse target poses with selective body visibility
    and historical pose tracking for masked motion imitation learning.
    """

    env: Mimic

    def __init__(self, config: MaskedMimicObsConfig, env: Mimic):
        """Initialize the masked mimic observation manager.

        Args:
            config: Configuration for masked mimic observations
            env: Reference to the parent Mimic environment
        """
        self.config = config
        self.env = env

        self.joint_masking_config = self.config.masked_mimic_masking.joint_masking
        self.target_pose_config = self.config.masked_mimic_target_pose
        self.historical_obs_config = self.config.historical_obs
        self.time_sampling_config = self.config.masked_mimic_masking.time_sampling

        self.num_conditionable_bodies = len(
            self.env.robot_config.trackable_bodies_subset
        )

        self._all_body_names = self.env.robot_config.kinematic_info.body_names
        self.conditionable_body_ids = build_body_ids_tensor(
            self._all_body_names,
            self.env.robot_config.trackable_bodies_subset,
            self.env.device,
        )

        num_future_steps = self.config.masked_mimic_target_pose.num_future_steps

        # Target poses for masked mimic learning
        self.masked_mimic_target_poses = None

        # Masks indicating which bodies are visible in each future pose
        self.masked_mimic_target_bodies_masks = torch.zeros(
            self.env.num_envs,
            self.num_conditionable_bodies * 2 * num_future_steps,
            dtype=torch.bool,
            device=self.env.device,
        )

        # Masks indicating which future poses are visible
        self.masked_mimic_target_poses_masks = torch.zeros(
            self.env.num_envs,
            num_future_steps,
            dtype=torch.bool,
            device=self.env.device,
        )

        # Historical pose observations
        self.historical_pose_obs = None
        assert (
            self.config.historical_obs.num_historical_conditioned_steps
            <= self.env.self_obs_cb.config.max_coords_obs.num_historical_steps
        ), (
            f"Requesting {self.config.historical_obs.num_historical_conditioned_steps} "
            f"historical steps, but self_obs is configured for only "
            f"{self.env.self_obs_cb.config.max_coords_obs.num_historical_steps} steps"
        )

        # Track when each time step should be reached for resampling
        self.target_times = torch.zeros(
            self.env.num_envs,
            num_future_steps,
            dtype=torch.float,
            device=self.env.device,
        )

        self.envs_requiring_reset = None

        self._initialized = False

    def _shift_and_sample_time_steps(self, env_ids):
        """Shift target time steps forward and sample a new future time step.

        Samples new time steps using a beta distribution to determine the offset from
        the last conditioned time. The new time is added to the end after shifting
        existing times forward.

        Args:
            env_ids: Tensor of environment indices to update
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
            self.time_sampling_config.alpha, self.time_sampling_config.beta
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

    def sample_body_masks(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample body visibility masks for conditioning.

        With a certain probability, repeats the previous mask; otherwise samples
        new masks to determine which bodies and their properties (translation/rotation)
        are visible for conditioning.

        Args:
            env_ids: Tensor of environment indices to sample masks for

        Returns:
            Tensor of shape (num_envs, num_conditionable_bodies * 2) with boolean masks
        """
        num_envs = len(env_ids)

        # Check if we should repeat previous masks
        repeat_mask_prob = (
            self.joint_masking_config.masked_mimic_repeat_mask_probability
        )
        repeat_mask = torch.rand(num_envs, device=self.env.device) < repeat_mask_prob
        if repeat_mask.any():
            # Get previous masks (last step)
            single_step_mask_size = self.num_conditionable_bodies * 2
            previous_masks = self.masked_mimic_target_bodies_masks[
                env_ids, -single_step_mask_size:
            ].view(num_envs, self.num_conditionable_bodies, 2)

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

    def _sample_new_body_masks(self, num_envs: int) -> torch.Tensor:
        """Sample fresh body masks for multiple environments.

        Determines the number of active bodies and their constraint states
        (translation only, rotation only, or both). Supports fixed conditioning
        or random sampling strategies.

        Args:
            num_envs: Number of environments to sample masks for

        Returns:
            Tensor of shape (num_envs, num_conditionable_bodies * 2) with boolean masks
        """
        # Sample number of active bodies
        num_bodies_mask = (
            torch.rand(num_envs, device=self.env.device)
            >= self.joint_masking_config.force_small_num_conditioned_bodies_prob
        )
        max_num_bodies = torch.where(num_bodies_mask, self.num_conditionable_bodies, 3)
        num_active_bodies = torch.round(
            torch.rand(num_envs, device=self.env.device) * (max_num_bodies - 1) + 1
        ).long()

        max_bodies_mask = (
            torch.rand(num_envs, device=self.env.device)
            <= self.joint_masking_config.force_max_conditioned_bodies_prob
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

        if self.joint_masking_config.masked_mimic_fixed_conditioning is None:
            # Parallelized sampling without replacement
            rand_values = torch.rand(
                num_envs, self.num_conditionable_bodies, device=self.env.device
            )
            # Get ranks: rank[i, j] is the rank of body j in env i among the random values
            ranks = torch.argsort(torch.argsort(rand_values, dim=1), dim=1)
            active_body_ids[:] = ranks < num_active_bodies.unsqueeze(1)
        else:
            fixed_conditioning = (
                self.joint_masking_config.masked_mimic_fixed_conditioning
            )
            body_names = [entry.body_name for entry in fixed_conditioning]
            conditioned_body_ids = build_body_ids_tensor(
                self._all_body_names, body_names, self.env.device
            ).tolist()
            fixed_body_indices = [
                self.conditionable_body_ids.tolist().index(body_id)
                for body_id in conditioned_body_ids
            ]

            for body_index in fixed_body_indices:
                body_name = self.env.robot_config.trackable_bodies_subset[body_index]
                conditioned_body_names = [
                    entry.body_name for entry in fixed_conditioning
                ]
                constraint_index = conditioned_body_names.index(body_name)

                active_body_ids[:, body_index] = True
                constraint_states[:, body_index] = fixed_conditioning[
                    constraint_index
                ].constraint_state

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

    def build_sparse_target_poses_masked_with_time(
        self, env_ids: torch.Tensor, num_future_steps: int
    ) -> torch.Tensor:
        """Build sparse target poses with body masks and time offsets.

        Combines target poses with visibility masks and time information for each
        future step. The output includes masked pose data and temporal information
        for conditioning.

        Args:
            env_ids: Tensor of environment indices
            num_future_steps: Number of future time steps to generate

        Returns:
            Tensor of shape (num_envs, num_future_steps * obs_per_pose) containing
            masked poses concatenated with time information
        """
        num_envs = env_ids.shape[0]

        # Use dynamically sampled time steps from beta distribution
        future_times = self.target_times[env_ids]

        motion_ids = (
            self.env.motion_manager.motion_ids[env_ids]
            .unsqueeze(-1)
            .tile([1, num_future_steps])
        )

        obs = self.build_sparse_target_poses(env_ids, future_times).view(
            num_envs, num_future_steps, self.num_conditionable_bodies, 2, 12
        )

        mask = self.masked_mimic_target_bodies_masks[env_ids].view(
            num_envs, num_future_steps, self.num_conditionable_bodies, 2, 1
        )

        masked_obs = obs * mask
        masked_obs_with_joints = torch.cat((masked_obs, mask), dim=-1).view(
            num_envs, num_future_steps, -1
        )

        flat_motion_ids = motion_ids.view(-1)
        motion_lengths = self.env.motion_lib.get_motion_length(flat_motion_ids)
        current_times = self.env.motion_manager.motion_times[env_ids].unsqueeze(-1)

        # Reshape motion_lengths to match future_times shape
        motion_lengths = motion_lengths.view(num_envs, num_future_steps)

        times = (torch.minimum(future_times, motion_lengths) - current_times).unsqueeze(
            -1
        )
        ones_vec = torch.ones(num_envs, num_future_steps, 1, device=self.env.device)
        times_with_mask = torch.cat((times, ones_vec), dim=-1)
        combined_sparse_future_pose_obs = torch.cat(
            (masked_obs_with_joints, times_with_mask), dim=-1
        )

        return combined_sparse_future_pose_obs.view(num_envs, -1)

    def build_sparse_target_poses(
        self, env_ids: torch.Tensor, future_times: torch.Tensor
    ) -> torch.Tensor:
        """Build sparse target poses relative to the current robot state.

        Extracts reference motion states at specified future times and converts them
        to poses relative to the current robot state. Accounts for terrain height
        and respawn offsets.

        Args:
            env_ids: Tensor of environment indices
            future_times: Tensor of shape (num_envs, num_future_steps) with target times

        Returns:
            Tensor of sparse target poses for conditionable bodies
        """
        num_envs = env_ids.shape[0]
        num_future_steps = future_times.shape[1]

        motion_ids = (
            self.env.motion_manager.motion_ids[env_ids]
            .unsqueeze(-1)
            .tile([1, num_future_steps])
        )
        flat_motion_ids = motion_ids.view(-1)
        motion_lengths = self.env.motion_lib.get_motion_length(flat_motion_ids)
        flat_future_times = torch.minimum(future_times.view(-1), motion_lengths)

        ref_state = self.env.motion_lib.get_motion_state(
            flat_motion_ids, flat_future_times
        )
        flat_target_pos = ref_state.rigid_body_pos
        flat_target_rot = ref_state.rigid_body_rot

        current_body_state = self.env.simulator.get_robot_state(env_ids)

        # Adjust current global translations to be relative to the data origin
        current_body_state.rigid_body_pos[:, :, -1:] -= (
            self.env.terrain.get_ground_heights(
                current_body_state.rigid_body_pos[:, 0]
            ).view(num_envs, 1, 1)
        )
        current_body_state.rigid_body_pos[..., :2] -= (
            self.env.respawn_root_offset[env_ids].clone()[..., :2].view(num_envs, 1, 2)
        )

        return build_sparse_target_poses(
            cur_gt=current_body_state.rigid_body_pos,
            cur_gr=current_body_state.rigid_body_rot,
            flat_target_pos=flat_target_pos,
            flat_target_rot=flat_target_rot,
            masked_mimic_conditionable_bodies_ids=self.conditionable_body_ids,
            num_future_steps=num_future_steps,
            num_envs=num_envs,
            w_last=True,
        )

    def compute_observations(self, env_ids: torch.Tensor):
        """Compute and update masked mimic observations for specified environments.

        Handles environment resets by initializing target times, then builds masked
        future poses and historical pose observations with temporal information.

        Args:
            env_ids: Tensor of environment indices to compute observations for
        """
        self._initialized = True

        if self.envs_requiring_reset is not None:
            new_times = self.env.motion_manager.motion_times[self.envs_requiring_reset]

            # Initialize time steps from current time
            self.target_times[self.envs_requiring_reset] = new_times.unsqueeze(-1)

            # Sample all future steps
            for _ in range(self.target_pose_config.num_future_steps):
                self._shift_and_sample_time_steps(self.envs_requiring_reset)
                self._shift_and_sample_body_masks(self.envs_requiring_reset)

            self.envs_requiring_reset = None

        future_poses = self.build_sparse_target_poses_masked_with_time(
            env_ids, self.target_pose_config.num_future_steps
        )
        
        if self.masked_mimic_target_poses is None:
            self.masked_mimic_target_poses = torch.zeros(
                self.env.num_envs,
                future_poses.shape[-1],
                dtype=torch.float,
                device=self.env.device,
            )

        self.masked_mimic_target_poses[env_ids] = future_poses

        num_envs = env_ids.shape[0]
        reshaped_masked_bodies_masks = self.masked_mimic_target_bodies_masks[
            env_ids
        ].view(num_envs, self.target_pose_config.num_future_steps, -1)

        for i in range(self.target_pose_config.num_future_steps):
            any_visible_joint = reshaped_masked_bodies_masks[:, i].any(dim=-1)
            self.masked_mimic_target_poses_masks[env_ids, i] = any_visible_joint

        if self.historical_obs_config.use_reduced_coords_obs:
            total_stored_historical_steps = (
                self.env.self_obs_cb.config.reduced_coords_obs.num_historical_steps
            )
            historical_poses = self.env.self_obs_cb.humanoid_reduced_coords_obs_hist_buf.get_all_flattened()[
                env_ids
            ]
        else:
            total_stored_historical_steps = (
                self.env.self_obs_cb.config.max_coords_obs.num_historical_steps
            )
            historical_poses = self.env.self_obs_cb.humanoid_max_coords_obs_hist_buf.get_all_flattened()[
                env_ids
            ]

        sub_sampling_factor = (
            total_stored_historical_steps
            // self.historical_obs_config.num_historical_conditioned_steps
        )
        historical_poses = historical_poses.view(
            num_envs, total_stored_historical_steps, -1
        )[:, ::sub_sampling_factor]

        times = (
            torch.arange(total_stored_historical_steps, device=self.env.device)
            * self.env.dt
        )
        time_offsets = (
            times[::sub_sampling_factor]
            .view(1, self.historical_obs_config.num_historical_conditioned_steps, 1)
            .expand(num_envs, -1, -1)
        )
        historical_poses_with_time = torch.cat(
            [historical_poses, time_offsets], dim=-1
        ).view(num_envs, -1)
        
        if self.historical_pose_obs is None:
            self.historical_pose_obs = torch.zeros(
                self.env.num_envs,
                historical_poses_with_time.shape[-1],
                device=self.env.device,
                dtype=torch.float,
            )
        
        self.historical_pose_obs[env_ids] = historical_poses_with_time

    def post_physics_step(self):
        """Update masks and target times after each physics simulation step.

        Identifies environments where the current time has passed the first target
        time and triggers resampling of time steps and body masks for those environments.
        """
        current_time = self.env.motion_manager.motion_times
        resample_env_ids = []

        outdated_target_times = current_time >= self.target_times[:, 0]
        resample_env_ids = torch.nonzero(outdated_target_times).squeeze(-1)

        if len(resample_env_ids) > 0:
            self._shift_and_sample_time_steps(resample_env_ids)
            self._shift_and_sample_body_masks(resample_env_ids)

    def _shift_and_sample_body_masks(self, env_ids: torch.Tensor):
        """Shift body masks forward and sample new masks for the last time step.

        Moves existing body masks one position forward (removing the first) and
        samples fresh masks for the newly added future time step.

        Args:
            env_ids: Tensor of environment indices to update masks for
        """
        single_step_mask_size = self.num_conditionable_bodies * 2
        num_future_steps = self.target_pose_config.num_future_steps

        # Reshape masks to (num_envs, num_future_steps, single_step_mask_size)
        masks = self.masked_mimic_target_bodies_masks[env_ids].view(
            len(env_ids), num_future_steps, single_step_mask_size
        )

        shifted_masks = masks.roll(shifts=-1, dims=1)

        # Sample new masks for the last position
        new_body_masks = self.sample_body_masks(env_ids)
        shifted_masks[:, -1] = new_body_masks.view(len(env_ids), single_step_mask_size)

        # Update the masks
        self.masked_mimic_target_bodies_masks[env_ids] = shifted_masks.view(
            len(env_ids), -1
        )

    def get_obs(self) -> Dict[str, torch.Tensor]:
        """Get the current masked mimic observations.

        Returns:
            Dictionary containing:
                - masked_mimic_target_poses: Masked future target poses
                - masked_mimic_target_bodies_masks: Body visibility masks for future steps
                - masked_mimic_target_poses_masks: Step-level visibility masks
                - historical_pose_obs: Historical pose observations with time offsets
        """
        if not self._initialized:
            self.compute_observations(torch.arange(self.env.num_envs, device=self.env.device))
        
        return {
            "masked_mimic_target_poses": self.masked_mimic_target_poses.clone(),
            "masked_mimic_target_bodies_masks": self.masked_mimic_target_bodies_masks.clone(),
            "masked_mimic_target_poses_masks": self.masked_mimic_target_poses_masks.clone(),
            "historical_pose_obs": self.historical_pose_obs.clone(),
        }
