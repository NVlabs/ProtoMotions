# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
from typing import TYPE_CHECKING, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from isaac_utils import rotations, torch_utils
from phys_anim.envs.mimic.mimic_utils import (
    build_max_coords_target_poses,
    build_max_coords_target_poses_future_rel,
    build_max_coords_object_target_poses,
    build_max_coords_object_target_poses_future_rel,
    dof_to_local,
    exp_tracking_reward,
)
from phys_anim.envs.humanoid.humanoid_utils import quat_diff_norm
from phys_anim.envs.env_utils.general import StepTracker

if TYPE_CHECKING:
    from phys_anim.envs.mimic.isaacgym import MimicHumanoid
else:
    MimicHumanoid = object


class BaseMimic(MimicHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config, device)

        self.reward_joint_weights = self.get_reward_joint_weights()

        # Used by the tl in eval.
        self.disable_reset_track = False

        # This mask is used to filter out motions we can't start from due to missing scenes.
        self.motion_sampling_mask = torch.zeros(
            self.motion_lib.num_sub_motions(), dtype=torch.bool, device=self.device
        )

        self.mimic_phase = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device
        )

        if self.config.mimic_target_pose.enabled:
            self.mimic_target_poses = torch.zeros(
                self.num_envs,
                self.config.mimic_target_pose.num_future_steps
                * self.config.mimic_target_pose.num_obs_per_target_pose,
                dtype=torch.float,
                device=self.device,
            )

        if self.config.mimic_conditionable_bodies is not None:
            self.mimic_conditionable_bodies_ids = self.build_body_ids_tensor(
                self.config.mimic_conditionable_bodies
            )
        else:
            self.mimic_conditionable_bodies_ids = torch.arange(
                self.config.robot.num_bodies, dtype=torch.long, device=self.device
            )

        self.failed_due_bad_reward = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        if self.config.mimic_dynamic_sampling.enabled:
            self.setup_dynamic_sampling()

        # Sampling vectors
        self.init_start_probs = (
            torch.ones(self.num_envs, dtype=torch.float, device=self.device)
            * self.config.mimic_motion_sampling.init_start_prob
        )
        self.init_random_probs = (
            torch.ones(self.num_envs, dtype=torch.float, device=self.device)
            * self.config.mimic_motion_sampling.init_random_prob
        )

        self.reset_track_steps = StepTracker(
            self.num_envs,
            min_steps=self.config.mimic_reset_track.steps_min,
            max_steps=self.config.mimic_reset_track.steps_max,
            device=self.device,
        )

        self.reset_track(
            torch.arange(0, self.num_envs, device=self.device, dtype=torch.long)
        )

        # For dynamic sampling, we record whether the motion was respawned on a flat terrain.
        # We do not record failures on irregular terrain for prioritized sampling as there are no guarantees it should have succeeded.
        self.respawned_on_flat = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    def sample_valid_motions(self, num_motions, new_motion_ids=None):
        self.motion_sampling_mask[:] = 1  # Reset, all motions are valid to sample from.
        invalid_envs = torch.ones(num_motions, dtype=torch.bool, device=self.device)
        new_motions_defined = new_motion_ids is not None
        if not new_motions_defined:
            new_motion_ids = torch.zeros(
                num_motions, dtype=torch.long, device=self.device
            )
        new_scenes = torch.zeros(num_motions, dtype=torch.long, device=self.device)
        while invalid_envs.any():
            assert self.motion_sampling_mask.any(), "No valid motions to sample from."

            if not new_motions_defined:
                new_motion_ids[invalid_envs] = self.motion_lib.sample_motions(
                    invalid_envs.sum(), self.motion_sampling_mask
                )
            if self.config.fixed_motion_id is not None:
                new_motion_ids = (
                    torch.zeros_like(new_motion_ids) + self.config.fixed_motion_id
                )
            new_scenes[invalid_envs], valid_mask = self.sample_scene_ids(
                new_motion_ids[invalid_envs]
            )

            if self.scene_lib is not None:
                self.scene_lib.mark_scene_in_use(new_scenes[invalid_envs][valid_mask])

            if new_motions_defined:
                assert (
                    valid_mask.all()
                ), f"Provided motion ids lack a valid scene. Motion ids: {new_motion_ids[invalid_envs]}"

            invalid_envs[invalid_envs == 1] = ~valid_mask
            self.motion_sampling_mask[new_motion_ids[invalid_envs]] = 0

        return new_motion_ids, new_scenes

    def setup_dynamic_sampling(self):
        num_buckets_list = []
        bucket_motion_ids_list = []
        bucket_starts_list = []
        bucket_lengths_list = []
        bucket_width = self.config.mimic_dynamic_sampling.bucket_width

        start_times = self.motion_lib.state.motion_timings[:, 0].tolist()
        end_times = self.motion_lib.state.motion_timings[:, 1].tolist()
        for motion_id, (start_time, end_time) in enumerate(zip(start_times, end_times)):
            if self.config.fixed_motion_id is not None:
                # When training on a fixed motion, only add buckets for that motion.
                if motion_id != self.config.fixed_motion_id:
                    continue

            length = end_time - start_time
            buckets = math.ceil(length / bucket_width)
            num_buckets_list.append(buckets)
            for j in range(buckets):
                bucket_motion_ids_list.append(motion_id)
                start = j * bucket_width + start_time
                end = min(end_time, start + bucket_width)
                assert end - start > 0
                bucket_starts_list.append(start)
                bucket_lengths_list.append(end - start)

        num_buckets = torch.tensor(
            num_buckets_list, dtype=torch.long, device=self.device
        )

        rolled = num_buckets.roll(1)
        rolled[0] = 0
        self.bucket_offsets = rolled.cumsum(0)

        total_num_buckets = sum(num_buckets_list)

        self.bucket_scores = torch.zeros(
            total_num_buckets, dtype=torch.float, device=self.device
        )
        self.bucket_frames_spent = torch.zeros(
            total_num_buckets, dtype=torch.long, device=self.device
        )

        # Minimal score equivalent to 1/total_num_buckets.
        # If a single bucket always fails and the rest always succeed,
        # there's an equal chance to sample the bad bucket, or any of the good buckets.
        # Should combine with periodic refresh of the weights to avoid getting stuck.
        # Typically, SLURM jobs are restarted every few hours, which takes care of this.
        self._min_bucket_weight = torch.ones(
            total_num_buckets, dtype=torch.float, device=self.device
        ) * (
            self.config.mimic_dynamic_sampling.min_bucket_weight
            if self.config.mimic_dynamic_sampling.min_bucket_weight is not None
            else 1e-6
        )
        self.bucket_weights = torch.zeros(
            total_num_buckets, dtype=torch.float, device=self.device
        )

        # The motion id each bucket corresponds to.
        self.bucket_motion_ids = torch.tensor(
            bucket_motion_ids_list, dtype=torch.long, device=self.device
        )

        # The start time (in seconds) of the bucket with respect to its corresponding motion.
        self.bucket_starts = torch.tensor(
            bucket_starts_list, dtype=torch.float, device=self.device
        )

        # The length of each bucket, in seconds (some bucket at the end of clips may be shorter than the set bucket width).
        self.bucket_lengths = torch.tensor(
            bucket_lengths_list, dtype=torch.float, device=self.device
        )

    def dynamic_sample(self, n: int):
        """
        Dynamically sample motion sequences based on their difficulty.

        This method implements a weighted sampling strategy to select motion sequences,
        prioritizing more challenging motions. It uses a bucket system where each bucket
        represents a portion of a motion sequence.

        Args:
            n (int): The number of motion sequences to sample.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - motion_ids (Tensor): The IDs of the sampled motions.
                - motion_times (Tensor): The start times for each sampled motion.
                - new_scenes (Tensor): The scene IDs associated with each sampled motion.

        The method works as follows:
        1. Initialize tensors to store the sampling results.
        2. Prepare the sampling weights, ensuring a minimum weight for each bucket.
        3. Sample buckets using the prepared weights.
        4. For each sampled bucket, determine the corresponding motion ID and time.
        5. Sample appropriate scenes for the selected motions.
        6. If sampling fails (e.g., no valid scenes), fall back to a simpler sampling method.
        7. Update the scene usage if using a scene library.
        8. Repeat until the requested number of samples is obtained.

        Note:
        - The method uses the current state of bucket weights, which are updated elsewhere
          based on the success/failure of previous attempts at these motions.
        - It includes logic to handle cases where some buckets have never been visited,
          ensuring exploration of new motion segments.
        """
        # Initialize tensors to store the sampling results
        new_scenes = torch.zeros(n, dtype=torch.long, device=self.device)
        motion_ids = torch.zeros(n, dtype=torch.long, device=self.device)
        motion_times = torch.zeros(n, dtype=torch.float, device=self.device)

        # Prepare the sampling weights, ensuring a minimum weight for each bucket
        weights = self.bucket_weights.clone().clamp(min=self._min_bucket_weight)
        if (self.bucket_weights == 0).any():
            # Bucket weights are always >= 0 for any visited bucket.
            # If a bucket has never been visited, it should be sampled with equal probability.
            # This optimization is for SLURM-based autoresume, where the job is restarted every few hours.
            # It ensures we quickly sample from unvisited buckets and figure out what are the challenging motions.
            weights[self.bucket_weights == 0] = 1.0
            weights[self.bucket_weights != 0] = 0.0
            norm_weight = weights
        else:
            # Normalize weights and clamp to a maximum value if specified
            min_weight = weights.min()
            norm_weight = weights / min_weight
            if self.config.mimic_dynamic_sampling.dynamic_weight_max is not None:
                norm_weight = norm_weight.clamp(
                    max=self.config.mimic_dynamic_sampling.dynamic_weight_max
                )

        valid_samples = 0
        while valid_samples < n:
            remaining = n - valid_samples
            if norm_weight.max() <= 0:
                print(
                    f"Dynamic sampling failed after {valid_samples} valid sampled motions. Resort to sample_valid_motions for the rest."
                )
                # No valid buckets, resort to sample_valid_motions
                fallback_motion_ids, fallback_scenes = self.sample_valid_motions(
                    remaining
                )
                motion_ids[valid_samples:] = fallback_motion_ids
                motion_times[valid_samples:] = self.motion_lib.sample_time(
                    fallback_motion_ids,
                    self.motion_lib.state.first_contact_time[fallback_motion_ids],
                    truncate_time=self.dt,
                )
                new_scenes[valid_samples:] = fallback_scenes
                valid_samples += remaining
            else:
                # Sample buckets using the prepared weights
                chosen_bucket_indices = torch.multinomial(
                    norm_weight, num_samples=remaining, replacement=True
                )
                sampled_motion_ids = self.bucket_motion_ids[chosen_bucket_indices]
                bucket_starts = self.bucket_starts[chosen_bucket_indices]
                bucket_lengths = self.bucket_lengths[chosen_bucket_indices]

                # Determine the corresponding motion ID and time for each sampled bucket
                sampled_motion_times = (
                    torch.rand(remaining, device=self.device) * bucket_lengths
                    + bucket_starts
                )

                # Sample scene ids for the motions
                sampled_scenes, valid_mask = self.sample_scene_ids(sampled_motion_ids)

                if valid_mask.any():
                    valid_count = valid_mask.sum().item()
                    motion_ids[valid_samples : valid_samples + valid_count] = (
                        sampled_motion_ids[valid_mask]
                    )
                    motion_times[valid_samples : valid_samples + valid_count] = (
                        sampled_motion_times[valid_mask]
                    )
                    new_scenes[valid_samples : valid_samples + valid_count] = (
                        sampled_scenes[valid_mask]
                    )
                    valid_samples += valid_count

                    # Mark valid scenes as in use
                    if self.scene_lib is not None:
                        self.scene_lib.mark_scene_in_use(sampled_scenes[valid_mask])

                # Iterate through invalid motion IDs and mark their corresponding buckets
                if ~valid_mask.any():
                    for invalid_motion_id in sampled_motion_ids[~valid_mask]:
                        invalid_buckets = self.bucket_motion_ids == invalid_motion_id
                        norm_weight[invalid_buckets] = 0

        return motion_ids, motion_times, new_scenes

    def update_dynamic_stats(self):
        """
        Update dynamic statistics for motion sampling based on performance.
        This method is crucial for adaptive sampling strategies.
        """
        # Only update stats for motions that failed on flat terrain
        if torch.any(self.respawned_on_flat):
            # Extract relevant data for motions that failed on flat terrain
            on_flat_motion_ids = self.motion_ids[self.respawned_on_flat]
            on_flat_start_times = self.motion_lib.state.motion_timings[
                on_flat_motion_ids, 0
            ]
            on_flat_end_times = self.motion_lib.state.motion_timings[
                on_flat_motion_ids, 1
            ]

            # Ensure motion times are within valid range
            on_flat_motion_times = torch.clamp(
                self.motion_times[self.respawned_on_flat],
                min=on_flat_start_times,
                max=on_flat_end_times,
            )

            # Get performance metrics for failed motions
            on_flat_failed_due_bad_reward = self.failed_due_bad_reward[
                self.respawned_on_flat
            ]
            on_flat_rew_buf = self.rew_buf[self.respawned_on_flat]

            # Determine bucket indices for the failed motions
            if self.config.fixed_motion_id is not None:
                # If using a fixed motion, all buckets correspond to that motion
                bucket_indices = torch.zeros_like(on_flat_motion_ids)
            else:
                bucket_indices = on_flat_motion_ids

            # Calculate the exact bucket for each motion
            base_offsets = self.bucket_offsets[bucket_indices]
            sub_motion_delta = on_flat_motion_times - on_flat_start_times
            extra_offsets = torch.floor(
                sub_motion_delta / self.config.mimic_dynamic_sampling.bucket_width
            ).long()
            bucket_indices = base_offsets + extra_offsets

            # NOTE These two lines
            # self.bucket_frames_spent[bucket_indices] += 1
            # self.bucket_scores[bucket_indices] += self.rew_buf
            # are NOT what we want, see https://discuss.pytorch.org/t/how-to-do-atomic-add-on-slice-with-duplicate-indices/136193

            self.bucket_frames_spent.scatter_add_(
                0, bucket_indices, torch.ones_like(bucket_indices)
            )

            # Update bucket scores based on the chosen criteria
            if (
                self.config.mimic_dynamic_sampling.sampling_criteria
                == "mimic_early_termination"
            ):
                self.bucket_scores.scatter_add_(
                    0, bucket_indices, on_flat_failed_due_bad_reward
                )
            elif self.config.mimic_dynamic_sampling.sampling_criteria == "reward":
                self.bucket_scores.scatter_add_(0, bucket_indices, on_flat_rew_buf)
            else:
                raise NotImplementedError(
                    "Dynamic weight criteria can be either 'mimic_early_termination' or 'reward'"
                )

    def refresh_dynamic_weights(self):
        """
        Refresh dynamic weights for motion sampling.
        This method updates the dynamic weights based on the performance metrics of the sampled motions.
        It ensures that more challenging motions are sampled more frequently.
        """
        visited = self.bucket_frames_spent > 0
        average_score = self.bucket_scores[visited] / self.bucket_frames_spent[visited]
        if (
            self.config.mimic_dynamic_sampling.sampling_criteria
            == "mimic_early_termination"
        ):
            weight = torch.pow(
                average_score, self.config.mimic_dynamic_sampling.dynamic_weight_pow
            )
        elif self.config.mimic_dynamic_sampling.sampling_criteria == "reward":
            weight = torch.pow(
                1 / average_score, self.config.mimic_dynamic_sampling.dynamic_weight_pow
            )
        else:
            raise NotImplementedError(
                "Dynamic weight criteria can be either 'mimic_early_termination' or 'reward"
            )

        self.bucket_weights[visited] = torch.clamp(
            weight + self.bucket_weights[visited] * 0.7,
            min=self._min_bucket_weight[visited],
        )

        # A bucket_weight can be 0 if and only if that bucket hasn't been sampled yet.
        # So long as such a bucket exists, force respawn on flat.
        # This ensures we first evaluate all buckets BEFORE diving deeper into training.
        self.force_respawn_on_flat = (self.bucket_weights == 0).any()

        tensors_of_interest = {
            "bucket_frames_spent": self.bucket_frames_spent.float(),
            "bucket_average_score": average_score,
            "bucket_scores": self.bucket_scores,
            "bucket_weights": self.bucket_weights,
            "bucket_added_weights": weight,
        }

        for k, v in tensors_of_interest.items():
            if v.shape[0] > 0:
                self.log_dict[f"{k}_min"] = v.min()
                self.log_dict[f"{k}_max"] = v.max()
                self.log_dict[f"{k}_mean"] = v.mean()

        self.bucket_frames_spent[:] = 0
        self.bucket_scores[:] = 0

    def reset_track(self, env_ids, new_motion_ids=None):
        """
        Reset the motion and scene for a set of environments.
        This method handles the process of resetting the motion and scene for a specified set of environments.
        It ensures that the reset process is correctly handled based on the current configuration.

        Args:
            env_ids (Tensor): Indices of the environments to reset.
            new_motion_ids (Tensor, optional): New motion IDs for the reset environments.
        Returns:
            Tuple[Tensor, Tensor]: New motion IDs and times for the reset environments.
        """
        if self.disable_reset_track:
            return

        # All reset envs will first relinquish their scene, then try to get a new one.
        if self.scene_lib is not None:
            self.scene_lib.mark_scene_not_in_use(self.scene_ids[env_ids])

        if self.config.mimic_fixed_motion_per_env:
            motion_index_offset = self.env.config.motion_index_offset
            if motion_index_offset is None:
                motion_index_offset = 0
            new_motion_ids = torch.fmod(
                env_ids + motion_index_offset,
                self.motion_lib.num_sub_motions(),
            )
            new_times = self.motion_lib.state.motion_timings[new_motion_ids, 0]

            new_scenes, valid_mask = self.sample_scene_ids(new_motion_ids)
            assert (
                valid_mask.all()
            ), f"Fixed motion per env. Motion ids {new_motion_ids[~valid_mask]} lack a valid scene."

        elif self.config.mimic_dynamic_sampling.enabled and new_motion_ids is None:
            new_motion_ids, new_times, new_scenes = self.dynamic_sample(len(env_ids))
        else:
            new_motion_ids, new_scenes = self.sample_valid_motions(
                len(env_ids), new_motion_ids
            )

            new_times = self.motion_lib.sample_time(
                new_motion_ids,
                truncate_time=self.dt,
            )

        if self.config.mimic_motion_sampling.init_start_prob > 0:
            init_start = torch.bernoulli(self.init_start_probs[: len(env_ids)])
            new_times = torch.where(
                init_start == 1,
                self.motion_lib.state.motion_timings[new_motion_ids, 0],
                new_times,
            )

        if self.config.mimic_motion_sampling.init_random_prob > 0:
            init_random = torch.bernoulli(self.init_random_probs[: len(env_ids)])
            new_times = torch.where(
                init_random == 1,
                self.motion_lib.sample_time(
                    new_motion_ids,
                    truncate_time=self.dt,
                ),
                new_times,
            )

        self.motion_ids[env_ids] = new_motion_ids
        self.motion_times[env_ids] = new_times
        self.scene_ids[env_ids] = new_scenes

        if self.config.scene_lib is not None:
            for env_id, scene_id in zip(env_ids, new_scenes):
                self.env_id_to_object_ids[env_id, :] = -1
                object_mask = self.object_id_to_scene_id == scene_id
                if object_mask.any():
                    object_ids = torch.where(object_mask)[0]
                    self.env_id_to_object_ids[env_id, : len(object_ids)] = object_ids

        self.reset_track_steps.reset_steps(env_ids)

        if not self.config.headless:
            self.randomize_color(env_ids)

        return self.motion_ids[env_ids], self.motion_times[env_ids]

    def reset_actors(self, env_ids):
        if env_ids.shape[0] > 0:
            # On reset actor, shift the counter backwards to reset the grace period
            self.reset_track_steps.shift_counter(
                env_ids, self.reset_track_steps.steps[env_ids]
            )

        if self.config.mimic_reset_track.reset_on_episode_reset:
            self.reset_track(env_ids)

        self.reset_ref_state_init(
            env_ids,
            motion_ids=self.motion_ids[env_ids],
            motion_times=self.motion_times[env_ids],
            scene_ids=self.scene_ids[env_ids],
        )

    def get_envs_respawn_position(
        self,
        env_ids,
        offset=0,
        rb_pos: torch.tensor = None,
        scene_ids: torch.tensor = None,
    ):
        respawn_position = super().get_envs_respawn_position(
            env_ids, offset=offset, rb_pos=rb_pos, scene_ids=scene_ids
        )

        ref_state = self.motion_lib.get_mimic_motion_state(
            self.motion_ids[env_ids], self.motion_times[env_ids]
        )
        target_cur_gt = ref_state.rb_pos
        target_cur_root_pos = target_cur_gt[:, 0, :]

        self.respawn_offset_relative_to_data[env_ids, :2] = (
            respawn_position[:, :2] - target_cur_root_pos[:, :2]
        )

        if self.terrain is not None:
            # Check if spawned on flat, for prioritized sampling
            new_root_pos = (
                respawn_position[..., :2].clone().reshape(env_ids.shape[0], 1, 2)
            )
            new_root_pos = (new_root_pos / self.terrain.horizontal_scale).long()
            px = new_root_pos[:, :, 0].view(-1)
            py = new_root_pos[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

            self.respawned_on_flat[env_ids] = self.terrain.flat_field_raw[px, py] == 0
            # if scene interaction motion -- also consider as "flat"
            if scene_ids is not None:
                scene_interaction_envs_mask = scene_ids != -1
                self.respawned_on_flat[env_ids[scene_interaction_envs_mask]] = True
        else:
            self.respawned_on_flat[env_ids] = True

        return respawn_position

    def compute_reset(self):
        super().compute_reset()

        if self.config.mimic_early_termination is not None:
            reward_too_bad = torch.zeros_like(self.reset_buf).bool()
            for entry in self.config.mimic_early_termination:
                if entry.get("from_other", False):
                    from_dict = self.last_other_rewards
                elif entry.use_scaled:
                    from_dict = self.last_scaled_rewards
                else:
                    from_dict = self.last_unscaled_rewards

                if entry.less_than:
                    entry_too_bad = (
                        from_dict[entry.mimic_early_termination_key]
                        < entry.mimic_early_termination_thresh
                    )
                    entry_on_flat_too_bad = (
                        from_dict[entry.mimic_early_termination_key]
                        < entry.mimic_early_termination_thresh_on_flat
                    )
                else:
                    entry_too_bad = (
                        from_dict[entry.mimic_early_termination_key]
                        > entry.mimic_early_termination_thresh
                    )
                    entry_on_flat_too_bad = (
                        from_dict[entry.mimic_early_termination_key]
                        > entry.mimic_early_termination_thresh_on_flat
                    )

                no_scene_interaction = self.scene_ids[:] < 0
                tight_tracking_threshold = torch.logical_and(
                    no_scene_interaction, self.respawned_on_flat
                )

                entry_too_bad[tight_tracking_threshold] = entry_on_flat_too_bad[
                    tight_tracking_threshold
                ]
                # entry_too_bad[self.respawned_on_flat] = entry_on_flat_too_bad[
                #     self.respawned_on_flat
                # ]

                reward_too_bad = torch.logical_or(reward_too_bad, entry_too_bad)

            has_reset_grace = (
                self.reset_track_steps.steps
                <= self.config.mimic_reset_track.grace_period
            )

            reward_too_bad = torch.logical_and(
                reward_too_bad, torch.logical_not(has_reset_grace)
            )

            self.failed_due_bad_reward[:] = 0
            self.failed_due_bad_reward[reward_too_bad] = 1

            self.reset_buf[reward_too_bad] = 1
            self.terminate_buf[reward_too_bad] = 1

            self.log_dict["reward_too_bad"] = reward_too_bad.float().mean()

        end_times = self.motion_lib.state.motion_timings[self.motion_ids, 1]
        done_clip = (self.motion_times + self.dt) >= end_times
        if self.config.mimic_reset_track.reset_episode_on_reset_track:
            self.reset_buf[done_clip] = 1
        done_ids = torch.nonzero(done_clip == 1, as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self.reset_track(done_ids)

    def handle_reset_track(self):
        self.reset_track_steps.advance()
        reset_track_ids = self.reset_track_steps.done_indices()

        if len(reset_track_ids) > 0:
            self.reset_track(reset_track_ids)

            if self.config.mimic_reset_track.reset_episode_on_reset_track:
                self.reset_buf[reset_track_ids] = 1

    def post_physics_step(self):
        self.motion_times += self.dt
        end_times = self.motion_lib.state.motion_timings[self.motion_ids, 1].clone()
        start_times = self.motion_lib.state.motion_timings[self.motion_ids, 0].clone()

        super().post_physics_step()

        # Don't update stats while in eval mode.
        if self.config.mimic_dynamic_sampling.enabled and not self.disable_reset:
            self.update_dynamic_stats()

        # Remove start time before fmod and then add it back.
        self.motion_times = (
            torch.fmod(self.motion_times - start_times, end_times - start_times)
            + start_times
        )

        if not self.disable_reset_track:
            self.handle_reset_track()

    def store_motion_data(self, skip=False):
        super().store_motion_data()
        if skip:
            return

        if "target_poses" not in self.motion_recording:
            self.motion_recording["target_poses"] = []

        ref_state = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.motion_times
        )
        target_pos = ref_state.rb_pos
        target_pos += self.respawn_offset_relative_to_data.clone().view(
            self.num_envs, 1, 3
        )

        self.motion_recording["target_poses"].append(target_pos.cpu().numpy())

    def process_kb(self, gt: Tensor, gr: Tensor):
        kb = gt[:, self.key_body_ids]

        if self.config.mimic_reward_config.relative_kb_pos:
            rt = gt[:, 0]
            rr = gr[:, 0]
            kb = kb - rt.unsqueeze(1)

            heading_rot = torch_utils.calc_heading_quat_inv(rr, self.w_last)
            rr_expand = heading_rot.unsqueeze(1).expand(rr.shape[0], kb.shape[1], 4)
            kb = rotations.quat_rotate(
                rr_expand.reshape(-1, 4), kb.view(-1, 3), self.w_last
            ).view(kb.shape)

        return kb

    def rotate_pos_to_local(self, pos: Tensor, heading: Optional[Tensor] = None):
        if heading is None:
            raise NotImplementedError("Heading is required for local rotation")
            # root_rot = self.rigid_body_rot[:, 0]
            root_rot = self.get_bodies_state().body_rot[:, 0]
            heading = torch_utils.calc_heading_quat_inv(root_rot, self.w_last)

        pos_num_dims = len(pos.shape)
        expanded_heading = heading.view(
            [heading.shape[0]] + [1] * (pos_num_dims - 2) + [heading.shape[1]]
        ).expand(pos.shape[:-1] + (4,))

        rotated = rotations.quat_rotate(
            expanded_heading.reshape(-1, 4), pos.reshape(-1, 3), self.w_last
        ).view(pos.shape)
        return rotated

    def compute_reward(self, actions):
        """
        Abbreviations:

        gt = global translation
        gr = global rotation
        rt = root translation
        rr = root rotation
        kb = key bodies
        dv = dof (degrees of freedom velocity)
        """
        ref_state = self.motion_lib.get_mimic_motion_state(
            self.motion_ids, self.motion_times
        )
        ref_gt = ref_state.rb_pos
        ref_gr = ref_state.rb_rot
        ref_lr = ref_state.local_rot
        ref_gv = ref_state.rb_vel
        ref_gav = ref_state.rb_ang_vel
        ref_dv = ref_state.dof_vel

        ref_lr = ref_lr[:, self.dof_body_ids]
        ref_kb = self.process_kb(ref_gt, ref_gr)

        current_state = self.get_bodies_state()
        gt, gr, gv, gav = (
            current_state.body_pos,
            current_state.body_rot,
            current_state.body_vel,
            current_state.body_ang_vel,
        )
        # first remove height based on current position
        gt[:, :, -1:] -= self.get_ground_heights(gt[:, 0, :2]).view(self.num_envs, 1, 1)
        # then remove offset to get back to the ground-truth data position
        gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[..., :2].view(
            self.num_envs, 1, 2
        )

        kb = self.process_kb(gt, gr)

        rt = gt[:, 0]
        ref_rt = ref_gt[:, 0]

        if self.config.mimic_reward_config.rt_ignore_height:
            rt = rt[..., :2]
            ref_rt = ref_rt[..., :2]

        rr = gr[:, 0]
        ref_rr = ref_gr[:, 0]

        inv_heading = torch_utils.calc_heading_quat_inv(rr, self.w_last)
        ref_inv_heading = torch_utils.calc_heading_quat_inv(ref_rr, self.w_last)

        rv = gv[:, 0]
        ref_rv = ref_gv[:, 0]

        rav = gav[:, 0]
        ref_rav = ref_gav[:, 0]

        dp, dv = self.get_dof_state()
        lr = dof_to_local(dp, self.get_dof_offsets(), self.w_last)

        if self.config.mimic_reward_config.add_rr_to_lr:
            rr = gr[:, 0]
            ref_rr = ref_gr[:, 0]

            lr = torch.cat([rr.unsqueeze(1), lr], dim=1)
            ref_lr = torch.cat([ref_rr.unsqueeze(1), ref_lr], dim=1)

        rew_dict = exp_tracking_reward(
            gt=gt,
            rt=rt,
            kb=kb,
            gr=gr,
            lr=lr,
            rv=rv,
            rav=rav,
            gv=gv,
            gav=gav,
            dv=dv,
            ref_gt=ref_gt,
            ref_rt=ref_rt,
            ref_kb=ref_kb,
            ref_gr=ref_gr,
            ref_lr=ref_lr,
            ref_rv=ref_rv,
            ref_rav=ref_rav,
            ref_gv=ref_gv,
            ref_gav=ref_gav,
            ref_dv=ref_dv,
            joint_reward_weights=self.reward_joint_weights,
            config=self.config.mimic_reward_config,
            w_last=self.w_last,
        )

        current_contact_forces = self.get_bodies_contact_buf()
        forces_delta = torch.clip(
            self.prev_contact_forces - current_contact_forces, min=0
        )[
            :, self.non_termination_contact_body_ids, 2
        ]  # get the Z axis
        kbf_rew = (
            forces_delta.sum(-1)
            .mul(self.config.mimic_reward_config.component_coefficients.kbf_rew_c)
            .exp()
        )

        rew_dict["kbf_rew"] = kbf_rew

        dof_forces = self.get_dof_forces()
        power = torch.abs(torch.multiply(dof_forces, dv)).sum(dim=-1)
        pow_rew = -power

        has_reset_grace = (
            self.reset_track_steps.steps <= self.config.mimic_reset_track.grace_period
        )
        pow_rew[has_reset_grace] = 0

        rew_dict["pow_rew"] = pow_rew

        self.last_scaled_rewards: Dict[str, Tensor] = {
            k: v * getattr(self.config.mimic_reward_config.component_weights, f"{k}_w")
            for k, v in rew_dict.items()
        }

        tracking_rew = sum(self.last_scaled_rewards.values())

        self.rew_buf = tracking_rew + self.config.mimic_reward_config.positive_constant

        for rew_name, rew in rew_dict.items():
            self.log_dict[f"raw/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"raw/{rew_name}_std"] = rew.std()

        for rew_name, rew in self.last_scaled_rewards.items():
            self.log_dict[f"scaled/{rew_name}_mean"] = rew.mean()
            self.log_dict[f"scaled/{rew_name}_std"] = rew.std()

        local_ref_gt = self.rotate_pos_to_local(ref_gt, ref_inv_heading)
        local_gt = self.rotate_pos_to_local(gt, inv_heading)
        cartesian_err = (
            ((local_ref_gt - local_ref_gt[:, 0:1]) - (local_gt - local_gt[:, 0:1]))
            .pow(2)
            .sum(-1)
            .sqrt()
            .mean(-1)
        )

        translation_mask_coeff = self.num_bodies
        rotation_mask_coeff = self.num_bodies

        gt_err = (ref_gt - gt).pow(2).sum(-1).sqrt().sum(-1).div(translation_mask_coeff)
        max_joint_err = (ref_gt - gt).pow(2).sum(-1).sqrt().max(-1)[0]

        gr_diff = quat_diff_norm(gr, ref_gr, self.w_last)
        gr_err = gr_diff.sum(-1).div(rotation_mask_coeff)
        gr_err_degrees = gr_err * 180 / torch.pi
        max_gr_err = gr_diff.max(-1)[0]
        max_gr_err_degrees = max_gr_err * 180 / torch.pi

        lr_diff = quat_diff_norm(lr, ref_lr, self.w_last)
        lr_err = lr_diff.sum(-1).div(rotation_mask_coeff)
        lr_err_degrees = lr_err * 180 / torch.pi
        max_lr_err = lr_diff.max(-1)[0]
        max_lr_err_degrees = max_lr_err * 180 / torch.pi

        other_log_terms = {
            "tracking_rew": tracking_rew,
            "total_rew": self.rew_buf,
            "cartesian_err": cartesian_err,
            "gt_err": gt_err,
            "gr_err": gr_err,
            "gr_err_degrees": gr_err_degrees,
            "lr_err_degrees": lr_err_degrees,
            "max_joint_err": max_joint_err,
            "max_lr_err_degrees": max_lr_err_degrees,
            "max_gr_err_degrees": max_gr_err_degrees,
        }

        for rew_name, rew in other_log_terms.items():
            self.log_dict[f"{rew_name}_mean"] = rew.mean()
            self.log_dict[f"{rew_name}_std"] = rew.std()

        self.last_unscaled_rewards: Dict[str, Tensor] = rew_dict
        self.last_scaled_rewards = self.last_scaled_rewards
        self.last_other_rewards = other_log_terms

    def get_phase_obs(self, motion_ids: Tensor, motion_times: Tensor):
        phase = (
            motion_times - self.motion_lib.state.motion_timings[motion_ids, 0]
        ) / self.motion_lib.get_sub_motion_length(motion_ids)
        sin_phase = phase.sin().unsqueeze(-1)
        cos_phase = phase.cos().unsqueeze(-1)

        phase_obs = torch.cat([sin_phase, cos_phase], dim=-1)
        return phase_obs

    def compute_observations(self, env_ids=None):
        super().compute_observations(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device).long()

        self.mimic_phase[env_ids] = self.get_phase_obs(
            self.motion_ids[env_ids], self.motion_times[env_ids]
        )

        if self.config.mimic_target_pose.enabled:
            # TODO: take env_ids as input here
            self.mimic_target_poses[:] = self.build_target_poses(
                self.config.mimic_target_pose.num_future_steps,
                self.config.mimic_target_pose.type,
                self.config.mimic_target_pose.with_time,
            )

    def on_epoch_end(self, current_epoch: int):
        super().on_epoch_end(current_epoch)
        if (
            self.config.mimic_dynamic_sampling.enabled
            and current_epoch > 0
            and current_epoch
            % self.config.mimic_dynamic_sampling.update_dynamic_weight_epochs
            == 0
        ):
            self.refresh_dynamic_weights()

    def build_target_poses(self, num_future_steps, target_pose_type, with_time):
        time_offsets = (
            torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
            * self.dt
        )

        raw_future_times = self.motion_times.unsqueeze(-1) + time_offsets.unsqueeze(0)
        motion_ids = self.motion_ids.unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        lengths = self.motion_lib.get_motion_length(flat_ids)
        flat_times = torch.minimum(raw_future_times.view(-1), lengths)

        ref_state = self.motion_lib.get_mimic_motion_state(flat_ids, flat_times)
        flat_target_pos = ref_state.rb_pos
        flat_target_rot = ref_state.rb_rot

        current_state = self.get_bodies_state()
        cur_gt, cur_gr = (
            current_state.body_pos,
            current_state.body_rot,
        )

        # First remove the height based on the current terrain, then remove the offset to get back to the ground-truth data position
        cur_gt[:, :, -1:] -= self.get_ground_heights(cur_gt[:, 0, :2]).view(
            self.num_envs, 1, 1
        )
        cur_gt[..., :2] -= self.respawn_offset_relative_to_data.clone()[..., :2].view(
            self.num_envs, 1, 2
        )

        if target_pose_type == "max-coords":
            target_pose_obs = build_max_coords_target_poses(
                cur_gt=cur_gt,
                cur_gr=cur_gr,
                flat_target_pos=flat_target_pos,
                flat_target_rot=flat_target_rot,
                num_envs=self.num_envs,
                num_future_steps=num_future_steps,
                mimic_conditionable_bodies_ids=self.mimic_conditionable_bodies_ids,
                w_last=self.w_last,
            )
        elif target_pose_type == "max-coords-future-rel":
            target_pose_obs = build_max_coords_target_poses_future_rel(
                cur_gt=cur_gt,
                cur_gr=cur_gr,
                flat_target_pos=flat_target_pos,
                flat_target_rot=flat_target_rot,
                num_envs=self.num_envs,
                num_future_steps=num_future_steps,
                mimic_conditionable_bodies_ids=self.mimic_conditionable_bodies_ids,
                w_last=self.w_last,
            )
        else:
            raise ValueError(f"Unknown target pose type '{target_pose_type}'")

        if with_time:
            target_pose_obs = self.add_time_to_target_poses(
                env_ids=torch.arange(self.num_envs, device=self.device),
                target_pose_obs=target_pose_obs,
                num_future_steps=num_future_steps,
            )

        return target_pose_obs

    def add_time_to_target_poses(self, env_ids, target_pose_obs, num_future_steps):
        num_envs = env_ids.shape[0]
        target_pose_obs = target_pose_obs.view(num_envs, num_future_steps, -1)

        time_offsets = (
            torch.arange(1, num_future_steps + 1, device=self.device, dtype=torch.long)
            * self.dt
        )

        raw_future_times = self.motion_times[env_ids].unsqueeze(
            -1
        ) + time_offsets.unsqueeze(0)
        motion_ids = self.motion_ids[env_ids].unsqueeze(-1).tile([1, num_future_steps])
        flat_ids = motion_ids.view(-1)

        lengths = self.motion_lib.get_motion_length(flat_ids)

        times = torch.minimum(raw_future_times.view(-1), lengths).view(
            num_envs, num_future_steps, 1
        ) - self.motion_times[env_ids].view(num_envs, 1, 1)

        obs = torch.cat([target_pose_obs, times], dim=-1).view(num_envs, -1)

        return obs

    def pre_physics_step(self, actions):
        if self.config.mimic_residual_control:
            actions = self.residual_actions_to_actual(actions)

        self.prev_contact_forces = self.get_bodies_contact_buf()

        return super().pre_physics_step(actions)

    def residual_actions_to_actual(
        self,
        residual_actions: Tensor,
        target_ids: Optional[Tensor] = None,
        target_times: Optional[Tensor] = None,
    ):
        if target_ids is None:
            target_ids = self.motion_ids

        if target_times is None:
            target_times = self.motion_times + self.dt

        ref_state = self.motion_lib.get_motion_state(target_ids, target_times)

        target_local_rot = dof_to_local(
            ref_state.dof_pos, self.get_dof_offsets(), self.w_last
        )
        residual_actions_as_quats = dof_to_local(
            residual_actions, self.get_dof_offsets(), self.w_last
        )

        actions_as_quats = rotations.quat_mul(
            residual_actions_as_quats, target_local_rot, self.w_last
        )
        actions = torch_utils.quat_to_exp_map(actions_as_quats, self.w_last).view(
            self.num_envs, -1
        )

        return actions

    def get_reward_joint_weights(self):
        """
        Calculate the weights for each joint in the reward function.

        This method assigns different weights to joints based on their importance
        in the overall motion. For the SMPLX humanoid model, it gives equal importance
        to each finger joint within a hand, while other joints maintain their default weight.

        Returns:
            torch.Tensor: A tensor of joint weights, with shape (num_bodies,).

        Note:
            - This method only applies unequal weights for the SMPLX humanoid model.
            - The weights are used to balance the contribution of different joints
              in the reward calculation, preventing over-emphasis on joints with many
              degrees of freedom (like hands).
        """
        # Return uniform weights if unequal weighting is disabled or not using SMPLX model
        if (
            not self.config.mimic_reward_config.unequal_reward_joint_weights
            or "smplx" not in self.config.robot.asset.asset_file_name
        ):
            return 1

        # Define groups of joints that should be weighted equally within the group
        joint_groups = [
            [
                "L_Wrist",
                "L_Index1",
                "L_Index2",
                "L_Index3",
                "L_Middle1",
                "L_Middle2",
                "L_Middle3",
                "L_Pinky1",
                "L_Pinky2",
                "L_Pinky3",
                "L_Ring1",
                "L_Ring2",
                "L_Ring3",
                "L_Thumb1",
                "L_Thumb2",
                "L_Thumb3",
            ],
            [
                "R_Wrist",
                "R_Index1",
                "R_Index2",
                "R_Index3",
                "R_Middle1",
                "R_Middle2",
                "R_Middle3",
                "R_Pinky1",
                "R_Pinky2",
                "R_Pinky3",
                "R_Ring1",
                "R_Ring2",
                "R_Ring3",
                "R_Thumb1",
                "R_Thumb2",
                "R_Thumb3",
            ],
        ]

        # Initialize all weights to 1
        joint_weights = torch.ones(
            self.num_bodies, device=self.device, dtype=torch.float
        )

        # Assign weights to joints in the defined groups
        for joint_index, joint_name in enumerate(self.body_names):
            for joint_group in joint_groups:
                if joint_name in joint_group:
                    # Set the weight to be 1 divided by the number of joints in the group
                    joint_weights[joint_index] = 1.0 / len(joint_group)

        # Normalize joint weights to sum to the total number of bodies
        total_bodies = len(self.body_names)
        current_sum = joint_weights.sum()
        joint_weights *= total_bodies / current_sum

        return joint_weights
