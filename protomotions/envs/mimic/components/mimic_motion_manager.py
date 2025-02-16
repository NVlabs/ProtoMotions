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

import torch
import math

from protomotions.envs.base_env.env_utils.general import StepTracker
from protomotions.envs.base_env.components.motion_manager import MotionManager


class MimicMotionManager(MotionManager):
    def __init__(self, config, env):
        super().__init__(config, env)
        self.disable_reset_track = False
        
        # For dynamic sampling in motion tracking, we record whether the motion was respawned on a flat terrain.
        # We do not record failures on irregular terrain for prioritized sampling as there are no guarantees it should have succeeded.
        self.envs_tracked_for_dynamic_sampling = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        self.dynamic_sampling_tracked_failures = torch.zeros(
            self.env.num_envs, dtype=torch.float, device=self.env.device
        )

        # Sampling vectors
        self.init_random_probs = (
            torch.ones(self.env.num_envs, dtype=torch.float, device=self.env.device)
            * self.config.motion_sampling.init_random_prob
        )

        self.reset_track_steps = StepTracker(
            self.env.num_envs,
            min_steps=self.config.reset_track.steps_min,
            max_steps=self.config.reset_track.steps_max,
            device=self.env.device,
        )

        if self.config.dynamic_sampling.enabled:
            self.setup_dynamic_sampling()

    def get_done_tracks(self):
        end_times = (
            self.env.motion_lib.state.motion_lengths[self.motion_ids]
            - self.env.dt  # Remove 1 frame to avoid overflowing
        )
        done_clip = (self.motion_times + self.env.dt) >= end_times
        return done_clip

    def get_has_reset_grace(self):
        return self.reset_track_steps.steps <= self.config.reset_track.grace_period

    def post_physics_step(self):
        self.motion_times += self.env.dt

    def handle_reset_track(self):
        if self.disable_reset_track:
            return
        self.reset_track_steps.advance()

    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_track(env_ids)

    def setup_dynamic_sampling(self):
        num_buckets_list = []
        bucket_motion_ids_list = []
        bucket_starts_list = []
        bucket_lengths_list = []
        bucket_width = self.config.dynamic_sampling.bucket_width

        lengths = self.env.motion_lib.state.motion_lengths.tolist()
        for motion_id, length in enumerate(lengths):
            if self.config.fixed_motion_id is not None:
                # When training on a fixed motion, only add buckets for that motion.
                if motion_id != self.config.fixed_motion_id:
                    continue
            buckets = math.ceil(length / bucket_width)
            num_buckets_list.append(buckets)
            for j in range(buckets):
                bucket_motion_ids_list.append(motion_id)
                start = j * bucket_width
                end = min(length, start + bucket_width)
                assert end - start > 0
                bucket_starts_list.append(start)
                bucket_lengths_list.append(end - start)

        num_buckets = torch.tensor(
            num_buckets_list, dtype=torch.long, device=self.env.device
        )

        rolled = num_buckets.roll(1)
        rolled[0] = 0
        self.bucket_offsets = rolled.cumsum(0)

        total_num_buckets = sum(num_buckets_list)

        self.bucket_scores = torch.zeros(
            total_num_buckets, dtype=torch.float, device=self.env.device
        )
        self.bucket_frames_spent = torch.zeros(
            total_num_buckets, dtype=torch.long, device=self.env.device
        )

        # Minimal score equivalent to 1/total_num_buckets.
        # If a single bucket always fails and the rest always succeed,
        # there's an equal chance to sample the bad bucket, or any of the good buckets.
        self._min_bucket_weight = torch.ones(
            total_num_buckets, dtype=torch.float, device=self.env.device
        ) * (
            self.config.dynamic_sampling.min_bucket_weight
            if self.config.dynamic_sampling.min_bucket_weight is not None
            else 1e-6
        )
        self.bucket_weights = torch.zeros(
            total_num_buckets, dtype=torch.float, device=self.env.device
        )

        # The motion id each bucket corresponds to.
        self.bucket_motion_ids = torch.tensor(
            bucket_motion_ids_list, dtype=torch.long, device=self.env.device
        )

        # The start time (in seconds) of the bucket with respect to its corresponding motion.
        self.bucket_starts = torch.tensor(
            bucket_starts_list, dtype=torch.float, device=self.env.device
        )

        # The length of each bucket, in seconds (some bucket at the end of clips may be shorter than the set bucket width).
        self.bucket_lengths = torch.tensor(
            bucket_lengths_list, dtype=torch.float, device=self.env.device
        )

    def dynamic_sample(self, n: int):
        """
        Dynamically sample motion sequences based on their difficulty.

        This method implements a weighted sampling strategy to select motion sequences,
        prioritizing more challenging motions. It uses a bucket system where each bucket
        represents a portion of a motion sequence.

        Args:
            n (int): The number of motion sequences to sample.
        Note:
        - The method uses the current state of bucket weights, which are updated elsewhere
          based on the success/failure of previous attempts at these motions.
        - It includes logic to handle cases where some buckets have never been visited,
          ensuring exploration of new motion segments.
        """

        # Prepare the sampling weights, ensuring a minimum weight for each bucket
        weights = self.bucket_weights.clone().clamp(min=self._min_bucket_weight)
        if (self.bucket_weights == 0).any():
            # Bucket weights are always >= 0 for any visited bucket.
            # If a bucket has never been visited, it should be sampled with equal probability.
            weights[self.bucket_weights == 0] = 1.0
            weights[self.bucket_weights != 0] = 0.0
            norm_weight = weights
        else:
            # Normalize weights and clamp to a maximum value if specified
            min_weight = weights.min()
            norm_weight = weights / min_weight
            if self.config.dynamic_sampling.dynamic_weight_max is not None:
                norm_weight = norm_weight.clamp(
                    max=self.config.dynamic_sampling.dynamic_weight_max
                )

        # Sample buckets using the prepared weights
        chosen_bucket_indices = torch.multinomial(
            norm_weight, num_samples=n, replacement=True
        )
        motion_ids = self.bucket_motion_ids[chosen_bucket_indices]
        bucket_starts = self.bucket_starts[chosen_bucket_indices]
        bucket_lengths = self.bucket_lengths[chosen_bucket_indices]

        # Determine the corresponding motion ID and time for each sampled bucket
        motion_times = (
            torch.rand(n, device=self.env.device) * bucket_lengths + bucket_starts
        )

        return motion_ids, motion_times

    def update_dynamic_stats(self):
        """
        Update dynamic statistics for motion sampling based on performance.
        This method is crucial for adaptive sampling strategies.
        """
        if not self.config.dynamic_sampling.enabled:
            return
        if self.env.disable_reset:
            return

        # Only update stats for motions that failed on flat terrain
        if torch.any(self.envs_tracked_for_dynamic_sampling):
            # Extract relevant data for motions that failed on and are valid for over-sampling
            valid_motion_ids = self.motion_ids[self.envs_tracked_for_dynamic_sampling]
            valid_motion_times = self.motion_times[
                self.envs_tracked_for_dynamic_sampling
            ]

            # Get performance metrics for failed motions
            valid_failed_due_bad_reward = self.dynamic_sampling_tracked_failures[
                self.envs_tracked_for_dynamic_sampling
            ]

            # Determine bucket indices for the failed motions
            if self.config.fixed_motion_id is not None:
                # If using a fixed motion, all buckets correspond to that motion
                valid_motion_ids = torch.zeros_like(valid_motion_ids)

            # Calculate the exact bucket for each motion
            base_offsets = self.bucket_offsets[valid_motion_ids]
            extra_offsets = torch.floor(
                valid_motion_times / self.config.dynamic_sampling.bucket_width
            ).long()
            bucket_indices = base_offsets + extra_offsets

            # NOTE These two lines
            # self.bucket_frames_spent[bucket_indices] += 1
            # self.bucket_scores[bucket_indices] += self.rew_buf
            # are NOT what we want, see https://discuss.pytorch.org/t/how-to-do-atomic-add-on-slice-with-duplicate-indices/136193

            self.bucket_frames_spent.scatter_add_(
                0, bucket_indices, torch.ones_like(bucket_indices)
            )

            # Update bucket scores
            self.bucket_scores.scatter_add_(
                0, bucket_indices, valid_failed_due_bad_reward
            )

    def refresh_dynamic_weights(self, current_epoch: int):
        """
        Refresh dynamic weights for motion sampling.
        This method updates the dynamic weights based on the performance metrics of the sampled motions.
        It ensures that more challenging motions are sampled more frequently.
        """
        if not (
            self.config.dynamic_sampling.enabled
            and current_epoch > 0
            and current_epoch
            % self.config.dynamic_sampling.update_dynamic_weight_epochs
            == 0
        ):
            return

        visited = self.bucket_frames_spent > 0
        average_score = self.bucket_scores[visited] / self.bucket_frames_spent[visited]
        weight = torch.pow(
            average_score,
            self.config.dynamic_sampling.dynamic_weight_pow,
        )

        self.bucket_weights[visited] = torch.clamp(
            weight + self.bucket_weights[visited] * 0.7,
            min=self._min_bucket_weight[visited],
        )

        tensors_of_interest = {
            "bucket_frames_spent": self.bucket_frames_spent.float(),
            "bucket_average_score": average_score,
            "bucket_scores": self.bucket_scores,
            "bucket_weights": self.bucket_weights,
            "bucket_added_weights": weight,
        }

        for k, v in tensors_of_interest.items():
            if v.shape[0] > 0:
                self.env.log_dict[f"{k}_min"] = v.min()
                self.env.log_dict[f"{k}_max"] = v.max()
                self.env.log_dict[f"{k}_mean"] = v.mean()

        self.bucket_frames_spent[:] = 0
        self.bucket_scores[:] = 0

    def sample_motions(self, env_ids, new_motion_ids=None):
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

        if self.config.fixed_motion_per_env:
            # We typically use this for recording. Maybe worth moving into the motion manager logic.
            motion_index_offset = self.config.motion_index_offset
            if motion_index_offset is None:
                motion_index_offset = 0
            new_motion_ids = torch.fmod(
                env_ids + motion_index_offset,
                self.env.motion_lib.num_motions(),
            )
            new_times = torch.zeros_like(
                self.env.motion_lib.state.motion_lengths[new_motion_ids]
            )
        elif self.config.dynamic_sampling.enabled and new_motion_ids is None:
            new_motion_ids, new_times = self.dynamic_sample(len(env_ids))
        else:
            if new_motion_ids is None:
                new_motion_ids = self.env.motion_lib.sample_motions(len(env_ids))
            if self.config.fixed_motion_id is not None:
                new_motion_ids = (
                    torch.zeros_like(new_motion_ids) + self.config.fixed_motion_id
                )
            new_times = self.env.motion_lib.sample_time(
                new_motion_ids, truncate_time=self.env.dt
            )

        if self.config.motion_sampling.init_start_prob > 0:
            init_start = torch.bernoulli(self.init_start_probs[: len(env_ids)])
            new_times = torch.where(
                init_start == 1,
                torch.zeros_like(new_times),
                new_times,
            )

        if self.config.motion_sampling.init_random_prob > 0:
            init_random = torch.bernoulli(self.init_random_probs[: len(env_ids)])
            new_times = torch.where(
                init_random == 1,
                self.env.motion_lib.sample_time(
                    new_motion_ids,
                    truncate_time=self.env.dt,
                ),
                new_times,
            )

        self.motion_ids[env_ids] = new_motion_ids
        self.motion_times[env_ids] = new_times

    def reset_track(self, env_ids):
        self.reset_track_steps.reset_steps(env_ids)

    def get_state_dict(self):
        if self.config.dynamic_sampling.enabled:
            state_dict = {
                "bucket_weights": self.bucket_weights.cpu().clone(),
            }
        else:
            state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict):
        if self.config.dynamic_sampling.enabled:
            self.bucket_weights[:] = state_dict["bucket_weights"].to(self.bucket_weights.device)
