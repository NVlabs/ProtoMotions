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

import random
from pathlib import Path

from rich.progress import track

from phys_anim.agents.amp import AMP
from phys_anim.envs.mimic.common import MimicHumanoid
import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional


class Mimic(AMP):
    env: MimicHumanoid

    def play_steps(self):
        # This only happens after training is resumed. This makes sure the motions are quickly evaluated such that
        # the training focuses longer on the harder data.
        if self.env.config.mimic_dynamic_sampling.enabled:
            self.env.force_respawn_on_flat = torch.any(self.env.bucket_weights == 0)
        super().play_steps()

    def map_motions_to_iterations(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Map motion IDs to iterations for distributed processing.

        This method distributes motion IDs across available ranks and creates a mapping
        of motions to be processed in each iteration.

        Returns:
            Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]:
                - A list of tuples, each containing motion IDs and scene IDs for an iteration.
                - The number of motions assigned to this rank.
        """
        world_size = self.fabric.world_size
        global_rank = self.fabric.global_rank
        num_motions = self.motion_lib.num_sub_motions()

        if self.env.config.fixed_motion_id is None:
            # Calculate motions per rank, ensuring at least 1 motion per rank
            motions_per_rank = max(math.ceil(num_motions * 1.0 / world_size), 1)
            start_motion = motions_per_rank * global_rank
            end_motion = motions_per_rank * (global_rank + 1)

            # Adjust end_motion for the last rank to cover all remaining motions
            if global_rank == world_size - 1:
                end_motion = num_motions

        # Initialize tensor to track remaining motions for this rank
        remaining_motions = torch.zeros(
            num_motions, dtype=torch.bool, device=self.device
        )
        if self.env.config.fixed_motion_id is not None:
            remaining_motions[self.env.config.fixed_motion_id] = True
            num_motions = 1
        else:
            remaining_motions[start_motion:end_motion] = True
            num_motions = end_motion - start_motion

        motion_map = []
        motions_so_far = 0

        # Distribute motions across iterations
        while remaining_motions.any():
            num_motions_this_iter = min(remaining_motions.sum().item(), self.num_envs)

            if self.env.scene_lib is not None:
                # Sample motions with scene awareness if scene library is available
                available_scenes = torch.ones(
                    len(self.env.scene_lib.scenes), device=self.device, dtype=torch.bool
                )
                motion_ids, scene_ids = self.env.motion_lib.sample_motions_scene_aware(
                    num_motions_this_iter,
                    available_scenes,
                    self.env.scene_lib.single_robot_in_scene,
                    with_replacement=False,
                    available_motion_mask=remaining_motions,
                )
            else:
                # Simple sequential assignment if no scenes
                motion_ids = torch.arange(
                    start_motion + motions_so_far,
                    start_motion + motions_so_far + num_motions_this_iter,
                    device=self.device,
                )
                scene_ids = -1  # Indicate no specific scene

            motion_map.append((motion_ids, scene_ids))
            remaining_motions[motion_ids] = False
            motions_so_far += num_motions_this_iter

        return motion_map, num_motions

    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        self.eval()

        # Mark all scenes not in use
        if self.env.scene_lib is not None:
            self.env.scene_lib.mark_scene_not_in_use(
                torch.arange(len(self.env.scene_lib.scenes), device=self.device)
            )

        if self.env.config.fixed_motion_id is not None:
            num_motions = 1
        else:
            num_motions = self.motion_lib.num_sub_motions()

        metrics = {
            "reward_too_bad": torch.zeros(num_motions, device=self.device),
            "max_average_deviation": torch.zeros(num_motions, device=self.device),
            "active": torch.zeros(num_motions, device=self.device, dtype=torch.bool),
        }
        for k in self.config.eval_metric_keys:
            metrics[k] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_max"] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_min"] = 3 * torch.ones(num_motions, device=self.device)

        # Compute how many motions each rank should evaluate
        root_dir = Path(self.fabric.loggers[0].root_dir)
        motion_map, remaining_motions = self.map_motions_to_iterations()
        num_outer_iters = len(motion_map)
        # Maximal number of iters any of the ranks needs to perform.
        max_iters = max(self.fabric.all_gather(len(motion_map)))

        for outer_iter in track(
            range(max_iters),
            description=f"Evaluating... {remaining_motions} motions remain...",
        ):
            motion_pointer = outer_iter % num_outer_iters
            motion_ids, scene_ids = motion_map[motion_pointer]
            num_motions_this_iter = len(motion_ids)
            metrics["active"][motion_ids] = True

            self.env.scene_ids[:] = -1
            self.env.motion_ids[:num_motions_this_iter] = motion_ids
            self.env.scene_ids[:num_motions_this_iter] = scene_ids
            self.env.force_respawn_on_flat = True

            env_ids = torch.arange(
                0, num_motions_this_iter, dtype=torch.long, device=self.device
            )

            dt: float = self.env.dt
            motion_lengths = self.motion_lib.get_sub_motion_length(motion_ids)
            motion_num_frames = (motion_lengths / dt).floor().long()

            max_len = (
                motion_num_frames.max().item()
                if self.config.eval_length is None
                else self.config.eval_length
            )

            for eval_episode in range(self.config.eval_num_episodes):
                steps = torch.zeros(
                    num_motions_this_iter, dtype=torch.float, device=self.device
                )

                elapsed_time = (
                    torch.rand_like(self.motion_lib.state.motion_timings[motion_ids, 0])
                    * dt
                )
                self.env.motion_times[:num_motions_this_iter] = (
                    self.motion_lib.state.motion_timings[motion_ids, 0] + elapsed_time
                )
                self.env.reset_track_steps.reset_steps(env_ids)
                self.env.disable_reset = True
                self.env.disable_reset_track = True

                obs = self.env.reset(
                    # Force reset all envs to ensure we don't have any residual effects from previous iterations
                    #   for example, this ensures all untracked envs do not spawn near any objects.
                    torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
                )

                actor_state = self.create_actor_state()
                actor_state["obs"] = obs
                actor_state = self.get_extra_obs_from_env(actor_state)

                for l in range(max_len):
                    actor_state = self.pre_eval_env_step(actor_state)

                    actor_state = self.env_step(actor_state)

                    actor_state = self.post_eval_env_step(actor_state)
                    elapsed_time += dt
                    clip_done = (motion_lengths - dt) < elapsed_time
                    clip_not_done = torch.logical_not(clip_done)
                    for k in self.config.eval_metric_keys:
                        if k in self.env.last_unscaled_rewards:
                            value = self.env.last_unscaled_rewards[k].detach()
                        elif k in self.env.last_other_rewards:
                            value = self.env.last_other_rewards[k].detach()
                        else:
                            raise ValueError(
                                f"Key {k} not found in last_unscaled_rewards or last_other_rewards"
                            )

                        metric = value[:num_motions_this_iter]
                        metrics[k][motion_ids[clip_not_done]] += metric[clip_not_done]
                        metrics[f"{k}_max"][motion_ids[clip_not_done]] = torch.maximum(
                            metrics[f"{k}_max"][motion_ids[clip_not_done]],
                            metric[clip_not_done],
                        )
                        metrics[f"{k}_min"][motion_ids[clip_not_done]] = torch.minimum(
                            metrics[f"{k}_min"][motion_ids[clip_not_done]],
                            metric[clip_not_done],
                        )

                    current_gt_err = self.env.last_other_rewards["gt_err"][
                        :num_motions_this_iter
                    ]
                    # Update max_average_deviation for non-done motions
                    metrics["max_average_deviation"][motion_ids[clip_not_done]] = (
                        torch.maximum(
                            metrics["max_average_deviation"][motion_ids[clip_not_done]],
                            current_gt_err[clip_not_done],
                        )
                    )

                    if self.env.config.mimic_early_termination is not None:
                        reward_too_bad = torch.zeros(
                            num_motions_this_iter, device=self.device, dtype=bool
                        )
                        for entry in self.env.config.mimic_early_termination:
                            if entry.get("from_other", False):
                                from_dict = self.env.last_other_rewards
                            elif entry.use_scaled:
                                from_dict = self.env.last_scaled_rewards
                            else:
                                from_dict = self.env.last_unscaled_rewards

                            if entry.less_than:
                                entry_too_bad = (
                                    from_dict[entry.mimic_early_termination_key][
                                        :num_motions_this_iter
                                    ]
                                    < entry.mimic_early_termination_thresh
                                )
                            else:
                                entry_too_bad = (
                                    from_dict[entry.mimic_early_termination_key][
                                        :num_motions_this_iter
                                    ]
                                    > entry.mimic_early_termination_thresh
                                )
                            del from_dict

                            reward_too_bad = torch.logical_or(
                                reward_too_bad, entry_too_bad
                            )
                            reward_too_bad = torch.logical_and(
                                reward_too_bad, torch.logical_not(clip_done)
                            )

                        # Don't early track terminate if we very recently switched
                        # the tracking clip.
                        reward_too_bad = torch.logical_and(
                            reward_too_bad,
                            steps >= self.env.config.mimic_reset_track.grace_period,
                        )
                        reward_too_bad = torch.logical_and(
                            reward_too_bad, torch.logical_not(clip_done)
                        )

                        steps += 1

                        metrics["reward_too_bad"][motion_ids[reward_too_bad]] += 1
                        del reward_too_bad

        print("Evaluation done, now aggregating data.")

        if self.env.config.fixed_motion_id is None:
            # This means we potentially ran multiple episodes, each time with a subset of motions.
            # We need to aggregate the data from all episodes. So now we reference all the motion_ids.
            motion_lengths = (
                self.motion_lib.state.motion_timings[:, 1]
                - self.motion_lib.state.motion_timings[:, 0]
            )
            motion_num_frames = (motion_lengths / dt).floor().long()

        # Save to device so we can then load and aggregate. distributed.all_gather does not support dictionaries
        with open(root_dir / f"{self.fabric.global_rank}_metrics.pt", "wb") as f:
            torch.save(metrics, f)
        self.fabric.barrier()
        # Now rank 0 should load all the data and aggregate it
        if self.fabric.global_rank == 0:
            for rank in range(1, self.fabric.world_size):
                with open(root_dir / f"{rank}_metrics.pt", "rb") as f:
                    other_metrics = torch.load(f, map_location=self.device)
                for k in other_metrics.keys():
                    missing_motions = torch.logical_not(metrics["active"])
                    metrics[k][missing_motions] = other_metrics[k][missing_motions]
                metrics["active"] = torch.logical_or(
                    metrics["active"], other_metrics["active"]
                )

            assert metrics["active"].all(), "Not all motions were evaluated."
        self.fabric.barrier()
        # Once it has done, each rank should remove the file it created
        (root_dir / f"{self.fabric.global_rank}_metrics.pt").unlink()

        to_log = {}
        for k in self.config.eval_metric_keys:
            mean_tracking_errors = metrics[k] / (
                motion_num_frames * self.config.eval_num_episodes
            )
            to_log[f"eval/{k}"] = mean_tracking_errors.detach().mean().item()
            to_log[f"eval/{k}_max"] = metrics[f"{k}_max"].detach().mean().item()
            to_log[f"eval/{k}_min"] = metrics[f"{k}_min"].detach().mean().item()

        mean_reset_errors = (metrics["reward_too_bad"] > 0).float()
        to_log["eval/reward_too_bad"] = mean_reset_errors.detach().mean().item()
        to_log["eval/mean_reward_too_bad"] = (
            (
                metrics["reward_too_bad"]
                / (motion_num_frames * self.config.eval_num_episodes)
            )
            .detach()
            .mean()
            .item()
        )

        tracking_failures = (metrics["max_average_deviation"] > 0.5).float()
        to_log["eval/tracking_success_rate"] = (
            1.0 - tracking_failures.detach().mean().item()
        )

        # get indices of failed motions and save list to file
        failed_motions = torch.nonzero(tracking_failures).flatten().tolist()

        # save failed_motions in different files for each rank to avoid race conditions
        print(
            f"Saving to: {root_dir / f'failed_motions_{self.fabric.global_rank}.txt'}"
        )
        with open(root_dir / f"failed_motions_{self.fabric.global_rank}.txt", "w") as f:
            for motion_id in failed_motions:
                f.write(f"{motion_id}\n")

        stop_early = (
            self.config.training_early_termination.early_terminate_cart_err is not None
            or self.config.training_early_termination.early_terminate_reward_too_bad_prob
            is not None
        )
        if self.config.training_early_termination.early_terminate_cart_err is not None:
            cart_err = to_log["eval/cartesian_err"]
            stop_early = stop_early and (
                cart_err
                <= self.config.training_early_termination.early_terminate_cart_err
            )
        if (
            self.config.training_early_termination.early_terminate_reward_too_bad_prob
            is not None
        ):
            early_term_prob = to_log["eval/mean_reward_too_bad"]

            stop_early = stop_early and (
                early_term_prob
                <= self.config.training_early_termination.early_terminate_reward_too_bad_prob
            )

        if stop_early:
            print(
                f"Stopping early! Target error reached, cart_err: {to_log['eval/cartesian_err'].item()}, early_term_prob: {to_log['eval/reward_too_bad'].item()}"
            )
            # Rank 0 will broadcast the best score to all ranks. This ensures all ranks are synchronized before saving.
            evaluated_score = self.fabric.broadcast(
                to_log["eval/tracking_success_rate"], src=0
            )
            self.best_evaluated_score = evaluated_score

            self.save(new_high_score=True)
            self.terminate_early()

        self.env.disable_reset = False
        self.env.disable_reset_track = False
        self.env.force_respawn_on_flat = False

        all_ids = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        self.env.reset_track(all_ids)

        self.force_full_restart = True

        schedule = self.config.actor.config.sigma_schedule
        if schedule is not None:
            if schedule.get("mimic_success", False):
                total_failing_motions = torch.sum(
                    (metrics["reward_too_bad"] > 0).float()
                )
                percent_failure = total_failing_motions / num_motions

                new_logstd = self.config.actor.config.init_logstd - percent_failure * (
                    self.config.config.actor.init_logstd - schedule.end_logstd
                )
            else:
                new_logstd = schedule.init_logstd + min(
                    max(0, self.current_epoch - schedule.get("start_epoch", 0))
                    / schedule.end_epoch,
                    1,
                ) * (schedule.end_logstd - schedule.init_logstd)

            self.actor.set_logstd(new_logstd)

        return to_log, to_log["eval/tracking_success_rate"]

    @torch.no_grad()
    def evaluate_mimic(self):
        self.eval()
        self.create_eval_callbacks()
        self.pre_evaluate_policy(reset_env=False)

        motion_index_offset = self.env.config.motion_index_offset
        if motion_index_offset is None:
            motion_index_offset = 0
        num_motions = self.motion_lib.num_sub_motions() - motion_index_offset
        # This is a bit hacky... we set fixed_motion_id when we want to evaluate one problem across multiple envs
        # and we set num_motions to the number of envs when we want to evaluate multiple problems in parallel
        if num_motions > self.num_envs or self.env.config.fixed_motion_id is not None:
            num_motions = self.num_envs

        motion_ids = (
            torch.arange(0, num_motions, dtype=torch.long, device=self.device)
            + motion_index_offset
        )
        if self.env.config.fixed_motion_id is not None:
            motion_ids = torch.zeros_like(motion_ids) + self.env.config.fixed_motion_id

        if hasattr(self, "vae_noise"):
            self.reset_vae_noise(None)

        total_successes = torch.zeros(num_motions, device=self.device)
        tracked_successes = torch.zeros(
            num_motions, self.config.eval_num_episodes, device=self.device
        )

        self.env.motion_ids[:num_motions] = motion_ids

        env_ids = torch.arange(0, num_motions, dtype=torch.long, device=self.device)

        dt: float = self.env.dt
        motion_lengths = self.motion_lib.get_sub_motion_length(motion_ids)
        motion_num_frames = (motion_lengths / dt).floor().long()

        metrics = {
            "reward_too_bad": torch.zeros(num_motions, device=self.device),
            "max_average_deviation": torch.zeros(num_motions, device=self.device),
            "max_max_deviation": torch.zeros(num_motions, device=self.device),
            "min_object_distance": torch.ones(
                self.config.eval_num_episodes, num_motions, device=self.device
            )
            * 1000,
            "success_object": torch.zeros(
                self.config.eval_num_episodes, num_motions, device=self.device
            ),
        }
        for k in self.config.eval_metric_keys:
            metrics[k] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_max"] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_min"] = 3 * torch.ones(num_motions, device=self.device)
        max_len = (
            motion_num_frames.max().item()
            if self.config.eval_length is None
            else self.config.eval_length
        )

        for eval_episode in range(self.config.eval_num_episodes):
            torch.cuda.empty_cache()

            random_offset = random.randint(0, 8000)

            steps = torch.zeros(num_motions, dtype=torch.float, device=self.device)

            elapsed_time = (
                torch.rand_like(self.motion_lib.state.motion_timings[motion_ids, 0])
                * dt
            )
            self.env.motion_times[:num_motions] = (
                self.motion_lib.state.motion_timings[motion_ids, 0] + elapsed_time
            )
            self.env.reset_track_steps.reset_steps(env_ids)
            self.env.disable_reset = True
            self.env.disable_reset_track = True
            if self.env.scene_lib is not None:
                self.env.scene_lib.mark_scene_not_in_use(
                    self.env.scene_ids[:num_motions]
                )
            self.env.scene_ids[:num_motions], valid_scene_masks = (
                self.env.sample_scene_ids(self.env.motion_ids[:num_motions])
            )
            # Make sure all scenes are validly sampled
            assert valid_scene_masks.all(), "Invalid scene sampled"
            if self.env.scene_lib is not None:
                self.env.scene_lib.mark_scene_in_use(
                    self.env.scene_ids[:num_motions][valid_scene_masks]
                )
            obs = self.env.reset(
                torch.arange(0, num_motions, dtype=torch.long, device=self.device)
            )

            actor_state = self.create_actor_state()
            actor_state["obs"] = obs
            actor_state = self.get_extra_obs_from_env(actor_state)

            if "motion_ids" in self.extra_obs_inputs:
                if self.actor.mu_model.extra_input_models[
                    "motion_ids"
                ].config.random_embedding.use_random_embeddings:
                    actor_state["motion_ids"] = (
                        actor_state["motion_ids"].clone() + random_offset
                    )

            if hasattr(self, "vae_noise"):
                self.reset_vae_noise(actor_state["done_indices"])

            for l in track(
                range(max_len),
                description=f"Evaluating episode {eval_episode}/{self.config.eval_num_episodes}...",
            ):
                actor_state = self.pre_eval_env_step(actor_state)

                actor_state = self.env_step(actor_state)
                if "motion_ids" in self.extra_obs_inputs:
                    if self.actor.mu_model.extra_input_models[
                        "motion_ids"
                    ].config.random_embedding.use_random_embeddings:
                        actor_state["motion_ids"] = (
                            actor_state["motion_ids"].clone() + random_offset
                        )

                actor_state = self.post_eval_env_step(actor_state)
                elapsed_time += dt
                clip_done = (motion_lengths - dt) < elapsed_time
                clip_not_done = torch.logical_not(clip_done)
                for k in self.config.eval_metric_keys:
                    if k in self.env.last_unscaled_rewards:
                        value = self.env.last_unscaled_rewards[k].detach()
                    elif k in self.env.last_other_rewards:
                        value = self.env.last_other_rewards[k].detach()
                    else:
                        raise ValueError(
                            f"Key {k} not found in last_unscaled_rewards or last_other_rewards"
                        )

                    metric = value[:num_motions]
                    metric *= 1 - clip_done.long()
                    metrics[k] += metric
                    metrics[f"{k}_max"][clip_not_done] = torch.maximum(
                        metrics[f"{k}_max"], metric
                    )[clip_not_done]
                    metrics[f"{k}_min"][clip_not_done] = torch.minimum(
                        metrics[f"{k}_min"], metric
                    )[clip_not_done]

                metrics["max_average_deviation"][clip_not_done] = torch.maximum(
                    metrics["max_average_deviation"][clip_not_done],
                    self.env.last_other_rewards["gt_err"][:num_motions][clip_not_done],
                )
                metrics["max_max_deviation"][clip_not_done] = torch.maximum(
                    metrics["max_max_deviation"][clip_not_done],
                    self.env.last_other_rewards["max_joint_err"][:num_motions][
                        clip_not_done
                    ],
                )

                if "success_object_position" in self.config.eval_metric_keys:
                    non_zero_distance = (
                        self.env.last_other_rewards["distance_to_object_position"][
                            :num_motions
                        ]
                        > 0
                    )
                    metrics["min_object_distance"][eval_episode][non_zero_distance] = (
                        torch.minimum(
                            metrics["min_object_distance"][eval_episode],
                            self.env.last_other_rewards["distance_to_object_position"][
                                :num_motions
                            ],
                        )[non_zero_distance]
                    )
                    metrics["success_object"][eval_episode][non_zero_distance] = (
                        torch.torch.maximum(
                            metrics["success_object"][eval_episode],
                            self.env.last_other_rewards["success_object_position"][
                                :num_motions
                            ],
                        )[non_zero_distance]
                    )

                if self.env.config.mimic_early_termination is not None:
                    reward_too_bad = torch.zeros(
                        num_motions, device=self.device, dtype=bool
                    )
                    for entry in self.env.config.mimic_early_termination:
                        if entry.get("from_other", False):
                            from_dict = self.env.last_other_rewards
                        elif entry.use_scaled:
                            from_dict = self.env.last_scaled_rewards
                        else:
                            from_dict = self.env.last_unscaled_rewards

                        if entry.less_than:
                            entry_too_bad = (
                                from_dict[entry.mimic_early_termination_key][
                                    :num_motions
                                ]
                                < entry.early_reward_end_term_thresh
                            )
                        else:
                            entry_too_bad = (
                                from_dict[entry.mimic_early_termination_key][
                                    :num_motions
                                ]
                                > entry.early_reward_end_term_thresh
                            )

                        reward_too_bad = torch.logical_or(reward_too_bad, entry_too_bad)
                        reward_too_bad = torch.logical_and(
                            reward_too_bad, torch.logical_not(clip_done)
                        )
                        del from_dict

                    # Don't early track terminate if we very recently switched
                    # the tracking clip.
                    reward_too_bad = torch.logical_and(
                        reward_too_bad,
                        steps >= self.env.config.mimic_reset_track.grace_period,
                    )

                    steps += 1

                    metrics["reward_too_bad"][reward_too_bad] += 1
                    del reward_too_bad

            mean_tracking_errors = metrics["max_average_deviation"] < 0.5
            total_successes[mean_tracking_errors] += 1
            tracked_successes[mean_tracking_errors, eval_episode] += 1

        to_log = {}
        for k in self.config.eval_metric_keys:
            mean_tracking_errors = metrics[k] / (
                motion_num_frames * self.config.eval_num_episodes
            )
            to_log[f"eval/{k}"] = mean_tracking_errors.detach().mean()
            to_log[f"eval/{k}_max"] = metrics[f"{k}_max"].detach().mean()
            to_log[f"eval/{k}_min"] = metrics[f"{k}_min"].detach().mean()

        if "reach_success" in self.env.last_other_rewards:
            to_log["eval/reach_success"] = torch.tensor(
                self.env.last_other_rewards["reach_success"]
            )
            to_log["eval/reach_distance"] = torch.tensor(
                self.env.last_other_rewards["reach_distance"]
            )

        mean_tracking_errors = metrics["max_average_deviation"]
        to_log["eval/max_average_deviation"] = mean_tracking_errors.detach().mean()
        mean_tracking_errors = metrics["max_max_deviation"]
        to_log["eval/max_max_deviation"] = mean_tracking_errors.detach().mean()

        mean_tracking_errors = (metrics["max_average_deviation"] > 0.5).float()
        to_log["eval/count_bound_max_average_deviation"] = (
            mean_tracking_errors.detach().mean()
        )
        mean_tracking_errors = (metrics["max_max_deviation"] > 0.5).float()
        to_log["eval/count_bound_max_max_deviation"] = (
            mean_tracking_errors.detach().mean()
        )

        successful_envs = metrics["max_average_deviation"] < 0.5
        gt_err = metrics["gt_err"] / (motion_num_frames * self.config.eval_num_episodes)
        to_log["eval/gt_err_success"] = gt_err[successful_envs].detach().mean()
        gr_err = metrics["gr_err"] / (motion_num_frames * self.config.eval_num_episodes)
        to_log["eval/gr_err_success"] = gr_err[successful_envs].detach().mean()

        mean_reset_errors = (metrics["reward_too_bad"] > 0).float()
        to_log["eval/reward_too_bad"] = mean_reset_errors.detach().mean()
        to_log["eval/mean_reward_too_bad"] = (
            (
                metrics["reward_too_bad"]
                / (motion_num_frames * self.config.eval_num_episodes)
            )
            .detach()
            .mean()
        )

        to_log["eval/success_rate"] = (
            total_successes.detach().mean() / self.config.eval_num_episodes
        )
        any_success = (total_successes > 0).float()
        to_log["eval/success_rate_top_k"] = any_success.detach().mean()

        average_object_success = metrics["success_object"].mean()
        to_log["eval/success_object"] = average_object_success.detach().mean()

        average_min_object_distance = metrics["min_object_distance"].mean()
        to_log["eval/min_object_distance"] = average_min_object_distance.detach().mean()

        indices = [idx for idx in range(self.config.eval_num_episodes)]
        for i in range(self.config.eval_num_episodes):
            successes = []
            for _ in range(50):
                np.random.shuffle(indices)
                count_undereq_i = (
                    tracked_successes[:, indices[: i + 1]].detach().sum(dim=1) > 0
                ).float()
                successes.append(count_undereq_i.cpu().detach().mean())

            # store quantiles for successes
            to_log[f"eval/success_rate_top_{i}"] = np.mean(successes)
            to_log[f"eval/success_rate_top_{i}_25q"] = np.quantile(successes, 0.25)
            to_log[f"eval/success_rate_top_{i}_75q"] = np.quantile(successes, 0.75)

        print("--- EVAL MIMIC RESULTS ---")
        for key, value in to_log.items():
            print(f"{key}: {value.item()}")

        print(f"Object motion results: {total_successes.int().detach().cpu().numpy()}")

        self.post_evaluate_policy()
