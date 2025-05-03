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
from torch import Tensor
import logging
import time
from lightning.fabric import Fabric

from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Tuple, Dict, List, Optional
from pathlib import Path
from rich.progress import track

from protomotions.agents.utils.data_utils import DictDataset, ExperienceBuffer
from protomotions.envs.mimic.env import Mimic as MimicEnv
from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.model import PPOModel
from protomotions.agents.masked_mimic.model import VaeDeterministicOutputModel
from protomotions.agents.common.common import weight_init

log = logging.getLogger(__name__)


class MaskedMimic(PPO):
    env: MimicEnv

    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def __init__(self, fabric: Fabric, env: MimicEnv, config):
        super().__init__(fabric, env, config)

        # Initialize VAE noise for each environment.
        self.vae_noise = torch.zeros(
            self.num_envs,
            self.config.model.config.vae_latent_dim,
            dtype=torch.float,
            device=self.device,
        )

    def setup(self):
        # Instantiate model and optimizer.
        model: VaeDeterministicOutputModel = instantiate(self.config.model)
        model.apply(weight_init)
        optimizer = instantiate(
            self.config.model.config.optimizer,
            params=list(model.parameters()),
        )

        self.model, self.optimizer = self.fabric.setup(model, optimizer)
        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_vae_outputs")

        # Optionally load a pre-trained expert model if provided.
        if self.config.expert_model_path is not None:
            print(
                "Loading pre-trained full-body tracker from:",
                self.config.expert_model_path,
            )
            # "score_based.ckpt" is the name of the file that is saved when a new high score is achieved
            checkpoint_path = self.config.expert_model_path + "/score_based.ckpt"
            if not Path(checkpoint_path).exists():
                checkpoint_path = self.config.expert_model_path + "/last.ckpt"
            pre_trained_expert = torch.load(checkpoint_path, map_location='cpu')

            self.expert_model_config = OmegaConf.load(
                Path(self.config.expert_model_path) / "config.yaml"
            )
            assert (
                self.env.mimic_obs_cb.config.mimic_target_pose.num_future_steps
                == self.expert_model_config.env.config.mimic_target_pose.num_future_steps
            )
            assert (
                self.env.mimic_obs_cb.config.mimic_target_pose.type
                == self.expert_model_config.env.config.mimic_target_pose.type
            )
            assert (
                self.env.mimic_obs_cb.config.mimic_target_pose.with_time
                == self.expert_model_config.env.config.mimic_target_pose.with_time
            )
            expert_model: PPOModel = instantiate(
                self.expert_model_config.agent.config.model
            )
            
            # load parameters before setting up the model
            expert_model.load_state_dict(pre_trained_expert["model"])

            self.expert_model = self.fabric.setup(expert_model)
            self.expert_model.mark_forward_method("act")
            for param in self.expert_model.parameters():
                param.requires_grad = False
            self.expert_model.eval()  # Just incase
        else:
            self.expert_model = None

    # -----------------------------
    # Experience Buffer Registration
    # -----------------------------
    def register_extra_experience_buffer_keys(self):
        pass

    # -----------------------------
    # VAE Noise Management
    # -----------------------------
    def reset_vae_noise(self, env_ids):
        """Reset the VAE noise tensor based on the selected noise type."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if type(env_ids) is list:
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)

        noise_type = self.config.model.config.vae_noise_type
        if noise_type == "normal":
            epsilon = torch.randn(
                env_ids.shape[0], self.model.config.vae_latent_dim, device=self.device
            )  # sampling epsilon
        elif noise_type == "uniform":
            epsilon = torch.rand(
                env_ids.shape[0], self.model.config.vae_latent_dim, device=self.device
            )  # sampling epsilon
        elif noise_type == "zeros":
            epsilon = torch.zeros(
                env_ids.shape[0], self.model.config.vae_latent_dim, device=self.device
            )  # no noise
        else:
            raise NotImplementedError
        self.vae_noise[env_ids] = epsilon

    # -----------------------------
    # Environment Step and Reset Handling
    # -----------------------------
    def handle_reset(self, done_indices=None):
        """Reset environment states and VAE noise, then update the observation accordingly."""
        self.reset_vae_noise(done_indices)
        obs = super().handle_reset(done_indices)
        obs["vae_noise"] = self.vae_noise.clone()
        return obs

    def env_step(self, actions):
        """Perform an environment step and inject current VAE noise into observations."""
        obs, rewards, dones, terminated, extras = super().env_step(actions)
        obs["vae_noise"] = self.vae_noise.clone()

        return obs, rewards, dones, terminated, extras

    def load_parameters(self, state_dict):
        self.current_epoch = state_dict["epoch"]

        if "step_count" in state_dict:
            self.step_count = state_dict["step_count"]
        if "run_start_time" in state_dict:
            self.fit_start_time = state_dict["run_start_time"]

        self.best_evaluated_score = state_dict.get("best_evaluated_score", None)

        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

        if self.config.normalize_values:
            self.running_val_norm.load_state_dict(state_dict["running_val_norm"])

        self.episode_reward_meter.load_state_dict(state_dict["episode_reward_meter"])
        self.episode_length_meter.load_state_dict(state_dict["episode_length_meter"])

    # -----------------------------
    # Training Loop and Dataset Processing
    # -----------------------------
    def fit(self):
        # Setup experience buffer
        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(
            self.device
        )
        self.experience_buffer.register_key(
            "self_obs", shape=(self.env.config.robot.self_obs_size,)
        )
        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("extra_rewards")
        self.experience_buffer.register_key("total_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key(
            "expert_actions", shape=(self.env.config.robot.number_of_actions,)
        )
        self.register_extra_experience_buffer_keys()

        if self.config.get("extra_inputs", None) is not None:
            obs = self.env.get_obs()
            for key in self.config.extra_inputs.keys():
                assert (
                    key in obs
                ), f"Key {key} not found in obs returned from env: {obs.keys()}"
                env_tensor = obs[key]
                shape = env_tensor.shape
                dtype = env_tensor.dtype
                self.experience_buffer.register_key(key, shape=shape[1:], dtype=dtype)

        # Force reset on fit start
        done_indices = None
        if self.fit_start_time is None:
            self.fit_start_time = time.time()
        self.fabric.call("on_fit_start", self)

        while self.current_epoch < self.config.max_epochs:
            self.epoch_start_time = time.time()

            # Set networks in eval mode to ensure normalizers are not updated
            self.eval()
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)

                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch}, collecting data...",
                ):
                    obs = self.handle_reset(done_indices)

                    self.experience_buffer.update_data(
                        "self_obs", step, obs["self_obs"]
                    )
                    if self.config.get("extra_inputs", None) is not None:
                        for key in self.config.extra_inputs:
                            self.experience_buffer.update_data(key, step, obs[key])

                    # At training we use the encoder to obtain less-noisy latent codes
                    action = self.model.act(obs, with_encoder=True)

                    expert_action = self.expert_model.act(obs)
                    self.experience_buffer.update_data(
                        "expert_actions", step, expert_action
                    )

                    # Go over all keys in obs and check if any has nans
                    for key in obs.keys():
                        has_nan = False
                        if torch.isnan(obs[key]).any():
                            has_nan = True
                            print(f"NaN in {key}: {obs[key]}")
                        if has_nan:
                            raise ValueError("NaN in obs")
                    if torch.isnan(action).any():
                        raise ValueError(f"NaN in action: {action}")

                    # Step env
                    next_obs, rewards, dones, terminated, extras = self.env_step(action)

                    all_done_indices = dones.nonzero(as_tuple=False)
                    done_indices = all_done_indices.squeeze(-1)

                    # Updating episode logging metrics
                    self.post_train_env_step(rewards, dones, done_indices, extras, step)

                    self.experience_buffer.update_data("rewards", step, rewards)
                    self.experience_buffer.update_data("dones", step, dones)

                    self.step_count += self.get_step_count_increment()

                # After collecting data, calculate rewards and update experience buffer
                rewards = self.experience_buffer.rewards
                extra_rewards = self.calculate_extra_reward()
                self.experience_buffer.batch_update_data("extra_rewards", extra_rewards)
                total_rewards = rewards + extra_rewards
                self.experience_buffer.batch_update_data("total_rewards", total_rewards)

            training_log_dict = self.optimize_model()

            training_log_dict["epoch"] = self.current_epoch
            self.current_epoch += 1
            self.fabric.call("after_train", self)

            # Save model before running eval. Eval is often a long operation and has some stability/memory issues.
            # This ensures that we have a checkpoint saved before running eval.
            if self.current_epoch % self.config.manual_save_every == 0:
                self.save()

            if (
                self.config.eval_metrics_every is not None
                and self.current_epoch > 0
                and self.current_epoch % self.config.eval_metrics_every == 0
            ):
                eval_log_dict, evaluated_score = self.calc_eval_metrics()
                # Rank 0 will broadcast the best score to all ranks. This ensures all ranks are synchronized before saving.
                evaluated_score = self.fabric.broadcast(evaluated_score, src=0)
                if evaluated_score is not None:
                    if (
                        self.best_evaluated_score is None
                        or evaluated_score >= self.best_evaluated_score
                    ):
                        self.best_evaluated_score = evaluated_score
                        self.save(new_high_score=True)

                training_log_dict.update(eval_log_dict)
            self.post_epoch_logging(training_log_dict)

            self.env.on_epoch_end(self.current_epoch)

            if self.should_stop:
                self.save()
                return

        self.time_report.report()
        self.save()
        self.fabric.call("on_fit_end", self)

    @torch.no_grad()
    def process_dataset(self, dataset):
        dataset = DictDataset(self.config.batch_size, dataset, shuffle=True)
        return dataset

    def optimize_model(self) -> Dict:
        dataset = self.process_dataset(self.experience_buffer.make_dict())

        self.train()
        training_log_dict = {}
        for batch_idx in track(
            range(self.max_num_batches()),
            description=f"Epoch {self.current_epoch}, training...",
        ):
            iter_log_dict = {}

            dataset_idx = batch_idx % len(dataset)
            # Reshuffling the data at the beginning of each mini epoch.
            if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
                dataset.shuffle()
            batch_dict = dataset[dataset_idx]

            # Go over all keys in obs and check if any has nans
            for key in batch_dict.keys():
                has_nan = False
                if torch.isnan(batch_dict[key]).any():
                    has_nan = True
                    print(f"NaN in {key}: {batch_dict[key]}")
                if has_nan:
                    raise ValueError("NaN in training")

            # Update model
            loss, loss_dict = self.model_step(batch_dict)
            iter_log_dict.update(loss_dict)
            self.optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(loss)
            grad_clip_dict = self.handle_model_grad_clipping(
                self.model, self.optimizer, "model"
            )
            iter_log_dict.update(grad_clip_dict)
            self.optimizer.step()

            # Update extra optimization steps
            extra_opt_steps_dict = self.extra_optimization_steps(batch_dict, batch_idx)
            iter_log_dict.update(extra_opt_steps_dict)

            for k, v in iter_log_dict.items():
                if k in training_log_dict:
                    training_log_dict[k][0] += v
                    training_log_dict[k][1] += 1
                else:
                    training_log_dict[k] = [v, 1]

        for k, v in training_log_dict.items():
            training_log_dict[k] = v[0] / v[1]

        self.eval()

        return training_log_dict

    # -----------------------------
    # Model Forward Pass and Loss Computation
    # -----------------------------
    def model_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        actions, prior_outs, encoder_outs = self.model.get_action_and_vae_outputs(
            batch_dict
        )
        expert_actions = batch_dict["expert_actions"]
        bc_loss = torch.square(actions - expert_actions).mean()

        extra_loss, extra_log_dict = self.calculate_extra_loss(batch_dict, actions)

        vae_kld_schedule = self.config.vae.kld_schedule
        vae_kld_loss = self.model.kl_loss(prior_outs, encoder_outs)
        vae_kld_loss = torch.mean(torch.sum(vae_kld_loss, dim=-1))

        kld_coeff = vae_kld_schedule.init_kld_coeff + min(
            max(0, self.current_epoch - vae_kld_schedule.start_epoch)
            / (vae_kld_schedule.end_epoch - vae_kld_schedule.start_epoch),
            1,
        ) * (vae_kld_schedule.end_kld_coeff - vae_kld_schedule.init_kld_coeff)

        loss = bc_loss + extra_loss + vae_kld_loss * kld_coeff

        log_dict = {
            "model/bc_loss": bc_loss.detach(),
            "model/extra_loss": extra_loss.detach(),
            "model/kld_coeff": kld_coeff,
            "losses/model_loss": loss.detach(),
            "model/vae_kld_loss": vae_kld_loss.detach(),
        }
        log_dict.update(extra_log_dict)

        return loss, log_dict

    def calculate_extra_loss(self, batch_dict, actions) -> Tuple[Tensor, Dict]:
        return torch.tensor(0.0, device=self.device), {}

    # -----------------------------
    # State Saving and Restoration
    # -----------------------------
    def get_state_dict(self, state_dict):
        extra_state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "step_count": self.step_count,
            "run_start_time": self.fit_start_time,
            "episode_reward_meter": self.episode_reward_meter.state_dict(),
            "episode_length_meter": self.episode_length_meter.state_dict(),
            "best_evaluated_score": self.best_evaluated_score,
        }

        if self.config.normalize_values:
            extra_state_dict["running_val_norm"] = self.running_val_norm.state_dict()
        state_dict.update(extra_state_dict)
        return state_dict

    def map_motions_to_iterations(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Maps motion IDs to iterations for distributed processing.

        This method distributes motion IDs across available ranks and creates a mapping
        of motions to be processed in each iteration. It ensures equal distribution of
        motions across ranks and proper scene sampling.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: A list of tuples, each containing:
                - motion_ids: Tensor of motion IDs for an iteration.
                - requires_scene: Tensor of booleans indicating if the motion requires a scene.
        """
        world_size = self.fabric.world_size
        global_rank = self.fabric.global_rank
        num_motions = self.motion_lib.num_motions()

        # Handle fixed motion ID case
        if self.env.config.motion_manager.fixed_motion_id is not None:
            motion_ids = torch.tensor(
                [self.env.config.motion_manager.fixed_motion_id], device=self.device
            )
            requires_scene = self.env.get_motion_requires_scene(motion_ids)
            return [(motion_ids, requires_scene)], 1

        # Calculate motions per rank, ensuring even distribution
        base_motions_per_rank = num_motions // world_size
        extra_motions = num_motions % world_size

        # Ranks with index < extra_motions get one additional motion
        motions_per_rank = base_motions_per_rank + (
            1 if global_rank < extra_motions else 0
        )
        start_motion = base_motions_per_rank * global_rank + min(
            global_rank, extra_motions
        )
        end_motion = start_motion + motions_per_rank

        # Create tensor of motion IDs assigned to this rank
        motion_range = torch.arange(start_motion, end_motion, device=self.device)

        # Split motions into batches of size self.num_envs
        motion_map = []
        for i in range(0, len(motion_range), self.num_envs):
            batch_motion_ids = motion_range[i : i + self.num_envs]
            # Sample corresponding scene IDs
            requires_scene = self.env.get_motion_requires_scene(batch_motion_ids)
            motion_map.append((batch_motion_ids, requires_scene))

        return motion_map, motions_per_rank

    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        self.eval()
        if self.env.config.motion_manager.fixed_motion_id is not None:
            num_motions = 1
        else:
            num_motions = self.motion_lib.num_motions()

        metrics = {
            # Track which motions are evaluated (within time limit)
            "evaluated": torch.zeros(num_motions, device=self.device, dtype=torch.bool),
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
        vae_noise = torch.zeros(self.num_envs, self.config.model.config.vae_latent_dim, device=self.device, dtype=torch.float)

        for outer_iter in track(
            range(max_iters),
            description=f"Evaluating... {remaining_motions} motions remain...",
        ):
            motion_pointer = outer_iter % num_outer_iters
            motion_ids, requires_scene = motion_map[motion_pointer]
            num_motions_this_iter = len(motion_ids)
            metrics["evaluated"][motion_ids] = True

            # Define the task mapping for each agent.
            self.env.agent_in_scene[:] = False
            self.env.motion_manager.motion_ids[:num_motions_this_iter] = motion_ids
            self.env.agent_in_scene[:num_motions_this_iter] = requires_scene
            # We force the respawn to flat terrain to ensure the agent can properly reconstruct the motion.
            self.env.force_respawn_on_flat = True

            env_ids = torch.arange(
                0, num_motions_this_iter, dtype=torch.long, device=self.device
            )

            dt: float = self.env.dt
            motion_lengths = self.motion_lib.get_motion_length(motion_ids)
            motion_num_frames = (motion_lengths / dt).floor().long()

            max_len = (
                motion_num_frames.max().item()
                if self.config.eval_length is None
                else self.config.eval_length
            )

            for eval_episode in range(self.config.eval_num_episodes):
                # Sample random start time. Slight noise is added to the start time to ensure the agent does not start at the same time.
                elapsed_time = (
                    torch.rand_like(self.motion_lib.state.motion_lengths[motion_ids])
                    * dt
                )
                self.env.motion_manager.motion_times[:num_motions_this_iter] = (
                    elapsed_time
                )
                self.env.motion_manager.reset_track_steps.reset_steps(env_ids)
                # This disables automatic reset of the environment and the tracking clip.
                self.env.disable_reset = True
                self.env.motion_manager.disable_reset_track = True

                obs = self.env.reset(
                    # Force reset all envs to ensure we don't have any residual effects from previous iterations
                    #   for example, this ensures all untracked envs do not spawn near any objects.
                    torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
                )

                for l in range(max_len):
                    obs["vae_noise"] = vae_noise
                    actions = self.model.act(obs)

                    obs, rewards, dones, terminated, extras = self.env_step(actions)

                    elapsed_time += dt
                    clip_done = (motion_lengths - dt) < elapsed_time
                    clip_not_done = torch.logical_not(clip_done)
                    for k in self.config.eval_metric_keys:
                        if k in self.env.mimic_info_dict:
                            value = self.env.mimic_info_dict[k].detach()
                        else:
                            raise ValueError(f"Key {k} not found in mimic_info_dict")

                        # Only collect metrics for the motions that are not done.
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

        print("Evaluation done, now aggregating data.")

        if self.env.config.motion_manager.fixed_motion_id is None:
            # This means we potentially ran multiple episodes, each time with a subset of motions.
            # We need to aggregate the data from all episodes. So now we reference all the motion_ids.
            motion_lengths = self.motion_lib.state.motion_lengths[:]
            motion_num_frames = (motion_lengths / dt).floor().long()

                # Save metrics per rank; distributed all_gather does not support dictionaries.
        with open(root_dir / f"{self.fabric.global_rank}_metrics.pt", "wb") as f:
            torch.save(metrics, f)
        self.fabric.barrier()
        # All ranks aggregrate data from all ranks.
        for rank in range(self.fabric.world_size):
            with open(root_dir / f"{rank}_metrics.pt", "rb") as f:
                other_metrics = torch.load(f, map_location=self.device)
            other_evaluated_indices = torch.nonzero(other_metrics["evaluated"]).flatten()
            for k in other_metrics.keys():
                metrics[k][other_evaluated_indices] = other_metrics[k][other_evaluated_indices]
            metrics["evaluated"][other_evaluated_indices] = True

        assert metrics["evaluated"].all(), "Not all motions were evaluated."
        self.fabric.barrier()
        (root_dir / f"{self.fabric.global_rank}_metrics.pt").unlink()

        to_log = {}
        for k in self.config.eval_metric_keys:
            mean_tracking_errors = metrics[k] / (motion_num_frames * self.config.eval_num_episodes)
            to_log[f"eval/{k}"] = mean_tracking_errors.detach().mean().item()
            to_log[f"eval/{k}_max"] = metrics[f"{k}_max"].detach().mean().item()
            to_log[f"eval/{k}_min"] = metrics[f"{k}_min"].detach().mean().item()

        if "gt_err" in self.config.eval_metric_keys:
            tracking_failures = (metrics["gt_err_max"] > 0.5).float()
            to_log["eval/tracking_success_rate"] = 1.0 - tracking_failures.detach().mean().item()

            failed_motions = torch.nonzero(tracking_failures).flatten().tolist()
            print(f"Saving to: {root_dir / f'failed_motions_{self.fabric.global_rank}.txt'}")
            with open(root_dir / f"failed_motions_{self.fabric.global_rank}.txt", "w") as f:
                for motion_id in failed_motions:
                    f.write(f"{motion_id}\n")
                    
            new_weights = torch.ones(self.motion_lib.num_motions(), device=self.device) * 1e-4
            new_weights[failed_motions] = 1.0
            self.env.motion_manager.update_sampling_weights(new_weights)

        stop_early = (
            self.config.training_early_termination.early_terminate_cart_err is not None
            or self.config.training_early_termination.early_terminate_success_rate
            is not None
            and self.fabric.global_rank
            == 0  # Only rank 0 should determine early termination
        )
        if self.config.training_early_termination.early_terminate_cart_err is not None:
            cart_err = to_log["eval/cartesian_err"]
            stop_early = stop_early and (
                cart_err
                <= self.config.training_early_termination.early_terminate_cart_err
            )
        if (
            self.config.training_early_termination.early_terminate_success_rate
            is not None
        ):
            tracking_success_rate = to_log["eval/tracking_success_rate"]

            stop_early = stop_early and (
                tracking_success_rate
                >= self.config.training_early_termination.early_terminate_success_rate
            )

        if stop_early:
            print("Stopping early! Target error reached")
            if "tracking_success_rate" in self.config.eval_metric_keys:
                print(
                    f"tracking_success_rate: {to_log['eval/tracking_success_rate'].item()}"
                )
            if "cartesian_err" in self.config.eval_metric_keys:
                print(f"cartesian_err: {to_log['eval/cartesian_err'].item()}")
            # Rank 0 will broadcast the best score to all ranks. This ensures all ranks are synchronized before saving.
            evaluated_score = self.fabric.broadcast(
                to_log["eval/tracking_success_rate"], src=0
            )
            self.best_evaluated_score = evaluated_score

            self.save(new_high_score=True)
            self.terminate_early()

        # Reset the environment back to normal state
        self.env.disable_reset = False
        self.env.motion_manager.disable_reset_track = False
        self.env.force_respawn_on_flat = False

        all_ids = torch.arange(0, self.num_envs, dtype=torch.long, device=self.device)
        self.env.motion_manager.reset_envs(all_ids)
        self.force_full_restart = True

        return to_log, to_log.get(
            "eval/tracking_success_rate", to_log.get("eval/cartesian_err", None)
        )
