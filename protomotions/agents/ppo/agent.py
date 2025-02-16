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
import os
import logging

from torch import Tensor

import time
import math
from pathlib import Path
from typing import Optional, Tuple, Dict

from lightning.fabric import Fabric

from hydra.utils import instantiate
from isaac_utils import torch_utils

from protomotions.utils.time_report import TimeReport
from protomotions.utils.average_meter import AverageMeter, TensorAverageMeterDict
from protomotions.agents.utils.data_utils import DictDataset, ExperienceBuffer
from protomotions.agents.ppo.model import PPOModel
from protomotions.agents.common.common import weight_init, get_params
from protomotions.envs.base_env.env import BaseEnv
from protomotions.utils.running_mean_std import RunningMeanStd
from rich.progress import track
from protomotions.agents.ppo.utils import discount_values, bounds_loss

log = logging.getLogger(__name__)


class PPO:
    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def __init__(self, fabric: Fabric, env: BaseEnv, config):
        self.fabric = fabric
        self.device: torch.device = fabric.device
        self.env = env
        self.motion_lib = self.env.motion_lib
        self.config = config

        self.num_envs: int = self.env.config.num_envs
        self.num_steps: int = config.num_steps
        self.gamma: float = config.gamma
        self.tau: float = config.tau
        self.e_clip: float = config.e_clip
        self.num_mini_epochs: int = config.num_mini_epochs
        self.task_reward_w: float = config.task_reward_w
        self._should_stop: bool = False

        if self.config.normalize_values:
            self.running_val_norm = RunningMeanStd(
                shape=(1,),
                device=self.device,
                clamp_value=self.config.normalized_val_clamp_value,
            )
        else:
            self.running_val_norm = None

        # timer
        self.time_report = TimeReport()

        self.current_lengths = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.current_rewards = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

        self.episode_reward_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.episode_env_tensors = TensorAverageMeterDict()
        self.step_count = 0
        self.current_epoch = 0
        self.fit_start_time = None
        self.best_evaluated_score = None

        self.force_full_restart = False

    @property
    def should_stop(self):
        return self.fabric.broadcast(self._should_stop)

    def setup(self):
        model: PPOModel = instantiate(self.config.model)
        model.apply(weight_init)
        actor_optimizer = instantiate(
            self.config.model.config.actor_optimizer,
            params=list(model._actor.parameters()),
        )
        critic_optimizer = instantiate(
            self.config.model.config.critic_optimizer,
            params=list(model._critic.parameters()),
        )

        self.model, self.actor_optimizer, self.critic_optimizer = self.fabric.setup(
            model, actor_optimizer, critic_optimizer
        )
        self.model.mark_forward_method("act")
        self.model.mark_forward_method("get_action_and_value")

    def load(self, checkpoint: Path):
        if checkpoint is not None:
            checkpoint = Path(checkpoint).resolve()
            print(f"Loading model from checkpoint: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.load_parameters(state_dict)
            
            env_checkpoint = checkpoint.resolve().parent / f"env_{self.fabric.global_rank}.ckpt"
            if env_checkpoint.exists():
                print(f"Loading env checkpoint: {env_checkpoint}")
                env_state_dict = torch.load(env_checkpoint, map_location=self.device)
                self.env.load_state_dict(env_state_dict)

    def load_parameters(self, state_dict):
        self.current_epoch = state_dict["epoch"]

        if "step_count" in state_dict:
            self.step_count = state_dict["step_count"]
        if "run_start_time" in state_dict:
            self.fit_start_time = state_dict["run_start_time"]

        self.best_evaluated_score = state_dict.get("best_evaluated_score", None)

        self.model.load_state_dict(state_dict["model"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

        if self.config.normalize_values:
            self.running_val_norm.load_state_dict(state_dict["running_val_norm"])

        self.episode_reward_meter.load_state_dict(state_dict["episode_reward_meter"])
        self.episode_length_meter.load_state_dict(state_dict["episode_length_meter"])

    # -----------------------------
    # Model Saving and State Dict
    # -----------------------------
    def get_state_dict(self, state_dict):
        extra_state_dict = {
            "model": self.model.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
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

    def save(self, path=None, name="last.ckpt", new_high_score=False):
        if path is None:
            path = self.fabric.loggers[0].log_dir
        root_dir = Path.cwd() / Path(self.fabric.loggers[0].root_dir)
        save_dir = Path.cwd() / Path(path)
        state_dict = self.get_state_dict({})
        self.fabric.save(save_dir / name, state_dict)

        if self.fabric.global_rank == 0:
            if root_dir != save_dir:
                if (root_dir / "last.ckpt").is_symlink():
                    (root_dir / "last.ckpt").unlink()
                # Make root_dir / "last.ckpt" point to the new checkpoint.
                # Calculate the relative path and create a symbolic link.
                relative_path = Path(os.path.relpath(save_dir / name, root_dir))
                (root_dir / "last.ckpt").symlink_to(relative_path)
                log.info(f"saved checkpoint, {root_dir / 'last.ckpt'}")
        self.fabric.barrier()
        
        # Save env state for all ranks to the same directory.
        rank_0_path = (root_dir / "last.ckpt").resolve().parent
        env_checkpoint = rank_0_path / f"env_{self.fabric.global_rank}.ckpt"
        env_state_dict = self.env.get_state_dict()
        torch.save(env_state_dict, env_checkpoint)

        # Check if new high score flag is consistent across devices.
        gathered_high_score = self.fabric.all_gather(new_high_score)
        assert all(
            [x == gathered_high_score[0] for x in gathered_high_score]
        ), "New high score flag should be the same across all ranks."

        if new_high_score:
            score_based_name = "score_based.ckpt"
            self.fabric.save(save_dir / score_based_name, state_dict)
            print(
                f"New best performing controller found with score {self.best_evaluated_score}. Model saved to {save_dir / score_based_name}."
            )
            if self.fabric.global_rank == 0:
                if root_dir != save_dir:
                    if (root_dir / "score_based.ckpt").is_symlink():
                        (root_dir / "score_based.ckpt").unlink()
                    # Create symlink for the best score checkpoint.
                    relative_path = Path(os.path.relpath(save_dir / name, root_dir))
                    (root_dir / "score_based.ckpt").symlink_to(relative_path)

    # -----------------------------
    # Experience Buffer and Training Loop
    # -----------------------------
    def register_extra_experience_buffer_keys(self):
        pass

    def fit(self):
        # Setup experience buffer
        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(
            self.device
        )
        self.experience_buffer.register_key(
            "self_obs", shape=(self.env.config.robot.self_obs_size,)
        )
        self.experience_buffer.register_key(
            "actions", shape=(self.env.config.robot.number_of_actions,)
        )
        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("extra_rewards")
        self.experience_buffer.register_key("total_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key("values")
        self.experience_buffer.register_key("next_values")
        self.experience_buffer.register_key("returns")
        self.experience_buffer.register_key("advantages")
        self.experience_buffer.register_key("neglogp")
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

            # Set networks in eval mode so that normalizers are not updated
            self.eval()
            with torch.no_grad():
                self.fabric.call("before_play_steps", self)

                for step in track(
                    range(self.num_steps),
                    description=f"Epoch {self.current_epoch}, collecting data...",
                ):
                    obs = self.handle_reset(done_indices)
                    self.experience_buffer.update_data("self_obs", step, obs["self_obs"])
                    if self.config.get("extra_inputs", None) is not None:
                        for key in self.config.extra_inputs:
                            self.experience_buffer.update_data(key, step, obs[key])

                    action, neglogp, value = self.model.get_action_and_value(obs)
                    self.experience_buffer.update_data("actions", step, action)
                    self.experience_buffer.update_data("neglogp", step, neglogp)
                    if self.config.normalize_values:
                        value = self.running_val_norm.normalize(value, un_norm=True)
                    self.experience_buffer.update_data("values", step, value)

                    # Check for NaNs in observations and actions
                    for key in obs.keys():
                        if torch.isnan(obs[key]).any():
                            print(f"NaN in {key}: {obs[key]}")
                            raise ValueError("NaN in obs")
                    if torch.isnan(action).any():
                        raise ValueError(f"NaN in action: {action}")

                    # Step the environment
                    next_obs, rewards, dones, terminated, extras = self.env_step(action)

                    all_done_indices = dones.nonzero(as_tuple=False)
                    done_indices = all_done_indices.squeeze(-1)

                    # Update logging metrics with the environment feedback
                    self.post_train_env_step(rewards, dones, done_indices, extras, step)

                    self.experience_buffer.update_data("rewards", step, rewards)
                    self.experience_buffer.update_data("dones", step, dones)

                    next_value = self.model._critic(next_obs).flatten()
                    if self.config.normalize_values:
                        next_value = self.running_val_norm.normalize(
                            next_value, un_norm=True
                        )
                    next_value = next_value * (1 - terminated.float())
                    self.experience_buffer.update_data("next_values", step, next_value)

                    self.step_count += self.get_step_count_increment()

                # After data collection, compute rewards, advantages, and returns.
                rewards = self.experience_buffer.rewards
                extra_rewards = self.calculate_extra_reward()
                self.experience_buffer.batch_update_data("extra_rewards", extra_rewards)
                total_rewards = rewards + extra_rewards
                self.experience_buffer.batch_update_data("total_rewards", total_rewards)

                advantages = discount_values(
                    self.experience_buffer.dones,
                    self.experience_buffer.values,
                    total_rewards,
                    self.experience_buffer.next_values,
                    self.gamma,
                    self.tau,
                )
                returns = advantages + self.experience_buffer.values
                self.experience_buffer.batch_update_data("returns", returns)

                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                self.experience_buffer.batch_update_data("advantages", advantages)

            training_log_dict = self.optimize_model()
            training_log_dict["epoch"] = self.current_epoch
            self.current_epoch += 1
            self.fabric.call("after_train", self)

            # Save model checkpoint at specified intervals before evaluation.
            if self.current_epoch % self.config.manual_save_every == 0:
                self.save()

            if (
                self.config.eval_metrics_every is not None
                and self.current_epoch > 0
                and self.current_epoch % self.config.eval_metrics_every == 0
            ):
                eval_log_dict, evaluated_score = self.calc_eval_metrics()
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

    # -----------------------------
    # Environment Interaction Helpers
    # -----------------------------
    def handle_reset(self, done_indices=None):
        if self.force_full_restart:
            done_indices = None
            self.force_full_restart = False
        obs = self.env.reset(done_indices)
        return obs

    def env_step(self, actions):
        obs, rewards, dones, extras = self.env.step(actions)
        rewards = rewards * self.task_reward_w
        terminated = extras["terminate"]
        return obs, rewards, dones, terminated, extras

    def post_train_env_step(self, rewards, dones, done_indices, extras, step):
        self.current_rewards += rewards
        self.current_lengths += 1

        self.episode_reward_meter.update(self.current_rewards[done_indices])
        self.episode_length_meter.update(self.current_lengths[done_indices])

        not_dones = 1.0 - dones.float()
        self.current_rewards = self.current_rewards * not_dones
        self.current_lengths = self.current_lengths * not_dones

        self.episode_env_tensors.add(extras["to_log"])

    # -----------------------------
    # Optimization
    # -----------------------------
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

            # Reshuffle dataset at the beginning of each mini epoch if configured.
            if dataset_idx == 0 and batch_idx != 0 and dataset.do_shuffle:
                dataset.shuffle()
            batch_dict = dataset[dataset_idx]

            # Check for NaNs in the batch.
            for key in batch_dict.keys():
                if torch.isnan(batch_dict[key]).any():
                    print(f"NaN in {key}: {batch_dict[key]}")
                    raise ValueError("NaN in training")

            # Update actor
            actor_loss, actor_loss_dict = self.actor_step(batch_dict)
            iter_log_dict.update(actor_loss_dict)
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(actor_loss)
            actor_grad_clip_dict = self.handle_model_grad_clipping(
                self.model._actor, self.actor_optimizer, "actor"
            )
            iter_log_dict.update(actor_grad_clip_dict)
            self.actor_optimizer.step()

            # Update critic
            critic_loss, critic_loss_dict = self.critic_step(batch_dict)
            iter_log_dict.update(critic_loss_dict)
            self.critic_optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(critic_loss)
            critic_grad_clip_dict = self.handle_model_grad_clipping(
                self.model._critic, self.critic_optimizer, "critic"
            )
            iter_log_dict.update(critic_grad_clip_dict)
            self.critic_optimizer.step()

            # Extra optimization steps if needed.
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

    def actor_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        dist = self.model._actor(batch_dict)
        logstd = self.model._actor.logstd
        std = torch.exp(logstd)
        neglogp = self.model.neglogp(batch_dict["actions"], dist.mean, std, logstd)

        # Compute probability ratio between new and old policy.
        ratio = torch.exp(batch_dict["neglogp"] - neglogp)
        surr1 = batch_dict["advantages"] * ratio
        surr2 = batch_dict["advantages"] * torch.clamp(
            ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
        )
        ppo_loss = torch.max(-surr1, -surr2)
        clipped = torch.abs(ratio - 1.0) > self.e_clip
        clipped = clipped.detach().float().mean()

        if self.config.bounds_loss_coef > 0:
            b_loss: Tensor = bounds_loss(dist.mean) * self.config.bounds_loss_coef
        else:
            b_loss = torch.zeros(self.num_envs, device=self.device)

        actor_ppo_loss = ppo_loss.mean()
        b_loss = b_loss.mean()
        extra_loss, extra_actor_log_dict = self.calculate_extra_actor_loss(batch_dict, dist)
        actor_loss = actor_ppo_loss + b_loss + extra_loss

        log_dict = {
            "actor/ppo_loss": actor_ppo_loss.detach(),
            "actor/bounds_loss": b_loss.detach(),
            "actor/extra_loss": extra_loss.detach(),
            "actor/clip_frac": clipped.detach(),
            "losses/actor_loss": actor_loss.detach(),
        }
        log_dict.update(extra_actor_log_dict)
        return actor_loss, log_dict

    def calculate_extra_actor_loss(self, batch_dict, dist) -> Tuple[Tensor, Dict]:
        return torch.tensor(0.0, device=self.device), {}

    def critic_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        values = self.model._critic(batch_dict).flatten()
        if self.config.clip_critic_loss:
            critic_loss_unclipped = (values - batch_dict["returns"]).pow(2)
            v_clipped = batch_dict["values"] + torch.clamp(
                values - batch_dict["values"],
                -self.config.e_clip,
                self.config.e_clip,
            )
            critic_loss_clipped = (v_clipped - batch_dict["returns"]).pow(2)
            critic_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
            critic_loss = 0.5 * critic_loss_max.mean()
        else:
            critic_loss = 0.5 * (batch_dict["returns"] - values).pow(2).mean()
        log_dict = {"losses/critic_loss": critic_loss.detach()}
        return critic_loss, log_dict

    def handle_model_grad_clipping(self, model, optimizer, model_name):
        params = get_params(list(model.parameters()))
        grad_norm_before_clip = torch_utils.grad_norm(params)
        if self.config.check_grad_mag:
            bad_grads = (
                torch.isnan(grad_norm_before_clip) or grad_norm_before_clip > 1000000.0
            )
        else:
            bad_grads = torch.isnan(grad_norm_before_clip)

        bad_grads_count = 0
        if bad_grads:
            if self.config.fail_on_bad_grads:
                all_params = torch.cat(
                    [p.grad.view(-1) for p in params if p.grad is not None],
                    dim=0,
                )
                raise ValueError(
                    f"NaN gradient in {model_name}"
                    + f" {all_params.isfinite().logical_not().float().mean().item()}"
                    + f" {all_params.abs().min().item()}"
                    + f" {all_params.abs().max().item()}"
                    + f" {grad_norm_before_clip.item()}"
                )
            else:
                bad_grads_count = 1
                for p in params:
                    if p.grad is not None:
                        p.grad.zero_()

        if self.config.gradient_clip_val > 0:
            self.fabric.clip_gradients(
                model,
                optimizer,
                max_norm=self.config.gradient_clip_val,
                error_if_nonfinite=True,
            )
        grad_norm_after_clip = torch_utils.grad_norm(params)
        clip_dict = {
            f"{model_name}/grad_norm_before_clip": grad_norm_before_clip.detach(),
            f"{model_name}/grad_norm_after_clip": grad_norm_after_clip.detach(),
            f"{model_name}/bad_grads_count": bad_grads_count,
        }
        return clip_dict

    # -----------------------------
    # Evaluation and Logging
    # -----------------------------
    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        return {}, None

    @torch.no_grad()
    def evaluate_policy(self):
        self.eval()
        done_indices = None  # Force reset on first entry
        step = 0
        while self.config.max_eval_steps is None or step < self.config.max_eval_steps:
            obs = self.handle_reset(done_indices)
            # Obtain actor predictions
            actions = self.model.act(obs)
            # Step the environment
            obs, rewards, dones, terminated, extras = self.env_step(actions)
            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            step += 1

    def post_epoch_logging(self, training_log_dict: Dict):
        end_time = time.time()
        log_dict = {
            "info/episode_length": self.episode_length_meter.get_mean().item(),
            "info/episode_reward": self.episode_reward_meter.get_mean().item(),
            "info/frames": torch.tensor(self.step_count),
            "info/gframes": torch.tensor(self.step_count / (10**9)),
            "times/fps_last_epoch": (self.num_steps * self.get_step_count_increment())
            / (end_time - self.epoch_start_time),
            "times/fps_total": self.step_count / (end_time - self.fit_start_time),
            "times/training_hours": (end_time - self.fit_start_time) / 3600,
            "times/training_minutes": (end_time - self.fit_start_time) / 60,
            "times/last_epoch_seconds": (end_time - self.epoch_start_time),
            "rewards/task_rewards": self.experience_buffer.rewards.mean().item(),
            "rewards/extra_rewards": self.experience_buffer.extra_rewards.mean().item(),
            "rewards/total_rewards": self.experience_buffer.total_rewards.mean().item(),
        }
        env_log_dict = self.episode_env_tensors.mean_and_clear()
        env_log_dict = {f"env/{k}": v for k, v in env_log_dict.items()}
        if len(env_log_dict) > 0:
            log_dict.update(env_log_dict)
        log_dict.update(training_log_dict)
        self.fabric.log_dict(log_dict)

    # -----------------------------
    # Helper Functions
    # -----------------------------
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    @torch.no_grad()
    def calculate_extra_reward(self):
        return torch.zeros(self.num_steps, self.num_envs, device=self.device)

    def max_num_batches(self):
        return math.ceil(
            self.num_envs * self.num_steps * self.num_mini_epochs / self.config.batch_size
        )

    def get_step_count_increment(self):
        return self.num_envs * self.fabric.world_size  # fabric.world_size = num gpu * num nodes

    def extra_optimization_steps(self, batch_dict, batch_idx: int):
        return {}

    def terminate_early(self):
        self._should_stop = True

    @torch.no_grad()
    def process_dataset(self, dataset):
        if self.config.normalize_values:
            self.running_val_norm.update(dataset["values"])
            self.running_val_norm.update(dataset["returns"])

            dataset["values"] = self.running_val_norm.normalize(dataset["values"])
            dataset["returns"] = self.running_val_norm.normalize(dataset["returns"])

        dataset = DictDataset(self.config.batch_size, dataset, shuffle=True)
        return dataset
