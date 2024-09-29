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

from torch import nn, Tensor

import time
import math
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from lightning.fabric import Fabric

from hydra.utils import instantiate
from isaac_utils import torch_utils

from phys_anim.utils.time_report import TimeReport
from phys_anim.utils.average_meter import AverageMeter, TensorAverageMeterDict
from phys_anim.agents.utils.data_utils import DictDataset, ExperienceBuffer
from phys_anim.agents.models.actor import PPO_Actor
from phys_anim.agents.models.common import NormObsBase
from phys_anim.envs.humanoid.common import Humanoid
from phys_anim.utils.running_mean_std import RunningMeanStd
from phys_anim.agents.callbacks.base_callback import RL_EvalCallback
from rich.progress import track


def get_params(obj) -> List[nn.Parameter]:
    """
    Gets list of params from either a list of params
    (where nothing happens) or a list of param groups
    """
    as_list = list(obj)
    if isinstance(as_list[0], Tensor):
        return as_list
    else:
        params = []
        for group in as_list:
            params = params + list(group["params"])
        return params


class PPO:
    def __init__(self, fabric: Fabric, env: Humanoid, config):
        self.fabric = fabric
        self.device: torch.device = fabric.device
        self.env = env
        self.motion_lib = self.env.motion_lib
        self.config = config

        self.lr_schedulers = []

        self.num_envs: int = self.env.config.num_envs
        self.num_obs = self.env.config.robot.self_obs_max_coords
        self.num_act = self.env.config.robot.number_of_actions
        self.num_steps: int = config.num_steps
        self.gamma: float = config.gamma
        self.tau: float = config.tau
        self.e_clip: float = config.e_clip
        self.task_reward_w: float = config.task_reward_w
        self.num_mini_epochs: int = config.num_mini_epochs
        self._should_stop: bool = False

        self.experience_buffer = ExperienceBuffer(self.num_envs, self.num_steps).to(
            self.device
        )

        if self.config.normalize_values:
            self.running_val_norm = RunningMeanStd(
                shape=(1,), device=self.device, clamp_value=self.config.val_clamp_value
            )
        else:
            self.running_val_norm = None

        # timer
        self.time_report = TimeReport()

        if config.schedules is None:
            self.schedules = None
        else:
            self.schedules = [instantiate(s, obj=self) for s in config.schedules]

        self.experience_buffer.register_key("obs", shape=(self.num_obs,))
        self.experience_buffer.register_key("mus", shape=(self.num_act,))
        self.experience_buffer.register_key("sigmas", shape=(self.num_act,))
        self.experience_buffer.register_key("actions", shape=(self.num_act,))
        self.experience_buffer.register_key("rewards")
        self.experience_buffer.register_key("extra_rewards")
        self.experience_buffer.register_key("total_rewards")
        self.experience_buffer.register_key("dones", dtype=torch.long)
        self.experience_buffer.register_key("values")
        self.experience_buffer.register_key("next_values")
        self.experience_buffer.register_key("returns")
        self.experience_buffer.register_key("advantages")
        self.experience_buffer.register_key("neglogp")

        self.extra_obs_inputs = self.config.extra_inputs
        if self.extra_obs_inputs is not None:
            keys = list(self.extra_obs_inputs.keys())
            for key in keys:
                val = self.extra_obs_inputs[key]
                if not val.get("retrieve_from_env", True):
                    del self.extra_obs_inputs[key]
                    continue
                dtype = getattr(torch, val.get("dtype", "float"))
                self.experience_buffer.register_key(key, shape=(val.size,), dtype=dtype)

        self.use_rand_action_masks = self.config.use_rand_action_masks
        if self.use_rand_action_masks:
            self.experience_buffer.register_key("rand_action_mask", dtype=torch.long)
            all_env_ids = torch.arange(
                self.num_envs, dtype=torch.long, device=self.device
            )
            # self._rand_action_probs = 1.0 - env_ids / (num_envs - 1.0)
            self.rand_action_probs = 1.0 - torch.exp(
                10 * (all_env_ids / (self.num_envs - 1.0) - 1.0)
            )
            self.rand_action_probs[0] = 1.0
            self.rand_action_probs[-1] = 0.0

            self.rand_action_mask = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

        # Obs deliberately not on here, since its updated before env step
        self.actor_state_to_experience_buffer_list = [
            "mus",
            "sigmas",
            "actions",
            "neglogp",
            "rewards",
            "dones",
        ]

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
        self.best_evaluated_score = None

        self.force_full_restart = False

        self.eval_callbacks: list[RL_EvalCallback] = []

    @property
    def should_stop(self):
        return self.fabric.broadcast(self._should_stop)

    def setup(self):
        actor: PPO_Actor = instantiate(
            self.config.actor, num_in=self.num_obs, num_act=self.num_act
        )
        actor_optimizer = instantiate(
            self.config.actor_optimizer,
            params=list(actor.parameters()),
            _convert_="all",
        )

        self.actor, self.actor_optimizer = self.fabric.setup(actor, actor_optimizer)
        self.actor.mark_forward_method("eval_forward")
        self.actor.mark_forward_method("training_forward")

        critic: NormObsBase = instantiate(
            self.config.critic, num_in=self.num_obs, num_out=1
        )
        critic_optimizer = instantiate(
            self.config.critic_optimizer,
            params=list(critic.parameters()),
        )
        self.critic, self.critic_optimizer = self.fabric.setup(critic, critic_optimizer)

        if self.config.actor_lr_scheduler is not None:
            self.actor_lr_scheduler = instantiate(
                self.config.actor_lr_scheduler, optimizer=self.actor_optimizer
            )
            self.lr_schedulers.append(self.actor_lr_scheduler)

        if self.config.critic_lr_scheduler is not None:
            self.critic_lr_scheduler = instantiate(
                self.config.critic_lr_scheduler, optimizer=self.critic_optimizer
            )
            self.lr_schedulers.append(self.critic_lr_scheduler)

    def load(self, checkpoint: Path):
        if checkpoint is not None:
            print(f"Loading model from checkpoint: {checkpoint}")
            state_dict = torch.load(checkpoint, map_location=self.device)
            self.load_parameters(state_dict)

    def load_parameters(self, state_dict):
        self.current_epoch = state_dict["epoch"]
        self.best_evaluated_score = state_dict.get("best_evaluated_score", None)

        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        if self.config.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.load_state_dict(state_dict["actor_lr_scheduler"])
        if self.config.critic_lr_scheduler is not None:
            self.critic_lr_scheduler.load_state_dict(state_dict["critic_lr_scheduler"])

        if self.config.normalize_values:
            self.running_val_norm.load_state_dict(state_dict["running_val_norm"])

        self.episode_reward_meter.load_state_dict(state_dict["episode_reward_meter"])
        self.episode_length_meter.load_state_dict(state_dict["episode_length_meter"])

    def fit(self):
        self.env_reset()
        self.fit_start_time = time.time()
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("epoch")
        self.time_report.start_timer("algorithm")
        self.fabric.call("on_fit_start", self)

        while self.current_epoch < self.config.max_epochs:
            self.epoch_start_time = time.time()
            self.time_report.start_timer("epoch")

            self.fabric.call("before_play_steps", self)

            self.eval()
            self.play_steps()
            self.generate_datasets()
            self.train()

            training_log_dict = {}
            for batch_idx in track(
                range(self.max_num_batches()),
                description=f"Epoch {self.current_epoch}, training...",
            ):
                iter_log_dict = self.training_step(batch_idx)

                for k, v in iter_log_dict.items():
                    if k in training_log_dict:
                        training_log_dict[k][0] += v
                        training_log_dict[k][1] += 1
                    else:
                        training_log_dict[k] = [v, 1]

            for k, v in training_log_dict.items():
                training_log_dict[k] = v[0] / v[1]

            training_log_dict["epoch"] = self.current_epoch
            self.current_epoch += 1
            self.time_report.end_timer("epoch")
            self.fabric.call("after_train", self)

            # Saves memory
            if hasattr(self, "actor_critic_dataset"):
                del self.actor_critic_dataset

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
            if self.schedules is not None:
                for s in self.schedules:
                    s.step()

            self.env.on_epoch_end(self.current_epoch)

            if self.should_stop:
                self.save()
                exit(0)

        self.time_report.end_timer("algorithm")
        self.time_report.report()
        self.save()
        self.fabric.call("on_fit_end", self)

    @torch.no_grad()
    def play_steps(self):
        actor_state = self.create_actor_state()

        for i in track(
            range(self.num_steps),
            description=f"Epoch {self.current_epoch}, collecting data...",
        ):
            actor_state["step"] = i

            actor_state = self.handle_reset(actor_state)

            # Invoke actor and critic, generate actions/values
            actor_state = self.pre_env_step(actor_state)

            # Step env
            actor_state = self.env_step(actor_state)

            all_done_indices = actor_state["dones"].nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            actor_state["done_indices"] = done_indices

            # Store things in experience buffer
            actor_state = self.post_env_step(actor_state)
            actor_state = self.compute_next_values(actor_state)

        self.post_play_steps(actor_state)

    def training_step(self, batch_idx: int) -> Dict:
        iter_log_dict = {}

        if batch_idx == 0:
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

        is_accumulating = (
            ((batch_idx + 1) % self.config.gradient_accumulation_steps != 0)
            or self.config.gradient_accumulation_steps <= 0
        ) and ((batch_idx + 1) < self.ac_max_num_batches())
        num_accumulation_steps = min(
            self.config.gradient_accumulation_steps, self.ac_max_num_batches()
        )
        if num_accumulation_steps <= 0:
            num_accumulation_steps = self.ac_max_num_batches()

        if batch_idx < self.ac_max_num_batches():
            with self.fabric.no_backward_sync(self.actor, enabled=is_accumulating):
                actor_loss, actor_loss_dict = self.actor_step(batch_idx)
                scaled_actor_loss = actor_loss / num_accumulation_steps
                self.fabric.backward(scaled_actor_loss)

            if not is_accumulating:
                actor_grad_clip_dict = self.handle_actor_grad_clipping()
                iter_log_dict.update(actor_grad_clip_dict)
                self.actor_optimizer.step()
                self.actor_optimizer.zero_grad()
                self.actor.logstd_tick(self.current_epoch)

            iter_log_dict.update(actor_loss_dict)

            with self.fabric.no_backward_sync(self.critic, enabled=is_accumulating):
                critic_loss, critic_loss_dict = self.critic_step(batch_idx)
                scaled_critic_loss = critic_loss / num_accumulation_steps
                self.fabric.backward(scaled_critic_loss)

            if not is_accumulating:
                critic_grad_clip_dict = self.handle_critic_grad_clipping()
                iter_log_dict.update(critic_grad_clip_dict)
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()

            iter_log_dict.update(critic_loss_dict)

        extra_opt_steps_dict = self.extra_optimization_steps(batch_idx)

        iter_log_dict.update(extra_opt_steps_dict)

        if batch_idx == (self.max_num_batches() - 1):
            for lr in self.lr_schedulers:
                lr.step()

        return iter_log_dict

    def handle_reset(self, actor_state):
        done_indices = actor_state["done_indices"]
        if self.force_full_restart:
            done_indices = None
            self.force_full_restart = False

        obs = self.env_reset(done_indices)
        actor_state["obs"] = obs

        actor_state = self.get_extra_obs_from_env(actor_state)

        return actor_state

    def env_reset(self, env_ids=None):
        obs = self.env.reset(env_ids)
        return obs

    def env_step(self, actor_state):

        obs, rewards, dones, extras = self.env.step(actor_state["actions"])
        rewards = rewards * self.task_reward_w
        actor_state.update(
            {"obs": obs, "rewards": rewards, "dones": dones, "extras": extras}
        )

        actor_state = self.get_extra_obs_from_env(actor_state)

        return actor_state

    def pre_env_step(self, actor_state):
        self.experience_buffer.update_data(
            "obs", actor_state["step"], actor_state["obs"]
        )
        if self.extra_obs_inputs is not None:
            for key in self.extra_obs_inputs.keys():
                self.experience_buffer.update_data(
                    key, actor_state["step"], actor_state[key]
                )

        actor_inputs = self.create_actor_args(actor_state)
        actor_outs = self.actor.eval_forward(actor_inputs)

        if self.use_rand_action_masks:
            rand_action_mask = torch.bernoulli(self.rand_action_probs)
            deterministic_actions = rand_action_mask == 0
            actor_outs["actions"][deterministic_actions] = actor_outs["mus"][
                deterministic_actions
            ]
            self.experience_buffer.update_data(
                "rand_action_mask", actor_state["step"], rand_action_mask
            )

        critic_inputs = self.create_critic_args(actor_state)
        values = self.critic(critic_inputs)

        if self.config.normalize_values:
            values = self.running_val_norm.normalize(values, un_norm=True)

        actor_state.update(actor_outs)

        # We want unnormalized values here.
        self.experience_buffer.update_data(
            "values", actor_state["step"], values.view(-1)
        )

        return actor_state

    def pre_eval_env_step(self, actor_state: dict):
        actor_inputs = self.create_actor_args(actor_state)
        actor_outs = self.actor.eval_forward(actor_inputs)
        actor_state.update(actor_outs)
        actor_state["sampled_actions"] = actor_state["actions"]

        # By default, use deterministic policy in eval
        # (unless overriden in callbacks).
        actor_state["actions"] = actor_state["mus"]

        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)

        return actor_state

    def post_env_step(self, actor_state):
        self.current_rewards += actor_state["rewards"]
        self.current_lengths += 1

        done_indices = actor_state["done_indices"]

        self.episode_reward_meter.update(self.current_rewards[done_indices])
        self.episode_length_meter.update(self.current_lengths[done_indices])

        not_dones = 1.0 - actor_state["dones"].float()

        self.current_rewards = self.current_rewards * not_dones
        self.current_lengths = self.current_lengths * not_dones

        for k in self.actor_state_to_experience_buffer_list:
            self.experience_buffer.update_data(k, actor_state["step"], actor_state[k])

        self.episode_env_tensors.add(actor_state["extras"]["to_log"])

        return actor_state

    def post_eval_env_step(self, actor_state):
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state

    def create_actor_state(self):
        return {"done_indices": [], "stop": False}

    def discount_values(self, mb_fdones, mb_values, mb_rewards, mb_next_values):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)

        for t in reversed(range(self.num_steps)):
            not_done = 1.0 - mb_fdones[t]
            not_done = not_done

            delta = mb_rewards[t] + self.gamma * mb_next_values[t] - mb_values[t]
            lastgaelam = delta + self.gamma * self.tau * not_done * lastgaelam
            mb_advs[t] = lastgaelam

        return mb_advs

    def post_play_steps(self, actor_state):
        self.step_count += self.get_step_count_increment()

        rewards = self.experience_buffer.rewards
        self.last_scaled_task_rewards_mean = rewards.detach().mean()

        extra_rewards = self.calculate_extra_reward()

        self.experience_buffer.batch_update_data("extra_rewards", extra_rewards)
        total_rewards = rewards + extra_rewards

        self.experience_buffer.batch_update_data("total_rewards", total_rewards)

        advantages = self.discount_values(
            self.experience_buffer.dones,
            self.experience_buffer.values,
            total_rewards,
            self.experience_buffer.next_values,
        )
        returns = advantages + self.experience_buffer.values

        self.experience_buffer.batch_update_data("returns", returns)

        if self.config.normalize_advantage:
            if not self.use_rand_action_masks:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            else:
                adv_mask = (self.experience_buffer.rand_action_mask != 0).float()
                advantages = normalization_with_masks(advantages, adv_mask)

        self.experience_buffer.batch_update_data("advantages", advantages)

    @torch.no_grad()
    def generate_datasets(self):
        actor_critic_data_dict = self.experience_buffer.make_dict()

        if self.config.normalize_values:
            self.running_val_norm.update(actor_critic_data_dict["values"])
            self.running_val_norm.update(actor_critic_data_dict["returns"])

            actor_critic_data_dict["values"] = self.running_val_norm.normalize(
                actor_critic_data_dict["values"]
            )
            actor_critic_data_dict["returns"] = self.running_val_norm.normalize(
                actor_critic_data_dict["returns"]
            )

        self.actor_critic_dataset = DictDataset(
            self.config.batch_size, actor_critic_data_dict, shuffle=True
        )

    def actor_step(self, batch_idx) -> Tuple[Tensor, Dict]:
        dataset_idx = batch_idx % len(self.actor_critic_dataset)
        # Reshuffling the data at the beginning of each mini epoch.
        # Only doing this in the actor and not the critic to
        # avoid extra reshuffles.
        if dataset_idx == 0 and batch_idx != 0 and self.actor_critic_dataset.do_shuffle:
            self.actor_critic_dataset.shuffle()
        batch_dict = self.actor_critic_dataset[dataset_idx]

        actor_outs = self.actor.training_forward(batch_dict)
        actor_info = self.actor_loss(
            batch_dict["neglogp"],
            actor_outs["neglogp"],
            batch_dict["advantages"],
            self.e_clip,
        )
        actor_ppo_loss: Tensor = actor_info["actor_loss"]
        actor_clipped: Tensor = actor_info["actor_clipped"].float()

        if self.config.bounds_loss_coef > 0:
            bounds_loss: Tensor = (
                self.bounds_loss(actor_outs["mus"]) * self.config.bounds_loss_coef
            )
        else:
            bounds_loss = torch.zeros(self.num_envs, device=self.device)

        if self.use_rand_action_masks:
            rand_action_mask = batch_dict["rand_action_mask"]
            action_loss_mask = (rand_action_mask != 0).float()
            action_mask_sum = torch.sum(action_loss_mask)

            actor_ppo_loss = (actor_ppo_loss * action_loss_mask).sum() / action_mask_sum
            actor_clipped = (actor_clipped * action_loss_mask).sum() / action_mask_sum
            bounds_loss = (bounds_loss * action_loss_mask).sum() / action_mask_sum
        else:
            actor_ppo_loss = actor_ppo_loss.mean()
            actor_clipped = actor_clipped.mean()
            bounds_loss = bounds_loss.mean()

        extra_loss, extra_actor_log_dict = self.calculate_extra_actor_loss(
            batch_idx, batch_dict
        )
        actor_loss = actor_ppo_loss + bounds_loss + extra_loss

        log_dict = {
            "actor/ppo_loss": actor_ppo_loss.detach(),
            "actor/bounds_loss": bounds_loss.detach(),
            "actor/extra_loss": extra_loss.detach(),
            "actor/clip_frac": actor_clipped.detach(),
            "losses/actor_loss": actor_loss.detach(),
        }
        log_dict.update(extra_actor_log_dict)
        return actor_loss, log_dict

    def bounds_loss(self, mu: Tensor) -> Tensor:
        soft_bound = 1.0
        mu_loss_high = (
            torch.maximum(mu - soft_bound, torch.tensor(0, device=self.device)) ** 2
        )
        mu_loss_low = (
            torch.minimum(mu + soft_bound, torch.tensor(0, device=self.device)) ** 2
        )
        b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        return b_loss

    def calculate_extra_actor_loss(self, batch_idx, batch_dict) -> Tuple[Tensor, Dict]:
        return torch.tensor(0.0, device=self.device), {}

    def critic_train_forward(self, batch_dict):
        return self.critic(batch_dict)

    def critic_step(self, batch_idx) -> Tuple[Tensor, Dict]:
        batch_dict = self.actor_critic_dataset[
            batch_idx % len(self.actor_critic_dataset)
        ]
        values = self.critic_train_forward(batch_dict)

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

    def actor_loss(
        self, old_action_neglogprobs, action_neglogprobs, advantage, curr_e_clip
    ):
        # = p(actions) / p_old(actions)
        ratio = torch.exp(old_action_neglogprobs - action_neglogprobs)

        surr1 = advantage * ratio
        surr2 = advantage * torch.clamp(ratio, 1.0 - curr_e_clip, 1.0 + curr_e_clip)
        ppo_loss = torch.max(-surr1, -surr2)

        clipped = torch.abs(ratio - 1.0) > curr_e_clip
        clipped = clipped.detach()

        info = {"actor_loss": ppo_loss, "actor_clipped": clipped}
        return info

    def get_state_dict(self, state_dict):
        extra_state_dict = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "epoch": self.current_epoch,
            "episode_reward_meter": self.episode_reward_meter.state_dict(),
            "episode_length_meter": self.episode_length_meter.state_dict(),
            "best_evaluated_score": self.best_evaluated_score,
        }
        if self.config.actor_lr_scheduler is not None:
            extra_state_dict["actor_lr_scheduler"] = (
                self.actor_lr_scheduler.state_dict()
            )
        if self.config.critic_lr_scheduler is not None:
            extra_state_dict["critic_lr_scheduler"] = (
                self.critic_lr_scheduler.state_dict()
            )
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
                # Make root_dir / "last.ckpt" point to the new checkpoint
                (root_dir / "last.ckpt").symlink_to(save_dir / name)

        # The function fabric.save has to be called on ALL devices. We assert that the new_high_score flag has the same
        # value across all devices. If it is True, we save the model with the best score to the root directory.
        # Make sure to fun an all_gather/broadcast operation to ensure that the flag is the same across all devices.
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
                    # Make root_dir / "score_based.ckpt" point to the new checkpoint
                    (root_dir / "score_based.ckpt").symlink_to(
                        save_dir / score_based_name
                    )

    def handle_actor_grad_clipping(self):
        actor_params = get_params(list(self.actor.parameters()))
        actor_grad_norm_before_clip = torch_utils.grad_norm(actor_params)

        if self.config.check_grad_mag:
            bad_grads = (
                torch.isnan(actor_grad_norm_before_clip)
                or actor_grad_norm_before_clip > 1000000.0
            )
        else:
            bad_grads = torch.isnan(actor_grad_norm_before_clip)

        # sanity check
        actor_bad_grads_count = 0
        if bad_grads:

            if self.config.fail_on_bad_grads:
                all_params = torch.cat(
                    [p.grad.view(-1) for p in actor_params if p.grad is not None],
                    dim=0,
                )
                raise ValueError(
                    f"NaN gradient"
                    + f" {all_params.isfinite().logical_not().float().mean().item()}"
                    + f" {all_params.abs().min().item()}"
                    + f" {all_params.abs().max().item()}"
                    + f" {actor_grad_norm_before_clip.item()}"
                )
            else:
                actor_bad_grads_count = 1
                for p in actor_params:
                    if p.grad is not None:
                        p.grad.zero_()

        if self.config.gradient_clip_val > 0:
            self.fabric.clip_gradients(
                self.actor,
                self.actor_optimizer,
                max_norm=self.config.gradient_clip_val,
                error_if_nonfinite=False,
            )
        actor_grad_norm_after_clip = torch_utils.grad_norm(actor_params)

        clip_dict = {
            "actor/grad_norm_before_clip": actor_grad_norm_before_clip.detach(),
            "actor/grad_norm_after_clip": actor_grad_norm_after_clip.detach(),
            "actor/bad_grads_count": actor_bad_grads_count,
        }

        return clip_dict

    def handle_critic_grad_clipping(self):
        critic_params = get_params(list(self.critic.parameters()))
        critic_grad_norm_before_clip = torch_utils.grad_norm(critic_params)

        if self.config.check_grad_mag:
            bad_grads = (
                torch.isnan(critic_grad_norm_before_clip)
                or critic_grad_norm_before_clip > 1000000.0
            )
        else:
            bad_grads = torch.isnan(critic_grad_norm_before_clip)

        critic_bad_grads_count = 0
        # sanity check
        if bad_grads:
            if self.config.fail_on_bad_grads:
                all_params = torch.cat(
                    [p.grad.view(-1) for p in critic_params if p.grad is not None],
                    dim=0,
                )
                print(
                    "NaN gradient",
                    all_params.isfinite().logical_not().float().mean().item(),
                    all_params.abs().min().item(),
                    all_params.abs().max().item(),
                    critic_grad_norm_before_clip.item(),
                )
                raise ValueError
            else:
                critic_bad_grads_count = 1
                for p in critic_params:
                    if p.grad is not None:
                        p.grad.zero_()

        if self.config.gradient_clip_val > 0:
            self.fabric.clip_gradients(
                self.critic,
                self.critic_optimizer,
                max_norm=self.config.gradient_clip_val,
                error_if_nonfinite=False,
            )
        critic_grad_norm_after_clip = torch_utils.grad_norm(critic_params)

        clip_dict = {
            "critic/grad_norm_before_clip": critic_grad_norm_before_clip.detach(),
            "critic/grad_norm_after_clip": critic_grad_norm_after_clip.detach(),
            "critic/bad_grads_count": critic_bad_grads_count,
        }
        return clip_dict

    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        return {}, None

    def post_epoch_logging(self, training_log_dict: Dict):
        end_time = time.time()
        log_dict = {
            "info/episode_length": self.episode_length_meter.get_mean().item(),
            "info/episode_reward": self.episode_reward_meter.get_mean().item(),
            "info/frames": torch.tensor(self.step_count),
            "info/gframes": torch.tensor(self.step_count / (10**9)),
            "times/fps_last_epoch": self.get_step_count_increment()
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

    def create_actor_args(self, actor_state):
        actor_args = {"obs": actor_state["obs"]}

        if self.extra_obs_inputs is not None:
            for key in self.extra_obs_inputs.keys():
                if key in actor_state:
                    actor_args[key] = actor_state[key]

        return actor_args

    def create_critic_args(self, actor_state):
        critic_args = {"obs": actor_state["obs"]}

        if self.extra_obs_inputs is not None:
            for key in self.extra_obs_inputs.keys():
                if key in actor_state:
                    critic_args[key] = actor_state[key]

        return critic_args

    @torch.no_grad()
    def evaluate_policy(self):
        self.create_eval_callbacks()
        self.pre_evaluate_policy()

        actor_state = self.create_actor_state()
        step = 0
        games_count = 0
        while (
            not actor_state["stop"]
            and (self.config.num_games is None or games_count < self.config.num_games)
            and (
                self.config.max_eval_steps is None or step < self.config.max_eval_steps
            )
        ):
            actor_state["step"] = step
            actor_state["games_count"] = games_count

            actor_state = self.handle_reset(actor_state)

            # Invoke actor and critic, generate actions/values
            actor_state = self.pre_eval_env_step(actor_state)

            # Step env
            actor_state = self.env_step(actor_state)

            all_done_indices = actor_state["dones"].nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            actor_state["done_indices"] = done_indices

            actor_state = self.post_eval_env_step(actor_state)

            games_count += len(done_indices)
            step += 1

        self.post_evaluate_policy()

    def pre_evaluate_policy(self, reset_env=True):
        self.eval()
        if reset_env:
            self.env_reset()

        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    ### Helpers ###
    def get_extra_obs_from_env(self, actor_state):
        if self.extra_obs_inputs is not None:
            for key in self.extra_obs_inputs.keys():
                env_obs_name = self.extra_obs_inputs[key].get("env_obs_name", key)
                val = getattr(self.env, env_obs_name, None)
                assert val is not None, f"Env does not have attribute {env_obs_name}"
                actor_state[key] = val.view(-1, self.extra_obs_inputs[key].size)
        return actor_state

    def compute_next_values(self, actor_state):
        critic_inputs = self.create_critic_args(actor_state)
        values = self.critic(critic_inputs).view(-1)

        if self.config.normalize_values:
            values = self.running_val_norm.normalize(values, un_norm=True)

        next_values = values * (1 - actor_state["extras"]["terminate"].float())

        self.experience_buffer.update_data(
            "next_values", actor_state["step"], next_values
        )
        return actor_state

    def create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(cb, training_loop=self))

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    def calculate_extra_reward(self):
        return torch.zeros(self.num_steps, self.num_envs, device=self.device)

    def max_num_batches(self):
        return self.ac_max_num_batches()

    def get_step_count_increment(self):
        return (
            self.num_steps * self.num_envs * self.fabric.world_size
        )  # fabric.world_size = num gpu * num nodes

    def ac_max_num_batches(self):
        return math.ceil(
            self.num_envs
            * self.num_steps
            * self.num_mini_epochs
            / self.config.batch_size
        )

    def extra_optimization_steps(self, batch_idx: int):
        return {}

    def terminate_early(self):
        self._should_stop = True


def normalization_with_masks(values: Tensor, masks: Optional[Tensor]):
    if masks is None:
        return (values - values.mean()) / (values.std() + 1e-8)

    values_mean, values_var = get_mean_var_with_masks(values, masks)
    values_std = torch.sqrt(values_var)
    normalized_values = (values - values_mean) / (values_std + 1e-8)

    return normalized_values


def get_mean_var_with_masks(values: Tensor, masks: Tensor):
    sum_mask = masks.sum()
    values_mask = values * masks
    values_mean = values_mask.sum() / sum_mask
    min_sqr = (((values_mask) ** 2) / sum_mask).sum() - (
        (values_mask / sum_mask).sum()
    ) ** 2
    values_var = min_sqr * sum_mask / (sum_mask - 1)
    return values_mean, values_var
