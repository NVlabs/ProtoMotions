# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable agent-side AMP training component.

AMP is a learned training objective, not just an environment reward: it owns a
discriminator, optional discriminator critic, replay data, reward normalization,
optimizers, and checkpoint state.  This component keeps that behavior reusable
for PPO AMP, ASE, and fine-tuning/PEFT agents without forcing every agent to inherit the
full AMP agent class.
"""

from __future__ import annotations

import inspect
import logging
from typing import Callable, Dict, Optional, Tuple

import torch
from tensordict import TensorDict
from torch import Tensor

from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.optimizer.factory import instantiate_optimizer
from protomotions.agents.ppo.utils import discount_values
from protomotions.agents.utils.normalization import RewardRunningMeanStd
from protomotions.agents.utils.replay_buffer import ReplayBuffer

log = logging.getLogger(__name__)


class AMPTrainingComponent:
    """Owns AMP discriminator training state for a host agent."""

    def __init__(self, agent):
        self.agent = agent

        self.use_disc_critic = agent.config.amp_parameters.use_disc_critic
        self.replay_buffer = ReplayBuffer(
            agent.config.amp_parameters.discriminator_replay_size,
            device=agent.device,
        )
        rollout_size = agent.num_envs * agent.num_steps
        disc_bs = agent.config.amp_parameters.discriminator_batch_size
        if disc_bs <= rollout_size:
            assert disc_bs * agent.max_num_batches() <= rollout_size, (
                f"Discriminator needs {disc_bs} × {agent.max_num_batches()} = "
                f"{disc_bs * agent.max_num_batches()} expert samples per epoch, "
                f"but rollout only has {rollout_size} entries. "
                f"Reduce discriminator_batch_size or increase num_envs/num_steps."
            )

        self.num_cumulative_bad_transitions = torch.zeros(
            agent.num_envs,
            device=agent.device,
            dtype=torch.int32,
        )
        if agent.config.normalize_rewards:
            self.running_reward_norm = RewardRunningMeanStd(
                shape=(1,),
                fabric=agent.fabric,
                gamma=agent.gamma,
                device=agent.device,
                clamp_value=agent.config.normalized_reward_clamp_value,
            )
        else:
            self.running_reward_norm = None

    @classmethod
    def for_agent(cls, agent) -> "AMPTrainingComponent":
        component = getattr(agent, "amp_component", None)
        if component is None:
            raise AttributeError(
                f"{type(agent).__name__} has no amp_component. "
                "AMPAgentMixin.__init__ must run before AMP helpers are used."
            )
        return component

    @property
    def config(self):
        return self.agent.config

    @property
    def device(self):
        return self.agent.device

    def create_optimizers(self, model) -> None:
        discriminator_optimizer = instantiate_optimizer(
            self.config.model.discriminator_optimizer,
            model._discriminator,
            params=list(model._discriminator.parameters()),
        )
        self.discriminator, self.discriminator_optimizer = (
            self.agent._setup_model_optimizer(
                model._discriminator,
                discriminator_optimizer,
            )
        )
        if self.use_disc_critic:
            disc_critic_optimizer = instantiate_optimizer(
                self.config.model.disc_critic_optimizer,
                model._disc_critic,
                params=list(model._disc_critic.parameters()),
            )
            self.disc_critic, self.disc_critic_optimizer = (
                self.agent._setup_model_optimizer(
                    model._disc_critic,
                    disc_critic_optimizer,
                )
            )

    def load_training_state(self, state_dict) -> None:
        self.discriminator_optimizer.load_state_dict(
            state_dict["discriminator_optimizer"]
        )
        if self.use_disc_critic:
            self.disc_critic_optimizer.load_state_dict(
                state_dict["disc_critic_optimizer"]
            )
        if self.config.normalize_rewards:
            self.running_reward_norm.load_state_dict(
                state_dict["running_amp_reward_norm"]
            )

    def add_state_dict(self, state_dict) -> dict:
        state_dict["discriminator_optimizer"] = (
            self.discriminator_optimizer.state_dict()
        )
        if self.use_disc_critic:
            state_dict["disc_critic_optimizer"] = (
                self.disc_critic_optimizer.state_dict()
            )
        if self.config.normalize_rewards:
            state_dict["running_amp_reward_norm"] = (
                self.running_reward_norm.state_dict()
            )
        return state_dict

    def register_experience_buffer_keys(self) -> None:
        buffer = self.agent.experience_buffer
        buffer.register_key("amp_rewards")
        if self.config.normalize_rewards:
            buffer.register_key("unnormalized_amp_rewards")

        if self.use_disc_critic:
            if not hasattr(buffer, "value"):
                raise RuntimeError(
                    "AMP disc critic requires experience_buffer.value to be "
                    "registered before AMP registers discriminator keys."
                )
            value_shape = buffer.value.shape[2:]
            buffer.register_key("next_disc_value", shape=value_shape)
            buffer.register_key("disc_returns")
            if self.config.normalize_rewards:
                buffer.register_key("unnormalized_disc_value", shape=value_shape)
                buffer.register_key("unnormalized_next_disc_value", shape=value_shape)

    def update_disc_replay_buffer(self, data_dict) -> None:
        buf_size = self.replay_buffer.get_buffer_size()
        buf_total_count = len(self.replay_buffer)

        values = list(data_dict.values())
        numel = values[0].shape[0]

        for i in range(1, len(values)):
            assert numel == values[i].shape[0]

        if buf_total_count >= buf_size:
            keep_probs = (
                torch.ones(numel, device=self.device)
                * self.config.amp_parameters.discriminator_replay_keep_prob
            )
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            for k, v in data_dict.items():
                data_dict[k] = v[keep_mask]
            numel = keep_mask.sum().item()

        if numel > buf_size:
            rand_idx = torch.randperm(numel)
            rand_idx = rand_idx[:buf_size]
            for k, v in data_dict.items():
                data_dict[k] = v[rand_idx]

        self.replay_buffer.store(data_dict)

    @torch.no_grad()
    def augment_dataset(self, dataset):
        discriminator_keys = self.agent.model._discriminator.in_keys
        num_samples = dataset[discriminator_keys[0]].shape[0]

        if len(self.replay_buffer) == 0:
            replay_disc_obs = {}
            for key in discriminator_keys:
                replay_disc_obs[key] = dataset[key]
        else:
            replay_disc_obs = self.replay_buffer.sample(num_samples)

        disc_bs = self.config.amp_parameters.discriminator_batch_size
        num_expert = min(num_samples, disc_bs * self.agent.max_num_batches())
        expert_disc_obs = self.agent.get_expert_disc_obs(num_expert)

        discriminator_training_data_dict = {}
        for key in discriminator_keys:
            discriminator_training_data_dict[f"agent_{key}"] = dataset[key]
            discriminator_training_data_dict[f"replay_{key}"] = replay_disc_obs[key]
            discriminator_training_data_dict[f"expert_{key}"] = expert_disc_obs[key]

        dataset.update(discriminator_training_data_dict)

        disc_data_dict = {}
        for key in discriminator_keys:
            disc_data_dict[key] = dataset[key]
        self.update_disc_replay_buffer(disc_data_dict)
        return dataset

    def get_expert_disc_obs(self, num_samples: int):
        motion_ids = self.agent.motion_manager.sample_n_motion_ids(num_samples)
        motion_times = self.agent.motion_manager.sample_time(motion_ids)

        ref_obs_components = self.config.reference_obs_components or {}
        if not ref_obs_components:
            raise ValueError(
                "AMP requires reference_obs_components to be defined in AMPAgentConfig."
            )

        disc_in_keys = set(self.discriminator.module.in_keys)
        batch_size = self.agent.num_envs
        chunk_results = []

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            chunk_ids = motion_ids[start:end]
            chunk_times = motion_times[start:end]
            chunk_n = end - start
            runtime_context = self._build_expert_obs_context(
                chunk_n,
                chunk_ids,
                chunk_times,
            )

            chunk_obs = {}
            for obs_name, router in ref_obs_components.items():
                if obs_name not in disc_in_keys:
                    continue
                fn = router.get_compute_func()
                params = router.get_params().copy()
                obs = self._call_ref_obs_fn(fn, runtime_context, params)
                chunk_obs[obs_name] = obs.view(chunk_n, -1)

            chunk_results.append(chunk_obs)

        expert_obs = {}
        for key in chunk_results[0]:
            expert_obs[key] = torch.cat([c[key] for c in chunk_results], dim=0)
        return expert_obs

    def _build_expert_obs_context(
        self,
        num_samples: int,
        motion_ids: Tensor,
        motion_times: Tensor,
    ) -> Dict[str, object]:
        return {
            "motion_lib": self.agent.motion_lib,
            "motion_ids": motion_ids,
            "motion_times": motion_times,
            "dt": self.agent.env.simulator.dt,
            "num_state_history_steps": self.agent.env.config.num_state_history_steps,
            "contact_body_ids": getattr(self.agent.env, "contact_body_ids", None),
        }

    @staticmethod
    def _call_ref_obs_fn(fn, runtime_context, static_params):
        sig = inspect.signature(fn)
        available = {**runtime_context, **static_params}
        func_kwargs = {}
        missing = []

        for name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name in available:
                func_kwargs[name] = available[name]
            elif param.default is inspect.Parameter.empty:
                missing.append(name)

        if missing:
            raise TypeError(
                f"{fn.__name__}() missing required arguments: {missing}. "
                f"Available context keys: {sorted(runtime_context.keys())}. "
                f"Static params: {sorted(static_params.keys())}."
            )

        return fn(**func_kwargs)

    def post_env_step_modifications(self, dones, terminated, extras):
        discriminator_termination = (
            self.num_cumulative_bad_transitions
            >= self.config.amp_parameters.discriminator_max_cumulative_bad_transitions
        )

        terminated = terminated | discriminator_termination
        dones = dones | terminated

        extras["amp_cumulative_bad_transitions"] = (
            self.num_cumulative_bad_transitions
        )
        extras["amp_discriminator_termination"] = discriminator_termination
        return dones, terminated, extras

    @torch.no_grad()
    def record_rollout_step(
        self,
        next_obs_td,
        rewards,
        terminated,
        done_indices,
        extras,
        step,
    ) -> None:
        disc_logits = self.discriminator(next_obs_td)[
            self.discriminator.module.config.out_keys[0]
        ]
        extras["disc_logits"] = disc_logits.detach()
        amp_rewards = self.discriminator.module.compute_disc_reward(
            disc_logits
        ).flatten()
        bad_transition = (
            amp_rewards < self.config.amp_parameters.discriminator_reward_threshold
        )
        self.num_cumulative_bad_transitions[bad_transition] += 1
        self.num_cumulative_bad_transitions[~bad_transition] = 0

        if len(done_indices) > 0:
            self.num_cumulative_bad_transitions[done_indices] = 0

        if self.use_disc_critic:
            next_disc_value = self.disc_critic(next_obs_td)[
                self.disc_critic.module.config.out_keys[0]
            ]
            next_disc_value = next_disc_value * (1 - terminated.float()).unsqueeze(-1)
            self.agent.experience_buffer.update_data(
                "next_disc_value",
                step,
                next_disc_value,
            )

        if self.config.normalize_rewards:
            self.running_reward_norm.record_reward(amp_rewards, terminated)
        self.agent.experience_buffer.update_data("amp_rewards", step, amp_rewards)

    @torch.no_grad()
    def normalize_rewards_in_buffer(self) -> None:
        if not self.config.normalize_rewards:
            return

        buffer = self.agent.experience_buffer
        amp_rewards = buffer.amp_rewards
        buffer.batch_update_data("unnormalized_amp_rewards", amp_rewards.clone())
        buffer.batch_update_data(
            "amp_rewards",
            self.running_reward_norm.normalize(amp_rewards),
        )

        if not self.use_disc_critic:
            return

        disc_value = buffer.disc_value
        unnorm_disc_value = self.running_reward_norm.normalize(
            disc_value,
            un_norm=True,
        )
        buffer.batch_update_data("unnormalized_disc_value", unnorm_disc_value)

        next_disc_value = buffer.next_disc_value
        unnorm_next_disc_value = self.running_reward_norm.normalize(
            next_disc_value,
            un_norm=True,
        )
        buffer.batch_update_data(
            "unnormalized_next_disc_value",
            unnorm_next_disc_value,
        )

    @torch.no_grad()
    def add_advantages(self, advantages_dict):
        dones = self.agent.experience_buffer.dones

        if self.use_disc_critic:
            if self.config.normalize_rewards:
                disc_rewards = self.agent.experience_buffer.unnormalized_amp_rewards
                disc_values = (
                    self.agent.experience_buffer.unnormalized_disc_value.squeeze(-1)
                )
                disc_next_values = (
                    self.agent.experience_buffer.unnormalized_next_disc_value.squeeze(
                        -1
                    )
                )
            else:
                disc_rewards = self.agent.experience_buffer.amp_rewards
                disc_values = self.agent.experience_buffer.disc_value.squeeze(-1)
                disc_next_values = (
                    self.agent.experience_buffer.next_disc_value.squeeze(-1)
                )

            disc_advantages = discount_values(
                dones,
                disc_values,
                disc_rewards,
                disc_next_values,
                self.agent.gamma,
                self.agent.tau,
            )
            disc_returns = disc_advantages + disc_values

            if self.config.normalize_rewards:
                disc_returns = self.running_reward_norm.normalize(disc_returns)

            self.agent.experience_buffer.batch_update_data(
                "disc_returns",
                disc_returns,
            )
        else:
            if self.config.normalize_rewards:
                disc_advantages = (
                    self.agent.experience_buffer.unnormalized_amp_rewards
                )
            else:
                disc_advantages = self.agent.experience_buffer.amp_rewards

        task_advantages = advantages_dict["advantages"]
        weighted_disc_advantages = (
            disc_advantages * self.config.amp_parameters.discriminator_reward_w
        )
        advantages_dict["advantages"] = task_advantages + weighted_disc_advantages
        self.agent._diag_task_advantages = task_advantages.detach()
        self.agent._diag_disc_advantages = weighted_disc_advantages.detach()
        return advantages_dict

    def optimize_batch_tail(self, batch_dict, batch_idx: int, iter_log_dict: Dict):
        if self.agent._skip_actor_for_epoch:
            return iter_log_dict

        if self.use_disc_critic:
            disc_critic_loss, disc_critic_loss_dict = self.agent.disc_critic_step(
                batch_dict
            )
            iter_log_dict.update(disc_critic_loss_dict)
            disc_critic_grad_clip_dict = self.agent._step_optimizer(
                disc_critic_loss,
                model=self.disc_critic,
                optimizer=self.disc_critic_optimizer,
                model_name="disc_critic",
            )
            iter_log_dict.update(disc_critic_grad_clip_dict)

        if batch_idx % self.config.amp_parameters.discriminator_optimization_ratio == 0:
            discriminator_loss, discriminator_loss_dict = self.agent.discriminator_step(
                batch_dict
            )
            iter_log_dict.update(discriminator_loss_dict)
            discriminator_grad_clip_dict = self.agent._step_optimizer(
                discriminator_loss,
                model=self.discriminator,
                optimizer=self.discriminator_optimizer,
                model_name="discriminator",
            )
            iter_log_dict.update(discriminator_grad_clip_dict)

        return iter_log_dict

    def disc_critic_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.disc_critic(batch_td)
        values = batch_td[self.disc_critic.module.config.out_keys[0]]

        if self.config.clip_critic_loss:
            disc_critic_loss_unclipped = (
                values - batch_dict["disc_returns"].unsqueeze(-1)
            ).pow(2)
            v_clipped = batch_dict["disc_value"] + torch.clamp(
                values - batch_dict["disc_value"],
                -self.agent.e_clip,
                self.agent.e_clip,
            )
            disc_critic_loss_clipped = (
                v_clipped - batch_dict["disc_returns"].unsqueeze(-1)
            ).pow(2)
            disc_critic_loss = torch.max(
                disc_critic_loss_unclipped,
                disc_critic_loss_clipped,
            ).mean()
        else:
            disc_critic_loss = (
                batch_dict["disc_returns"].unsqueeze(-1) - values
            ).pow(2).mean()

        with torch.no_grad():
            disc_returns = batch_dict["disc_returns"].unsqueeze(-1)
            disc_errors = values.detach() - disc_returns
            disc_return_var = disc_returns.var()
            disc_explained_var = (
                1.0 - disc_errors.var() / (disc_return_var + 1e-8)
                if disc_return_var > 1e-8
                else torch.zeros(1, device=values.device)
            )

        return disc_critic_loss, {
            "losses/disc_critic_loss": disc_critic_loss.detach(),
            "disc_critic/explained_variance": disc_explained_var,
            "disc_critic/value_mean": values.detach().mean(),
            "disc_critic/value_std": values.detach().std(),
            "disc_critic/return_mean": disc_returns.mean(),
            "disc_critic/return_std": disc_returns.std(),
            "disc_critic/error_mean": disc_errors.mean(),
            "disc_critic/error_std": disc_errors.std(),
        }

    def discriminator_step(self, batch_dict):
        agent_obs = {}
        for key in batch_dict.keys():
            if "agent_" in key:
                agent_obs[key.replace("agent_", "")] = batch_dict[key][
                    : self.config.amp_parameters.discriminator_batch_size
                ]
        replay_obs = {}
        for key in batch_dict.keys():
            if "replay_" in key:
                replay_obs[key.replace("replay_", "")] = batch_dict[key][
                    : self.config.amp_parameters.discriminator_batch_size
                ]
        expert_obs = {}
        for key in batch_dict.keys():
            if "expert_" in key:
                val = batch_dict[key][
                    : self.config.amp_parameters.discriminator_batch_size
                ]
                if not val.is_floating_point():
                    val = val.float()
                expert_obs[key.replace("expert_", "")] = val
                expert_obs[key.replace("expert_", "")].requires_grad_(True)

        if self.config.amp_parameters.conditional_discriminator:
            negative_expert_obs = self.agent.produce_negative_expert_obs(batch_dict)

        expert_obs_td = TensorDict(
            expert_obs,
            batch_size=self.config.amp_parameters.discriminator_batch_size,
        )
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            expert_obs_td = self.discriminator(expert_obs_td)
        expert_logits = expert_obs_td[self.discriminator.module.config.out_keys[0]]
        expert_norm_obs = [
            expert_obs_td[key] for key in self.discriminator.module._grad_penalty_keys
        ]

        agent_obs_td = TensorDict(
            agent_obs,
            batch_size=self.config.amp_parameters.discriminator_batch_size,
        )
        agent_obs_td = self.discriminator(agent_obs_td)
        agent_logits = agent_obs_td[self.discriminator.module.config.out_keys[0]]

        replay_obs_td = TensorDict(
            replay_obs,
            batch_size=self.config.amp_parameters.discriminator_batch_size,
        )
        replay_obs_td = self.discriminator(replay_obs_td)
        replay_logits = replay_obs_td[self.discriminator.module.config.out_keys[0]]

        if self.config.amp_parameters.conditional_discriminator:
            negative_expert_obs_td = TensorDict(
                negative_expert_obs,
                batch_size=self.config.amp_parameters.discriminator_batch_size,
            )
            negative_expert_obs_td = self.discriminator(negative_expert_obs_td)
            negative_expert_logits = negative_expert_obs_td[
                self.discriminator.module.config.out_keys[0]
            ]

        expert_loss = -torch.nn.functional.logsigmoid(expert_logits).mean()
        unlabeled_loss = torch.nn.functional.softplus(agent_logits).mean()
        replay_loss = torch.nn.functional.softplus(replay_logits).mean()
        if self.config.amp_parameters.conditional_discriminator:
            negative_loss = torch.nn.functional.softplus(
                negative_expert_logits
            ).mean()
            negative_loss = 0.5 * (unlabeled_loss + replay_loss + negative_loss)
        else:
            negative_loss = 0.5 * (unlabeled_loss + replay_loss)
        class_loss = 0.5 * (expert_loss + negative_loss)

        disc_grads = torch.autograd.grad(
            expert_logits,
            expert_norm_obs,
            grad_outputs=torch.ones_like(expert_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )
        disc_grad_norm = sum(
            (
                g.flatten(start_dim=1).pow(2).sum(dim=-1)
                for g in disc_grads
                if g is not None
            ),
            torch.zeros(expert_logits.shape[0], device=expert_logits.device),
        )
        disc_grad_penalty = torch.mean(disc_grad_norm)
        grad_loss = (
            self.config.amp_parameters.discriminator_grad_penalty
            * disc_grad_penalty
        )

        if self.config.amp_parameters.discriminator_weight_decay > 0:
            all_weight_params = self.discriminator.module.all_discriminator_weights()
            total = sum([p.pow(2).sum() for p in all_weight_params])
            weight_decay_loss = (
                total * self.config.amp_parameters.discriminator_weight_decay
            )
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)
            total = torch.tensor(0.0, device=self.device)

        if self.config.amp_parameters.discriminator_logit_weight_decay > 0:
            logit_params = self.discriminator.module.logit_weights()
            logit_total = sum([p.pow(2).sum() for p in logit_params])
            logit_weight_decay_loss = (
                logit_total
                * self.config.amp_parameters.discriminator_logit_weight_decay
            )
        else:
            logit_weight_decay_loss = torch.tensor(0.0, device=self.device)
            logit_total = torch.tensor(0.0, device=self.device)

        loss = grad_loss + class_loss + weight_decay_loss + logit_weight_decay_loss

        with torch.no_grad():
            pos_acc = self.compute_pos_acc(expert_logits)
            agent_acc = self.compute_neg_acc(agent_logits)
            replay_acc = self.compute_neg_acc(replay_logits)
            neg_acc = 0.5 * (agent_acc + replay_acc)
            log_dict = {
                "losses/discriminator_loss": loss.detach(),
                "discriminator/pos_acc": pos_acc.detach(),
                "discriminator/agent_acc": agent_acc.detach(),
                "discriminator/replay_acc": replay_acc.detach(),
                "discriminator/neg_acc": neg_acc.detach(),
                "discriminator/grad_penalty": disc_grad_penalty.detach(),
                "discriminator/grad_loss": grad_loss.detach(),
                "discriminator/class_loss": class_loss.detach(),
                "discriminator/l2_logit_total": logit_total.detach(),
                "discriminator/l2_logit_loss": logit_weight_decay_loss.detach(),
                "discriminator/l2_total": total.detach(),
                "discriminator/l2_loss": weight_decay_loss.detach(),
                "discriminator/expert_logit_mean": expert_logits.detach().mean(),
                "discriminator/agent_logit_mean": agent_logits.detach().mean(),
                "discriminator/replay_logit_mean": replay_logits.detach().mean(),
            }
            if self.config.amp_parameters.conditional_discriminator:
                log_dict["discriminator/negative_expert_logit_mean"] = (
                    negative_expert_logits.detach().mean()
                )
                log_dict["discriminator/negative_logit_mean"] = 0.5 * (
                    log_dict["discriminator/agent_logit_mean"]
                    + log_dict["discriminator/replay_logit_mean"]
                    + log_dict["discriminator/negative_expert_logit_mean"]
                )
            else:
                log_dict["discriminator/negative_logit_mean"] = 0.5 * (
                    log_dict["discriminator/agent_logit_mean"]
                    + log_dict["discriminator/replay_logit_mean"]
                )

        return loss, log_dict

    @staticmethod
    def compute_pos_acc(positive_logit: Tensor) -> Tensor:
        return (positive_logit > 0).float().mean()

    @staticmethod
    def compute_neg_acc(negative_logit: Tensor) -> Tensor:
        return (negative_logit < 0).float().mean()

    def add_epoch_logging(self, training_log_dict) -> None:
        amp_rewards = self.agent.experience_buffer.amp_rewards
        training_log_dict["rewards/amp_rewards"] = amp_rewards.mean()
        training_log_dict["rewards/amp_rewards_std"] = amp_rewards.std()
        training_log_dict["rewards/amp_rewards_min"] = amp_rewards.min()
        training_log_dict["rewards/amp_rewards_max"] = amp_rewards.max()

        if self.config.normalize_rewards:
            training_log_dict["rewards/unnormalized_amp_rewards"] = (
                self.agent.experience_buffer.unnormalized_amp_rewards.mean().item()
            )
            training_log_dict["amp_reward_norm/var"] = (
                self.running_reward_norm.var.item()
            )
            training_log_dict["amp_reward_norm/pre_norm"] = (
                self.agent.experience_buffer.unnormalized_amp_rewards.mean().item()
            )
            training_log_dict["amp_reward_norm/post_norm"] = (
                self.agent.experience_buffer.amp_rewards.mean().item()
            )

        if hasattr(self.agent, "_diag_task_advantages"):
            task_adv = self.agent._diag_task_advantages
            disc_adv = self.agent._diag_disc_advantages
            training_log_dict["advantages/task_mean"] = task_adv.mean()
            training_log_dict["advantages/task_std"] = task_adv.std()
            training_log_dict["advantages/disc_mean"] = disc_adv.mean()
            training_log_dict["advantages/disc_std"] = disc_adv.std()


class AMPAgentMixin:
    """Lifecycle mixin that installs and delegates to `AMPTrainingComponent`."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.amp_component = AMPTrainingComponent(self)

    def create_optimizers(self, model):
        super().create_optimizers(model)
        AMPTrainingComponent.for_agent(self).create_optimizers(model)

    def _load_model_state_dict(self, model_state_dict):
        has_amp_model = any(k.startswith("_discriminator.") for k in model_state_dict)
        self._warm_start_from_non_amp_checkpoint = not has_amp_model

        if has_amp_model:
            super()._load_model_state_dict(model_state_dict)
            return

        log.info(
            "Warm-starting AMP from non-AMP checkpoint: loading actor/critic "
            "weights only; discriminator networks stay randomly initialized."
        )

        actor_module = self._fixed_logstd_actor_module()
        has_fixed_logstd = actor_module is not None
        if has_fixed_logstd:
            current_logstd = actor_module.logstd.data.clone()

        missing, unexpected = self.model.load_state_dict(
            model_state_dict,
            strict=False,
        )
        if missing:
            log.info("Expected missing keys for new AMP networks: %d", len(missing))
        if unexpected:
            log.warning("Unexpected keys in checkpoint: %s", unexpected)

        if has_fixed_logstd:
            checkpoint_logstd = actor_module.logstd.data
            if not torch.allclose(current_logstd, checkpoint_logstd, atol=1e-6):
                actor_module.logstd.data = current_logstd

    def _fixed_logstd_actor_module(self):
        """Return the actor module whose fixed Gaussian logstd should be preserved."""
        actor_config = self.config.model.actor
        if not hasattr(actor_config, "learnable_std") or actor_config.learnable_std:
            return None
        if not hasattr(self, "actor"):
            return None
        actor_module = self.actor_module
        return actor_module if hasattr(actor_module, "logstd") else None

    def _actor_has_fixed_logstd(self) -> bool:
        """Whether warm-start loading should preserve a fixed Gaussian logstd."""
        return self._fixed_logstd_actor_module() is not None

    def _load_training_state(self, state_dict):
        if getattr(self, "_warm_start_from_non_amp_checkpoint", False):
            BaseAgent._load_training_state(self, state_dict)
            self._load_ppo_training_state(state_dict, require_optimizers=False)
            return

        super()._load_training_state(state_dict)
        AMPTrainingComponent.for_agent(self).load_training_state(state_dict)

    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        return AMPTrainingComponent.for_agent(self).add_state_dict(state_dict)

    def register_algorithm_experience_buffer_keys(self):
        component = AMPTrainingComponent.for_agent(self)
        super().register_algorithm_experience_buffer_keys()
        component.register_experience_buffer_keys()

    def update_disc_replay_buffer(self, data_dict):
        return AMPTrainingComponent.for_agent(self).update_disc_replay_buffer(data_dict)

    @torch.no_grad()
    def process_dataset(self, dataset):
        dataset = AMPTrainingComponent.for_agent(self).augment_dataset(dataset)
        return super().process_dataset(dataset)

    def get_expert_disc_obs(self, num_samples: int):
        return AMPTrainingComponent.for_agent(self).get_expert_disc_obs(num_samples)

    def _build_expert_obs_context(
        self,
        num_samples: int,
        motion_ids: Tensor,
        motion_times: Tensor,
    ) -> Dict[str, object]:
        return AMPTrainingComponent.for_agent(self)._build_expert_obs_context(
            num_samples,
            motion_ids,
            motion_times,
        )

    @staticmethod
    def _call_ref_obs_fn(fn, runtime_context, static_params):
        return AMPTrainingComponent._call_ref_obs_fn(
            fn,
            runtime_context,
            static_params,
        )

    def post_env_step_modifications(self, dones, terminated, extras):
        dones, terminated, extras = AMPTrainingComponent.for_agent(
            self
        ).post_env_step_modifications(dones, terminated, extras)
        return super().post_env_step_modifications(dones, terminated, extras)

    @torch.no_grad()
    def record_rollout_step(
        self,
        next_obs_td,
        actions,
        rewards,
        dones,
        terminated,
        done_indices,
        extras,
        step,
    ):
        super().record_rollout_step(
            next_obs_td,
            actions,
            rewards,
            dones,
            terminated,
            done_indices,
            extras,
            step,
        )
        AMPTrainingComponent.for_agent(self).record_rollout_step(
            next_obs_td,
            rewards,
            terminated,
            done_indices,
            extras,
            step,
        )

    @torch.no_grad()
    def normalize_rewards_in_buffer(self):
        super().normalize_rewards_in_buffer()
        AMPTrainingComponent.for_agent(self).normalize_rewards_in_buffer()

    @torch.no_grad()
    def compute_advantages(self):
        advantages_dict = super().compute_advantages()
        return AMPTrainingComponent.for_agent(self).add_advantages(advantages_dict)

    def perform_optimization_step(self, batch_dict, batch_idx: int) -> Dict:
        iter_log_dict = super().perform_optimization_step(batch_dict, batch_idx)
        return AMPTrainingComponent.for_agent(self).optimize_batch_tail(
            batch_dict,
            batch_idx,
            iter_log_dict,
        )

    def disc_critic_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        return AMPTrainingComponent.for_agent(self).disc_critic_step(batch_dict)

    def discriminator_step(self, batch_dict):
        return AMPTrainingComponent.for_agent(self).discriminator_step(batch_dict)

    @staticmethod
    def compute_pos_acc(positive_logit: Tensor) -> Tensor:
        return AMPTrainingComponent.compute_pos_acc(positive_logit)

    @staticmethod
    def compute_neg_acc(negative_logit: Tensor) -> Tensor:
        return AMPTrainingComponent.compute_neg_acc(negative_logit)

    def post_epoch_logging(self, training_log_dict):
        AMPTrainingComponent.for_agent(self).add_epoch_logging(training_log_dict)
        super().post_epoch_logging(training_log_dict)
