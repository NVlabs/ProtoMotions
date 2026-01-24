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
"""Adversarial Motion Priors (AMP) agent implementation.

This module implements the AMP algorithm which extends PPO with a discriminator
network that provides style rewards. The discriminator learns to distinguish between
agent and reference motion data, encouraging the agent to produce naturalistic movements
while accomplishing tasks.

Key Classes:
    - AMP: Main AMP agent class extending PPO

References:
    Peng et al. "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control" (2021)
"""

import torch
import logging

from torch import Tensor
from tensordict import TensorDict
from lightning.fabric import Fabric
from protomotions.utils.hydra_replacement import instantiate

from protomotions.agents.utils.replay_buffer import ReplayBuffer
from protomotions.agents.amp.model import AMPModel
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.utils.normalization import RewardRunningMeanStd
from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.utils import discount_values
from protomotions.agents.amp.config import AMPAgentConfig
from pathlib import Path
from typing import Optional, Dict, Tuple

from protomotions.agents.utils.training import handle_model_grad_clipping

log = logging.getLogger(__name__)


class AMP(PPO):
    """Adversarial Motion Priors (AMP) agent.

    Extends PPO with a discriminator network that learns to distinguish between
    agent and reference motion data. The discriminator provides a style reward that
    encourages the agent to produce motions with similar characteristics to the
    reference dataset. This enables training agents that perform tasks while
    maintaining natural motion styles.

    The agent combines task rewards with discriminator-based style rewards:
    - Task reward: From environment (e.g., reaching a target)
    - Style reward: From discriminator (similarity to reference motions)

    Args:
        fabric: Lightning Fabric instance for distributed training.
        env: Environment instance with motion library for reference data.
        config: AMP-specific configuration including discriminator parameters.
        root_dir: Optional root directory for saving outputs.

    Attributes:
        amp_replay_buffer: Replay buffer storing agent transitions for discriminator training.
        discriminator: Network that distinguishes agent from reference motions.

    Example:
        >>> fabric = Fabric(devices=4)
        >>> env = Mimic(config, robot_config, simulator_config, device)
        >>> agent = AMP(fabric, env, config)
        >>> agent.setup()
        >>> agent.train()

    Note:
        Requires environment with motion library (motion_lib) for sampling reference data.
    """

    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    config: AMPAgentConfig

    def __init__(
        self, fabric: Fabric, env: BaseEnv, config, root_dir: Optional[Path] = None
    ):
        super().__init__(fabric, env, config, root_dir=root_dir)
        self.amp_replay_buffer = ReplayBuffer(
            self.config.amp_parameters.discriminator_replay_size, device=self.device
        )
        self.num_cumulative_bad_transitions = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.int32
        )
        if self.config.normalize_rewards:
            self.running_amp_reward_norm = RewardRunningMeanStd(
                shape=(1,),
                fabric=self.fabric,
                gamma=self.gamma,
                device=self.device,
                clamp_value=self.config.normalized_reward_clamp_value,
            )
        else:
            self.running_amp_reward_norm = None

    def create_optimizers(self, model: AMPModel):
        super().create_optimizers(model)
        discriminator_optimizer = instantiate(
            self.config.model.discriminator_optimizer,
            params=list(model._discriminator.parameters()),
        )
        self.discriminator, self.discriminator_optimizer = self.fabric.setup(
            model._discriminator, discriminator_optimizer
        )
        disc_critic_optimizer = instantiate(
            self.config.model.disc_critic_optimizer,
            params=list(model._disc_critic.parameters()),
        )
        self.disc_critic, self.disc_critic_optimizer = self.fabric.setup(
            model._disc_critic, disc_critic_optimizer
        )

    def load_parameters(self, state_dict):
        super().load_parameters(state_dict)
        self.discriminator_optimizer.load_state_dict(
            state_dict["discriminator_optimizer"]
        )
        self.disc_critic_optimizer.load_state_dict(
            state_dict["disc_critic_optimizer"]
        )
        if self.config.normalize_rewards:
            self.running_amp_reward_norm.load_state_dict(
                state_dict["running_amp_reward_norm"]
            )

    def get_state_dict(self, state_dict):
        state_dict = super().get_state_dict(state_dict)
        state_dict["discriminator_optimizer"] = (
            self.discriminator_optimizer.state_dict()
        )
        state_dict["disc_critic_optimizer"] = self.disc_critic_optimizer.state_dict()
        if self.config.normalize_rewards:
            state_dict["running_amp_reward_norm"] = (
                self.running_amp_reward_norm.state_dict()
            )
        return state_dict

    # -----------------------------
    # Experience Buffer and Dataset Processing
    # -----------------------------
    def register_algorithm_experience_buffer_keys(self):
        super().register_algorithm_experience_buffer_keys()
        self.experience_buffer.register_key("amp_rewards")
        if self.config.normalize_rewards:
            self.experience_buffer.register_key("unnormalized_amp_rewards")

        value_shape = getattr(self.experience_buffer, "value").shape[2:]
        self.experience_buffer.register_key("next_disc_value", shape=value_shape)
        self.experience_buffer.register_key("disc_returns")
        if self.config.normalize_rewards:
            self.experience_buffer.register_key("unnormalized_disc_value", shape=value_shape)
            self.experience_buffer.register_key("unnormalized_next_disc_value", shape=value_shape)

    def update_disc_replay_buffer(self, data_dict):
        buf_size = self.amp_replay_buffer.get_buffer_size()
        buf_total_count = len(self.amp_replay_buffer)

        values = list(data_dict.values())
        numel = values[0].shape[0]

        for i in range(1, len(values)):
            assert numel == values[i].shape[0]

        if buf_total_count > buf_size:
            keep_probs = (
                torch.ones(numel, device=self.device)
                * self.config.amp_parameters.discriminator_replay_keep_prob
            )
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            for k, v in data_dict.items():
                data_dict[k] = v[keep_mask]

        if numel > buf_size:
            rand_idx = torch.randperm(numel)
            rand_idx = rand_idx[:buf_size]
            for k, v in data_dict.items():
                data_dict[k] = v[rand_idx]

        self.amp_replay_buffer.store(data_dict)

    @torch.no_grad()
    def process_dataset(self, dataset):
        discriminator_keys = self.model._discriminator.in_keys

        num_samples = dataset[discriminator_keys[0]].shape[0]

        if len(self.amp_replay_buffer) == 0:
            replay_disc_obs = {}
            for key in discriminator_keys:
                replay_disc_obs[key] = dataset[key]
        else:
            replay_disc_obs = self.amp_replay_buffer.sample(num_samples)

        expert_disc_obs = self.get_expert_disc_obs(num_samples)

        discriminator_training_data_dict = {}
        for key in discriminator_keys:
            discriminator_training_data_dict[f"agent_{key}"] = dataset[key]
            discriminator_training_data_dict[f"replay_{key}"] = replay_disc_obs[key]
            discriminator_training_data_dict[f"expert_{key}"] = expert_disc_obs[key]

        dataset.update(discriminator_training_data_dict)

        # Add observations to the disc replay buffer
        disc_data_dict = {}
        for key in discriminator_keys:
            disc_data_dict[key] = dataset[key]
        self.update_disc_replay_buffer(disc_data_dict)

        return super().process_dataset(dataset)

    def get_expert_disc_obs(self, num_samples: int):
        """Build expert observations from motion library for discriminator training.
        
        Iterates over reference_obs_components defined in AMPAgentConfig
        and uses them to compute demo observations from sampled motions.
        """
        motion_ids = self.motion_manager.sample_n_motion_ids(num_samples)
        motion_times = self.motion_manager.sample_time(motion_ids)

        ref_obs_components = self.config.reference_obs_components or {}
        
        if not ref_obs_components:
            raise ValueError(
                "AMP requires reference_obs_components to be defined in AMPAgentConfig. "
                "Use factories like historical_max_coords_ref_obs_factory() to define them."
            )
        
        # Build context for evaluating variables
        # Provide motion library and sampled motion info
        context = {
            "motion_lib": self.motion_lib,
            "motion_ids": motion_ids,
            "motion_times": motion_times,
            "dt": self.env.simulator.dt,
            "num_historical_steps": self.env.config.num_state_history_steps,
        }
        
        all_demo_obs = {}
        
        # Iterate over reference observation components
        for obs_name, component in ref_obs_components.items():
            if component.function is None:
                continue
            
            # Build kwargs for the observation function
            func_kwargs = {}
            for arg_name, var_value in component.variables.items():
                if isinstance(var_value, str):
                    # Look up string values in context
                    if var_value in context:
                        func_kwargs[arg_name] = context[var_value]
                    else:
                        raise ValueError(
                            f"Variable '{var_value}' not found in context for "
                            f"reference obs component '{obs_name}'"
                        )
                else:
                    # Non-string values are passed directly as constants
                    func_kwargs[arg_name] = var_value
            
            # Call the observation function
            obs = component.function(**func_kwargs)
            all_demo_obs[obs_name] = obs.view(num_samples, -1)
        
        # Filter to only keys the discriminator uses
        expert_obs = {}
        for key, value in all_demo_obs.items():
            if key in self.discriminator.in_keys:
                expert_obs[key] = value

        return expert_obs

    # -----------------------------
    # Reward Calculation
    # -----------------------------
    def post_env_step_modifications(self, dones, terminated, extras):
        """Add AMP-specific discriminator-based termination."""

        discriminator_termination = (
            self.num_cumulative_bad_transitions
            >= self.config.amp_parameters.discriminator_max_cumulative_bad_transitions
        )

        terminated = terminated | discriminator_termination
        dones = dones | terminated

        extras["amp_cumulative_bad_transitions"] = self.num_cumulative_bad_transitions
        extras["amp_discriminator_termination"] = discriminator_termination

        return dones, terminated, extras

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
            next_obs_td, actions, rewards, dones, terminated, done_indices, extras, step
        )

        disc_logits = self.discriminator(next_obs_td)[
            self.discriminator.config.out_keys[0]
        ]
        amp_rewards = self.discriminator.compute_disc_reward(disc_logits).flatten()
        bad_transition = (
            amp_rewards < self.config.amp_parameters.discriminator_reward_threshold
        )
        self.num_cumulative_bad_transitions[bad_transition] += 1
        self.num_cumulative_bad_transitions[~bad_transition] = 0

        if len(done_indices) > 0:
            self.num_cumulative_bad_transitions[done_indices] = 0

        next_disc_value = self.disc_critic(next_obs_td)[self.disc_critic.config.out_keys[0]]
        next_disc_value = next_disc_value * (1 - terminated.float()).unsqueeze(-1)
        self.experience_buffer.update_data("next_disc_value", step, next_disc_value)

        if self.config.normalize_rewards:
            self.running_amp_reward_norm.record_reward(amp_rewards, terminated)
        self.experience_buffer.update_data("amp_rewards", step, amp_rewards)

    @torch.no_grad()
    def normalize_rewards_in_buffer(self):
        super().normalize_rewards_in_buffer()
        if not self.config.normalize_rewards:
            return

        amp_rewards = self.experience_buffer.amp_rewards
        self.experience_buffer.batch_update_data(
            "unnormalized_amp_rewards", amp_rewards.clone()
        )
        self.experience_buffer.batch_update_data(
            "amp_rewards", self.running_amp_reward_norm.normalize(amp_rewards)
        )

        disc_value = self.experience_buffer.disc_value
        unnorm_disc_value = self.running_amp_reward_norm.normalize(
            disc_value, un_norm=True
        )
        self.experience_buffer.batch_update_data("unnormalized_disc_value", unnorm_disc_value)

        next_disc_value = self.experience_buffer.next_disc_value
        unnorm_next_disc_value = self.running_amp_reward_norm.normalize(
            next_disc_value, un_norm=True
        )
        self.experience_buffer.batch_update_data(
            "unnormalized_next_disc_value", unnorm_next_disc_value
        )

    @torch.no_grad()
    def compute_advantages(self):
        advantages_dict = super().compute_advantages()
        dones = self.experience_buffer.dones

        if self.config.normalize_rewards:
            disc_rewards = self.experience_buffer.unnormalized_amp_rewards
            disc_values = self.experience_buffer.unnormalized_disc_value.squeeze(-1)
            disc_next_values = self.experience_buffer.unnormalized_next_disc_value.squeeze(-1)
        else:
            disc_rewards = self.experience_buffer.amp_rewards
            disc_values = self.experience_buffer.disc_value.squeeze(-1)
            disc_next_values = self.experience_buffer.next_disc_value.squeeze(-1)

        disc_advantages = discount_values(
            dones, disc_values, disc_rewards, disc_next_values, self.gamma, self.tau
        )
        disc_returns = disc_advantages + disc_values

        if self.config.normalize_rewards:
            disc_returns = self.running_amp_reward_norm.normalize(disc_returns)

        self.experience_buffer.batch_update_data("disc_returns", disc_returns)

        advantages_dict["advantages"] = (
            advantages_dict["advantages"]
            + disc_advantages * self.config.amp_parameters.discriminator_reward_w
        )
        return advantages_dict

    # -----------------------------
    # Optimization
    # -----------------------------
    def perform_optimization_step(self, batch_dict, batch_idx: int) -> Dict:
        iter_log_dict = super().perform_optimization_step(batch_dict, batch_idx)

        disc_critic_loss, disc_critic_loss_dict = self.disc_critic_step(batch_dict)
        iter_log_dict.update(disc_critic_loss_dict)
        self.disc_critic_optimizer.zero_grad(set_to_none=True)
        self.fabric.backward(disc_critic_loss)
        disc_critic_grad_clip_dict = handle_model_grad_clipping(
            config=self.config,
            fabric=self.fabric,
            model=self.disc_critic,
            optimizer=self.disc_critic_optimizer,
            model_name="disc_critic",
        )
        iter_log_dict.update(disc_critic_grad_clip_dict)
        self.disc_critic_optimizer.step()

        if batch_idx % self.config.amp_parameters.discriminator_optimization_ratio == 0:
            discriminator_loss, discriminator_loss_dict = self.discriminator_step(
                batch_dict
            )
            iter_log_dict.update(discriminator_loss_dict)
            self.discriminator_optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(discriminator_loss)
            discriminator_grad_clip_dict = handle_model_grad_clipping(
                config=self.config,
                fabric=self.fabric,
                model=self.discriminator,
                optimizer=self.discriminator_optimizer,
                model_name="discriminator",
            )
            iter_log_dict.update(discriminator_grad_clip_dict)
            self.discriminator_optimizer.step()

        return iter_log_dict

    def disc_critic_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.disc_critic(batch_td)
        values = batch_td[self.disc_critic.config.out_keys[0]]

        if self.config.clip_critic_loss:
            disc_critic_loss_unclipped = (
                values - batch_dict["disc_returns"].unsqueeze(-1)
            ).pow(2)
            v_clipped = batch_dict["disc_value"] + torch.clamp(
                values - batch_dict["disc_value"],
                -self.config.e_clip,
                self.config.e_clip,
            )
            disc_critic_loss_clipped = (
                v_clipped - batch_dict["disc_returns"].unsqueeze(-1)
            ).pow(2)
            disc_critic_loss_max = torch.max(
                disc_critic_loss_unclipped, disc_critic_loss_clipped
            )
            disc_critic_loss = disc_critic_loss_max.mean()
        else:
            disc_critic_loss = (
                (batch_dict["disc_returns"].unsqueeze(-1) - values).pow(2).mean()
            )

        log_dict = {"losses/disc_critic_loss": disc_critic_loss.detach()}
        return disc_critic_loss, log_dict

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
                expert_obs[key.replace("expert_", "")] = batch_dict[key][
                    : self.config.amp_parameters.discriminator_batch_size
                ]
                expert_obs[key.replace("expert_", "")].requires_grad_(True)

        if self.config.amp_parameters.conditional_discriminator:
            negative_expert_obs = self.produce_negative_expert_obs(batch_dict)

        expert_obs_td = TensorDict(
            expert_obs, batch_size=self.config.amp_parameters.discriminator_batch_size
        )
        expert_obs_td = self.discriminator(expert_obs_td)
        expert_logits = expert_obs_td[self.discriminator.config.out_keys[0]]

        expert_norm_obs = []
        for key in expert_obs_td.keys():
            if "norm_" in key:
                expert_norm_obs.append(expert_obs_td[key])

        agent_obs_td = TensorDict(
            agent_obs, batch_size=self.config.amp_parameters.discriminator_batch_size
        )
        agent_obs_td = self.discriminator(agent_obs_td)
        agent_logits = agent_obs_td[self.discriminator.config.out_keys[0]]

        replay_obs_td = TensorDict(
            replay_obs, batch_size=self.config.amp_parameters.discriminator_batch_size
        )
        replay_obs_td = self.discriminator(replay_obs_td)
        replay_logits = replay_obs_td[self.discriminator.config.out_keys[0]]

        if self.config.amp_parameters.conditional_discriminator:
            negative_expert_obs_td = TensorDict(
                negative_expert_obs,
                batch_size=self.config.amp_parameters.discriminator_batch_size,
            )
            negative_expert_obs_td = self.discriminator(negative_expert_obs_td)
            negative_expert_logits = negative_expert_obs_td[
                self.discriminator.config.out_keys[0]
            ]

        expert_loss = -torch.nn.functional.logsigmoid(expert_logits).mean()
        unlabeled_loss = torch.nn.functional.softplus(agent_logits).mean()
        replay_loss = torch.nn.functional.softplus(replay_logits).mean()
        if self.config.amp_parameters.conditional_discriminator:
            neg_loss = torch.nn.functional.softplus(negative_expert_logits).mean()

        if self.config.amp_parameters.conditional_discriminator:
            neg_loss = 0.5 * (unlabeled_loss + replay_loss + neg_loss)
        else:
            neg_loss = 0.5 * (unlabeled_loss + replay_loss)
        class_loss = 0.5 * (expert_loss + neg_loss)

        # Gradient penalty
        disc_grad = torch.autograd.grad(
            expert_logits,
            expert_norm_obs,
            grad_outputs=torch.ones_like(expert_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        disc_grad_norm = torch.sum(torch.square(disc_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_grad_norm)
        grad_loss: Tensor = (
            self.config.amp_parameters.discriminator_grad_penalty * disc_grad_penalty
        )

        if self.config.amp_parameters.discriminator_weight_decay > 0:
            all_weight_params = self.discriminator.all_discriminator_weights()
            total: Tensor = sum([p.pow(2).sum() for p in all_weight_params])
            weight_decay_loss: Tensor = (
                total * self.config.amp_parameters.discriminator_weight_decay
            )
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)
            total = torch.tensor(0.0, device=self.device)

        if self.config.amp_parameters.discriminator_logit_weight_decay > 0:
            logit_params = self.discriminator.logit_weights()
            logit_total = sum([p.pow(2).sum() for p in logit_params])
            logit_weight_decay_loss: Tensor = (
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

    # -----------------------------
    # Discriminator Metrics and Utility
    # -----------------------------
    @staticmethod
    def compute_pos_acc(positive_logit: Tensor) -> Tensor:
        return (positive_logit > 0).float().mean()

    @staticmethod
    def compute_neg_acc(negative_logit: Tensor) -> Tensor:
        return (negative_logit < 0).float().mean()

    # -----------------------------
    # Termination and Logging
    # -----------------------------
    def terminate_early(self):
        self._should_stop = True

    def post_epoch_logging(self, training_log_dict):
        training_log_dict["rewards/amp_rewards"] = (
            self.experience_buffer.amp_rewards.mean()
        )
        if self.config.normalize_rewards:
            training_log_dict["rewards/unnormalized_amp_rewards"] = (
                self.experience_buffer.unnormalized_amp_rewards.mean().item()
            )

        super().post_epoch_logging(training_log_dict)
