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
"""Proximal Policy Optimization (PPO) agent implementation.

This module implements the PPO algorithm for reinforcement learning. PPO is an
on-policy algorithm that uses clipped surrogate objectives for stable policy updates.
It collects experience through environment interaction and performs multiple epochs
of minibatch updates using Generalized Advantage Estimation (GAE).

Key Classes:
    - PPO: Main PPO agent class extending BaseAgent

References:
    Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
from torch import Tensor
from tensordict import TensorDict

import logging

from pathlib import Path
from typing import Optional, Tuple, Dict

from lightning.fabric import Fabric

from protomotions.utils.hydra_replacement import instantiate, get_class

from protomotions.agents.ppo.model import PPOModel
from protomotions.agents.common.common import weight_init
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.utils.normalization import combine_moments
from protomotions.agents.ppo.utils import discount_values
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.utils.training import bounds_loss, handle_model_grad_clipping

log = logging.getLogger(__name__)


class PPO(BaseAgent):
    """Proximal Policy Optimization (PPO) agent.

    Implements the PPO algorithm for training reinforcement learning policies.
    PPO uses clipped surrogate objectives to enable stable policy updates while
    maintaining sample efficiency. This implementation supports actor-critic
    architecture with separate optimizers for policy and value networks.

    The agent collects experience through environment interaction, computes
    advantages using Generalized Advantage Estimation (GAE), and performs
    multiple epochs of minibatch updates on the collected data.

    Args:
        fabric: Lightning Fabric instance for distributed training.
        env: Environment instance to train on.
        config: PPO-specific configuration including learning rates, clip parameters, etc.
        root_dir: Optional root directory for saving outputs.

    Attributes:
        tau: GAE lambda parameter for advantage estimation.
        e_clip: PPO clipping parameter for policy updates.
        actor: Policy network.
        critic: Value network.

    Example:
        >>> fabric = Fabric(devices=4)
        >>> env = Steering(config, robot_config, simulator_config, device)
        >>> agent = PPO(fabric, env, config)
        >>> agent.setup()
        >>> agent.train()
    """

    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    config: PPOAgentConfig

    def __init__(
        self,
        fabric: Fabric,
        env: BaseEnv,
        config: PPOAgentConfig,
        root_dir: Optional[Path] = None,
    ):
        super().__init__(fabric, env, config, root_dir)
        self.tau: float = self.config.tau
        self.e_clip: float = self.config.e_clip

        # Initialize EMA for advantage normalization
        if (
            self.config.advantage_normalization.enabled
            and self.config.advantage_normalization.use_ema
        ):
            self.adv_mean_ema = torch.zeros(1, device=self.device, dtype=torch.float32)
            self.adv_std_ema = torch.ones(1, device=self.device, dtype=torch.float32)
        else:
            self.adv_mean_ema = None
            self.adv_std_ema = None

    def create_model(self):
        """Create PPO actor-critic model.

        Instantiates the PPO model with actor and critic networks, applies
        weight initialization, and returns the model.

        Returns:
            PPOModel instance with initialized weights.
        """
        PPOModelClass = get_class(self.config.model._target_)
        model: PPOModel = PPOModelClass(config=self.config.model)
        model.apply(weight_init)
        return model

    def create_optimizers(self, model: PPOModel):
        """Create separate optimizers for actor and critic.

        Sets up Adam optimizers for policy and value networks with independent
        learning rates. Uses Fabric for distributed training setup.

        Args:
            model: PPOModel with actor and critic networks.
        """
        actor_optimizer = instantiate(
            self.config.model.actor_optimizer,
            params=list(model._actor.parameters()),
        )
        self.actor, self.actor_optimizer = self.fabric.setup(
            model._actor, actor_optimizer
        )
        # Actor now only has forward() method

        critic_optimizer = instantiate(
            self.config.model.critic_optimizer,
            params=list(model._critic.parameters()),
        )
        self.critic, self.critic_optimizer = self.fabric.setup(
            model._critic, critic_optimizer
        )

    def load_parameters(self, state_dict):
        """Load PPO-specific parameters from checkpoint.

        Loads actor, critic, and optimizer states. Preserves config overrides
        for actor_logstd if specified at command line.

        Args:
            state_dict: Checkpoint state dictionary containing model and optimizer states.
        """
        # Save current logstd value to preserve config overrides
        current_logstd = self.actor.logstd.data.clone()
        current_actor_logstd_config = self.config.model.actor.actor_logstd

        super().load_parameters(state_dict)

        # Restore logstd if it was overridden in config (different from checkpoint)
        checkpoint_logstd = self.actor.logstd.data
        if not torch.allclose(current_logstd, checkpoint_logstd, atol=1e-6):
            print(
                f"Preserving overridden actor_logstd: {current_actor_logstd_config} "
                f"(checkpoint had: {checkpoint_logstd[0].item():.3f})"
            )
            self.actor.logstd.data = current_logstd

        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

        # Load EMA state if available
        if (
            self.config.advantage_normalization.enabled
            and self.config.advantage_normalization.use_ema
        ):
            if "adv_mean_ema" in state_dict:
                self.adv_mean_ema.copy_(state_dict["adv_mean_ema"])
            if "adv_std_ema" in state_dict:
                self.adv_std_ema.copy_(state_dict["adv_std_ema"])

    # -----------------------------
    # Model Saving and State Dict
    # -----------------------------
    def get_state_dict(self, state_dict):
        extra_state_dict = super().get_state_dict(state_dict)
        extra_state_dict.update(
            {
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            }
        )

        # Save EMA state
        if (
            self.config.advantage_normalization.enabled
            and self.config.advantage_normalization.use_ema
        ):
            extra_state_dict["adv_mean_ema"] = self.adv_mean_ema
            extra_state_dict["adv_std_ema"] = self.adv_std_ema

        state_dict.update(extra_state_dict)
        return state_dict

    # -----------------------------
    # Experience Buffer and Training Loop
    # -----------------------------
    def register_algorithm_experience_buffer_keys(self):
        super().register_algorithm_experience_buffer_keys()
        # PPO-specific keys that are computed after rollout (not from model forward)
        self.experience_buffer.register_key(
            "next_value", shape=(getattr(self.experience_buffer, "value").shape[2:])
        )  # Computed in post_train_env_step
        self.experience_buffer.register_key(
            "returns"
        )  # Computed in pre_process_dataset
        self.experience_buffer.register_key(
            "advantages"
        )  # Computed in pre_process_dataset

        if self.config.normalize_rewards:
            self.experience_buffer.register_key(
                "unnormalized_value",
                shape=(getattr(self.experience_buffer, "value").shape[2:]),
            )
            self.experience_buffer.register_key(
                "unnormalized_next_value",
                shape=(getattr(self.experience_buffer, "value").shape[2:]),
            )

    # -----------------------------
    # Environment Interaction Helpers
    # -----------------------------
    def record_rollout_step(
        self,
        next_obs_td: TensorDict,
        actions,
        rewards,
        dones,
        terminated,
        done_indices,
        extras,
        step,
    ):
        """Record PPO-specific data: next value estimates for GAE computation."""
        super().record_rollout_step(
            next_obs_td, actions, rewards, dones, terminated, done_indices, extras, step
        )

        # Use model forward to get next value
        next_output_td = self.model._critic(next_obs_td)
        next_value = next_output_td["value"] * (1 - terminated.float()).unsqueeze(-1)
        self.experience_buffer.update_data("next_value", step, next_value)

    @torch.no_grad()
    def normalize_rewards_in_buffer(self):
        """Normalize rewards and denormalize critic values after data collection."""
        super().normalize_rewards_in_buffer()
        if not self.config.normalize_rewards:
            return

        value = self.experience_buffer.value
        unnorm_value = self.running_reward_norm.normalize(value, un_norm=True)
        self.experience_buffer.batch_update_data("unnormalized_value", unnorm_value)

        next_value = self.experience_buffer.next_value
        unnorm_next_value = self.running_reward_norm.normalize(next_value, un_norm=True)
        self.experience_buffer.batch_update_data("unnormalized_next_value", unnorm_next_value)

    # -----------------------------
    # Optimization
    # -----------------------------
    def perform_optimization_step(self, batch_dict, batch_idx) -> Dict:
        """Perform one PPO optimization step on a minibatch.

        Computes actor and critic losses, performs backpropagation, clips gradients,
        and updates both networks.

        Args:
            batch_dict: Dictionary containing minibatch data (obs, actions, advantages, etc.).
            batch_idx: Index of current batch (unused but kept for compatibility).

        Returns:
            Dictionary of training metrics (losses, clip fraction, etc.).
        """
        iter_log_dict = {}
        # Update actor
        actor_loss, actor_loss_dict = self.actor_step(batch_dict)
        iter_log_dict.update(actor_loss_dict)

        # Check if we should skip actor update for this epoch
        # Once triggered, skip all remaining batches (same distribution)
        if (
            not self._skip_actor_for_epoch
            and self.config.actor_clip_frac_threshold is not None
        ):
            clip_frac = actor_loss_dict["actor/clip_frac"].item()
            # Synchronize clip_frac across all GPUs
            if self.fabric.world_size > 1 and torch.distributed.is_initialized():
                clip_frac_tensor = torch.tensor(clip_frac, device=self.device)
                torch.distributed.all_reduce(
                    clip_frac_tensor, op=torch.distributed.ReduceOp.SUM
                )
                clip_frac = (clip_frac_tensor / self.fabric.world_size).item()

            if clip_frac > self.config.actor_clip_frac_threshold:
                self._skip_actor_for_epoch = True
                if self.fabric.global_rank == 0:
                    log.warning(
                        f"Epoch {self.current_epoch}: Skipping actor updates for remaining batches "
                        f"(clip_frac {clip_frac:.3f} > {self.config.actor_clip_frac_threshold})"
                    )

        if not self._skip_actor_for_epoch:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.fabric.backward(actor_loss)
            actor_grad_clip_dict = handle_model_grad_clipping(
                config=self.config,
                fabric=self.fabric,
                model=self.actor,
                optimizer=self.actor_optimizer,
                model_name="actor",
            )
            iter_log_dict.update(actor_grad_clip_dict)
            self.actor_optimizer.step()
            iter_log_dict["actor/update_skipped"] = torch.tensor(
                0.0, device=self.device
            )
        else:
            iter_log_dict["actor/update_skipped"] = torch.tensor(
                1.0, device=self.device
            )

        # Update critic
        critic_loss, critic_loss_dict = self.critic_step(batch_dict)
        iter_log_dict.update(critic_loss_dict)
        self.critic_optimizer.zero_grad(set_to_none=True)
        self.fabric.backward(critic_loss)
        critic_grad_clip_dict = handle_model_grad_clipping(
            config=self.config,
            fabric=self.fabric,
            model=self.critic,
            optimizer=self.critic_optimizer,
            model_name="critic",
        )
        iter_log_dict.update(critic_grad_clip_dict)
        self.critic_optimizer.step()

        return iter_log_dict

    def actor_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        """Compute actor loss and perform policy update.

        Computes PPO clipped surrogate objective plus optional bounds loss and
        extra algorithm-specific losses.

        Args:
            batch_dict: Minibatch containing obs, actions, old neglogp, advantages.

        Returns:
            Tuple of (actor_loss, log_dict) where:
                - actor_loss: Total actor loss for backprop
                - log_dict: Dictionary of actor metrics for logging
        """
        # Forward through actor to get current policy's distribution
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.actor(batch_td)

        mean_action = batch_td["mean_action"]

        # Recompute neglogp for the actions that were actually taken (from experience buffer)
        # We need the current policy's evaluation, not the sampled action's neglogp
        mu = mean_action  # Already tanh-bounded
        std = torch.exp(self.actor.logstd)
        dist = torch.distributions.Normal(mu, std)
        current_neglogp = -dist.log_prob(batch_dict["action"]).sum(dim=-1)

        # Compute probability ratio between new and old policy
        ratio = torch.exp(batch_dict["neglogp"] - current_neglogp)
        surr1 = batch_dict["advantages"] * ratio
        surr2 = batch_dict["advantages"] * torch.clamp(
            ratio, 1.0 - self.e_clip, 1.0 + self.e_clip
        )
        ppo_loss = torch.max(-surr1, -surr2)
        clipped = torch.abs(ratio - 1.0) > self.e_clip
        clipped = clipped.detach().float().mean()

        if self.config.bounds_loss_coef > 0:
            b_loss: Tensor = bounds_loss(mean_action) * self.config.bounds_loss_coef
        else:
            b_loss = torch.zeros(self.num_envs, device=self.device)

        actor_ppo_loss = ppo_loss.mean()
        b_loss = b_loss.mean()
        extra_loss, extra_actor_log_dict = self.calculate_extra_actor_loss(batch_td)

        actor_loss = actor_ppo_loss + b_loss + extra_loss

        log_dict = {
            "actor/ppo_loss": actor_ppo_loss.detach(),
            "actor/bounds_loss": b_loss.detach(),
            "actor/extra_loss": extra_loss.detach(),
            "actor/clip_frac": clipped.detach(),
            "losses/actor_loss": actor_loss.detach(),
        }
        log_dict.update(extra_actor_log_dict)

        # Memory optimization: Detach intermediate tensors that won't be used for gradients
        # This prevents unnecessary gradient graph retention
        ratio = ratio.detach()
        surr1 = surr1.detach()
        surr2 = surr2.detach()
        ppo_loss = ppo_loss.detach()

        return actor_loss, log_dict

    def calculate_extra_actor_loss(self, batch_td) -> Tuple[Tensor, Dict]:
        """Calculate additional actor losses beyond PPO objective.

        Subclasses can override to add custom actor losses (e.g., entropy bonus,
        auxiliary losses). Default implementation returns zero loss.

        Args:
            batch_td: Minibatch data.

        Returns:
            Tuple of (extra_loss, log_dict) with additional loss and metrics.
        """
        return torch.tensor(0.0, device=self.device), {}

    def critic_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        # Convert to TensorDict for model processing
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.critic(batch_td)
        values = batch_td["value"]

        if self.config.clip_critic_loss:
            critic_loss_unclipped = (values - batch_dict["returns"].unsqueeze(-1)).pow(
                2
            )
            v_clipped = batch_dict["value"] + torch.clamp(
                values - batch_dict["value"],
                -self.config.e_clip,
                self.config.e_clip,
            )
            critic_loss_clipped = (v_clipped - batch_dict["returns"].unsqueeze(-1)).pow(
                2
            )
            critic_loss_max = torch.max(critic_loss_unclipped, critic_loss_clipped)
            critic_loss = critic_loss_max.mean()

            # Memory optimization: Detach intermediate tensors
            critic_loss_unclipped = critic_loss_unclipped.detach()
            v_clipped = v_clipped.detach()
            critic_loss_clipped = critic_loss_clipped.detach()
            critic_loss_max = critic_loss_max.detach()
        else:
            critic_loss = (batch_dict["returns"].unsqueeze(-1) - values).pow(2).mean()

        log_dict = {"losses/critic_loss": critic_loss.detach()}

        # Memory optimization: Detach values tensor if not needed for gradients
        values = values.detach()

        return critic_loss, log_dict

    # -----------------------------
    # Optimization Override
    # -----------------------------
    def optimize_model(self) -> Dict:
        # Reset epoch-level actor skip flag
        self._skip_actor_for_epoch = False

        training_log_dict = super().optimize_model()
        # Merge advantage normalization logs if available
        if hasattr(self, "_adv_norm_log"):
            training_log_dict.update(self._adv_norm_log)
        return training_log_dict

    # -----------------------------
    # Helper Functions
    # -----------------------------
    @torch.no_grad()
    def compute_advantages(self):
        """Compute GAE advantages and returns, storing them in experience buffer."""
        dones = self.experience_buffer.dones

        if self.config.normalize_rewards:
            rewards = self.experience_buffer.unnormalized_rewards
            values = self.experience_buffer.unnormalized_value.squeeze(-1)
            next_values = self.experience_buffer.unnormalized_next_value.squeeze(-1)
        else:
            rewards = self.experience_buffer.rewards
            values = self.experience_buffer.value.squeeze(-1)
            next_values = self.experience_buffer.next_value.squeeze(-1)

        advantages = discount_values(
            dones, values, rewards, next_values, self.gamma, self.tau
        )
        returns = advantages + values

        if self.config.normalize_rewards:
            returns = self.running_reward_norm.normalize(returns)

        assert torch.all(torch.isfinite(returns)), f"Returns are not finite: {returns}"
        
        return {
            "returns": returns,
            "advantages": advantages * self.config.task_reward_w,
        }

    @torch.no_grad()
    def pre_process_dataset(self):
        """Pre-process the dataset to compute advantages and returns."""
        advantages_dict = self.compute_advantages()
        for key, value in advantages_dict.items():
            self.experience_buffer.batch_update_data(key, value)
        
        advantages = self.experience_buffer.advantages

        adv_norm_log = {}
        # Log raw advantage stats
        adv_norm_log["adv_norm/raw_mean"] = advantages.mean().item()
        adv_norm_log["adv_norm/raw_std"] = advantages.std().item()
        adv_norm_log["adv_norm/raw_min"] = advantages.min().item()
        adv_norm_log["adv_norm/raw_max"] = advantages.max().item()
        if self.config.advantage_normalization.enabled:
            if self.config.advantage_normalization.use_ema:
                # EMA-based advantage normalization with clamping

                # Apply safety minimum std to prevent extreme normalization
                adv_std_safe = torch.clamp(
                    self.adv_std_ema, min=self.config.advantage_normalization.min_std
                )

                # Normalize advantages using EMA stats (from previous iteration)
                if self.config.advantage_normalization.shift_mean:
                    advantages_normalized = (
                        advantages - self.adv_mean_ema
                    ) / adv_std_safe
                else:
                    advantages_normalized = advantages / adv_std_safe

                # Clamp normalized advantages (z-scores)
                clamp_range = self.config.advantage_normalization.clamp_range
                advantages_clamped = torch.clamp(
                    advantages_normalized, -clamp_range, clamp_range
                )

                # Compute clamp fraction for logging
                clamp_frac = (
                    (torch.abs(advantages_normalized) > clamp_range).float().mean()
                )

                # De-normalize the clamped values to get the actual values used
                if self.config.advantage_normalization.shift_mean:
                    advantages_denorm = (
                        advantages_clamped * adv_std_safe + self.adv_mean_ema
                    )
                else:
                    advantages_denorm = advantages_clamped * adv_std_safe

                # Update EMA on GPU 0 using de-normalized clamped advantages
                if self.fabric.global_rank == 0:
                    batch_mean = advantages_denorm.mean()
                    batch_std = advantages_denorm.std() + 1e-8
                    # EMA update: new = alpha * batch + (1 - alpha) * old
                    ema_alpha = self.config.advantage_normalization.ema_alpha
                    self.adv_mean_ema = (
                        ema_alpha * batch_mean + (1 - ema_alpha) * self.adv_mean_ema
                    )
                    self.adv_std_ema = (
                        ema_alpha * batch_std + (1 - ema_alpha) * self.adv_std_ema
                    )

                # Broadcast EMA stats from GPU 0 to all GPUs
                if self.fabric.world_size > 1 and torch.distributed.is_initialized():
                    torch.distributed.broadcast(self.adv_mean_ema, src=0)
                    torch.distributed.broadcast(self.adv_std_ema, src=0)

                # Store advantage normalization stats for logging
                adv_norm_log["adv_norm/mean_ema"] = self.adv_mean_ema.item()
                adv_norm_log["adv_norm/std_ema"] = self.adv_std_ema.item()
                adv_norm_log["adv_norm/clamp_frac"] = clamp_frac.item()

                advantages = advantages_clamped
            else:
                # Original batch-based advantage normalization (no logging)
                mean = advantages.mean()
                var = advantages.var()
                count = advantages.numel()

                if self.fabric.world_size > 1:
                    all_means = self.fabric.all_gather(mean)
                    all_vars = self.fabric.all_gather(var)
                    all_counts = self.fabric.all_gather(count)

                    if self.fabric.global_rank == 0:
                        mean, var, count = combine_moments(
                            all_means, all_vars, all_counts
                        )

                # Fabric broadcast returns a tensor on the source rank, so we need to move it to the device of the current rank.
                updated_mean = self.fabric.broadcast(mean, src=0).to(self.device)
                updated_var = self.fabric.broadcast(var, src=0).to(self.device)

                if self.config.advantage_normalization.shift_mean:
                    advantages = (advantages - updated_mean) / (
                        torch.sqrt(updated_var) + 1e-8
                    )
                else:
                    advantages = advantages / (torch.sqrt(updated_var) + 1e-8)

        assert torch.all(
            torch.isfinite(advantages)
        ), f"Advantages are not finite: {advantages}"
        self.experience_buffer.batch_update_data("advantages", advantages)

        # Store logs for later use
        self._adv_norm_log = adv_norm_log
