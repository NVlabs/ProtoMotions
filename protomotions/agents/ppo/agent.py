# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

from protomotions.utils.hydra_replacement import get_class

from protomotions.agents.ppo.model import PPOModel
from protomotions.agents.common.common import MODULE_INTERNALS_KEY, weight_init_trainable
from protomotions.agents.optimizer.factory import (
    instantiate_optimizer,
    optimizer_learning_rate,
    scale_optimizer_learning_rates,
)
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.utils.normalization import combine_moments
from protomotions.agents.ppo.utils import discount_values
from protomotions.agents.ppo.config import PPOAgentConfig
from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.utils.training import bounds_loss

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

    @property
    def has_critic(self) -> bool:
        """PPO owns a value critic by construction."""
        return True

    def create_model(self):
        """Create PPO actor-critic model.

        Instantiates the PPO model with actor and critic networks, applies
        weight initialization, and returns the model.

        Returns:
            PPOModel instance with initialized weights.
        """
        PPOModelClass = get_class(self.config.model._target_)
        model: PPOModel = PPOModelClass(config=self.config.model)
        model.apply(weight_init_trainable)
        return model

    def create_optimizers(self, model: PPOModel):
        """Create separate optimizers for actor and critic.

        Uses Fabric to prepare both model/optimizer pairs for distributed
        training, matching the public PPO setup path.

        Args:
            model: PPOModel with actor and critic networks.
        """
        actor_optimizer = instantiate_optimizer(
            self.config.model.actor_optimizer,
            model._actor,
        )
        self.actor, self.actor_optimizer = self._setup_model_optimizer(
            model._actor, actor_optimizer
        )

        critic_optimizer = instantiate_optimizer(
            self.config.model.critic_optimizer,
            model._critic,
        )
        self.critic, self.critic_optimizer = self._setup_model_optimizer(
            model._critic, critic_optimizer
        )

        # Store initial learning rates for adaptive KL scheduling
        if self.config.adaptive_lr.enabled:
            self.actor_lr = optimizer_learning_rate(
                self.config.model.actor_optimizer, self.actor_optimizer
            )
            self.critic_lr = optimizer_learning_rate(
                self.config.model.critic_optimizer, self.critic_optimizer
            )

    @property
    def actor_module(self):
        """Underlying actor module, unwrapped from DDP when needed."""
        return self.actor.module if hasattr(self.actor, "module") else self.actor

    @property
    def critic_module(self):
        """Underlying critic module, unwrapped from DDP when needed."""
        return self.critic.module if hasattr(self.critic, "module") else self.critic

    def _load_model_state_dict(self, model_state_dict):
        # Save current logstd value to preserve config overrides.
        # Only override the checkpoint's logstd when std is NOT learnable
        # (i.e., a fixed hyperparameter that may have been changed via CLI).
        # When learnable_std=True, the checkpoint contains the trained value
        # and must not be overwritten.
        has_fixed_logstd = not self.config.model.actor.learnable_std
        if has_fixed_logstd:
            current_logstd = self.actor_module.logstd.data.clone()
            current_actor_logstd_config = self.config.model.actor.actor_logstd

        super()._load_model_state_dict(model_state_dict)

        if has_fixed_logstd:
            checkpoint_logstd = self.actor_module.logstd.data
            # Fixed std: preserve config override if it differs from checkpoint
            if not torch.allclose(current_logstd, checkpoint_logstd, atol=1e-6):
                print(
                    f"Preserving overridden actor_logstd: {current_actor_logstd_config} "
                    f"(checkpoint had: {checkpoint_logstd[0].item():.3f})"
                )
                self.actor_module.logstd.data = current_logstd

    def _load_training_state(self, state_dict):
        """Restore PPO optimizer and advantage-normalization state."""
        super()._load_training_state(state_dict)
        self._load_ppo_training_state(state_dict, require_optimizers=True)

    def _load_ppo_training_state(self, state_dict, require_optimizers: bool):
        if require_optimizers or "actor_optimizer" in state_dict:
            self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        if require_optimizers or "critic_optimizer" in state_dict:
            self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])

        # Restore adaptive LR state
        if self.config.adaptive_lr.enabled and "adaptive_lr" in state_dict:
            old_actor_lr = getattr(
                self,
                "actor_lr",
                optimizer_learning_rate(
                    self.config.model.actor_optimizer,
                    self.actor_optimizer,
                ),
            )
            old_critic_lr = getattr(
                self,
                "critic_lr",
                optimizer_learning_rate(
                    self.config.model.critic_optimizer,
                    self.critic_optimizer,
                ),
            )
            self.actor_lr = state_dict["adaptive_lr"]["actor_lr"]
            self.critic_lr = state_dict["adaptive_lr"]["critic_lr"]
            scale_optimizer_learning_rates(
                self.actor_optimizer,
                old_lr=old_actor_lr,
                new_lr=self.actor_lr,
            )
            scale_optimizer_learning_rates(
                self.critic_optimizer,
                old_lr=old_critic_lr,
                new_lr=self.critic_lr,
            )

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

        # Save adaptive LR state
        if self.config.adaptive_lr.enabled:
            extra_state_dict["adaptive_lr"] = {
                "actor_lr": self.actor_lr,
                "critic_lr": self.critic_lr,
            }

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
        self.experience_buffer.batch_update_data(
            "unnormalized_next_value", unnorm_next_value
        )

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

        # Adaptive learning rate based on KL divergence
        if self.config.adaptive_lr.enabled and "actor/kl" in actor_loss_dict:
            self._update_learning_rate(actor_loss_dict["actor/kl"])
            iter_log_dict["info/actor_lr"] = torch.tensor(
                self.actor_lr, device=self.device
            )
            iter_log_dict["info/critic_lr"] = torch.tensor(
                self.critic_lr, device=self.device
            )

        # Check if we should skip actor update for this epoch
        # Once triggered, skip all remaining batches (same distribution)
        if (
            not self._skip_actor_for_epoch
            and self.config.actor_clip_frac_threshold is not None
        ):
            clip_frac = actor_loss_dict["actor/clip_frac"].item()
            # Synchronize clip_frac across all GPUs (weighted by batch size)
            if self.fabric.world_size > 1:
                batch_size = batch_dict["action"].shape[0]
                clip_sum = torch.tensor(
                    clip_frac * batch_size, device=self.device, dtype=torch.float32
                )
                clip_count = torch.tensor(
                    batch_size, device=self.device, dtype=torch.float32
                )
                all_sums = self.fabric.all_gather(clip_sum)
                all_counts = self.fabric.all_gather(clip_count)
                clip_frac = (all_sums.sum() / all_counts.sum()).item()

            if clip_frac > self.config.actor_clip_frac_threshold:
                self._skip_actor_for_epoch = True
                if self.fabric.global_rank == 0:
                    log.warning(
                        f"Epoch {self.current_epoch}: Skipping actor updates for remaining batches "
                        f"(clip_frac {clip_frac:.3f} > {self.config.actor_clip_frac_threshold})"
                    )

        if self._skip_actor_for_epoch:
            iter_log_dict["actor/update_skipped"] = torch.tensor(
                1.0, device=self.device
            )
            return iter_log_dict

        actor_grad_clip_dict = self._step_optimizer(
            loss=actor_loss,
            model=self.actor,
            optimizer=self.actor_optimizer,
            model_name="actor",
        )
        iter_log_dict.update(actor_grad_clip_dict)
        iter_log_dict["actor/update_skipped"] = torch.tensor(0.0, device=self.device)

        critic_loss, critic_loss_dict = self.critic_step(batch_dict)
        iter_log_dict.update(critic_loss_dict)
        critic_grad_clip_dict = self._step_optimizer(
            loss=critic_loss,
            model=self.critic,
            optimizer=self.critic_optimizer,
            model_name="critic",
        )
        iter_log_dict.update(critic_grad_clip_dict)

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
        batch_td = self.actor(batch_td, log_internals=True)

        mean_action = batch_td["mean_action"]

        # Recompute neglogp for the actions that were actually taken (from experience buffer)
        # We need the current policy's evaluation, not the sampled action's neglogp
        mu = mean_action  # Already tanh-bounded
        std = torch.exp(self.actor_module.logstd)
        dist = torch.distributions.Normal(mu, mu * 0 + std)
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
        model_loss, model_log_dict = self.actor_module.compute_model_loss(
            batch_td,
            current_epoch=self.current_epoch,
            zero_loss=actor_ppo_loss,
            log_prefix="actor_model",
        )

        actor_loss = actor_ppo_loss + b_loss + extra_loss + model_loss

        # Entropy bonus for learnable std exploration noise
        if self.config.model.actor.learnable_std:
            entropy_loss = dist.entropy().sum(dim=-1).mean()
            actor_loss = actor_loss - self.config.entropy_coef * entropy_loss
        else:
            entropy_loss = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            approx_kl = (ratio - 1.0 - torch.log(ratio + 1e-8)).mean()

        log_dict = {
            "actor/ppo_loss": actor_ppo_loss.detach(),
            "actor/bounds_loss": b_loss.detach(),
            "actor/extra_loss": extra_loss.detach(),
            "actor/model_loss": model_loss.detach(),
            "actor/entropy_loss": entropy_loss.detach(),
            "actor/clip_frac": clipped.detach(),
            "actor/approx_kl": approx_kl.detach(),
            "actor/ratio_mean": ratio.mean().detach(),
            "actor/ratio_std": ratio.std().detach(),
            "actor/ratio_max": ratio.max().detach(),
            "losses/actor_loss": actor_loss.detach(),
        }
        if self.config.model.actor.learnable_std:
            log_dict["actor/std_mean"] = std.mean().detach()
        module_internals = batch_td.get(MODULE_INTERNALS_KEY, None)
        if module_internals is not None:
            for key, value in module_internals.items():
                if isinstance(value, Tensor):
                    log_dict[f"actor/internals/{key}"] = (
                        value.float().mean().detach()
                    )
        log_dict.update(model_log_dict)
        log_dict.update(extra_actor_log_dict)

        # Compute KL divergence for adaptive learning rate
        if self.config.adaptive_lr.enabled:
            kl_mean = self._compute_kl(
                batch_dict["mean_action"].detach(),
                mu.detach(),
                std.detach(),
            )
            log_dict["actor/kl"] = kl_mean

        # Memory optimization: Detach intermediate tensors that won't be used for gradients
        # This prevents unnecessary gradient graph retention
        ratio = ratio.detach()
        surr1 = surr1.detach()
        surr2 = surr2.detach()
        ppo_loss = ppo_loss.detach()

        return actor_loss, log_dict

    def calculate_extra_actor_loss(self, batch_td) -> Tuple[Tensor, Dict]:
        """Calculate additional actor losses beyond PPO objective.

        Supports L2C2 regularization: penalizes Lipschitz ratio between actor
        outputs on noisy vs clean observations (Kobayashi 2022).

        Args:
            batch_td: Minibatch data (post actor forward, contains mean_action).

        Returns:
            Tuple of (extra_loss, log_dict) with additional loss and metrics.
        """
        extra_loss = torch.tensor(0.0, device=self.device)
        log_dict = {}

        # --- L2C2 ---
        if self.config.l2c2.enabled:
            mu_noisy = batch_td["mean_action"]

            # Build clean TensorDict and accumulate input perturbation
            input_ss = torch.tensor(0.0, device=self.device)
            input_n = 0
            clean_td_dict = {}
            for key in self.actor_module.in_keys:
                if key in self.config.l2c2.obs_pairs:
                    clean_key = self.config.l2c2.obs_pairs[key]
                    clean_td_dict[key] = batch_td[clean_key]
                    diff = batch_td[key] - batch_td[clean_key]
                    input_ss = input_ss + diff.pow(2).sum()
                    input_n += diff.numel()
                else:
                    clean_td_dict[key] = batch_td[key]
            clean_td = TensorDict(clean_td_dict, batch_size=mu_noisy.shape[0])

            input_dist = (input_ss / input_n).detach()

            clean_td = self.actor(clean_td)
            mu_clean = clean_td["mean_action"]

            output_dist = (mu_noisy - mu_clean).pow(2).mean()
            l2c2_loss = output_dist / (input_dist + 1e-8)
            l2c2_weighted = self.config.l2c2.lambda_l2c2 * l2c2_loss

            extra_loss = extra_loss + l2c2_weighted
            log_dict.update(
                {
                    "actor/l2c2_loss": l2c2_loss.detach(),
                    "actor/l2c2_weighted": l2c2_weighted.detach(),
                    "actor/l2c2_input_dist": input_dist.detach(),
                    "actor/l2c2_output_dist": output_dist.detach(),
                }
            )

        return extra_loss, log_dict

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

        model_loss, model_log_dict = self.critic_module.compute_model_loss(
            batch_td,
            current_epoch=self.current_epoch,
            zero_loss=critic_loss,
            log_prefix="critic_model",
        )
        critic_loss = critic_loss + model_loss

        with torch.no_grad():
            returns = batch_dict["returns"].unsqueeze(-1)
            errors = values.detach() - returns
            return_var = returns.var()
            explained_var = (
                1.0 - errors.var() / (return_var + 1e-8) if return_var > 1e-8 else torch.zeros(1, device=values.device)
            )

        log_dict = {
            "losses/critic_loss": critic_loss.detach(),
            "critic/model_loss": model_loss.detach(),
            "critic/explained_variance": explained_var,
            "critic/value_mean": values.detach().mean(),
            "critic/value_std": values.detach().std(),
            "critic/return_mean": returns.mean(),
            "critic/return_std": returns.std(),
            "critic/error_mean": errors.mean(),
            "critic/error_std": errors.std(),
        }
        log_dict.update(model_log_dict)

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
    def _compute_kl(self, old_mu, new_mu, std):
        """Compute mean KL divergence between old and new policy distributions.

        Uses the current std for both distributions (exact when learnable_std=False,
        close approximation when learnable_std=True since std changes slowly).

        Args:
            old_mu: Mean actions from rollout (experience buffer).
            new_mu: Mean actions from current policy.
            std: Current policy standard deviation.

        Returns:
            Scalar tensor with mean KL divergence (summed over action dims, averaged over batch).
        """
        old_dist = torch.distributions.Normal(old_mu, std)
        new_dist = torch.distributions.Normal(new_mu, std)
        kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(-1)
        kl_sum = kl.sum()
        kl_count = torch.tensor(kl.numel(), dtype=kl_sum.dtype, device=kl_sum.device)
        if self.fabric.world_size > 1:
            all_sums = self.fabric.all_gather(kl_sum)
            all_counts = self.fabric.all_gather(kl_count)
            return all_sums.sum() / all_counts.sum()
        return kl_sum / kl_count

    def _update_learning_rate(self, kl_mean):
        """Adjust actor and critic learning rates based on KL divergence.

        If KL exceeds 2x the target, learning rates are decreased by 1.5x.
        If KL is below 0.5x the target, learning rates are increased by 1.5x.
        Learning rates are clamped to config min/max bounds.

        Args:
            kl_mean: Mean KL divergence from _compute_kl.
        """
        old_actor_lr = self.actor_lr
        old_critic_lr = self.critic_lr
        if kl_mean > self.config.adaptive_lr.desired_kl * 2.0:
            self.actor_lr = max(self.config.adaptive_lr.min_lr, self.actor_lr / 1.5)
            self.critic_lr = max(self.config.adaptive_lr.min_lr, self.critic_lr / 1.5)
        elif kl_mean < self.config.adaptive_lr.desired_kl / 2.0 and kl_mean > 0.0:
            self.actor_lr = min(self.config.adaptive_lr.max_lr, self.actor_lr * 1.5)
            self.critic_lr = min(self.config.adaptive_lr.max_lr, self.critic_lr * 1.5)

        scale_optimizer_learning_rates(
            self.actor_optimizer,
            old_lr=old_actor_lr,
            new_lr=self.actor_lr,
        )
        scale_optimizer_learning_rates(
            self.critic_optimizer,
            old_lr=old_critic_lr,
            new_lr=self.critic_lr,
        )

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
