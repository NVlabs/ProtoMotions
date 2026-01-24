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
"""Adversarial Skill Embeddings (ASE) agent implementation.

This module implements the ASE algorithm which extends AMP with learned skill embeddings.
The discriminator encodes motions into a latent skill space, and the policy is conditioned
on these latent codes. This enables learning diverse skills from motion data and composing
them for complex tasks.

Key Classes:
    - ASE: Main ASE agent class extending AMP

References:
    Peng et al. "ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters" (2022)
"""

import torch
import logging

from torch import Tensor
from tensordict import TensorDict

from lightning.fabric import Fabric

from protomotions.agents.ase.model import ASEModel, ASEDiscriminatorEncoder
from protomotions.agents.utils.step_tracker import StepTracker
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.amp.agent import AMP
from typing import Tuple, Dict, Optional
from pathlib import Path
from protomotions.agents.ase.config import ASEAgentConfig
from protomotions.agents.utils.normalization import RewardRunningMeanStd
from protomotions.agents.ppo.utils import discount_values
from protomotions.utils.hydra_replacement import instantiate
from protomotions.agents.utils.training import handle_model_grad_clipping


log = logging.getLogger(__name__)


class ASE(AMP):
    """Adversarial Skill Embeddings (ASE) agent.

    Extends AMP with a low-level policy conditioned on learned skill embeddings.
    The discriminator learns to encode skills from motion data into a latent space,
    while the policy learns to execute behaviors conditioned on these latent codes.
    This enables learning diverse skills from motion data and composing them for tasks.

    Key components:
    - **Low-level policy**: Conditioned on latent skill codes
    - **Discriminator**: Encodes motions into skill embeddings
    - **Mutual information**: Encourages skill diversity
    - **Latent sampling**: Periodically samples new skills during rollouts

    Args:
        fabric: Lightning Fabric instance for distributed training.
        env: Environment instance with diverse motion library.
        config: ASE-specific configuration including latent dimensions.
        root_dir: Optional root directory for saving outputs.

    Attributes:
        latents: Current latent skill codes for each environment.
        latent_reset_steps: Steps until next latent resample.

    Example:
        >>> fabric = Fabric(devices=4)
        >>> env = Mimic(config, robot_config, simulator_config, device)
        >>> agent = ASE(fabric, env, config)
        >>> agent.setup()
        >>> agent.train()

    Note:
        Requires large diverse motion dataset for effective skill learning.
    """

    model: ASEModel
    discriminator: ASEDiscriminatorEncoder
    config: ASEAgentConfig

    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def __init__(
        self, fabric: Fabric, env: BaseEnv, config, root_dir: Optional[Path] = None
    ):
        super().__init__(fabric, env, config, root_dir=root_dir)
        if self.config.normalize_rewards:
            self.running_mi_enc_norm = RewardRunningMeanStd(
                shape=(1,),
                fabric=self.fabric,
                gamma=self.gamma,
                device=self.device,
                clamp_value=self.config.normalized_reward_clamp_value,
            )
        else:
            self.running_mi_enc_norm = None

    def setup(self):
        self.latents = torch.zeros(
            (self.num_envs, self.config.ase_parameters.latent_dim),
            dtype=torch.float,
            device=self.device,
        )

        self.latent_reset_steps = StepTracker(
            self.num_envs,
            min_steps=self.config.ase_parameters.latent_steps_min,
            max_steps=self.config.ase_parameters.latent_steps_max,
            device=self.device,
        )
        super().setup()

    def create_optimizers(self, model: ASEModel):
        super().create_optimizers(model)
        mi_critic_optimizer = instantiate(
            self.config.model.mi_critic_optimizer,
            params=list(model._mi_critic.parameters()),
        )
        self.mi_critic, self.mi_critic_optimizer = self.fabric.setup(
            model._mi_critic, mi_critic_optimizer
        )

    def load_parameters(self, state_dict):
        super().load_parameters(state_dict)
        self.mi_critic_optimizer.load_state_dict(state_dict["mi_critic_optimizer"])
        if self.config.normalize_rewards:
            self.running_mi_enc_norm.load_state_dict(state_dict["running_mi_enc_norm"])

    def get_state_dict(self, state_dict):
        extra_state_dict = super().get_state_dict(state_dict)
        extra_state_dict["mi_critic_optimizer"] = self.mi_critic_optimizer.state_dict()
        if self.config.normalize_rewards:
            extra_state_dict["running_mi_enc_norm"] = (
                self.running_mi_enc_norm.state_dict()
            )
        state_dict.update(extra_state_dict)
        return state_dict

    # -----------------------------
    # Latent Management
    # -----------------------------
    def update_latents(self):
        """Updates latent variables based on latent reset steps."""
        self.latent_reset_steps.advance()
        reset_ids = self.latent_reset_steps.done_indices()

        if reset_ids.numel() > 0:
            self.reset_latents(reset_ids)
            self.latent_reset_steps.reset_steps(reset_ids)

    def reset_latents(self, env_ids=None):
        """Resets latent variables for specified environments or all environments if None.

        Args:
            env_ids (torch.Tensor, optional): Environment indices to reset latents for. Defaults to None (all envs).
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        latents = self.sample_latents(len(env_ids))
        self.store_latents(latents, env_ids)

    def store_latents(self, latents, env_ids):
        """Stores latent variables for specified environments.

        Args:
            latents (torch.Tensor): Latent variables to store. Shape (num_envs, latent_dim).
            env_ids (torch.Tensor): Environment indices to store latents for. Shape (num_envs,).
        """
        self.latents[env_ids] = latents

    def sample_latents(self, n):
        """Samples new latent variables uniformly on the unit-sphere.

        Args:
            n (int): Number of latent variables to sample.

        Returns:
            torch.Tensor: Sampled latent variables. Shape (n, latent_dim).
        """
        zeros = torch.zeros(
            [n, self.config.ase_parameters.latent_dim], device=self.device
        )

        gaussian_sample = torch.normal(zeros)
        latents = torch.nn.functional.normalize(gaussian_sample, dim=-1)
        return latents

    def mi_enc_forward(self, obs_dict: dict) -> Tensor:
        """Forward pass through the Mutual Information encoder.

        Args:
            obs_dict: Dictionary containing observations.

        Returns:
            Tensor: Encoded observation tensor. Shape (batch_size, encoder_output_dim).
        """
        # Convert to TensorDict and forward through discriminator
        batch_size = obs_dict[list(obs_dict.keys())[0]].shape[0]
        obs_td = TensorDict(obs_dict, batch_size=batch_size)
        obs_td = self.discriminator(obs_td)
        return obs_td["mi_enc_output"]

    # -----------------------------
    # Experience Buffer and Dataset Processing
    # -----------------------------
    def register_algorithm_experience_buffer_keys(self):
        super().register_algorithm_experience_buffer_keys()
        if self.config.normalize_rewards:
            self.experience_buffer.register_key("unnormalized_mi_rewards")
        self.experience_buffer.register_key("mi_rewards")

        value_shape = getattr(self.experience_buffer, "value").shape[2:]
        self.experience_buffer.register_key("next_mi_value", shape=value_shape)
        self.experience_buffer.register_key("mi_returns")
        if self.config.normalize_rewards:
            self.experience_buffer.register_key("unnormalized_mi_value", shape=value_shape)
            self.experience_buffer.register_key("unnormalized_next_mi_value", shape=value_shape)

    # -----------------------------
    # Environment Interaction
    # -----------------------------
    def add_agent_info_to_obs(self, obs):
        """Perform an environment step and inject current latents into observations."""
        obs = super().add_agent_info_to_obs(obs)
        self.update_latents()
        obs["latents"] = self.latents.clone()
        return obs

    # -----------------------------
    # Reward Calculation
    # -----------------------------
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

        next_obs_td = self.discriminator(next_obs_td)

        mi_r = self.discriminator.compute_mi_reward(
            next_obs_td, self.config.ase_parameters.mi_hypersphere_reward_shift
        ).view(-1)

        next_mi_value = self.mi_critic(next_obs_td)[self.mi_critic.config.out_keys[0]]
        next_mi_value = next_mi_value * (1 - terminated.float()).unsqueeze(-1)
        self.experience_buffer.update_data("next_mi_value", step, next_mi_value)

        if self.config.normalize_rewards:
            self.running_mi_enc_norm.record_reward(mi_r, terminated)
        self.experience_buffer.update_data("mi_rewards", step, mi_r)

    @torch.no_grad()
    def normalize_rewards_in_buffer(self):
        super().normalize_rewards_in_buffer()
        if not self.config.normalize_rewards:
            return

        mi_rewards = self.experience_buffer.mi_rewards
        self.experience_buffer.batch_update_data(
            "unnormalized_mi_rewards", mi_rewards.clone()
        )
        self.experience_buffer.batch_update_data(
            "mi_rewards", self.running_mi_enc_norm.normalize(mi_rewards)
        )

        mi_value = self.experience_buffer.mi_value
        unnorm_mi_value = self.running_mi_enc_norm.normalize(mi_value, un_norm=True)
        self.experience_buffer.batch_update_data("unnormalized_mi_value", unnorm_mi_value)

        next_mi_value = self.experience_buffer.next_mi_value
        unnorm_next_mi_value = self.running_mi_enc_norm.normalize(
            next_mi_value, un_norm=True
        )
        self.experience_buffer.batch_update_data(
            "unnormalized_next_mi_value", unnorm_next_mi_value
        )

    @torch.no_grad()
    def compute_advantages(self):
        advantages_dict = super().compute_advantages()
        dones = self.experience_buffer.dones

        if self.config.normalize_rewards:
            mi_rewards = self.experience_buffer.unnormalized_mi_rewards
            mi_values = self.experience_buffer.unnormalized_mi_value.squeeze(-1)
            mi_next_values = self.experience_buffer.unnormalized_next_mi_value.squeeze(-1)
        else:
            mi_rewards = self.experience_buffer.mi_rewards
            mi_values = self.experience_buffer.mi_value.squeeze(-1)
            mi_next_values = self.experience_buffer.next_mi_value.squeeze(-1)

        mi_advantages = discount_values(
            dones, mi_values, mi_rewards, mi_next_values, self.gamma, self.tau
        )
        mi_returns = mi_advantages + mi_values

        if self.config.normalize_rewards:
            mi_returns = self.running_mi_enc_norm.normalize(mi_returns)

        self.experience_buffer.batch_update_data("mi_returns", mi_returns)

        advantages_dict["advantages"] = (
            advantages_dict["advantages"]
            + mi_advantages * self.config.ase_parameters.mi_reward_w
        )
        return advantages_dict

    # -----------------------------
    # Optimization
    # -----------------------------
    def perform_optimization_step(self, batch_dict, batch_idx: int) -> Dict:
        iter_log_dict = super().perform_optimization_step(batch_dict, batch_idx)

        mi_critic_loss, mi_critic_loss_dict = self.mi_critic_step(batch_dict)
        iter_log_dict.update(mi_critic_loss_dict)
        self.mi_critic_optimizer.zero_grad(set_to_none=True)
        self.fabric.backward(mi_critic_loss)
        mi_critic_grad_clip_dict = handle_model_grad_clipping(
            config=self.config,
            fabric=self.fabric,
            model=self.mi_critic,
            optimizer=self.mi_critic_optimizer,
            model_name="mi_critic",
        )
        iter_log_dict.update(mi_critic_grad_clip_dict)
        self.mi_critic_optimizer.step()

        return iter_log_dict

    def mi_critic_step(self, batch_dict) -> Tuple[Tensor, Dict]:
        batch_td = TensorDict(batch_dict, batch_size=batch_dict["action"].shape[0])
        batch_td = self.mi_critic(batch_td)
        values = batch_td[self.mi_critic.config.out_keys[0]]

        if self.config.clip_critic_loss:
            mi_critic_loss_unclipped = (
                values - batch_dict["mi_returns"].unsqueeze(-1)
            ).pow(2)
            v_clipped = batch_dict["mi_value"] + torch.clamp(
                values - batch_dict["mi_value"],
                -self.config.e_clip,
                self.config.e_clip,
            )
            mi_critic_loss_clipped = (
                v_clipped - batch_dict["mi_returns"].unsqueeze(-1)
            ).pow(2)
            mi_critic_loss_max = torch.max(mi_critic_loss_unclipped, mi_critic_loss_clipped)
            mi_critic_loss = mi_critic_loss_max.mean()
        else:
            mi_critic_loss = (
                (batch_dict["mi_returns"].unsqueeze(-1) - values).pow(2).mean()
            )

        log_dict = {"losses/mi_critic_loss": mi_critic_loss.detach()}
        return mi_critic_loss, log_dict

    def get_expert_disc_obs(self, num_samples: int):
        expert_disc_obs = super().get_expert_disc_obs(num_samples)
        # Add the predicted latents to the batch dict. This produces a conditional discriminator, conditioned on the latents.
        mi_enc_pred = self.mi_enc_forward(expert_disc_obs)
        expert_disc_obs["latents"] = mi_enc_pred.clone()
        return expert_disc_obs

    def produce_negative_expert_obs(self, batch_dict):
        negative_expert_obs = {}
        # Use discriminator's in_keys dynamically
        discriminator_keys = self.model._discriminator.in_keys
        for key in discriminator_keys:
            if key == "latents":
                continue  # Handle latents separately below
            negative_expert_obs[key] = batch_dict[f"expert_{key}"][
                : self.config.amp_parameters.discriminator_batch_size
            ]
        random_conditioned_latent = torch.rand_like(
            batch_dict["agent_latents"][: self.config.amp_parameters.discriminator_batch_size]
        )
        projected_latent = torch.nn.functional.normalize(
            random_conditioned_latent, dim=-1
        )
        negative_expert_obs["latents"] = projected_latent
        return negative_expert_obs

    def discriminator_step(self, batch_dict):
        """Performs a discriminator update step.

        Args:
            batch_dict (dict): Batch of data from the experience buffer.

        Returns:
            Tuple[Tensor, Dict]: Discriminator loss and logging dictionary.
        """
        discriminator_loss, discriminator_log_dict = super().discriminator_step(
            batch_dict
        )

        # Extract agent and expert observations dynamically (like AMP parent class)
        agent_obs = {}
        for key in batch_dict.keys():
            if key.startswith("agent_"):
                agent_obs[key.replace("agent_", "")] = batch_dict[key][
                    : self.config.amp_parameters.discriminator_batch_size
                ]

        expert_obs = {}
        for key in batch_dict.keys():
            if key.startswith("expert_"):
                expert_obs[key.replace("expert_", "")] = batch_dict[key][
                    : self.config.amp_parameters.discriminator_batch_size
                ]

        latents = agent_obs.get("latents", batch_dict.get("latents", None))
        if latents is None:
            raise KeyError("Could not find 'latents' in agent_obs or batch_dict")

        # Get the observation key (first non-latents key from discriminator)
        disc_in_keys = self.model._discriminator.in_keys
        obs_key = [k for k in disc_in_keys if k != "latents"][0]

        agent_obs_tensor = agent_obs[obs_key]
        expert_obs_tensor = expert_obs[obs_key]

        if self.config.ase_parameters.mi_enc_grad_penalty > 0:
            agent_obs_tensor.requires_grad_(True)

        # Compute MI encoder predictions for both agent and expert observations
        agent_disc_obs = {obs_key: agent_obs_tensor, "latents": latents}
        mi_enc_pred_agent = self.mi_enc_forward(agent_disc_obs)

        expert_disc_obs = {obs_key: expert_obs_tensor, "latents": latents}
        mi_enc_pred_expert = self.mi_enc_forward(expert_disc_obs)

        # Original MI encoder loss for agent observations
        mi_enc_err = self.discriminator.calc_von_mises_fisher_enc_error(
            mi_enc_pred_agent, latents
        )
        mi_enc_loss = torch.mean(mi_enc_err)

        # Uniformity loss for uniform coverage on unit sphere
        # Concatenate agent and expert encodings for uniform distribution
        all_encodings = torch.cat([mi_enc_pred_agent, mi_enc_pred_expert], dim=0)
        uniformity_loss = self.compute_uniformity_loss(all_encodings)
        total_mi_loss = (
            mi_enc_loss
            + uniformity_loss * self.config.ase_parameters.latent_uniformity_weight
        )

        if self.config.ase_parameters.mi_enc_weight_decay > 0:
            enc_weight_params = self.discriminator.enc_weights()
            total: Tensor = sum([p.pow(2).sum() for p in enc_weight_params])
            weight_decay_loss: Tensor = (
                total * self.config.ase_parameters.mi_enc_weight_decay
            )
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)

        if self.config.ase_parameters.mi_enc_grad_penalty > 0:
            mi_enc_obs_grad = torch.autograd.grad(
                mi_enc_err,
                agent_obs,
                grad_outputs=torch.ones_like(mi_enc_err),
                create_graph=True,
                retain_graph=True,
            )[0]

            mi_enc_obs_grad = torch.sum(torch.square(mi_enc_obs_grad), dim=-1)
            mi_enc_grad_penalty = torch.mean(mi_enc_obs_grad)

            grad_loss: Tensor = (
                mi_enc_grad_penalty * self.config.ase_parameters.mi_enc_grad_penalty
            )
        else:
            grad_loss = torch.tensor(0.0, device=self.device)

        mi_loss = total_mi_loss + weight_decay_loss + grad_loss

        log_dict = {
            "encoder/loss": mi_loss.detach(),
            "encoder/mi_enc_loss": mi_enc_loss.detach(),
            "encoder/uniformity_loss": uniformity_loss.detach(),
            "encoder/l2_loss": weight_decay_loss.detach(),
            "encoder/grad_penalty": grad_loss.detach(),
        }

        discriminator_log_dict.update(log_dict)

        return mi_loss + discriminator_loss, discriminator_log_dict

    def compute_uniformity_loss(self, encodings: Tensor) -> Tensor:
        """Computes uniformity loss to encourage uniform distribution on unit sphere.

        Args:
            encodings (Tensor): Normalized encodings on unit sphere. Shape (batch_size, latent_dim).

        Returns:
            Tensor: Uniformity loss value.
        """
        # Compute pairwise distances between all encodings
        pairwise_distances = torch.cdist(encodings, encodings, p=2)  # L2 distance

        # Convert distances to Gaussian kernel values
        t = self.config.ase_parameters.uniformity_kernel_scale
        kernel_values = torch.exp(-t * pairwise_distances**2)

        # Uniformity loss: log of average kernel value
        # Lower values indicate more uniform distribution
        uniformity_loss = torch.log(torch.mean(kernel_values))

        return uniformity_loss

    def calculate_extra_actor_loss(self, batch_td) -> Tuple[Tensor, Dict]:
        """Adds the diversity loss, if enabled.

        Args:
            batch_td (TensorDict): Batch of data from the experience buffer and the actor.

        Returns:
            Tuple[Tensor, Dict]: Extra actor loss and logging dictionary.
        """
        extra_loss, extra_actor_log_dict = super().calculate_extra_actor_loss(batch_td)

        if self.config.ase_parameters.diversity_bonus <= 0:
            return extra_loss, extra_actor_log_dict

        diversity_loss = self.diversity_loss(batch_td)

        extra_actor_log_dict["actor/diversity_loss"] = diversity_loss.detach()

        return (
            extra_loss + diversity_loss * self.config.ase_parameters.diversity_bonus,
            extra_actor_log_dict,
        )

    def diversity_loss(self, batch_td):
        """Calculates the diversity loss to encourage latents to lead to diverse behaviors.

        Args:
            batch_td (TensorDict): Batch of data from the experience buffer and the actor.

        Returns:
            Tensor: Diversity loss.
        """
        old_latents = batch_td["latents"].clone()
        clipped_old_mu = torch.clamp(batch_td["mean_action"], -1.0, 1.0)

        new_latents = self.sample_latents(batch_td.batch_size[0])
        batch_td["latents"] = new_latents
        batch_td = self.actor(batch_td)

        clipped_new_mu = torch.clamp(batch_td["mean_action"], -1.0, 1.0)

        mu_diff = clipped_new_mu - clipped_old_mu
        mu_diff = torch.mean(torch.square(mu_diff), dim=-1)

        z_diff = new_latents * old_latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        diversity_bonus = mu_diff / (z_diff + 1e-5)
        diversity_loss = torch.square(
            self.config.ase_parameters.diversity_tar - diversity_bonus
        ).mean()

        return diversity_loss

    # -----------------------------
    # Termination and Logging
    # -----------------------------
    def post_epoch_logging(self, training_log_dict: Dict):
        """Performs post epoch logging, including Mutual Information reward logging.

        Args:
            training_log_dict (Dict): Dictionary to update with logging information.
        """
        training_log_dict["rewards/mi_enc_rewards"] = (
            self.experience_buffer.mi_rewards.mean()
        )
        if self.config.normalize_rewards:
            training_log_dict["rewards/unnormalized_mi_enc_rewards"] = (
                self.experience_buffer.unnormalized_mi_rewards
            )

        super().post_epoch_logging(training_log_dict)
