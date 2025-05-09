import torch
import logging

from torch import Tensor

from lightning.fabric import Fabric

from protomotions.agents.ase.model import ASEModel
from protomotions.envs.base_env.env_utils.general import StepTracker
from protomotions.envs.base_env.env import BaseEnv
from protomotions.agents.amp.agent import AMP
from typing import Tuple, Dict


log = logging.getLogger(__name__)


class ASE(AMP):
    # -----------------------------
    # Initialization and Setup
    # -----------------------------
    def __init__(self, fabric: Fabric, env: BaseEnv, config):
        super().__init__(fabric, env, config)
        self.latents = None
        self.latent_reset_steps = None

    def setup(self):
        super().setup()
        self.model: ASEModel

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
        latents = torch.nn.functional.normalize(
            gaussian_sample, dim=-1
        )
        return latents

    def mi_enc_forward(self, obs: Tensor) -> Tensor:
        """Forward pass through the Mutual Information encoder.

        Args:
            obs (Tensor): Observation tensor. Shape (batch_size, obs_dim).

        Returns:
            Tensor: Encoded observation tensor. Shape (batch_size, encoder_output_dim).
        """
        return self.model._discriminator.mi_enc_forward(obs)

    # -----------------------------
    # Experience Buffer and Dataset Processing
    # -----------------------------
    def register_extra_experience_buffer_keys(self):
        super().register_extra_experience_buffer_keys()
        self.experience_buffer.register_key("mi_rewards")  # mi = mutual information
        self.experience_buffer.register_key(
            "latents", shape=(self.config.ase_parameters.latent_dim,)
        )

    def post_train_env_step(self, rewards, dones, done_indices, extras, step):
        """Post environment step processing during training. Updates experience buffer with latents and updates latents.

        Args:
            rewards (Tensor): Rewards from environment step. Shape (num_envs,).
            dones (Tensor): Done flags from environment step. Shape (num_envs,).
            done_indices (Tensor): Indices of environments that are done. Shape (num_done_envs,).
            extras (dict): Extra information from environment step.
            step (int): Current training step.
        """
        super().post_train_env_step(rewards, dones, done_indices, extras, step)
        self.experience_buffer.update_data("latents", step, self.latents)
        self.update_latents()

    # process_dataset is inherited from AMP and is relevant to dataset processing

    # -----------------------------
    # Environment Interaction
    # -----------------------------
    def handle_reset(self, done_indices=None):
        """Reset environment states and latents, then update the observation accordingly."""
        self.reset_latents(done_indices)
        obs = super().handle_reset(done_indices)
        obs["latents"] = self.latents.clone()
        return obs

    def env_step(self, actions):
        """Perform an environment step and inject current latents into observations."""
        obs, rewards, dones, terminated, extras = super().env_step(actions)
        obs["latents"] = self.latents.clone()
        return obs, rewards, dones, terminated, extras

    @torch.no_grad()
    def evaluate_policy(self):
        """Evaluates the policy in evaluation mode."""
        self.eval()
        done_indices = None  # Force reset on first entry
        step = 0
        while self.config.max_eval_steps is None or step < self.config.max_eval_steps:
            obs = self.handle_reset(done_indices)
            # Obtain actor predictions
            actions = self.model.act(obs)
            # Step the environment
            obs, rewards, dones, terminated, extras = self.env_step(actions)
            self.update_latents()
            all_done_indices = dones.nonzero(as_tuple=False)
            done_indices = all_done_indices.squeeze(-1)
            step += 1

    # -----------------------------
    # Reward Calculation
    # -----------------------------
    @torch.no_grad()
    def calculate_extra_reward(self):
        """Calculates extra reward based on Mutual Information."""
        rew = super().calculate_extra_reward()

        historical_self_obs = self.experience_buffer.historical_self_obs
        latents = self.experience_buffer.latents
        mi_r = self.model._discriminator.compute_mi_reward(
            {
                "historical_self_obs": historical_self_obs.view(
                    self.num_envs * self.num_steps, -1
                ),
            },
            latents=latents.view(self.num_envs * self.num_steps, -1),
        ).view(self.num_steps, self.num_envs)

        self.experience_buffer.batch_update_data("mi_rewards", mi_r)

        extra_reward = mi_r * self.config.ase_parameters.mi_reward_w + rew
        return extra_reward

    # -----------------------------
    # Optimization
    # -----------------------------
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

        agent_obs = batch_dict["agent_historical_self_obs"][
            : self.config.discriminator_batch_size
        ]
        latents = batch_dict["latents"][
            : self.config.discriminator_batch_size
        ]

        if self.config.ase_parameters.mi_enc_grad_penalty > 0:
            agent_obs.requires_grad_(True)

        mi_enc_pred = self.mi_enc_forward({"historical_self_obs": agent_obs})

        mi_enc_err = self.model._discriminator.calc_von_mises_fisher_enc_error(mi_enc_pred, latents)

        mi_enc_loss = torch.mean(mi_enc_err)

        if self.config.ase_parameters.mi_enc_weight_decay > 0:
            enc_weight_params = self.model._discriminator.enc_weights()
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
            )

            mi_enc_obs_grad = mi_enc_obs_grad[0]
            mi_enc_obs_grad = torch.sum(torch.square(mi_enc_obs_grad), dim=-1)
            mi_enc_grad_penalty = torch.mean(mi_enc_obs_grad)

            grad_loss: Tensor = mi_enc_grad_penalty * self.config.ase_parameters.mi_enc_grad_penalty
        else:
            grad_loss = torch.tensor(0.0, device=self.device)

        mi_loss = mi_enc_loss + weight_decay_loss + grad_loss

        log_dict = {
            "encoder/loss": mi_loss.detach(),
            "encoder/l2_loss": weight_decay_loss.detach(),
            "encoder/grad_penalty": grad_loss.detach(),
        }

        discriminator_log_dict.update(log_dict)

        return mi_loss + discriminator_loss, discriminator_log_dict

    def calculate_extra_actor_loss(self, batch_dict, dist) -> Tuple[Tensor, Dict]:
        """Adds the diversity loss, if enabled.

        Args:
            batch_dict (dict): Batch of data from the experience buffer.
            dist (torch.distributions.Distribution): Action distribution.

        Returns:
            Tuple[Tensor, Dict]: Extra actor loss and logging dictionary.
        """
        extra_loss, extra_actor_log_dict = super().calculate_extra_actor_loss(
            batch_dict, dist
        )

        if self.config.ase_parameters.diversity_bonus <= 0:
            return extra_loss, extra_actor_log_dict

        diversity_loss = self.diversity_loss(batch_dict)

        extra_actor_log_dict["actor/diversity_loss"] = diversity_loss.detach()

        return (
            extra_loss
            + diversity_loss * self.config.ase_parameters.diversity_bonus,
            extra_actor_log_dict,
        )

    def diversity_loss(self, batch_dict):
        """Calculates the diversity loss to encourage latents to lead to diverse behaviors.

        Args:
            batch_dict (dict): Batch of data from the experience buffer.

        Returns:
            Tensor: Diversity loss.
        """
        prev_latents = batch_dict["latents"]
        new_latents = self.sample_latents(batch_dict["self_obs"].shape[0])
        batch_dict["latents"] = new_latents
        new_dist = self.model._actor(batch_dict)

        batch_dict["latents"] = prev_latents
        old_dist = self.model._actor(batch_dict)

        clipped_new_mu = torch.clamp(new_dist.mean, -1.0, 1.0)
        clipped_old_mu = torch.clamp(old_dist.mean, -1.0, 1.0)

        mu_diff = clipped_new_mu - clipped_old_mu
        mu_diff = torch.mean(torch.square(mu_diff), dim=-1)

        z_diff = new_latents * prev_latents
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
        super().post_epoch_logging(training_log_dict)
