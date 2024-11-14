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

from lightning.fabric import Fabric

from torch import Tensor
from phys_anim.agents.amp import AMP
from phys_anim.envs.env_utils.general import StepTracker
from phys_anim.envs.amp.common import DiscHumanoid
from typing import Tuple, Dict


class InfoMax(AMP):
    def __init__(self, fabric: Fabric, env: DiscHumanoid, config):
        super().__init__(fabric, env, config)

        self.latents = torch.zeros(
            (self.num_envs, sum(self.config.infomax_parameters.latent_dim)),
            dtype=torch.float,
            device=self.device,
        )

        self.latent_reset_steps = StepTracker(
            self.num_envs,
            min_steps=self.config.infomax_parameters.latent_steps_min,
            max_steps=self.config.infomax_parameters.latent_steps_max,
            device=self.device,
        )

        self.experience_buffer.register_key("mi_rewards")  # mi = mutual information
        self.experience_buffer.register_key("latents", shape=(sum(self.config.infomax_parameters.latent_dim),))

    def update_latents(self):
        self.latent_reset_steps.advance()
        reset_ids = self.latent_reset_steps.done_indices()

        if reset_ids.numel() > 0:
            self.reset_latents(reset_ids)
            self.latent_reset_steps.reset_steps(reset_ids)

    def mi_enc_forward(self, obs: Tensor) -> Tensor:
        args = {"obs": obs}
        return self.discriminator(args, return_enc=True)

    def reset_latents(self, env_ids):
        self.env.randomize_color(env_ids)
        latents = self.sample_latents(len(env_ids))
        self.store_latents(latents, env_ids)

    def store_latents(self, latents, env_ids):
        self.latents[env_ids] = latents

    def sample_latents(self, n):
        latents = torch.zeros(
            [n, sum(self.config.infomax_parameters.latent_dim)], device=self.device
        )

        start = 0
        for idx, dim in enumerate(self.config.infomax_parameters.latent_dim):
            if self.config.infomax_parameters.latent_types[idx] == "gaussian":
                gaussian_sample = torch.normal(latents[:, start : start + dim])
                latents[:, start : start + dim] = gaussian_sample

            elif self.config.infomax_parameters.latent_types[idx] == "hypersphere":
                gaussian_sample = torch.normal(latents[:, start : start + dim])
                projected_gaussian_sample = torch.nn.functional.normalize(
                    gaussian_sample, dim=-1
                )
                latents[:, start : start + dim] = projected_gaussian_sample

            elif self.config.infomax_parameters.latent_types[idx] == "uniform":
                uniform_sample = torch.rand([n, dim], device=self.device)
                latents[:, start : start + dim] = uniform_sample

            elif self.config.infomax_parameters.latent_types[idx] == "categorical":
                categorical_sample = torch.multinomial(
                    latents[0, start : start + dim] + 1.0,
                    num_samples=n,
                    replacement=True,
                )
                b = torch.arange(n, device=self.device)
                latents[b, categorical_sample + start] = categorical_sample
            else:
                raise NotImplementedError

            start += dim

        return latents

    def create_actor_state(self):
        state = super().create_actor_state()
        state["latents"] = self.latents
        return state

    def handle_reset(self, actor_state):
        actor_state = super().handle_reset(actor_state)

        actor_state["latents"] = self.latents
        return actor_state

    def post_env_step(self, actor_state):
        actor_state = super().post_env_step(actor_state)
        self.update_latents()
        actor_state["latents"] = self.latents
        self.experience_buffer.update_data("latents", actor_state["step"],actor_state["latents"])
        return actor_state

    def post_eval_env_step(self, actor_state):
        self.update_latents()
        actor_state["latents"] = self.latents
        actor_state = super().post_eval_env_step(actor_state)
        return actor_state

    def create_actor_args(self, actor_state):
        actor_args = super().create_actor_args(actor_state)
        actor_args['latents'] = actor_state['latents']
        return actor_args

    def create_critic_args(self, actor_state):
        critic_args = super().create_critic_args(actor_state)
        critic_args['latents'] = actor_state['latents']
        return critic_args

    def calculate_extra_reward(self):
        rew = super().calculate_extra_reward()

        discriminator_obs = self.experience_buffer.discriminator_obs
        latents = self.experience_buffer.latents
        mi_r = self.calc_mi_reward(
            discriminator_obs.view(self.num_envs * self.num_steps, -1),
            latents.view(self.num_envs * self.num_steps, -1),
        ).view(self.num_steps, self.num_envs)

        self.experience_buffer.batch_update_data("mi_rewards", mi_r)

        extra_reward = mi_r + rew
        return extra_reward

    def calc_mi_reward(self, discriminator_obs, latents):
        """
        TODO: calculate reward for each distribution type
            Gaussian -- MSE (we assume variance 1)
            Hypersphere -- von Mises-Fisher
            Uniform -- MSE
            Categorical -- torch.nn.functional.cross_entropy()
        """
        mi_r = torch.zeros(self.num_steps * self.num_envs, 1, device=self.device)

        enc_pred = self.mi_enc_forward(discriminator_obs)
        cumulative_enc_dim = 0
        cumulative_latent_dim = 0
        for idx, latent_dim in enumerate(self.config.infomax_parameters.latent_dim):
            if self.config.infomax_parameters.latent_types[idx] == "gaussian":
                r = self.calc_gaussian_enc_reward(
                    enc_pred[..., cumulative_enc_dim : cumulative_enc_dim + latent_dim],
                    enc_pred[
                        ...,
                        cumulative_enc_dim
                        + cumulative_enc_dim
                        + latent_dim : cumulative_enc_dim
                        + latent_dim * 2,
                    ],
                    latents[
                        ..., cumulative_latent_dim : cumulative_latent_dim + latent_dim
                    ],
                )

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim * 2

            elif self.config.infomax_parameters.latent_types[idx] == "hypersphere":
                r = self.von_mises_fisher_reward(
                    enc_pred[..., cumulative_enc_dim : cumulative_enc_dim + latent_dim],
                    latents[
                        ..., cumulative_latent_dim : cumulative_latent_dim + latent_dim
                    ],
                )

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim

            elif self.config.infomax_parameters.latent_types[idx] == "uniform":
                r = self.calc_gaussian_enc_reward(
                    enc_pred[..., cumulative_enc_dim : cumulative_enc_dim + latent_dim],
                    enc_pred[
                        ...,
                        cumulative_enc_dim
                        + cumulative_enc_dim
                        + latent_dim : cumulative_enc_dim
                        + latent_dim * 2,
                    ],
                    latents[
                        ..., cumulative_latent_dim : cumulative_latent_dim + latent_dim
                    ],
                )

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim * 2

            elif self.config.infomax_parameters.latent_types[idx] == "categorical":
                r = self.calc_categorical_enc_reward(
                    enc_pred[..., cumulative_enc_dim : cumulative_enc_dim + latent_dim],
                    latents[
                        ..., cumulative_latent_dim : cumulative_latent_dim + latent_dim
                    ],
                )

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim
            else:
                raise NotImplementedError

            mi_r += r * self.config.infomax_parameters.mi_reward_w[idx]

        return mi_r / len(self.config.infomax_parameters.latent_dim)

    def von_mises_fisher_reward(self, enc_prediction, latents):
        neg_err = -self.calc_von_mises_fisher_enc_error(enc_prediction, latents)
        if self.config.infomax_parameters.mi_hypersphere_reward_shift:
            mi_r = (neg_err + 1) / 2
        else:
            mi_r = torch.clamp_min(neg_err, 0.0)
        return mi_r

    def calc_von_mises_fisher_enc_error(self, enc_pred, latent):
        err = enc_pred * latent
        err = -torch.sum(err, dim=-1, keepdim=True)
        return err

    def calc_gaussian_enc_reward(self, enc_pred_mu, enc_pred_var, latents):
        neg_err = -self.calc_gaussian_enc_error(enc_pred_mu, enc_pred_var, latents)
        mi_r = neg_err  # TODO: should we normalize this to [0, 1] somehow?
        return mi_r

    def calc_gaussian_enc_error(self, enc_pred_mu, enc_pred_var, latent):
        logli = -0.5 * (enc_pred_var.mul(2 * torch.pi) + 1e-6).log() - (
            latent - enc_pred_mu
        ).pow(2).div(enc_pred_var.mul(2.0) + 1e-6)
        nll = -torch.sum(logli, dim=-1, keepdim=True)
        return nll

    def calc_categorical_enc_reward(self, enc_pred, latents):
        neg_err = -self.calc_categorical_enc_err(enc_pred, latents)
        mi_r = torch.exp(neg_err)  # normalize to [0, 1]
        return mi_r

    def calc_categorical_enc_err(self, enc_pred, latent):
        # TODO: make sure the dimensions match and add up nicely
        err = torch.nn.functional.cross_entropy(
            enc_pred, latent, reduction="none"
        ).view(self.num_steps, self.num_envs)
        return err

    def calc_mi_enc_error(self, enc_pred, latents):
        cumulative_enc_dim = 0
        cumulative_latent_dim = 0

        total_error = []

        for idx, latent_dim in enumerate(self.config.infomax_parameters.latent_dim):
            if self.config.infomax_parameters.latent_types[idx] == "gaussian":
                err = self.calc_gaussian_enc_error(
                    enc_pred[cumulative_enc_dim : cumulative_enc_dim + latent_dim],
                    enc_pred[
                        cumulative_enc_dim
                        + cumulative_enc_dim
                        + latent_dim : cumulative_enc_dim
                        + latent_dim * 2
                    ],
                    latents[cumulative_latent_dim : cumulative_latent_dim + latent_dim],
                )

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim * 2

            elif self.config.infomax_parameters.latent_types[idx] == "hypersphere":
                err = self.calc_von_mises_fisher_enc_error(enc_pred, latents)

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim

            elif self.config.infomax_parameters.latent_types[idx] == "uniform":
                err = self.calc_gaussian_enc_error(
                    enc_pred[cumulative_enc_dim : cumulative_enc_dim + latent_dim],
                    enc_pred[
                        cumulative_enc_dim
                        + cumulative_enc_dim
                        + latent_dim : cumulative_enc_dim
                        + latent_dim * 2
                    ],
                    latents[cumulative_latent_dim : cumulative_latent_dim + latent_dim],
                )

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim * 2

            elif self.config.infomax_parameters.latent_types[idx] == "categorical":
                err = self.calc_categorical_enc_error(
                    enc_pred[cumulative_enc_dim : cumulative_enc_dim + latent_dim],
                    latents[cumulative_latent_dim : cumulative_latent_dim + latent_dim],
                )

                cumulative_latent_dim += latent_dim
                cumulative_enc_dim += latent_dim
            else:
                raise NotImplementedError

            total_error.append(err)

        return torch.cat(total_error, dim=-1)

    def discriminator_step(self, batch_idx: int) -> Tuple[Tensor, Dict]:
        discriminator_loss, discriminator_log_dict = super().discriminator_step(
            batch_idx
        )

        dataset_idx = batch_idx % len(self.actor_critic_dataset)
        batch_dict = self.actor_critic_dataset[dataset_idx]

        obs = batch_dict["discriminator_obs"]
        latents = batch_dict["latents"]
        if self.config.infomax_parameters.mi_enc_grad_penalty > 0:
            obs.requires_grad_(True)

        mi_enc_pred = self.mi_enc_forward(obs)

        mi_enc_err = self.calc_mi_enc_error(mi_enc_pred, latents)

        mi_enc_loss = torch.mean(mi_enc_err)

        if self.config.infomax_parameters.mi_enc_weight_decay > 0:
            enc_weight_params = self.discriminator.enc_weights()
            total: Tensor = sum([p.pow(2).sum() for p in enc_weight_params])
            weight_decay_loss: Tensor = (
                total * self.config.infomax_parameters.mi_enc_weight_decay
            )
        else:
            weight_decay_loss = torch.tensor(0.0, device=self.device)

        if self.config.infomax_parameters.mi_enc_grad_penalty > 0:
            mi_enc_obs_grad = torch.autograd.grad(
                mi_enc_err,
                obs,
                grad_outputs=torch.ones_like(mi_enc_err),
                create_graph=True,
                retain_graph=True,
            )

            mi_enc_obs_grad = mi_enc_obs_grad[0]
            mi_enc_obs_grad = torch.sum(torch.square(mi_enc_obs_grad), dim=-1)
            mi_enc_grad_penalty = torch.mean(mi_enc_obs_grad)

            grad_loss: Tensor = mi_enc_grad_penalty * self.mi_enc_grad_penalty
        else:
            grad_loss = torch.tensor(0.0, device=self.device)

        mi_loss = mi_enc_loss + weight_decay_loss + grad_loss

        log_dict = {
            "loss": mi_loss.detach(),
            "l2_loss": weight_decay_loss.detach(),
            "grad_penalty": grad_loss.detach(),
        }

        mi_enc_log_dict = {f"mi_enc/{k}": v for k, v in log_dict.items()}

        discriminator_log_dict.update(mi_enc_log_dict)

        return mi_loss + discriminator_loss, discriminator_log_dict

    def calculate_extra_actor_loss(self, batch_idx, batch_dict) -> Tuple[Tensor, Dict]:
        extra_loss, extra_actor_log_dict = super().calculate_extra_actor_loss(
            batch_idx, batch_dict
        )

        if self.config.infomax_parameters.diversity_bonus <= 0:
            return extra_loss, extra_actor_log_dict

        diversity_loss = self.diversity_loss(batch_dict)

        extra_actor_log_dict["actor/diversity_loss"] = diversity_loss.detach()

        return (
            extra_loss
            + diversity_loss * self.config.infomax_parameters.diversity_bonus,
            extra_actor_log_dict,
        )

    def diversity_loss(self, batch_dict):
        prev_latents = batch_dict["latents"]
        new_latents = self.sample_latents(batch_dict["obs"].shape[0])
        batch_dict["latents"] = new_latents
        new_outs = self.actor.training_forward(batch_dict)

        batch_dict["latents"] = prev_latents
        old_outs = self.actor.training_forward(batch_dict)

        clipped_new_mu = torch.clamp(new_outs["mus"], -1.0, 1.0)
        clipped_old_mu = torch.clamp(old_outs["mus"], -1.0, 1.0)

        mu_diff = clipped_new_mu - clipped_old_mu
        mu_diff = torch.mean(torch.square(mu_diff), dim=-1)

        z_diff = new_latents * prev_latents
        z_diff = torch.sum(z_diff, dim=-1)
        z_diff = 0.5 - 0.5 * z_diff

        diversity_bonus = mu_diff / (z_diff + 1e-5)
        diversity_loss = torch.square(
            self.config.infomax_parameters.diversity_tar - diversity_bonus
        ).mean()

        return diversity_loss

    def post_epoch_logging(self, training_log_dict: Dict):
        training_log_dict["rewards/mi_enc_rewards"] = (
            self.experience_buffer.mi_rewards.mean()
        )
        super().post_epoch_logging(training_log_dict)
