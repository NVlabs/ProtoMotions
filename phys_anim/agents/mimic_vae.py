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

from phys_anim.agents.mimic import Mimic
from phys_anim.envs.mimic.common import MimicHumanoid
from phys_anim.agents.models.actor import ActorFixedSigmaVAE

import torch
from torch import Tensor
from lightning.fabric import Fabric

from typing import Tuple, Dict, Optional


class MimicVAE(Mimic):
    env: MimicHumanoid
    actor: ActorFixedSigmaVAE

    def __init__(self, fabric: Fabric, env: MimicHumanoid, config):
        super().__init__(fabric, env, config)

        self.experience_buffer.register_key(
            "vae_noise", shape=(self.config.actor.config.vae_latent_dim,)
        )
        self.vae_noise = torch.zeros(
            self.num_envs,
            self.config.actor.config.vae_latent_dim,
            dtype=torch.float,
            device=self.device,
        )

    def setup(self):
        super().setup()
        self.actor.mark_forward_method("kl_loss")

    def reset_vae_noise(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if type(env_ids) == list:
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)

        if self.config.vae.vae_noise_type == "normal":
            epsilon = torch.randn(
                env_ids.shape[0], self.actor.vae_latent_dim, device=self.device
            )  # sampling epsilon
        elif self.config.vae.vae_noise_type == "uniform":
            epsilon = torch.rand(
                env_ids.shape[0], self.actor.vae_latent_dim, device=self.device
            )  # sampling epsilon
        elif self.config.vae.vae_noise_type == "zeros":
            epsilon = torch.zeros(
                env_ids.shape[0], self.actor.vae_latent_dim, device=self.device
            )  # no noise
        else:
            raise NotImplementedError
        self.vae_noise[env_ids] = epsilon

    def handle_reset(self, actor_state):
        done_indices = actor_state["done_indices"]
        self.reset_vae_noise(done_indices)
        actor_state = super().handle_reset(actor_state)
        actor_state["vae_noise"] = self.vae_noise

        return actor_state

    def create_actor_state(self):
        state = super().create_actor_state()
        state["vae_noise"] = self.vae_noise
        return state

    def create_actor_args(self, actor_state):
        actor_inputs = super().create_actor_args(actor_state)
        actor_inputs["vae_noise"] = actor_state["vae_noise"]
        return actor_inputs

    def pre_env_step(self, actor_state) -> Tensor:
        self.experience_buffer.update_data(
            "vae_noise", actor_state["step"], actor_state["vae_noise"]
        )

        return super().pre_env_step(actor_state)

    def calculate_extra_actor_loss(self, batch_idx, batch_dict) -> Tuple[Tensor, Dict]:
        extra_loss, extra_actor_log_dict = super().calculate_extra_actor_loss(
            batch_idx, batch_dict
        )

        vae_kld_schedule = self.config.vae.vae_kld_schedule
        if vae_kld_schedule is not None:
            vae_kld_loss = self.actor.kl_loss(batch_dict)
            vae_kld_loss = torch.mean(torch.sum(vae_kld_loss, dim=-1))

            extra_actor_log_dict["actor/vae_kld_loss"] = vae_kld_loss.detach()

            kld_coeff = vae_kld_schedule.init_kld_coeff + min(
                max(0, self.current_epoch - vae_kld_schedule.start_epoch)
                / vae_kld_schedule.end_epoch,
                1,
            ) * (vae_kld_schedule.end_kld_coeff - vae_kld_schedule.init_kld_coeff)

            extra_actor_log_dict["actor/kld_coeff"] = kld_coeff

            return extra_loss + vae_kld_loss * kld_coeff, extra_actor_log_dict
        else:
            return extra_loss, extra_actor_log_dict

    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:
        vae_latent_from_prior = None
        if hasattr(self.actor, "vae_latent_from_prior"):
            vae_latent_from_prior = self.actor.vae_latent_from_prior
            self.actor.vae_latent_from_prior = True

        eval_metrics_dict, evaluated_score = super().calc_eval_metrics()

        if vae_latent_from_prior is not None:
            self.actor.vae_latent_from_prior = vae_latent_from_prior

        return eval_metrics_dict, evaluated_score
