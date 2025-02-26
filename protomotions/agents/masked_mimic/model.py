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
from torch import nn
from hydra.utils import instantiate
from protomotions.agents.common.mlp import MultiHeadedMLP


class VaeModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self._trunk = instantiate(self.config.trunk)
        self._mu_head = instantiate(self.config.mu_head)
        self._logvar_head = instantiate(self.config.logvar_head)

    def forward(self, input_dict):
        trunk_out = self._trunk(input_dict)
        mu = self._mu_head(trunk_out)
        logvar = self._logvar_head(trunk_out)
        return {"mu": mu, "logvar": logvar}


class VaeDeterministicOutputModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # create networks
        self._encoder: VaeModule = instantiate(
            self.config.encoder,
        )
        self._prior: VaeModule = instantiate(
            self.config.prior,
        )
        self._trunk: MultiHeadedMLP = instantiate(
            self.config.trunk,
        )

    def reparameterization(self, mean, std, vae_noise):
        z = mean + std * vae_noise  # reparameterization trick
        return z

    def act(self, input_dict: dict, with_encoder: bool = False):
        prior_out = self._prior(input_dict)
        if with_encoder:
            encoder_out = self._encoder(input_dict)
            mu = prior_out["mu"] + encoder_out["mu"]
            logvar = prior_out["logvar"] + encoder_out["logvar"]
        else:
            mu = prior_out["mu"]
            logvar = prior_out["logvar"]

        z = self.reparameterization(
            mu,
            torch.exp(0.5 * logvar),
            input_dict["vae_noise"],
        )
        input_dict["vae_latent"] = z
        action = self._trunk(input_dict)
        action = torch.tanh(action)
        
        return action

    def get_action_and_vae_outputs(self, input_dict: dict):
        prior_out = self._prior(input_dict)
        encoder_out = self._encoder(input_dict)

        mu = prior_out["mu"] + encoder_out["mu"]
        logvar = prior_out["logvar"] + encoder_out["logvar"]

        if "vae_noise" not in input_dict:
            # During training, we randomly re-sample the noise.
            input_dict["vae_noise"] = torch.randn_like(mu)

        z = self.reparameterization(
            mu,
            torch.exp(0.5 * logvar),
            input_dict["vae_noise"],
        )

        input_dict["vae_latent"] = z
        action = self._trunk(input_dict)
        action = torch.tanh(action)

        return action, prior_out, encoder_out

    @staticmethod
    def kl_loss(prior_outs, encoder_outs):
        return 0.5 * (
            prior_outs["logvar"]
            - encoder_outs["logvar"]
            + torch.exp(encoder_outs["logvar"]) / torch.exp(prior_outs["logvar"])
            + encoder_outs["mu"] ** 2 / torch.exp(prior_outs["logvar"])
            - 1
        )
