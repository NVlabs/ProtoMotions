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
from torch import distributions, nn
from hydra.utils import instantiate
from protomotions.agents.common.mlp import MultiHeadedMLP


class PPOActor(nn.Module):
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config
        self.logstd = nn.Parameter(
            torch.ones(num_out) * config.actor_logstd,
            requires_grad=False,
        )
        self.mu: MultiHeadedMLP = instantiate(self.config.mu_model, num_out=num_out)

    def forward(self, input_dict):
        mu = self.mu(input_dict)
        mu = torch.tanh(mu)
        std = torch.exp(self.logstd)
        dist = distributions.Normal(mu, std)
        return dist


class PPOModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # create networks
        self._actor: PPOActor = instantiate(
            self.config.actor,
        )
        self._critic: MultiHeadedMLP = instantiate(
            self.config.critic,
        )

    def get_action_and_value(self, input_dict: dict):
        dist = self._actor(input_dict)
        action = dist.sample()
        value = self._critic(input_dict).flatten()

        logstd = self._actor.logstd
        std = torch.exp(logstd)

        neglogp = self.neglogp(action, dist.mean, std, logstd)
        return action, neglogp, value.flatten()

    def act(self, input_dict: dict, mean: bool = True) -> torch.Tensor:
        dist = self._actor(input_dict)
        if mean:
            return dist.mean
        return dist.sample()

    @staticmethod
    def neglogp(x, mean, std, logstd):
        dist = distributions.Normal(mean, std)
        return -dist.log_prob(x).sum(dim=-1)
