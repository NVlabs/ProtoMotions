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
from typing import List
from hydra.utils import instantiate
from protomotions.agents.common.mlp import MLP_WithNorm
from protomotions.agents.ppo.model import PPOModel


class Discriminator(MLP_WithNorm):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in, num_out)

    def forward(self, input_dict: dict) -> torch.Tensor:
        outs = super().forward(input_dict)
        return torch.sigmoid(outs)

    def compute_logits(
        self, input_dict: dict, return_norm_obs: bool = False
    ) -> torch.Tensor:
        outs = super().forward(input_dict, return_norm_obs=return_norm_obs)
        return outs

    def compute_reward(self, input_dict: dict, eps: float = 1e-7) -> torch.Tensor:
        s = self.forward(input_dict)
        s = torch.clamp(s, eps, 1 - eps)
        reward = -(1 - s).log()
        return reward

    def all_discriminator_weights(self):
        weights: list[nn.Parameter] = []
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        return weights

    def logit_weights(self) -> List[nn.Parameter]:
        return [self.mlp[-1].weight]


class AMPModel(PPOModel):
    def __init__(self, config):
        super().__init__(config)
        self._discriminator: Discriminator = instantiate(
            self.config.discriminator,
        )
