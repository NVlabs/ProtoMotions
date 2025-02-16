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
from torch import nn, Tensor
from hydra.utils import instantiate
from protomotions.agents.common.common import NormObsBase
from protomotions.utils import model_utils


def build_mlp(config, num_in: int, num_out: int):
    indim = num_in
    layers = []
    for i, layer in enumerate(config.layers):
        layers.append(nn.Linear(indim, layer.units))
        if layer.use_layer_norm and i == 0:
            layers.append(nn.LayerNorm(layer.units))
        layers.append(model_utils.get_activation_func(layer.activation))
        indim = layer.units

    layers.append(nn.Linear(indim, num_out))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__()
        self.config = config
        self.mlp = build_mlp(self.config, num_in, num_out)

    def forward(self, input_dict, *args, **kwargs):
        if isinstance(input_dict, torch.Tensor):
            return self.mlp(input_dict)
        return self.mlp(input_dict[self.config.obs_key])


class MLP_WithNorm(NormObsBase):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in, num_out)
        self.mlp = build_mlp(self.config, num_in, num_out)

    def forward(self, input_dict, return_norm_obs=False):
        obs = super().forward(input_dict[self.config.obs_key])
        outs: Tensor = self.mlp(obs)

        if return_norm_obs:
            return {"outs": outs, f"norm_{self.config.obs_key}": obs}
        else:
            return outs


class MultiHeadedMLP(nn.Module):
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config
        self.num_out = num_out

        input_models = {}
        self.feature_size = 0
        for key, input_cfg in self.config.input_models.items():
            model = instantiate(input_cfg)
            input_models[key] = model
            self.feature_size += model.num_out
        self.input_models = nn.ModuleDict(input_models)

        self.trunk: MLP = instantiate(self.config.trunk, num_in=self.feature_size)

    def forward(self, input_dict, return_norm_obs=False):
        if return_norm_obs:
            norm_obs = {}
        outs = []

        for key, model in self.input_models.items():
            out = model(input_dict, return_norm_obs=return_norm_obs)
            if return_norm_obs:
                out, norm_obs[f"norm_{model.config.obs_key}"] = (
                    out["outs"],
                    out[f"norm_{model.config.obs_key}"],
                )
            outs.append(out)

        outs = torch.cat(outs, dim=-1)

        outs: Tensor = self.trunk(outs)

        if return_norm_obs:
            ret_dict = {**{"outs": outs}, **norm_obs}
            return ret_dict
        else:
            return outs
