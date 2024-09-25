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

import numpy as np
import torch
from hydra.utils import instantiate
from torch import nn, Tensor

from phys_anim.agents.models.common import NormObsBase
from phys_anim.utils import model_utils


def default_init(m: nn.Linear):
    m.bias.data.zero_()
    return m


INIT_DICT = {
    "orthogonal": lambda m: model_utils.init(
        m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
    ),
    "default": default_init,
}


def build_mlp(config, num_in: int, num_out: int):
    init = INIT_DICT[config.initializer]

    indim = num_in
    layers = []
    for outdim in config.units:
        layers.append(init(nn.Linear(indim, outdim)))
        layers.append(model_utils.get_activation_func(config.activation))
        if config.use_layer_norm:
            layers.append(nn.LayerNorm(outdim))
        indim = outdim

    layers.append(init(nn.Linear(outdim, num_out)))
    mlp = nn.Sequential(*layers)
    return mlp


class MLP_WithNorm(NormObsBase):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in)
        self.num_out = num_out
        self.trunk = build_mlp(self.config, self.num_input_units(), num_out)

    def num_input_units(self):
        return self.num_in

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        obs = (
            self.maybe_normalize_obs(input_dict["obs"])
            if not already_normalized
            else input_dict["obs"]
        )

        outs: Tensor = self.trunk(obs)

        if return_norm_obs:
            assert not already_normalized
            return {"outs": outs, "norm_obs": obs}
        else:
            return outs


class MultiHeadedMLP(MLP_WithNorm):
    def __init__(self, config, num_in: int, num_out: int):
        self.extra_input_keys = []
        self.connected_keys = {}
        if config.extra_inputs is not None:
            self.extra_input_keys = sorted(config.extra_inputs.keys())

            for extra_input_key, extra_input in config.extra_inputs.items():
                if extra_input is None:
                    self.extra_input_keys.remove(extra_input_key)
                    continue
                if extra_input.config.get("connected_keys", None) is not None:
                    self.connected_keys[extra_input_key] = sorted(
                        extra_input.config.connected_keys
                    )
                    for connected_key in self.connected_keys[extra_input_key]:
                        self.extra_input_keys.remove(connected_key)
                else:
                    self.connected_keys[extra_input_key] = []

        self.feature_size = num_in
        extra_input_models = {}
        for key in self.extra_input_keys:
            model = instantiate(config.extra_inputs[key])
            extra_input_models[key] = model
            self.feature_size += config.extra_inputs[key].num_out

        super().__init__(config, num_in, num_out)
        self.extra_input_models = nn.ModuleDict(extra_input_models)

    def num_input_units(self):
        return self.feature_size

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        obs = (
            self.maybe_normalize_obs(input_dict["obs"])
            if not already_normalized
            else input_dict["obs"]
        )

        cat_obs = obs
        for key in self.extra_input_keys:
            key_obs = {"obs": input_dict[key]}
            for connected_key in self.connected_keys[key]:
                key_obs[connected_key] = input_dict[connected_key]
            cat_obs = torch.cat(
                [cat_obs, self.extra_input_models[key](key_obs)], dim=-1
            )

        outs: Tensor = self.trunk(cat_obs)

        if return_norm_obs:
            assert not already_normalized
            return {"outs": outs, "norm_obs": obs}
        else:
            return outs


class MultiOutputNetwork(NormObsBase):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in)
        self.trunk = instantiate(self.config.trunk, num_in=self.num_input_units())

        self.output_keys = sorted(config.outputs.keys())

        output_models = {}
        for key in self.output_keys:
            model = instantiate(config.outputs[key])
            output_models[key] = model
        self.output_models = nn.ModuleDict(output_models)

    def num_input_units(self):
        return self.num_in

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        outs: Tensor = self.trunk(input_dict, already_normalized, return_norm_obs)

        if return_norm_obs:
            outs, obs = outs["outs"], outs["norm_obs"]

        outputs = {}
        for output_model_key in self.output_models.keys():
            outputs[output_model_key] = self.output_models[output_model_key](
                {"obs": outs}
            )

        if return_norm_obs:
            assert not already_normalized
            return {"outs": outputs, "norm_obs": obs}
        else:
            return outputs
