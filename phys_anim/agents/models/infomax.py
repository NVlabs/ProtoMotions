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
from phys_anim.utils import model_utils

from phys_anim.agents.models.discriminator import DISC_LOGIT_INIT_SCALE
from phys_anim.agents.models.mlp import MLP_WithNorm

from typing import List

ENC_LOGIT_INIT_SCALE = 0.1


class NormalizedLinearEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, normalization):
        super().__init__()

        self.normalization = normalization
        self.enc = torch.nn.Linear(input_dim, output_dim)
        torch.nn.init.uniform_(
            self.enc.weight, -ENC_LOGIT_INIT_SCALE, ENC_LOGIT_INIT_SCALE
        )
        torch.nn.init.zeros_(self.enc.bias)

    def forward(self, x):
        out = self.enc(x)
        if self.normalization == "von Mises-Fisher":
            out = torch.nn.functional.normalize(out, dim=-1)
        elif self.normalization == "softmax":
            out = torch.nn.functional.softmax(out, dim=-1)
        else:
            raise NotImplementedError
        return out


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.enc = torch.nn.Linear(input_dim, output_dim)
        self.var_enc = torch.nn.Linear(input_dim, output_dim)
        torch.nn.init.uniform_(
            self.enc.weight, -ENC_LOGIT_INIT_SCALE, ENC_LOGIT_INIT_SCALE
        )
        torch.nn.init.zeros_(self.enc.bias)

    def forward(self, x):
        mu = self.enc(x)
        var = torch.exp(self.var_enc(x))
        return mu, var


class JointDiscWithMutualInformationEncMLP(MLP_WithNorm):
    def __init__(self, config, num_in: int, num_out: int = 1):
        super().__init__(config.shared, num_in, num_out=config.shared.out_dim)

        self.mi_enc_config = config.mi_enc
        self.trunk.append(model_utils.get_activation_func(config.shared.activation))

        self.disc_logits = torch.nn.Linear(config.shared.out_dim, 1)
        torch.nn.init.uniform_(
            self.disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE
        )

        encoders = []
        for idx, latent_dim in enumerate(config.mi_enc.latent_dim):
            if config.mi_enc.latent_types[idx] == "gaussian":
                enc = GaussianEncoder(config.shared.out_dim, latent_dim)
            elif config.mi_enc.latent_types[idx] == "hypersphere":
                enc = NormalizedLinearEncoder(
                    config.shared.out_dim, latent_dim, normalization="von Mises-Fisher"
                )
            elif config.mi_enc.latent_types[idx] == "uniform":
                enc = GaussianEncoder(config.shared.out_dim, latent_dim)
            elif config.mi_enc.latent_types[idx] == "categorical":
                enc = NormalizedLinearEncoder(
                    config.shared.out_dim, latent_dim, normalization="softmax"
                )
            else:
                raise NotImplementedError
            encoders.append(enc)

        self.encoders = nn.ModuleList(encoders)

    def forward(
        self,
        input_dict,
        already_normalized=False,
        return_norm_obs=False,
        return_enc=False,
    ):
        obs = (
            self.maybe_normalize_obs(input_dict["obs"])
            if not already_normalized
            else input_dict["obs"]
        )

        outs: Tensor = self.trunk(obs)

        if return_enc:
            enc_outs = []
            for idx, enc in enumerate(self.encoders):
                out = enc(outs)

                if self.mi_enc_config.latent_types[idx] == "gaussian":
                    out = torch.cat(out, dim=-1)
                elif self.mi_enc_config.latent_types[idx] == "hypersphere":
                    pass
                elif self.mi_enc_config.latent_types[idx] == "uniform":
                    out = torch.cat(out, dim=-1)
                elif self.mi_enc_config.latent_types[idx] == "categorical":
                    pass
                else:
                    raise NotImplementedError
                enc_outs.append(out)

            outs = torch.cat(enc_outs, dim=0)
        else:
            outs = self.disc_logits(outs)

        if return_norm_obs:
            assert not already_normalized
            return {"outs": outs, "norm_obs": obs}
        else:
            return outs

    def all_weights(self):
        weights: list[nn.Parameter] = []
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        return weights

    def all_discriminator_weights(self):
        weights: list[nn.Parameter] = []
        for mod in self.trunk.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        weights.append(self.logit_weights()[0])
        return weights

    def logit_weights(self) -> List[nn.Parameter]:
        return [self.trunk[-2].weight]

    def all_enc_weights(self):
        weights: list[nn.Parameter] = []
        for mod in self.trunk.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        weights.append(self.enc_weights()[0])
        return weights

    def enc_weights(self) -> List[nn.Parameter]:
        return [self.enc.weight]
