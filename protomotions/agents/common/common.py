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
import torch.nn.functional as F
from typing import List

from protomotions.utils.running_mean_std import RunningMeanStd


def get_params(obj) -> List[nn.Parameter]:
    """
    Gets list of params from either a list of params
    (where nothing happens) or a list of param groups
    """
    as_list = list(obj)
    if isinstance(as_list[0], Tensor):
        return as_list
    else:
        params = []
        for group in as_list:
            params = params + list(group["params"])
        return params


def weight_init(m, orthogonal=False):
    if isinstance(m, nn.Linear):
        if orthogonal:
            nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


class NormObsBase(nn.Module):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__()
        self.config = config
        self.num_in = num_in
        self.num_out = num_out
        self.build_norm()

    def build_norm(self):
        if self.config.normalize_obs:
            self.running_obs_norm = RunningMeanStd(
                shape=(self.num_in,),
                device="cpu",
                clamp_value=self.config.norm_clamp_value,
            )

    def forward(self, obs, *args, **kwargs):
        if torch.isnan(obs).any():
            raise ValueError("NaN in obs")
        if self.config.normalize_obs:
            # Only update obs during training
            if self.training:
                self.running_obs_norm.update(obs)
            obs = self.running_obs_norm.normalize(obs)
        if torch.isnan(obs).any():
            raise ValueError("NaN in obs")
        return obs


class Flatten(NormObsBase):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in, num_out)
        self.flatten = nn.Flatten()

    def forward(self, input_dict, *args, **kwargs):
        obs = input_dict[self.config.obs_key]
        obs = self.flatten(obs)
        obs = super().forward(obs)

        if "return_norm_obs" in kwargs and kwargs["return_norm_obs"]:
            ret_dict = {"outs": obs, f"norm_{self.config.obs_key}": obs}
            return ret_dict

        return obs


class Embedding(NormObsBase):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in, num_out)
        self.embedding = nn.Embedding(self.num_in, self.num_out)

        if self.config.random_embedding.use_random_embeddings:
            self.random_embedding_vectors = torch.rand(
                (self.config.random_embedding.num_random_embeddings, self.num_out)
            )

    def _apply(self, fn):
        super()._apply(fn)
        if self.config.random_embedding.use_random_embeddings:
            self.random_embedding_vectors = fn(self.random_embedding_vectors)
        return self

    def forward(self, input_dict, *args, **kwargs):
        obs = input_dict[self.config.obs_key]
        obs = super().forward(obs)

        if self.config.random_embedding.use_random_embeddings:
            embedding = self.embedding(obs * 0)  # CT hack: ensure indices within range
            embedding = self.random_embedding_vectors[
                torch.fmod(
                    obs.view(-1), self.config.random_embedding.num_random_embeddings
                )
            ].reshape(embedding.shape)
        else:
            embedding = self.embedding(obs)

        if self.config.normalized_embedding:
            embedding = F.normalize(embedding, dim=-1)
        else:
            embedding = embedding

        return embedding
