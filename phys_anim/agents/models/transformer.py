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

from copy import copy

import torch
from torch import nn, Tensor
import torch.nn.functional as f
import numpy as np

from phys_anim.agents.models.mlp import build_mlp, MLP_WithNorm
from phys_anim.utils.model_utils import get_activation_func

from hydra.utils import instantiate


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1], : x.shape[2]]
        return x


class TransformerWithNorm(nn.Module):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__()
        self.config = config

        # Input pre-processing
        self.obs_encoder = MLP_WithNorm(
            config.obs_mlp, num_in, config.latent_dim - config.type_embedding_dim
        )
        num_entries = 1

        self.extra_input_keys = []
        self.mask_keys = {}
        if config.extra_inputs is not None:
            self.extra_input_keys = sorted(config.extra_inputs.keys())

            for extra_input_key, extra_input in config.extra_inputs.items():
                if extra_input is None:
                    self.extra_input_keys.remove(extra_input_key)
                    continue
                if extra_input.config.get("mask_key", None) is not None:
                    self.mask_keys[extra_input_key] = extra_input.config.mask_key
                    self.extra_input_keys.remove(extra_input.config.mask_key)
                else:
                    self.mask_keys[extra_input_key] = None

            num_entries += len(self.extra_input_keys)

        self.type_embedding = Embedding(
            num_embeddings=num_entries, embedding_dim=config.type_embedding_dim
        )

        self.feature_size = num_in
        extra_input_models = {}
        for key in self.extra_input_keys:
            num_in = config.extra_inputs[key].config.encoder_input_dim
            model = instantiate(config.extra_inputs[key], num_in=num_in)
            extra_input_models[key] = model

            for operation in self.config.extra_inputs[key].config.get("operations", []):
                if operation.type == "embedding_per_entry":
                    embedding_per_entry = Embedding(
                        num_embeddings=operation.num_embeddings,
                        embedding_dim=operation.embedding_dim,
                    )
                    extra_input_models[key + "_" + operation.name] = embedding_per_entry

            self.feature_size += config.extra_inputs[key].num_out

        self.extra_input_models = nn.ModuleDict(extra_input_models)

        # Transformer layers
        self.sequence_pos_encoder = PositionalEncoding(
            config.latent_dim - config.type_embedding_dim
        )
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=config.latent_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_size,
            dropout=config.dropout,
            activation=get_activation_func(config.activation, return_type="functional"),
        )
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=config.num_layers
        )

        # Map transformer layer to actions
        if config.output_mlp.get("num_out", None) is not None:
            num_out = config.output_mlp.num_out
        self.output_mapping = build_mlp(
            self.config.output_mlp, config.latent_dim, num_out
        )

    def get_extracted_features(
        self, input_dict, already_normalized=False, return_norm_obs=False
    ):
        type_array = torch.arange(
            len(self.extra_input_keys) + 1,
            dtype=torch.long,
            device=input_dict["obs"].device,
        )

        batch_size = input_dict["obs"].shape[0]

        cur_encoded_state = self.obs_encoder(
            {"obs": input_dict["obs"]}, already_normalized, return_norm_obs
        )
        if return_norm_obs:
            cur_encoded_state, obs = (
                cur_encoded_state["outs"],
                cur_encoded_state["norm_obs"],
            )

        cur_state_embedding = self.type_embedding(type_array[0].unsqueeze(0)).expand(
            cur_encoded_state.shape[0], -1
        )
        cat_obs = torch.cat([cur_encoded_state, cur_state_embedding], dim=1).unsqueeze(
            1
        )

        cur_mask = torch.zeros(
            batch_size, 1, dtype=torch.bool, device=input_dict["obs"].device
        )

        for key_idx, key in enumerate(self.extra_input_keys):
            key_obs = input_dict[key]
            used_mask_multiply = False

            for operation in self.config.extra_inputs[key].config.get("operations", []):
                if operation.type == "permute":
                    key_obs = key_obs.permute(*operation.new_order)
                elif operation.type == "reshape":
                    new_shape = copy(operation.new_shape)
                    if new_shape[0] == "batch_size":
                        new_shape[0] = batch_size
                    key_obs = key_obs.reshape(*new_shape)
                elif operation.type == "squeeze":
                    key_obs = key_obs.squeeze(dim=operation.squeeze_dim)
                elif operation.type == "unsqueeze":
                    key_obs = key_obs.unsqueeze(dim=operation.unsqueeze_dim)
                elif operation.type == "expand":
                    key_obs = key_obs.expand(*operation.expand_shape)
                elif operation.type == "positional_encoding":
                    key_obs = self.sequence_pos_encoder(key_obs)
                elif operation.type == "encode":
                    key_obs = {"obs": key_obs}
                    key_obs = self.extra_input_models[key](key_obs)
                elif operation.type == "embedding_per_entry":
                    embedding_per_entry_module = self.extra_input_models[
                        key + "_" + operation.name
                    ]
                    key_type_array = torch.arange(
                        embedding_per_entry_module.num_embeddings,
                        dtype=torch.long,
                        device=input_dict["obs"].device,
                    )
                    key_obs = torch.cat(
                        [
                            key_obs,
                            embedding_per_entry_module(
                                key_type_array.unsqueeze(0)
                            ).expand(key_obs.shape[0], -1, -1),
                        ],
                        dim=-1,
                    )
                elif operation.type == "type_embedding":
                    key_type_embedding = self.type_embedding(
                        type_array[key_idx + 1].unsqueeze(0).unsqueeze(0)
                    ).expand(key_obs.shape[0], key_obs.shape[1], -1)
                    key_obs = torch.cat([key_obs, key_type_embedding], dim=-1)
                elif operation.type == "mask_multiply":
                    num_mask_dims = len(input_dict[self.mask_keys[key]].shape)
                    num_obs_dims = len(key_obs.shape)
                    extra_needed_dims = num_obs_dims - num_mask_dims
                    key_obs = key_obs * input_dict[self.mask_keys[key]].view(
                        *input_dict[self.mask_keys[key]].shape,
                        *((1,) * extra_needed_dims),
                    )
                    used_mask_multiply = True
                elif operation.type == "mask_multiply_concat":
                    num_mask_dims = len(input_dict[self.mask_keys[key]].shape)
                    num_obs_dims = len(key_obs.shape)
                    extra_needed_dims = num_obs_dims - num_mask_dims
                    key_obs = key_obs * input_dict[self.mask_keys[key]].view(
                        *input_dict[self.mask_keys[key]].shape,
                        *((1,) * extra_needed_dims),
                    )
                    key_obs = torch.cat(
                        [
                            key_obs,
                            input_dict[self.mask_keys[key]].view(
                                *input_dict[self.mask_keys[key]].shape,
                                *((1,) * extra_needed_dims),
                            ),
                        ],
                        dim=-1,
                    )
                    used_mask_multiply = True
                else:
                    raise NotImplementedError(f"Operation {operation} not implemented")

            cat_obs = torch.cat([cat_obs, key_obs], dim=1)

            if self.mask_keys[key] is not None and not used_mask_multiply:
                key_mask = input_dict[self.mask_keys[key]]
                if not self.config.extra_inputs[key].config.get(
                    "mask_valid_as_zeros", True
                ):
                    key_mask = key_mask.logical_not()
            else:
                key_mask = torch.zeros(
                    batch_size,
                    key_obs.shape[1],
                    dtype=torch.bool,
                    device=input_dict["obs"].device,
                )
            cur_mask = torch.cat([cur_mask, key_mask], dim=1)

        # obs creation works in batch_first but transformer expects seq_len first
        cat_obs = cat_obs.permute(1, 0, 2).contiguous()  # [seq_len, bs, d]

        cur_mask = cur_mask.unsqueeze(1).expand(-1, cat_obs.shape[0], -1)
        cur_mask = torch.repeat_interleave(cur_mask, self.config.num_heads, dim=0)

        output = self.seqTransEncoder(cat_obs, mask=cur_mask)[0]  # [bs, d]

        if return_norm_obs:
            assert not already_normalized
            return {"outs": output, "norm_obs": obs}
        else:
            return output

    def forward(self, input_dict, already_normalized=False, return_norm_obs=False):
        output = self.get_extracted_features(
            input_dict, already_normalized, return_norm_obs
        )
        if return_norm_obs:
            extracted_features, obs = output["outs"], output["norm_obs"]
        else:
            extracted_features = output

        if self.config.get("output_decoder", True):
            output = self.output_mapping(extracted_features)  # [bs, output]
        else:
            output = extracted_features

        if return_norm_obs:
            assert not already_normalized
            return {"outs": output, "norm_obs": obs}
        else:
            return output

    def get_features_size(self):
        return self.config.latent_dim


class Embedding(nn.Module):
    # TODO: replace this with models.base_interface.Embedding
    """Implementation of embedding using one hot encoded input and fully connected layer"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(Embedding, self).__init__()
        self.projection = nn.Linear(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings

    def forward(self, e: Tensor) -> Tensor:
        e_ohe = f.one_hot(e, num_classes=self.num_embeddings).float()
        return self.projection(e_ohe)
