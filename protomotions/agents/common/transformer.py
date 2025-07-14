from copy import copy

import torch
from torch import nn
import numpy as np

from protomotions.utils.model_utils import get_activation_func

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


class Transformer(nn.Module):
    def __init__(self, config, num_out: int):
        super().__init__()
        self.config = config

        self.mask_keys = {}
        input_models = {}

        for input_key, input_config in config.input_models.items():
            input_models[input_key] = instantiate(input_config)
            self.mask_keys[input_key] = input_config.config.get("mask_key", None)

        self.input_models = nn.ModuleDict(input_models)
        self.feature_size = self.config.transformer_token_size * len(input_models)

        # Transformer layers
        self.sequence_pos_encoder = PositionalEncoding(config.latent_dim)
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

        if config.get("output_model", None) is not None:
            self.output_model = instantiate(config.output_model)

    def get_extracted_features(self, input_dict):
        batch_size = next(iter(input_dict.values())).shape[0]
        device = next(iter(input_dict.values())).device
        cat_obs = []
        cat_mask = []

        for model_name, input_model in self.input_models.items():
            input_key = input_model.config.obs_key
            # print(input_dict)
            if input_key not in input_dict:
                print(f"Transformer expected to see key {input_key} in input_dict.")
                # Transformer token will not be created for this key
                # This acts similar to masking out the token
                continue

            key_obs = input_dict[input_key]

            if input_model.config.get("operations", None) is not None:
                for operation in input_model.config.get("operations", []):
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
                        key_obs = {input_key: key_obs}
                        key_obs = input_model(key_obs)
                    elif operation.type == "mask_multiply":
                        num_mask_dims = len(
                            input_dict[self.mask_keys[model_name]].shape
                        )
                        num_obs_dims = len(key_obs.shape)
                        extra_needed_dims = num_obs_dims - num_mask_dims
                        key_obs = key_obs * input_dict[self.mask_keys[model_name]].view(
                            *input_dict[self.mask_keys[model_name]].shape,
                            *((1,) * extra_needed_dims),
                        )
                    elif operation.type == "mask_multiply_concat":
                        num_mask_dims = len(
                            input_dict[self.mask_keys[model_name]].shape
                        )
                        num_obs_dims = len(key_obs.shape)
                        extra_needed_dims = num_obs_dims - num_mask_dims
                        key_obs = key_obs * input_dict[self.mask_keys[model_name]].view(
                            *input_dict[self.mask_keys[model_name]].shape,
                            *((1,) * extra_needed_dims),
                        )
                        key_obs = torch.cat(
                            [
                                key_obs,
                                input_dict[self.mask_keys[model_name]].view(
                                    *input_dict[self.mask_keys[model_name]].shape,
                                    *((1,) * extra_needed_dims),
                                ),
                            ],
                            dim=-1,
                        )
                    elif operation.type == "concat_obs":
                        to_add_obs = input_dict[operation.obs_key]
                        if len(to_add_obs.shape) != len(key_obs.shape):
                            to_add_obs = to_add_obs.unsqueeze(1).expand(
                                to_add_obs.shape[0],
                                key_obs.shape[1],
                                to_add_obs.shape[-1],
                            )
                        key_obs = torch.cat([key_obs, to_add_obs], dim=-1)
                    else:
                        raise NotImplementedError(
                            f"Operation {operation} not implemented"
                        )
            else:
                key_obs = {input_key: key_obs}
                key_obs = input_model(key_obs)

            if len(key_obs.shape) == 2:
                # Add a sequence dimension
                key_obs = key_obs.unsqueeze(1)

            cat_obs.append(key_obs)

            if self.mask_keys[model_name] is not None:
                key_mask = input_dict[self.mask_keys[model_name]]
                # Our mask is 1 for valid and 0 for invalid
                # The transformer expects the mask to be 0 for valid and 1 for invalid
                key_mask = key_mask.logical_not()
            else:
                key_mask = torch.zeros(
                    batch_size,
                    key_obs.shape[1],
                    dtype=torch.bool,
                    device=device,
                )
            cat_mask.append(key_mask)

        # Concatenate all the features
        cat_obs = torch.cat(cat_obs, dim=1)
        cat_mask = torch.cat(cat_mask, dim=1)

        # obs creation works in batch_first but transformer expects seq_len first
        cat_obs = cat_obs.permute(1, 0, 2).contiguous()  # [seq_len, bs, d]

        cur_mask = cat_mask.unsqueeze(1).expand(-1, cat_obs.shape[0], -1)
        cur_mask = torch.repeat_interleave(cur_mask, self.config.num_heads, dim=0)

        output = self.seqTransEncoder(cat_obs, mask=cur_mask)[0]  # [bs, d]

        return output

    def forward(self, input_dict):
        output = self.get_extracted_features(input_dict)

        if self.config.get("output_model", None) is not None:
            output = self.output_model(output)

        return output
