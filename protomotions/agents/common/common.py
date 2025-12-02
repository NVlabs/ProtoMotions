# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Common neural network components and utilities for agents.

This module provides shared building blocks used across different agent architectures,
including observation normalization, weight initialization, and specialized layers.

Key Classes:
    - NormObsBase: Base class for modules with observation normalization
    - Flatten: Flattening layer with flexible dimensions
    - Embedding: Embedding layer for discrete inputs

Key Functions:
    - weight_init: Initialize network weights
    - get_params: Extract parameters from optimizer groups
"""

import torch
from torch import nn, Tensor
from protomotions.utils.hydra_replacement import get_class
from copy import copy
from typing import List, Dict
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.utils.normalization import RunningMeanStd
from protomotions.agents.common.config import (
    NormObsBaseConfig,
    FlattenConfig,
    SequentialModuleConfig,
    MultiInputModuleConfig,
    ModuleOperationConfig,
    ModuleOperationForwardConfig,
    ModuleOperationPermuteConfig,
    ModuleOperationReshapeConfig,
    ModuleOperationSqueezeConfig,
    ModuleOperationUnsqueezeConfig,
    ModuleOperationExpandConfig,
    ModuleOperationSphereProjectionConfig,
    MultiOutputModuleConfig,
)


def get_params(obj) -> List[nn.Parameter]:
    """Extract parameters from optimizer parameter groups.

    Handles both flat lists of parameters and grouped parameters (as used by optimizers).

    Args:
        obj: Either a list of nn.Parameter or a list of parameter groups (dicts).

    Returns:
        Flat list of all parameters.
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
    """Initialize weights for neural network modules.

    Applies appropriate initialization to linear layers and other modules.
    Linear layers get orthogonal or default initialization with zero bias.

    Args:
        m: Neural network module to initialize.
        orthogonal: If True, use orthogonal initialization for linear layers.
    """
    if isinstance(m, nn.Linear):
        if orthogonal:
            nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


class NormObsBase(nn.Module):
    """Base class for modules with observation normalization.

    Provides running mean/std normalization of observations using exponential
    moving averages. Normalization statistics are updated during training and
    frozen during evaluation.

    Uses lazy initialization - input shape is inferred on first forward pass.

    This is a simple tensor-to-tensor module. Subclasses handle TensorDict extraction/insertion.

    Args:
        config: Configuration specifying output dimensions and normalization parameters.

    Attributes:
        running_obs_norm: RunningMeanStd module for observation normalization (lazy).
        num_out: Output dimension.
    """

    def __init__(self, config: NormObsBaseConfig):
        super().__init__()
        self.config = config
        self.build_norm()

    def build_norm(self):
        if self.config.normalize_obs:
            # Use lazy initialization - shape inferred on first forward
            self.running_obs_norm = RunningMeanStd(
                fabric=None,
                shape=None,  # Lazy: inferred from first input
                device="cpu",
                clamp_value=self.config.norm_clamp_value,
            )

    def forward(self, obs: Tensor) -> Tensor:
        """Forward pass that normalizes observations.

        Args:
            obs: Observation tensor to normalize.

        Returns:
            Normalized observation tensor.
        """
        if self.config.normalize_obs:
            norm_obs = self.running_obs_norm.normalize(obs)
            if self.training:
                self.running_obs_norm.record_moments(obs)
        else:
            norm_obs = obs

        return norm_obs


def apply_module_operations(
    obs: Tensor,
    module_operations: List[ModuleOperationConfig],
    forward_model: nn.Module,
    normalizer: NormObsBase,
) -> Dict[str, Tensor]:
    batch_size = obs.shape[0]
    norm_obs = None
    for operation in module_operations:
        if isinstance(operation, ModuleOperationPermuteConfig):
            obs = obs.permute(*operation.new_order)
        elif isinstance(operation, ModuleOperationReshapeConfig):
            new_shape = copy(operation.new_shape)
            if isinstance(new_shape[0], str) and "batch_size" in new_shape[0]:
                new_shape[0] = eval(new_shape[0].replace("batch_size", str(batch_size)))
            obs = obs.reshape(*new_shape)
        elif isinstance(operation, ModuleOperationSqueezeConfig):
            obs = obs.squeeze(dim=operation.squeeze_dim)
        elif isinstance(operation, ModuleOperationUnsqueezeConfig):
            obs = obs.unsqueeze(dim=operation.unsqueeze_dim)
        elif isinstance(operation, ModuleOperationExpandConfig):
            obs = obs.expand(*operation.expand_shape)
        elif isinstance(operation, ModuleOperationSphereProjectionConfig):
            obs = torch.nn.functional.normalize(obs, dim=-1)
        elif isinstance(operation, ModuleOperationForwardConfig):
            obs_shape = obs.shape
            if len(obs_shape) > 2:
                # Normalizer and MLP forward modules expect 2D inputs
                obs = obs.reshape(-1, obs_shape[-1])
            if normalizer is not None:
                obs = norm_obs = normalizer(obs)
            if len(obs_shape) > 2:
                obs = obs.reshape(*obs_shape[:-1], -1)
                if norm_obs is not None:
                    norm_obs = norm_obs.reshape(*obs_shape[:-1], -1)
            obs = forward_model(obs)
        else:
            raise NotImplementedError(f"Operation {operation} not implemented")
    return_dict = {"output": obs}
    if norm_obs is not None:
        return_dict["norm_obs"] = norm_obs
    return return_dict


class Flatten(TensorDictModuleBase):
    """Flatten layer with observation normalization for TensorDict inputs.

    Flattens input tensor and optionally normalizes it.

    Args:
        config: Configuration specifying obs_key, normalization parameters, etc.

    Attributes:
        norm: NormObsBase module for normalization.
        flatten: Flatten layer.
        in_keys: List containing obs_key.
        out_keys: List containing out_key.
    """

    config: FlattenConfig

    def __init__(self, config: FlattenConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        # Validate TensorDict keys
        assert len(config.out_keys) == 1, "Flatten requires exactly one output key"

        # Create the normalization base module (plain nn.Module)
        self.norm = NormObsBase(config)
        self.flatten = nn.Flatten()

        # Set up TensorDict keys
        self.in_keys = config.in_keys
        self.out_keys = config.out_keys

    def forward(self, tensordict: TensorDict, *args, **kwargs) -> TensorDict:
        """Forward pass that flattens and normalizes observations.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with flattened and normalized observations.
        """
        # Extract, flatten, normalize, and store back
        obs = torch.cat([tensordict[key] for key in self.in_keys], dim=-1)

        result = apply_module_operations(
            obs, self.config.module_operations, self.flatten, self.norm
        )
        tensordict[self.out_keys[0]] = result["output"]
        if result["norm_obs"] is not None:
            tensordict[f"norm_{self.in_keys[0]}"] = result["norm_obs"]

        return tensordict


class SequentialModule(TensorDictModuleBase):
    """Sequential model with multiple input models and a trunk."""

    config: SequentialModuleConfig

    def __init__(self, config: SequentialModuleConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        sequential_models = []
        for input_model in config.input_models:
            model = get_class(input_model._target_)(config=input_model)
            sequential_models.append(model)
        self.sequential_models = nn.Sequential(*sequential_models)

        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys

        for in_key in self.sequential_models[0].in_keys:
            assert (
                in_key in self.in_keys
            ), f"SequentialModule input key {in_key} not in in_keys {self.in_keys}"
        for out_key in self.sequential_models[-1].out_keys:
            assert (
                out_key in self.out_keys
            ), f"SequentialModule output key {out_key} not in out_keys {self.out_keys}"

    def forward(self, tensordict: TensorDict, *args, **kwargs) -> TensorDict:
        for model in self.sequential_models:
            tensordict = model(tensordict)
        return tensordict


class MultiInputModule(TensorDictModuleBase):
    config: MultiInputModuleConfig

    def __init__(self, config: MultiInputModuleConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        self.input_models = nn.ModuleList()
        for input_model in config.input_models:
            model = get_class(input_model._target_)(config=input_model)
            self.input_models.append(model)

            for in_key in model.in_keys:
                assert (
                    in_key in self.config.in_keys
                ), f"MultiInputModule input key {in_key} not in in_keys {self.config.in_keys}"
            for out_key in model.out_keys:
                assert (
                    out_key in self.config.out_keys
                ), f"MultiInputModule output key {out_key} not in out_keys {self.config.out_keys}"

        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys

    def forward(self, tensordict: TensorDict, *args, **kwargs) -> TensorDict:
        for model in self.input_models:
            tensordict = model(tensordict)
        return tensordict


class MultiOutputModule(TensorDictModuleBase):
    """Takes single input key, passes to multiple output heads in parallel.

    Opposite of MultiInputModule - one input, multiple outputs.
    Useful for multi-head architectures like ASE (discriminator + encoder heads).
    """

    config: MultiOutputModuleConfig

    def __init__(self, config: MultiOutputModuleConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        self.output_models = nn.ModuleList()
        for output_model in self.config.output_models:
            model = get_class(output_model._target_)(config=output_model)
            for key in model.in_keys:
                assert (
                    key in self.config.in_keys
                ), f"MultiOutputModule input key {key} not in in_keys {self.config.in_keys}"
            for key in model.out_keys:
                assert (
                    key in self.config.out_keys
                ), f"MultiOutputModule output key {key} not in out_keys {self.config.out_keys}"
            self.output_models.append(model)

        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys

    def forward(self, tensordict: TensorDict, *args, **kwargs) -> TensorDict:
        """Forward through all output heads in parallel.

        Each head reads from in_keys and writes to its own out_key.
        """
        for model in self.output_models:
            tensordict = model(tensordict)
        return tensordict
