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
    - ObsProcessor: General observation processor for reshaping and normalizing
    - Embedding: Embedding layer for discrete inputs

Key Functions:
    - weight_init: Initialize network weights
    - get_params: Extract parameters from optimizer groups
"""

import torch
from torch import nn, Tensor
from protomotions.utils.hydra_replacement import get_class
from copy import copy
from typing import List, Dict, Optional
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.utils.normalization import RunningMeanStd
from protomotions.agents.common.config import (
    NormObsBaseConfig,
    ObsProcessorConfig,
    ModuleContainerConfig,
    ModuleOperationConfig,
    ModuleOperationForwardConfig,
    ModuleOperationPermuteConfig,
    ModuleOperationReshapeConfig,
    ModuleOperationSqueezeConfig,
    ModuleOperationUnsqueezeConfig,
    ModuleOperationExpandConfig,
    ModuleOperationSphereProjectionConfig,
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
    normalizer: NormObsBase,
    forward_model: Optional[nn.Module] = None,
) -> Dict[str, Tensor]:
    batch_size = obs.shape[0]
    norm_obs = None
    for operation in module_operations:
        if isinstance(operation, ModuleOperationPermuteConfig):
            obs = obs.permute(*operation.new_order)
        elif isinstance(operation, ModuleOperationReshapeConfig):
            new_shape = copy(operation.new_shape)
            if isinstance(new_shape[0], str) and new_shape[0] == "batch_size":
                # Use actual tensor dimension for ONNX tracing compatibility
                new_shape[0] = batch_size
            elif isinstance(new_shape[0], str) and "batch_size" in new_shape[0]:
                # Handle expressions like "batch_size * 2" - only works for concrete values
                try:
                    new_shape[0] = eval(new_shape[0].replace("batch_size", str(int(batch_size))))
                except (TypeError, ValueError):
                    # During tracing, batch_size may not be convertible to int
                    # Fall back to using -1 which PyTorch will infer
                    new_shape[0] = -1
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
                obs = obs.reshape(-1, obs_shape[-1])
            if normalizer is not None:
                obs = norm_obs = normalizer(obs)
            if len(obs_shape) > 2:
                obs = obs.reshape(*obs_shape[:-1], -1)
                if norm_obs is not None:
                    norm_obs = norm_obs.reshape(*obs_shape[:-1], -1)
            if forward_model is not None:
                obs = forward_model(obs)
        else:
            raise NotImplementedError(f"Operation {operation} not implemented")
    return_dict = {"output": obs}
    if norm_obs is not None:
        return_dict["norm_obs"] = norm_obs
    return return_dict


class ObsProcessor(TensorDictModuleBase):
    """General observation processor - applies operations and normalization.
    
    ForwardConfig applies normalization but skips the forward model (no MLP).
    """

    config: ObsProcessorConfig

    def __init__(self, config: ObsProcessorConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        assert len(config.out_keys) == 1, "ObsProcessor requires exactly one output key"

        self.norm = NormObsBase(config) if config.normalize_obs else None
        self.in_keys = config.in_keys
        self.out_keys = config.out_keys

    def forward(self, tensordict: TensorDict, *args, **kwargs) -> TensorDict:
        obs = torch.cat([tensordict[key] for key in self.in_keys], dim=-1)

        result = apply_module_operations(
            obs, self.config.module_operations, forward_model=None, normalizer=self.norm
        )
        tensordict[self.out_keys[0]] = result["output"]

        return tensordict


class ModuleContainer(TensorDictModuleBase):
    """Generic container that runs a list of modules sequentially on a TensorDict.
    
    With TensorDict, the distinction between "sequential", "parallel input",
    and "parallel output" is implicit in how keys flow between modules.
    """

    config: ModuleContainerConfig

    def __init__(self, config: ModuleContainerConfig):
        TensorDictModuleBase.__init__(self)
        self.config = config

        self.models = nn.ModuleList()
        
        # Build models and validate data flow
        # Container in_keys are "available" at the start (like sources)
        available_keys = set(self.config.in_keys)
        
        for i, model_cfg in enumerate(config.models):
            model = get_class(model_cfg._target_)(config=model_cfg)
            self.models.append(model)
            
            # Validate: each model's in_keys must be available
            # (either from container in_keys or a previous model's out_keys)
            for in_key in model.in_keys:
                assert in_key in available_keys, (
                    f"ModuleContainer: model {i} ({type(model).__name__}) requires "
                    f"input key '{in_key}' but it's not available.\n"
                    f"  Available keys: {sorted(available_keys)}\n"
                    f"  Model in_keys: {model.in_keys}"
                )
            
            # Add this model's outputs to available keys for subsequent models
            available_keys.update(model.out_keys)

        # Validate: every container out_key must have been produced
        for out_key in self.config.out_keys:
            assert out_key in available_keys, (
                f"ModuleContainer: container promises out_key '{out_key}' "
                f"but no model produces it.\n"
                f"  Available keys after all models: {sorted(available_keys)}"
            )

        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys

    def forward(self, tensordict: TensorDict, *args, **kwargs) -> TensorDict:
        """Forward through all models sequentially."""
        for model in self.models:
            tensordict = model(tensordict)
        return tensordict
