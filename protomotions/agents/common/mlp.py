# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-Layer Perceptron (MLP) network implementations.

This module provides MLP architectures used throughout the codebase.
All MLPs support optional observation normalization and operate on TensorDict inputs.

These are the building blocks for actor and critic networks in RL agents.

Key Classes:
    - MLPWithConcat: Feedforward network with optional observation normalization

Functions:
    - build_mlp: Factory function to construct MLP from configuration
"""

import torch
from torch import nn
from tensordict import TensorDict
from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.common.common import NormObsBase, apply_module_operations
from protomotions.agents.utils.training import get_activation_func
from protomotions.agents.common.config import (
    MLPWithConcatConfig,
)


def build_mlp(config: MLPWithConcatConfig):
    """Build a multi-layer perceptron from configuration using LazyLinear.

    Uses LazyLinear for automatic input size inference. The first forward pass
    will materialize the layers with the correct input dimensions.

    Args:
        config: MLP configuration specifying layers, activations, and output dimensions.

    Returns:
        Sequential neural network module with lazy initialization.
    """
    layers = []
    for i, layer in enumerate(config.layers):
        # Use LazyLinear - input size inferred on first forward
        layers.append(nn.LazyLinear(layer.units))
        if layer.use_layer_norm and i == 0:
            layers.append(nn.LayerNorm(layer.units))
        layers.append(get_activation_func(layer.activation))

    # Final layer also uses LazyLinear
    layers.append(nn.LazyLinear(config.num_out))
    return nn.Sequential(*layers)


class MLPWithConcat(ProtoMotionsTensorDictModule):
    """Multi-layer perceptron network with optional observation normalization.

    Feedforward network that processes observations through multiple
    fully-connected layers with configurable activations. Optionally
    normalizes inputs using running mean/std statistics.

    REQUIRES explicit obs_key and out_key to prevent key collisions.
    Always operates on TensorDict for clean, traceable data flow.

    Args:
        config: Configuration specifying input/output dimensions, hidden layers,
               and normalization settings. Both obs_key and out_key must be explicitly set.

    Attributes:
        norm: NormObsBase module for optional normalization (plain nn.Module).
        mlp: Sequential network of linear layers and activations.
        in_keys: List of input keys (always non-empty).
        out_keys: List of output keys (always non-empty).
    """

    config: MLPWithConcatConfig

    def __init__(self, config: MLPWithConcatConfig):
        ProtoMotionsTensorDictModule.__init__(self)
        self.config = config

        # Validate TensorDict keys
        assert config.in_keys, "MLP requires obs_key to be explicitly set."
        assert config.out_keys, "MLP requires out_key to be explicitly set."

        # Create normalization module (plain nn.Module with lazy init)
        self.norm = NormObsBase(config)
        self.mlp = build_mlp(self.config)

        self.output_activation = None
        if self.config.output_activation is not None:
            self.output_activation = get_activation_func(self.config.output_activation)

        # Set up TensorDict keys
        self.in_keys = self.config.in_keys
        self.out_keys = self.config.out_keys
        assert len(self.out_keys) == 1, "MLP requires exactly one output key"

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Forward pass with optional normalization.

        Args:
            tensordict: TensorDict containing observations.
            log_internals: Accepted for the common TensorDict-module contract.

        Returns:
            TensorDict with processed outputs.
        """
        combined_obs = torch.cat(
            [tensordict[key] for key in self.config.in_keys], dim=-1
        )

        result = apply_module_operations(
            combined_obs,
            self.config.module_operations,
            normalizer=self.norm,
            forward_model=self.mlp,
        )

        outs = result["output"]

        if self.output_activation is not None:
            outs = self.output_activation(outs)

        tensordict[self.config.out_keys[0]] = outs
        if self.config.normalize_obs and result["norm_obs"] is not None:
            norm_obs = result["norm_obs"]
            # Only store if batch dimension matches (reshape operations may change it)
            if norm_obs.shape[0] == tensordict.batch_size[0]:
                tensordict[f"norm_{self.config.in_keys[0]}"] = norm_obs

        return tensordict
