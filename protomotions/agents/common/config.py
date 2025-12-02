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
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Union
from protomotions.utils.config_builder import ConfigBuilder

# =============================================================================
# Base Configuration for Normalized Observations
# =============================================================================


@dataclass
class NormObsBaseConfig(ConfigBuilder):
    """Base configuration for modules that support optional observation normalization.

    With LazyLinear, only num_out is needed - input sizes are inferred automatically.
    This is purely about normalization settings and output dimensions.
    Individual TensorDictModules add their own obs_key/out_key fields as needed.
    """

    normalize_obs: bool = False
    norm_clamp_value: float = 5.0


# =============================================================================
# Common Module Configurations (from common.py)
# =============================================================================


@dataclass
class ModuleOperationConfig(ConfigBuilder):
    """Configuration for module operations."""


@dataclass
class ModuleOperationForwardConfig(ModuleOperationConfig):
    """Configuration for module operation forward."""


@dataclass
class ModuleOperationPermuteConfig(ModuleOperationConfig):
    """Configuration for module operation permute."""

    new_order: List[int]


@dataclass
class ModuleOperationReshapeConfig(ModuleOperationConfig):
    """Configuration for module operation reshape."""

    new_shape: List[Union[int, str]]


@dataclass
class ModuleOperationSqueezeConfig(ModuleOperationConfig):
    """Configuration for module operation squeeze."""

    squeeze_dim: int


@dataclass
class ModuleOperationUnsqueezeConfig(ModuleOperationConfig):
    """Configuration for module operation unsqueeze."""

    unsqueeze_dim: int


@dataclass
class ModuleOperationExpandConfig(ModuleOperationConfig):
    """Configuration for module operation expand."""

    expand_shape: List[int]


@dataclass
class ModuleOperationSphereProjectionConfig(ModuleOperationConfig):
    """Configuration for sphere projection operation (L2 normalization to unit sphere)."""


@dataclass
class FlattenConfig(NormObsBaseConfig):
    """Configuration for Flatten module."""

    _target_: str = "protomotions.agents.common.common.Flatten"
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(default_factory=list)
    module_operations: List[ModuleOperationConfig] = field(
        default_factory=lambda: [ModuleOperationForwardConfig()]
    )


# =============================================================================
# MLP Configurations (from mlp.py)
# =============================================================================


@dataclass
class MLPLayerConfig(ConfigBuilder):
    """Configuration for a single MLP layer."""

    units: int = 512
    activation: str = "relu"
    use_layer_norm: bool = False


@dataclass
class MLPWithConcatConfig(NormObsBaseConfig):
    """Configuration for Multi-Layer Perceptron with optional normalization.

    Unified MLP configuration that supports optional input normalization.
    Set normalize_obs=False if you don't want normalization (default is False).
    obs_key and out_key are optional in config but validated in MLP module.
    """

    num_out: int = None
    layers: List[MLPLayerConfig] = None
    # For example:
    # field(default_factory=lambda: [
    #     MLPLayerConfig(units=1024, activation="relu", use_layer_norm=False),
    #     MLPLayerConfig(units=1024, activation="relu", use_layer_norm=False),
    #     MLPLayerConfig(units=512, activation="relu", use_layer_norm=False)
    # ])
    _target_: str = "protomotions.agents.common.mlp.MLPWithConcat"
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(default_factory=list)
    output_activation: Optional[str] = None
    module_operations: List[ModuleOperationConfig] = field(
        default_factory=lambda: [ModuleOperationForwardConfig()]
    )

    def __post_init__(self):
        assert self.num_out is not None, "num_out must be provided"
        assert self.layers is not None, "layers must be provided"


@dataclass
class MultiInputModuleConfig(ConfigBuilder):
    """Configuration for Multi-Headed MLP."""

    input_models: List[Any]
    _target_: str = "protomotions.agents.common.common.MultiInputModule"
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(default_factory=list)


@dataclass
class SequentialModuleConfig(ConfigBuilder):
    """Configuration for a sequential model."""

    input_models: List[Any]
    _target_: str = "protomotions.agents.common.common.SequentialModule"
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(default_factory=list)


@dataclass
class MultiOutputModuleConfig(ConfigBuilder):
    """Configuration for a multi-output model (one input, many outputs)."""

    output_models: List[Any]
    _target_: str = "protomotions.agents.common.common.MultiOutputModule"
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(default_factory=list)


# =============================================================================
# Transformer Configurations (from transformer.py)
# =============================================================================


@dataclass
class TransformerConfig(ConfigBuilder):
    """Configuration for Transformer encoder."""

    _target_: str = "protomotions.agents.common.transformer.Transformer"
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(default_factory=list)
    input_and_mask_mapping: Optional[Dict[str, str]] = None

    transformer_token_size: int = 512
    latent_dim: int = 512
    num_heads: int = 4
    ff_size: int = 1024
    num_layers: int = 4
    dropout: float = 0  # By default turned off, as RL has enough noise already
    activation: str = "relu"

    output_activation: Optional[str] = None

    def __post_init__(self):
        if self.input_and_mask_mapping is not None:
            mask_keys = self.input_and_mask_mapping.values()
            for key in mask_keys:
                assert (
                    key in self.in_keys
                ), f"Key {key} is defined as a mask key but not in in_keys {self.in_keys}"

            input_keys = self.input_and_mask_mapping.keys()
            for key in input_keys:
                assert (
                    key in self.in_keys
                ), f"Key {key} is defined as an input key to be masked but not in in_keys {self.in_keys}"

        assert len(self.out_keys) == 1, "Transformer supports exactly one output key"
