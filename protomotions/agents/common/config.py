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
from typing import Any, List, Dict, Optional, Union
from dataclasses import dataclass, field

# =============================================================================
# Base Configuration for Normalized Observations
# =============================================================================


@dataclass
class NormObsBaseConfig:
    """Base configuration for modules that support optional observation normalization.

    With LazyLinear, only num_out is needed - input sizes are inferred automatically.
    This is purely about normalization settings and output dimensions.
    Individual TensorDictModules add their own obs_key/out_key fields as needed.
    """

    normalize_obs: bool = field(
        default=False,
        metadata={"help": "Whether to normalize observations using running statistics."}
    )
    norm_clamp_value: float = field(
        default=5.0,
        metadata={
            "help": "Clamp normalized values to [-value, value] to prevent extreme outliers.",
            "min": 0.0,
        }
    )


# =============================================================================
# Common Module Configurations (from common.py)
# =============================================================================


@dataclass
class ModuleOperationConfig:
    """Configuration for module operations."""


@dataclass
class ModuleOperationForwardConfig(ModuleOperationConfig):
    """Configuration for module operation forward."""


@dataclass
class ModuleOperationPermuteConfig(ModuleOperationConfig):
    """Configuration for module operation permute."""

    new_order: List[int] = field(metadata={"help": "New dimension order, e.g. [0, 2, 1] swaps dims 1 and 2."})


@dataclass
class ModuleOperationReshapeConfig(ModuleOperationConfig):
    """Configuration for module operation reshape."""

    new_shape: List[Union[int, str]] = field(metadata={"help": "New shape. Use 'batch_size' for dynamic batch dim."})


@dataclass
class ModuleOperationSqueezeConfig(ModuleOperationConfig):
    """Configuration for module operation squeeze."""

    squeeze_dim: int = field(metadata={"help": "Dimension to squeeze (remove if size is 1)."})


@dataclass
class ModuleOperationUnsqueezeConfig(ModuleOperationConfig):
    """Configuration for module operation unsqueeze."""

    unsqueeze_dim: int = field(metadata={"help": "Position where to insert new dimension of size 1."})


@dataclass
class ModuleOperationExpandConfig(ModuleOperationConfig):
    """Configuration for module operation expand."""

    expand_shape: List[int] = field(metadata={"help": "Target shape to expand to. Use -1 to keep original size."})


@dataclass
class ModuleOperationSphereProjectionConfig(ModuleOperationConfig):
    """Configuration for sphere projection operation (L2 normalization to unit sphere)."""


@dataclass
class ObsProcessorConfig(NormObsBaseConfig):
    """General observation processor - applies operations and normalization.
    
    Supports all module_operations. ForwardConfig applies normalization but skips the forward
    model (no MLP). Useful for reshaping, normalizing, and other tensor manipulations.
    """

    _target_: str = "protomotions.agents.common.common.ObsProcessor"
    in_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Input tensor keys to read from TensorDict."}
    )
    out_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Output tensor keys to write to TensorDict."}
    )
    module_operations: List[ModuleOperationConfig] = field(
        default_factory=list,
        metadata={"help": "Sequence of operations to apply (reshape, permute, etc)."}
    )


# =============================================================================
# MLP Configurations (from mlp.py)
# =============================================================================


@dataclass
class MLPLayerConfig:
    """Configuration for a single MLP layer."""

    units: int = field(
        default=512,
        metadata={"help": "Number of neurons in this layer.", "min": 1}
    )
    activation: str = field(
        default="relu",
        metadata={
            "help": "Activation function for this layer.",
            "options": ["relu", "tanh", "elu", "selu", "gelu", "silu", "sigmoid", None],
        }
    )
    use_layer_norm: bool = field(
        default=False,
        metadata={"help": "Whether to apply layer normalization after activation."}
    )


@dataclass
class MLPWithConcatConfig(NormObsBaseConfig):
    """Configuration for Multi-Layer Perceptron with optional normalization.

    Unified MLP configuration that supports optional input normalization.
    Set normalize_obs=False if you don't want normalization (default is False).
    """

    num_out: int = field(
        default=None,
        metadata={"help": "Output dimension of the MLP. Required.", "min": 1}
    )
    layers: List[MLPLayerConfig] = field(
        default_factory=list,
        metadata={"help": "List of layer configurations defining the MLP architecture."}
    )
    _target_: str = "protomotions.agents.common.mlp.MLPWithConcat"
    in_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Input tensor keys to read and concatenate from TensorDict."}
    )
    out_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Output tensor keys to write to TensorDict."}
    )
    output_activation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Activation function for the output layer (None for linear output).",
            "options": ["relu", "tanh", "elu", "selu", "gelu", "silu", "sigmoid", None],
        }
    )
    module_operations: List[ModuleOperationConfig] = field(
        default_factory=lambda: [ModuleOperationForwardConfig()],
        metadata={"help": "Sequence of operations including forward pass and reshapes."}
    )

    def __post_init__(self):
        assert self.num_out is not None, "num_out must be provided"
        assert self.layers is not None, "layers must be provided"


@dataclass
class ModuleContainerConfig:
    """Configuration for a container of modules that are executed sequentially.
    
    Modules are processed in order, with each module's outputs available to subsequent modules.
    Input keys are passed through, and all specified output keys must be produced by internal modules.
    """

    models: List[Any] = field(
        default_factory=list,
        metadata={"help": "List of module configurations to execute sequentially."}
    )
    _target_: str = "protomotions.agents.common.common.ModuleContainer"
    in_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Input tensor keys required by this container."}
    )
    out_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Output tensor keys produced by this container."}
    )


# =============================================================================
# Transformer Configurations (from transformer.py)
# =============================================================================


@dataclass
class TransformerConfig:
    """Configuration for Transformer encoder.
    
    Multi-head self-attention transformer that processes tokenized inputs.
    Supports optional masking for variable-length sequences.
    """

    _target_: str = "protomotions.agents.common.transformer.Transformer"
    in_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Input tensor keys (tokens and optional masks)."}
    )
    out_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Output tensor key for transformer output (exactly one)."}
    )
    input_and_mask_mapping: Optional[Dict[str, str]] = field(
        default=None,
        metadata={"help": "Maps input token keys to their mask keys for attention masking."}
    )

    transformer_token_size: int = field(
        default=512,
        metadata={"help": "Expected input token dimension size.", "min": 1}
    )
    latent_dim: int = field(
        default=512,
        metadata={"help": "Internal/output dimension of transformer.", "min": 1}
    )
    num_heads: int = field(
        default=4,
        metadata={"help": "Number of attention heads. Must divide latent_dim evenly.", "min": 1}
    )
    ff_size: int = field(
        default=1024,
        metadata={"help": "Feed-forward network hidden dimension.", "min": 1}
    )
    num_layers: int = field(
        default=4,
        metadata={"help": "Number of transformer encoder layers.", "min": 1}
    )
    dropout: float = field(
        default=0.0,
        metadata={
            "help": "Dropout probability. Default 0 since RL has enough noise.",
            "min": 0.0,
            "max": 1.0,
        }
    )
    activation: str = field(
        default="relu",
        metadata={
            "help": "Activation function for feed-forward layers.",
            "options": ["relu", "gelu", "silu"],
        }
    )
    output_activation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional activation for transformer output.",
            "options": ["relu", "tanh", "gelu", "silu", None],
        }
    )

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
