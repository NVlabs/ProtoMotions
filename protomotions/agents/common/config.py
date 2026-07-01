# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, List, Dict, Optional, Union
from dataclasses import dataclass, field

from protomotions.agents.base_agent.config import BaseModelConfig

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
        metadata={
            "help": "Whether to normalize observations using running statistics."
        },
    )
    norm_clamp_value: float = field(
        default=5.0,
        metadata={
            "help": "Clamp normalized values to [-value, value] to prevent extreme outliers.",
            "min": 0.0,
        },
    )
    norm_ema_decay: Optional[float] = field(
        default=0.999,
        metadata={
            "help": "EMA decay for obs normalization (None = Welford). "
            "Default 0.999 tracks a ~1000-sample moving window.",
        },
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

    new_order: List[int] = field(
        metadata={"help": "New dimension order, e.g. [0, 2, 1] swaps dims 1 and 2."}
    )


@dataclass
class ModuleOperationReshapeConfig(ModuleOperationConfig):
    """Configuration for module operation reshape."""

    new_shape: List[Union[int, str]] = field(
        metadata={"help": "New shape. Use 'batch_size' for dynamic batch dim."}
    )


@dataclass
class ModuleOperationSqueezeConfig(ModuleOperationConfig):
    """Configuration for module operation squeeze."""

    squeeze_dim: int = field(
        metadata={"help": "Dimension to squeeze (remove if size is 1)."}
    )


@dataclass
class ModuleOperationUnsqueezeConfig(ModuleOperationConfig):
    """Configuration for module operation unsqueeze."""

    unsqueeze_dim: int = field(
        metadata={"help": "Position where to insert new dimension of size 1."}
    )


@dataclass
class ModuleOperationExpandConfig(ModuleOperationConfig):
    """Configuration for module operation expand."""

    expand_shape: List[int] = field(
        metadata={"help": "Target shape to expand to. Use -1 to keep original size."}
    )


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
        metadata={"help": "Input tensor keys to read from TensorDict."},
    )
    out_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Output tensor keys to write to TensorDict."},
    )
    module_operations: List[ModuleOperationConfig] = field(
        default_factory=list,
        metadata={"help": "Sequence of operations to apply (reshape, permute, etc)."},
    )


# =============================================================================
# MLP Configurations (from mlp.py)
# =============================================================================


@dataclass
class MLPLayerConfig:
    """Configuration for a single MLP layer."""

    units: int = field(
        default=512, metadata={"help": "Number of neurons in this layer.", "min": 1}
    )
    activation: str = field(
        default="relu",
        metadata={
            "help": "Activation function for this layer.",
            "options": ["relu", "tanh", "elu", "selu", "gelu", "silu", "sigmoid", None],
        },
    )
    use_layer_norm: bool = field(
        default=False,
        metadata={"help": "Whether to apply layer normalization after activation."},
    )


@dataclass
class MLPWithConcatConfig(NormObsBaseConfig):
    """Configuration for Multi-Layer Perceptron with optional normalization.

    Unified MLP configuration that supports optional input normalization.
    Set normalize_obs=False if you don't want normalization (default is False).
    """

    num_out: int = field(
        default=None,
        metadata={"help": "Output dimension of the MLP. Required.", "min": 1},
    )
    layers: List[MLPLayerConfig] = field(
        default_factory=list,
        metadata={
            "help": "List of layer configurations defining the MLP architecture."
        },
    )
    _target_: str = "protomotions.agents.common.mlp.MLPWithConcat"
    in_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Input tensor keys to read and concatenate from TensorDict."},
    )
    out_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Output tensor keys to write to TensorDict."},
    )
    output_activation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Activation function for the output layer (None for linear output).",
            "options": ["relu", "tanh", "elu", "selu", "gelu", "silu", "sigmoid", None],
        },
    )
    module_operations: List[ModuleOperationConfig] = field(
        default_factory=lambda: [ModuleOperationForwardConfig()],
        metadata={
            "help": "Sequence of operations including forward pass and reshapes."
        },
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
        metadata={"help": "List of module configurations to execute sequentially."},
    )
    _target_: str = "protomotions.agents.common.common.ModuleContainer"
    in_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Input tensor keys required by this container."},
    )
    out_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Output tensor keys produced by this container."},
    )


@dataclass
class PretrainedModelConfig:
    """Configuration for loading a model from a checkpoint and returning a submodule."""

    checkpoint_path: str = field(
        default="",
        metadata={"help": "Checkpoint containing state dict."},
    )
    module_path: str = field(
        default="",
        metadata={
            "help": "Dotted module path to return from the loaded model, e.g. "
            "'actor.mu' or 'actor.mu.decoder'. Empty string returns the whole "
            "loaded model."
        },
    )
    module_config_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional dotted config path for the returned module inside "
            "the pretrained checkpoint's resolved configs."
        },
    )
    module_config: Optional[Any] = field(
        default=None,
        metadata={
            "help": "Optional embedded config for the returned module. Inference "
            "bundles use this to instantiate frozen modules whose weights are "
            "saved in the owning checkpoint."
        },
    )
    config_path: str = field(
        default="agent.model",
        metadata={"help": "Dotted config path inside resolved_configs.pt."},
    )
    state_dict_key: str = field(
        default="model",
        metadata={"help": "Top-level checkpoint key containing model weights."},
    )
    strict: bool = field(
        default=True,
        metadata={"help": "Use strict state-dict loading for the model."},
    )
    freeze: bool = field(
        default=True,
        metadata={"help": "Freeze returned module parameters after loading."},
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
        metadata={"help": "Input tensor keys (tokens and optional masks)."},
    )
    out_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "Output tensor key for transformer output (exactly one)."},
    )
    input_and_mask_mapping: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": "Maps input token keys to their mask keys for attention masking."
        },
    )

    transformer_token_size: int = field(
        default=512, metadata={"help": "Expected input token dimension size.", "min": 1}
    )
    latent_dim: int = field(
        default=512,
        metadata={"help": "Internal/output dimension of transformer.", "min": 1},
    )
    num_heads: int = field(
        default=4,
        metadata={
            "help": "Number of attention heads. Must divide latent_dim evenly.",
            "min": 1,
        },
    )
    ff_size: int = field(
        default=1024,
        metadata={"help": "Feed-forward network hidden dimension.", "min": 1},
    )
    num_layers: int = field(
        default=4, metadata={"help": "Number of transformer encoder layers.", "min": 1}
    )
    dropout: float = field(
        default=0.0,
        metadata={
            "help": "Dropout probability. Default 0 since RL has enough noise.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    activation: str = field(
        default="relu",
        metadata={
            "help": "Activation function for feed-forward layers.",
            "options": ["relu", "gelu", "silu"],
        },
    )
    output_activation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional activation for transformer output.",
            "options": ["relu", "tanh", "gelu", "silu", None],
        },
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


@dataclass
class DiscreteAutoregressiveTransformerConfig:
    """Configuration for categorical autoregressive transformer modules."""

    _target_: str = (
        "protomotions.agents.common.autoregressive."
        "DiscreteAutoregressiveTransformer"
    )

    in_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "TensorDict input keys. Defaults to context_key and token_key."},
    )
    out_keys: List[str] = field(
        default_factory=list,
        metadata={"help": "TensorDict output keys. Defaults to logits_key."},
    )
    context_key: str = field(
        default="context",
        metadata={"help": "TensorDict key containing context observations."},
    )
    token_key: str = field(
        default="tokens",
        metadata={"help": "TensorDict key containing teacher-forced tokens."},
    )
    logits_key: str = field(
        default="logits",
        metadata={"help": "TensorDict key for token logits."},
    )
    generated_tokens_key: Optional[str] = field(
        default=None,
        metadata={"help": "TensorDict key for generated token indices."},
    )
    logprob_key: Optional[str] = field(
        default=None,
        metadata={"help": "Optional TensorDict key for generated-token log probabilities."},
    )
    context_embedding_key: str = field(
        default="_ar_context_embedding",
        metadata={"help": "Internal TensorDict key for context embeddings."},
    )
    token_embedding_key: str = field(
        default="_ar_token_embedding",
        metadata={"help": "Internal TensorDict key for token embeddings."},
    )
    hidden_key: str = field(
        default="_ar_hidden",
        metadata={"help": "Internal TensorDict key for transformer hidden states."},
    )

    context_encoder: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Context projection container."},
    )
    token_encoder: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Teacher-token projection container."},
    )
    output_head: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Logit projection container."},
    )

    d_model: int = field(default=1024, metadata={"help": "Transformer hidden size."})
    num_heads: int = field(default=4, metadata={"help": "Number of attention heads."})
    num_layers: int = field(default=6, metadata={"help": "Number of transformer layers."})
    ff_size: int = field(default=4096, metadata={"help": "Feed-forward hidden size."})
    dropout: float = field(default=0.1, metadata={"help": "Dropout probability."})
    activation: str = field(default="gelu", metadata={"help": "Transformer activation."})
    num_tokens: int = field(
        default=1,
        metadata={
            "help": "Number of generated tokens. Use 0 when an owning module derives "
            "the token shape before instantiating the transformer."
        },
    )
    vocab_size: int = field(
        default=2,
        metadata={
            "help": "Categorical vocabulary size. Use 0 when an owning module derives "
            "the token shape before instantiating the transformer."
        },
    )
    max_seq_len: int = field(
        default=0,
        metadata={"help": "Maximum sequence length including context. 0 uses num_tokens + 1."},
    )

    def __post_init__(self):
        def container_in_keys(container):
            if container.in_keys:
                return list(container.in_keys)
            if container.models:
                return list(getattr(container.models[0], "in_keys", []))
            return []

        def container_out_keys(container):
            if container.out_keys:
                return list(container.out_keys)
            if container.models:
                return list(getattr(container.models[-1], "out_keys", []))
            return []

        context_in_keys = container_in_keys(self.context_encoder)
        context_out_keys = container_out_keys(self.context_encoder)
        if context_out_keys:
            assert len(context_out_keys) == 1, (
                "DiscreteAutoregressiveTransformer context_encoder must produce "
                f"one embedding key, got {context_out_keys}"
            )
            self.context_embedding_key = context_out_keys[0]
        elif self.context_encoder.models:
            raise AssertionError(
                "DiscreteAutoregressiveTransformer context_encoder must declare "
                "one output key."
            )
        else:
            context_in_keys = [self.context_key]
            context_out_keys = [self.context_embedding_key]

        if not self.context_encoder.in_keys:
            self.context_encoder.in_keys = context_in_keys
        if not self.context_encoder.out_keys:
            self.context_encoder.out_keys = context_out_keys

        if not self.in_keys:
            self.in_keys = list(dict.fromkeys(context_in_keys + [self.token_key]))
        if not self.out_keys:
            self.out_keys = [self.logits_key]
        elif self.logits_key == "logits":
            self.logits_key = self.out_keys[0]

        missing_context_keys = [
            key for key in context_in_keys if key not in self.in_keys
        ]
        assert not missing_context_keys, (
            f"context keys {missing_context_keys} must be in in_keys {self.in_keys}"
        )
        assert self.token_key in self.in_keys, (
            f"token_key '{self.token_key}' must be in in_keys {self.in_keys}"
        )
        assert (
            len(self.out_keys) == 1
        ), "DiscreteAutoregressiveTransformer supports one logits output key"
        assert self.num_tokens >= 0, "num_tokens must be non-negative"
        assert self.vocab_size >= 0, "vocab_size must be non-negative"

        if not self.context_encoder.models:
            self.context_encoder = ModuleContainerConfig(
                in_keys=[self.context_key],
                out_keys=[self.context_embedding_key],
                models=[
                    MLPWithConcatConfig(
                        in_keys=[self.context_key],
                        out_keys=[self.context_embedding_key],
                        num_out=self.d_model,
                        layers=[
                            MLPLayerConfig(
                                units=self.d_model,
                                activation=self.activation,
                            )
                        ],
                    )
                ],
            )
        if not self.token_encoder.models:
            self.token_encoder = ModuleContainerConfig(
                in_keys=[self.token_key],
                out_keys=[self.token_embedding_key],
                models=[
                    MLPWithConcatConfig(
                        in_keys=[self.token_key],
                        out_keys=[self.token_embedding_key],
                        num_out=self.d_model,
                        layers=[],
                    )
                ],
            )
        if not self.output_head.models:
            self.output_head = ModuleContainerConfig(
                in_keys=[self.hidden_key],
                out_keys=[self.logits_key],
                models=[
                    MLPWithConcatConfig(
                        in_keys=[self.hidden_key],
                        out_keys=[self.logits_key],
                        num_out=self.vocab_size,
                        layers=[],
                    )
                ],
            )
