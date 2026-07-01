# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for shared autoencoder-shaped models."""

from dataclasses import dataclass, field
from typing import List

from protomotions.agents.base_agent.config import BaseModelConfig
from protomotions.agents.common.config import ModuleContainerConfig


@dataclass
class AutoEncoderConfig(BaseModelConfig):
    """Generic encoder-bottleneck-decoder module configuration."""

    _target_: str = "protomotions.agents.common.autoencoder.AutoEncoder"
    encoder_out_keys: List[str] = field(
        default_factory=list, metadata={"help": "Encoder output keys."}
    )
    decoder_out_keys: List[str] = field(
        default_factory=list, metadata={"help": "Decoder output keys."}
    )
    encoder: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Encoder network configuration."},
    )
    decoder: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Decoder network configuration."},
    )
    latent_key: str = field(
        default="latent",
        metadata={"help": "TensorDict key used to pass latents to decoder."},
    )
