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
"""Configuration classes for MaskedMimic agent.

MaskedMimic uses a VAE-based architecture for versatile motion imitation
with masked conditioning and latent space learning.
"""

from typing import Union, Optional
from enum import Enum
from protomotions.agents.common.config import ModuleContainerConfig
from protomotions.agents.base_agent.config import (
    OptimizerConfig,
    BaseAgentConfig,
    BaseModelConfig,
)
from dataclasses import dataclass, field


@dataclass
class KLDScheduleConfig:
    """Configuration for KL divergence scheduling in VAE training."""

    init_kld_coeff: float = field(
        default=0.0001,
        metadata={"help": "Initial KL divergence coefficient.", "min": 0.0}
    )
    end_kld_coeff: float = field(
        default=0.01,
        metadata={"help": "Final KL divergence coefficient.", "min": 0.0}
    )
    start_epoch: int = field(
        default=3000,
        metadata={"help": "Epoch to start KLD coefficient annealing.", "min": 0}
    )
    end_epoch: int = field(
        default=6000,
        metadata={"help": "Epoch to end KLD coefficient annealing.", "min": 0}
    )


class VaeNoiseType(Enum):
    """Type of noise for VAE sampling."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    ZEROS = "zeros"

    @classmethod
    def from_str(cls, value: str) -> "VaeNoiseType":
        """Create enum from string, case-insensitive."""
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. "
                f"Valid values are: {[e.value for e in cls]}"
            )
        return cls(value)


@dataclass
class VaeConfig:
    """Configuration for VAE-specific parameters."""

    kld_schedule: KLDScheduleConfig = field(
        default_factory=KLDScheduleConfig,
        metadata={"help": "KL divergence annealing schedule."}
    )
    vae_latent_dim: int = field(
        default=64,
        metadata={"help": "Dimension of VAE latent space.", "min": 1}
    )
    vae_noise_type: VaeNoiseType = field(
        default=VaeNoiseType.NORMAL,
        metadata={"help": "Type of noise for latent sampling: normal, uniform, or zeros."}
    )


@dataclass
class FeedForwardModelConfig(BaseModelConfig):
    """Configuration for FeedForwardModel (non-VAE variant)."""

    _target_: str = "protomotions.agents.masked_mimic.model.FeedForwardModel"
    trunk: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Main trunk network for forward pass."}
    )


@dataclass
class MaskedMimicModelConfig(BaseModelConfig):
    """Configuration for MaskedMimic Model (VAE-based imitation learning)."""

    _target_: str = "protomotions.agents.masked_mimic.model.MaskedMimicModel"

    encoder: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "VAE encoder network (maps observations to latent)."}
    )
    prior: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Prior network for latent distribution."}
    )
    trunk: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Decoder trunk network (latent to actions)."}
    )

    vae: VaeConfig = field(
        default_factory=VaeConfig,
        metadata={"help": "VAE configuration (latent dim, KLD schedule, etc)."}
    )

    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for model training."}
    )


@dataclass
class MaskedMimicAgentConfig(BaseAgentConfig):
    """Main configuration class for MaskedMimic Agent."""

    _target_: str = "protomotions.agents.masked_mimic.agent.MaskedMimic"

    model: Union[MaskedMimicModelConfig, FeedForwardModelConfig] = field(
        default_factory=MaskedMimicModelConfig,
        metadata={"help": "Model configuration (VAE or FeedForward variant)."}
    )

    expert_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-trained expert model checkpoint."}
    )
