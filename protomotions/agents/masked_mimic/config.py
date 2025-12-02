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
from typing import Union, Optional
from enum import Enum
from protomotions.utils.config_builder import ConfigBuilder
from protomotions.agents.common.config import (
    MultiOutputModuleConfig,
    SequentialModuleConfig,
)
from protomotions.agents.base_agent.config import (
    OptimizerConfig,
    BaseAgentConfig,
    BaseModelConfig,
)


@dataclass
class KLDScheduleConfig(ConfigBuilder):
    """Configuration for KL divergence scheduling in VAE training."""

    init_kld_coeff: float = 0.0001
    end_kld_coeff: float = 0.01
    start_epoch: int = 3000
    end_epoch: int = 6000


class VaeNoiseType(Enum):
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
class VaeConfig(ConfigBuilder):
    """Configuration for VAE-specific parameters."""

    kld_schedule: KLDScheduleConfig = field(default_factory=KLDScheduleConfig)
    vae_latent_dim: int = 64
    vae_noise_type: VaeNoiseType = VaeNoiseType.NORMAL


@dataclass
class FeedForwardModelConfig(BaseModelConfig):
    """Configuration for FeedForwardModel."""

    _target_: str = "protomotions.agents.masked_mimic.model.FeedForwardModel"
    trunk: SequentialModuleConfig = field(default_factory=SequentialModuleConfig)


@dataclass
class MaskedMimicModelConfig(BaseModelConfig):
    """Configuration for MaskedMimic Model (VAE-based imitation learning)."""

    _target_: str = "protomotions.agents.masked_mimic.model.MaskedMimicModel"

    # VAE components
    encoder: MultiOutputModuleConfig = field(default_factory=MultiOutputModuleConfig)
    prior: MultiOutputModuleConfig = field(default_factory=MultiOutputModuleConfig)
    trunk: SequentialModuleConfig = field(default_factory=SequentialModuleConfig)

    vae: VaeConfig = field(default_factory=VaeConfig)

    # Optimizer
    optimizer: OptimizerConfig = field(default_factory=lambda: OptimizerConfig(lr=2e-5))


@dataclass
class MaskedMimicAgentConfig(BaseAgentConfig):
    """Main configuration class for MaskedMimic Agent."""

    _target_: str = "protomotions.agents.masked_mimic.agent.MaskedMimic"

    # Model configuration
    model: Union[MaskedMimicModelConfig, FeedForwardModelConfig] = field(
        default_factory=MaskedMimicModelConfig
    )

    expert_model_path: Optional[str] = None
