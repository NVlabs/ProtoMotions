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
from typing import List
from protomotions.utils.config_builder import ConfigBuilder
from protomotions.agents.amp.config import AMPAgentConfig, DiscriminatorConfig


@dataclass
class ASEParametersConfig(ConfigBuilder):
    """Configuration for ASE-specific hyperparameters."""

    # Latent variable configuration
    latent_dim: int = 64
    latent_steps_min: int = 1
    latent_steps_max: int = 150

    # Mutual Information reward configuration
    mi_reward_w: float = 0.5
    mi_hypersphere_reward_shift: bool = True

    # Encoder regularization
    mi_enc_weight_decay: float = 0
    mi_enc_grad_penalty: float = 0

    # Diversity bonus configuration
    diversity_bonus: float = 0.01
    diversity_tar: float = 1.0

    # Uniformity loss configuration
    latent_uniformity_weight: float = 0.1
    uniformity_kernel_scale: float = 1.0


@dataclass
class ASEDiscriminatorEncoderConfig(DiscriminatorConfig):
    """Configuration for ASE Discriminator-Encoder network (extends SequentialModuleConfig)."""

    encoder_out_size: int = None  # Should match latent_dim
    _target_: str = "protomotions.agents.ase.model.ASEDiscriminatorEncoder"
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(
        default_factory=lambda: ["disc_logits", "mi_enc_output"]
    )

    def __post_init__(self):
        getattr(
            super(), "__post_init__", lambda: None
        )()  # Call super's __post_init__ if it exists
        assert self.encoder_out_size is not None, "encoder_out_size must be provided"


@dataclass
class ASEAgentConfig(AMPAgentConfig):
    """Main configuration class for ASE Agent."""

    _target_: str = "protomotions.agents.ase.agent.ASE"

    # ASE-specific parameters
    ase_parameters: ASEParametersConfig = field(default_factory=ASEParametersConfig)
