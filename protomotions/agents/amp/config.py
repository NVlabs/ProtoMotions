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
from protomotions.agents.common.config import SequentialModuleConfig
from protomotions.agents.ppo.config import (
    PPOModelConfig,
    PPOAgentConfig,
    OptimizerConfig,
)
from protomotions.utils.config_builder import ConfigBuilder


@dataclass
class AMPParametersConfig(ConfigBuilder):
    """Configuration for AMP-specific hyperparameters."""

    # Conditional discriminator configuration
    conditional_discriminator: bool = False

    # AMP-specific parameters
    discriminator_reward_w: float = 1.0  # Weight for discriminator reward

    # Discriminator training parameters
    discriminator_weight_decay: float = 0.0001
    discriminator_logit_weight_decay: float = 0.01
    discriminator_batch_size: int = 4096
    discriminator_grad_penalty: float = 5.0
    discriminator_optimization_ratio: int = (
        1  # Policy optimization steps : discriminator optimization steps
    )

    # Replay buffer parameters
    discriminator_replay_keep_prob: float = 0.01
    discriminator_replay_size: int = 200000

    # Discriminator termination parameters
    discriminator_reward_threshold: float = 0.05
    discriminator_max_cumulative_bad_transitions: int = 10


@dataclass
class DiscriminatorConfig(SequentialModuleConfig):
    """Configuration for AMP Discriminator network."""

    _target_: str = "protomotions.agents.amp.model.Discriminator"
    out_keys: List[str] = field(default_factory=lambda: ["disc_logits"])


@dataclass
class AMPModelConfig(PPOModelConfig):
    """Configuration for AMP Model (Actor-Critic with Discriminator)."""

    _target_: str = "protomotions.agents.amp.model.AMPModel"
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    discriminator_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4)
    )


@dataclass
class AMPAgentConfig(PPOAgentConfig):
    """Main configuration class for AMP Agent."""

    _target_: str = "protomotions.agents.amp.agent.AMP"

    # Override model to use AMPModel
    model: AMPModelConfig = field(default_factory=AMPModelConfig)

    # AMP-specific parameters
    amp_parameters: AMPParametersConfig = field(default_factory=AMPParametersConfig)
