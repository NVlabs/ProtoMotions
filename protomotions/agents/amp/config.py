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
"""Configuration classes for AMP (Adversarial Motion Priors) agent.

This module defines configurations for the AMP algorithm which uses a discriminator
to learn motion priors from reference motions.
"""

from typing import List, Dict
from protomotions.agents.common.config import ModuleContainerConfig
from protomotions.agents.ppo.config import (
    PPOModelConfig,
    PPOAgentConfig,
    OptimizerConfig,
)
from protomotions.envs.obs.observation_component import ObservationComponentConfig
from dataclasses import dataclass, field


@dataclass
class AMPParametersConfig:
    """Configuration for AMP-specific hyperparameters."""

    conditional_discriminator: bool = field(
        default=False,
        metadata={"help": "Whether to use conditional discriminator based on motion state."}
    )

    discriminator_reward_w: float = field(
        default=1.0,
        metadata={"help": "Weight for discriminator reward in total reward.", "min": 0.0}
    )

    discriminator_weight_decay: float = field(
        default=0.0001,
        metadata={"help": "L2 weight decay for discriminator parameters.", "min": 0.0}
    )
    discriminator_logit_weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay specifically for discriminator logit layer.", "min": 0.0}
    )
    discriminator_batch_size: int = field(
        default=4096,
        metadata={"help": "Batch size for discriminator training.", "min": 1}
    )
    discriminator_grad_penalty: float = field(
        default=5.0,
        metadata={"help": "Gradient penalty coefficient for discriminator stability.", "min": 0.0}
    )
    discriminator_optimization_ratio: int = field(
        default=1,
        metadata={"help": "Ratio of discriminator updates to policy updates.", "min": 1}
    )

    discriminator_replay_keep_prob: float = field(
        default=0.01,
        metadata={"help": "Probability to keep samples in replay buffer.", "min": 0.0, "max": 1.0}
    )
    discriminator_replay_size: int = field(
        default=200000,
        metadata={"help": "Maximum size of discriminator replay buffer.", "min": 1}
    )

    discriminator_reward_threshold: float = field(
        default=0.05,
        metadata={"help": "Threshold for discriminator reward termination.", "min": 0.0, "max": 1.0}
    )
    discriminator_max_cumulative_bad_transitions: int = field(
        default=10,
        metadata={"help": "Max bad transitions before termination.", "min": 1}
    )


@dataclass
class DiscriminatorConfig(ModuleContainerConfig):
    """Configuration for AMP Discriminator network."""

    _target_: str = "protomotions.agents.amp.model.Discriminator"
    out_keys: List[str] = field(
        default_factory=lambda: ["disc_logits"],
        metadata={"help": "Output key for discriminator logits."}
    )


@dataclass
class AMPModelConfig(PPOModelConfig):
    """Configuration for AMP Model (Actor-Critic with Discriminator)."""

    _target_: str = "protomotions.agents.amp.model.AMPModel"
    discriminator: DiscriminatorConfig = field(
        default_factory=DiscriminatorConfig,
        metadata={"help": "Discriminator network for motion prior learning."}
    )
    discriminator_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4),
        metadata={"help": "Optimizer settings for discriminator."}
    )
    disc_critic: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Critic network for discriminator reward."}
    )
    disc_critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4),
        metadata={"help": "Optimizer settings for discriminator critic."}
    )


@dataclass
class AMPAgentConfig(PPOAgentConfig):
    """Main configuration class for AMP Agent."""

    _target_: str = "protomotions.agents.amp.agent.AMP"

    model: AMPModelConfig = field(
        default_factory=AMPModelConfig,
        metadata={"help": "AMP model configuration including discriminator."}
    )

    amp_parameters: AMPParametersConfig = field(
        default_factory=AMPParametersConfig,
        metadata={"help": "AMP-specific training parameters."}
    )

    reference_obs_components: Dict[str, ObservationComponentConfig] = field(
        default_factory=dict,
        metadata={"help": "Observation components for computing reference motion features."}
    )
