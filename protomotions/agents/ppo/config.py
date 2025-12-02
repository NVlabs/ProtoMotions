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
"""Configuration classes for PPO agent.

This module defines all configuration dataclasses for the Proximal Policy Optimization (PPO)
algorithm, including actor-critic architecture parameters, optimization settings, and
training hyperparameters.

Key Classes:
    - PPOAgentConfig: Main PPO agent configuration
    - PPOModelConfig: PPO model (actor-critic) configuration
    - PPOActorConfig: Policy network configuration
    - AdvantageNormalizationConfig: Advantage normalization settings
"""

from dataclasses import dataclass, field
from typing import List, Optional
from protomotions.utils.config_builder import ConfigBuilder
from protomotions.agents.common.config import (
    SequentialModuleConfig,
    # TransformerConfig
)
from protomotions.agents.base_agent.config import (
    OptimizerConfig,
    BaseAgentConfig,
    BaseModelConfig,
)


@dataclass
class PPOActorConfig(ConfigBuilder):
    """Configuration for PPO Actor network."""

    mu_key: str  # The key of the output of the mu model
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(
        default_factory=lambda: ["action", "mean_action", "neglogp"]
    )
    _target_: str = "protomotions.agents.ppo.model.PPOActor"
    # mu_model: Union[TransformerConfig, MultiHeadedMLPConfig, DictConfig, Dict] = field(default_factory=MultiHeadedMLPConfig)
    mu_model: SequentialModuleConfig = field(default_factory=SequentialModuleConfig)
    num_out: int = None  # Will be set based on robot.number_of_actions
    actor_logstd: float = -2.9


@dataclass
class PPOModelConfig(BaseModelConfig):
    """Configuration for PPO Model (Actor-Critic)."""

    _target_: str = "protomotions.agents.ppo.model.PPOModel"
    out_keys: List[str] = field(
        default_factory=lambda: ["action", "mean_action", "neglogp", "value"]
    )
    actor: PPOActorConfig = field(default_factory=PPOActorConfig)
    # critic: Union[TransformerConfig, MultiHeadedMLPConfig, DictConfig, Dict] = field(default_factory=MultiHeadedMLPConfig)
    critic: SequentialModuleConfig = field(default_factory=SequentialModuleConfig)
    actor_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5)
    )
    critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4)
    )


@dataclass
class AdvantageNormalizationConfig(ConfigBuilder):
    """Configuration for advantage normalization."""

    enabled: bool = True
    shift_mean: bool = True
    # EMA parameters
    use_ema: bool = True
    ema_alpha: float = 0.05  # EMA weight for new data
    min_std: float = 0.02  # Safety minimum std to prevent extreme normalization
    clamp_range: float = (
        4.0  # Clamp normalized advantages (z-scores) to [-clamp_range, clamp_range]
    )


@dataclass
class PPOAgentConfig(BaseAgentConfig):
    """Main configuration class for PPO Agent."""

    _target_: str = "protomotions.agents.ppo.agent.PPO"

    # Model configuration
    model: PPOModelConfig = field(default_factory=PPOModelConfig)

    # PPO hyperparameters
    tau: float = 0.95
    e_clip: float = 0.2
    clip_critic_loss: bool = True

    # Actor update control
    actor_clip_frac_threshold: Optional[float] = (
        0.6  # Skip actor update if clip_frac > threshold (e.g., 0.5)
    )

    # Value normalization
    advantage_normalization: AdvantageNormalizationConfig = field(
        default_factory=AdvantageNormalizationConfig
    )
