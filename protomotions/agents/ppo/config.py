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

from typing import List, Optional
from dataclasses import dataclass, field
from protomotions.agents.common.config import (
    ModuleContainerConfig,
)
from protomotions.agents.base_agent.config import (
    OptimizerConfig,
    BaseAgentConfig,
    BaseModelConfig,
)


@dataclass
class PPOActorConfig:
    """Configuration for PPO Actor network."""

    mu_key: str = field(metadata={"help": "The key of the output of the mu model."})
    in_keys: List[str] = field(default_factory=list, metadata={"help": "Input observation keys."})
    out_keys: List[str] = field(
        default_factory=lambda: ["action", "mean_action", "neglogp"],
        metadata={"help": "Output keys: action, mean_action, neglogp."}
    )
    _target_: str = "protomotions.agents.ppo.model.PPOActor"
    mu_model: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Neural network model for action mean."}
    )
    num_out: int = field(default=None, metadata={"help": "Number of actions. Set from robot config."})
    actor_logstd: float = field(default=-2.9, metadata={"help": "Initial log std for action distribution."})


@dataclass
class PPOModelConfig(BaseModelConfig):
    """Configuration for PPO Model (Actor-Critic)."""

    _target_: str = "protomotions.agents.ppo.model.PPOModel"
    out_keys: List[str] = field(
        default_factory=lambda: ["action", "mean_action", "neglogp", "value"],
        metadata={"help": "Output keys including actions and value estimate."}
    )
    actor: PPOActorConfig = field(
        default_factory=PPOActorConfig,
        metadata={"help": "Actor (policy) network configuration."}
    )
    critic: ModuleContainerConfig = field(
        default_factory=ModuleContainerConfig,
        metadata={"help": "Critic (value) network configuration."}
    )
    actor_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=2e-5),
        metadata={"help": "Optimizer settings for actor network."}
    )
    critic_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-4),
        metadata={"help": "Optimizer settings for critic network."}
    )


@dataclass
class AdvantageNormalizationConfig:
    """Configuration for advantage normalization."""

    enabled: bool = field(default=True, metadata={"help": "Whether to normalize advantages."})
    shift_mean: bool = field(default=True, metadata={"help": "Subtract mean from advantages."})
    # EMA parameters
    use_ema: bool = field(default=True, metadata={"help": "Use EMA for normalization statistics."})
    ema_alpha: float = field(default=0.05, metadata={"help": "EMA weight for new data."})
    min_std: float = field(default=0.02, metadata={"help": "Minimum std to prevent extreme normalization."})
    clamp_range: float = field(default=4.0, metadata={"help": "Clamp normalized advantages to [-range, range]."})


@dataclass
class PPOAgentConfig(BaseAgentConfig):
    """Main configuration class for PPO Agent."""

    _target_: str = "protomotions.agents.ppo.agent.PPO"

    # Model configuration
    model: PPOModelConfig = field(default_factory=PPOModelConfig, metadata={"help": "Model configuration."})

    # PPO hyperparameters
    tau: float = field(default=0.95, metadata={"help": "GAE lambda for advantage estimation."})
    e_clip: float = field(default=0.2, metadata={"help": "PPO clipping parameter epsilon."})
    clip_critic_loss: bool = field(default=True, metadata={"help": "Clip critic loss similar to actor."})

    # Actor update control
    actor_clip_frac_threshold: Optional[float] = field(
        default=0.6, metadata={"help": "Skip actor update if clip_frac > threshold (e.g., 0.5)."}
    )

    # Value normalization
    advantage_normalization: AdvantageNormalizationConfig = field(
        default_factory=AdvantageNormalizationConfig, metadata={"help": "Advantage normalization settings."}
    )
