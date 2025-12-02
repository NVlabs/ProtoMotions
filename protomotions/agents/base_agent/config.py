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
"""Configuration classes for base agent.

This module defines the configuration dataclasses used by the base agent and all
derived agents. These configurations specify training parameters, optimization
settings, and evaluation parameters.

Key Classes:
    - BaseAgentConfig: Main agent configuration
    - BaseModelConfig: Model architecture configuration
    - OptimizerConfig: Optimizer parameters
    - MaxEpisodeLengthManagerConfig: Episode length curriculum
"""

from dataclasses import dataclass, field
from typing import Optional, List
from protomotions.utils.config_builder import ConfigBuilder
from protomotions.agents.evaluators.config import EvaluatorConfig


@dataclass
class MaxEpisodeLengthManagerConfig(ConfigBuilder):
    """Configuration for managing max episode length during training."""

    # Example for configuration for agent to slowly increase the max episode length
    # max_episode_length_manager:
    #   start_length: 5
    #   end_length: 300
    #   transition_epochs: 100000
    start_length: int = 5
    end_length: int = 300
    transition_epochs: int = 100000

    def current_max_episode_length(self, current_epoch: int) -> int:
        """
        Returns the current max episode length based on linear interpolation.

        Args:
            current_step: Current step in the episode

        Returns:
            Interpolated max episode length
        """
        if self.transition_epochs == 0:
            # No interpolation, return the fixed value
            return self.start_length

        # Linear interpolation between start and end values
        progress = min(current_epoch / self.transition_epochs, 1.0)
        return int(self.start_length + progress * (self.end_length - self.start_length))


@dataclass
class OptimizerConfig(ConfigBuilder):
    """Configuration for optimizers."""

    _target_: str = "torch.optim.Adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    eps: float = 1e-8
    betas: tuple = field(default_factory=lambda: (0.9, 0.999))


@dataclass
class BaseModelConfig(ConfigBuilder):
    """Configuration for PPO Model (Actor-Critic)."""

    _target_: str = "protomotions.agents.base_agent.model.BaseModel"
    in_keys: List[str] = field(default_factory=list)
    out_keys: List[str] = field(default_factory=list)


@dataclass
class BaseAgentConfig(ConfigBuilder):
    """Main configuration class for PPO Agent."""

    batch_size: int
    training_max_steps: int

    _target_: str = "protomotions.agents.base_agent.agent.BaseAgent"

    # Model configuration
    model: BaseModelConfig = field(default_factory=BaseModelConfig)

    # Base agent hyperparameters
    num_steps: int = 32
    gradient_clip_val: float = 0.0
    fail_on_bad_grads: bool = False
    check_grad_mag: bool = True
    gamma: float = 0.99

    # Bounds and regularization
    bounds_loss_coef: float = (
        0.0  # Default policy uses tanh outputs, so we don't need the bounds loss.
    )

    # Training configuration
    task_reward_w: float = 1.0
    num_mini_epochs: int = 1

    training_early_termination: Optional[int] = None

    # Checkpoint saving configuration
    save_epoch_checkpoint_every: Optional[int] = (
        1000  # Save epoch_xxx.ckpt every N epochs (None = disabled)
    )
    save_last_checkpoint_every: int = 10  # Save/overwrite last.ckpt every K epochs

    # Episode length management
    max_episode_length_manager: Optional[MaxEpisodeLengthManagerConfig] = None

    # Evaluator configuration
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)

    # Reward normalization
    normalize_rewards: bool = True
    normalized_reward_clamp_value: float = 5.0
