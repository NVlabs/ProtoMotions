# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from protomotions.agents.evaluators.config import EvaluatorConfig


@dataclass
class MaxEpisodeLengthManagerConfig:
    """Configuration for managing max episode length during training."""

    # Example for configuration for agent to slowly increase the max episode length
    # max_episode_length_manager:
    #   start_length: 5
    #   end_length: 300
    #   transition_epochs: 100000
    start_length: int = field(
        default=5, metadata={"help": "Initial max episode length."}
    )
    end_length: int = field(default=300, metadata={"help": "Final max episode length."})
    transition_epochs: int = field(
        default=100000, metadata={"help": "Epochs to transition."}
    )

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
class OptimizerConfig:
    """Configuration for optimizers."""

    _target_: str = "torch.optim.Adam"
    lr: float = field(default=1e-4, metadata={"help": "Learning rate."})
    weight_decay: float = field(default=0.0, metadata={"help": "L2 weight decay."})
    eps: float = field(
        default=1e-8, metadata={"help": "Epsilon for numerical stability."}
    )
    betas: tuple = field(
        default_factory=lambda: (0.9, 0.999), metadata={"help": "Adam betas."}
    )


@dataclass
class MuonWithAuxAdamConfig(OptimizerConfig):
    """Configuration for Muon with AdamW fallback parameter groups."""

    _target_: str = "protomotions.agents.optimizer.muon.MuonWithAuxAdam"
    lr: float = field(
        default=1e-3,
        metadata={"help": "Primary Muon learning rate."},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Primary Muon weight decay."},
    )
    eps: float = field(default=1e-10, metadata={"help": "Adam fallback eps."})
    betas: Tuple[float, float] = field(
        default=(0.9, 0.95), metadata={"help": "Adam fallback betas."}
    )
    momentum: float = field(default=0.95, metadata={"help": "Muon momentum."})
    adam_lr: float = field(default=2e-4, metadata={"help": "Adam fallback LR."})
    adam_betas: Optional[Tuple[float, float]] = field(
        default=None, metadata={"help": "Adam fallback betas. Defaults to betas."}
    )
    adam_eps: Optional[float] = field(
        default=None, metadata={"help": "Adam fallback eps. Defaults to eps."}
    )
    adam_weight_decay: float = field(
        default=0.01, metadata={"help": "Adam fallback weight decay."}
    )
    adam_fallback_module_patterns: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Optional module-name patterns that should use Adam fallback."
        },
    )
    adam_fallback_parameter_patterns: List[str] = field(
        default_factory=list,
        metadata={
            "help": "Optional parameter-name patterns that should use Adam fallback."
        },
    )
    use_adam_for_sequential_projections: bool = field(
        default=True,
        metadata={
            "help": "Use Adam fallback for input/output projections in nn.Sequential blocks."
        },
    )


@dataclass
class BaseModelConfig:
    """Configuration for PPO Model (Actor-Critic)."""

    _target_: str = "protomotions.agents.base_agent.model.BaseModel"
    in_keys: List[str] = field(default_factory=list, metadata={"help": "Input keys."})
    out_keys: List[str] = field(default_factory=list, metadata={"help": "Output keys."})


@dataclass
class BaseAgentConfig:
    """Main configuration class for PPO Agent."""

    batch_size: int = field(metadata={"help": "Training batch size."})
    training_max_steps: int = field(metadata={"help": "Maximum training steps."})

    _target_: str = "protomotions.agents.base_agent.agent.BaseAgent"

    # Model configuration
    model: BaseModelConfig = field(
        default_factory=BaseModelConfig, metadata={"help": "Model config."}
    )

    # Base agent hyperparameters
    num_steps: int = field(
        default=32, metadata={"help": "Environment steps per update."}
    )
    gradient_clip_val: float = field(
        default=0.0, metadata={"help": "Max gradient norm. 0=disabled."}
    )
    fail_on_bad_grads: bool = field(
        default=False, metadata={"help": "Fail on NaN/Inf gradients."}
    )
    check_grad_mag: bool = field(
        default=True, metadata={"help": "Log gradient magnitude."}
    )
    gamma: float = field(default=0.99, metadata={"help": "Discount factor."})

    # Bounds and regularization
    bounds_loss_coef: float = field(
        default=0.0, metadata={"help": "Action bounds loss. 0 for tanh outputs."}
    )  # Default policy uses tanh outputs, so we don't need the bounds loss.

    # Training configuration
    task_reward_w: float = field(default=1.0, metadata={"help": "Task reward weight."})
    num_mini_epochs: int = field(
        default=1, metadata={"help": "Mini-epochs per update."}
    )

    training_early_termination: Optional[int] = field(
        default=None, metadata={"help": "Stop early at this step. None=disabled."}
    )

    # Checkpoint saving configuration
    save_epoch_checkpoint_every: Optional[int] = field(
        default=1000, metadata={"help": "Save epoch_xxx.ckpt every N epochs."}
    )  # Save epoch_xxx.ckpt every N epochs (None = disabled)
    save_last_checkpoint_every: int = field(
        default=10, metadata={"help": "Save last.ckpt every K epochs."}
    )  # Save/overwrite last.ckpt every K epochs
    save_inference_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Also save inference_<name>.ckpt without optimizer or other "
            "training-only state."
        },
    )

    # Episode length management
    max_episode_length_manager: Optional[MaxEpisodeLengthManagerConfig] = field(
        default=None, metadata={"help": "Episode length curriculum."}
    )

    # Evaluator configuration
    evaluator: EvaluatorConfig = field(
        default_factory=EvaluatorConfig, metadata={"help": "Evaluation config."}
    )

    # Reward normalization
    normalize_rewards: bool = field(
        default=True, metadata={"help": "Normalize rewards."}
    )
    normalized_reward_clamp_value: float = field(
        default=5.0, metadata={"help": "Clamp normalized rewards to [-val, val]."}
    )
    reward_norm_ema_decay: Optional[float] = field(
        default=None,
        metadata={
            "help": "EMA decay for reward normalization (None = Welford). "
            "Set to e.g. 0.99 to track non-stationary reward distributions."
        },
    )
