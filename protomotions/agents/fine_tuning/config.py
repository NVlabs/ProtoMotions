# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for fine-tuning agents."""

from dataclasses import dataclass, field
from typing import Dict

from protomotions.agents.base_agent.config import BaseModelConfig
from protomotions.agents.common.config import PretrainedModelConfig
from protomotions.agents.ppo.config import PPOAgentConfig


@dataclass
class FineTuningAgentConfig(PPOAgentConfig):
    """Base config for agents that train on top of frozen pretrained modules."""

    _target_: str = "protomotions.agents.fine_tuning.agent.FineTuningAgent"

    model: BaseModelConfig = field(
        default_factory=BaseModelConfig,
        metadata={"help": "Model config supplied by concrete fine-tuning agents."},
    )

    pretrained_modules: Dict[str, PretrainedModelConfig] = field(
        default_factory=dict,
        metadata={
            "help": "Frozen lower-stage modules keyed by name. Each entry is "
            "loaded before create_model() runs."
        },
    )
