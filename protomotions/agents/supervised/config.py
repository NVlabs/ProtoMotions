# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for generic supervised rollout training."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from protomotions.agents.base_agent.config import BaseAgentConfig, BaseModelConfig
from protomotions.agents.common.supervision import SupervisionLossConfig


class RolloutActor(Enum):
    """Policy source used to step the environment during supervised rollout collection."""

    STUDENT = "student"
    EXPERT = "expert"

    @classmethod
    def from_str(cls, value: str) -> "RolloutActor":
        try:
            return next(
                member for member in cls if member.value.lower() == value.lower()
            )
        except StopIteration:
            valid = [member.value for member in cls]
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. Valid values are: {valid}"
            )


@dataclass
class SupervisedAgentConfig(BaseAgentConfig):
    """Generic supervised imitation agent configuration.

    Experiment files choose the rollout actor, optional external expert
    checkpoint, and supervised loss keys. The agent loop stays independent of
    the specific student model.
    """

    _target_: str = "protomotions.agents.supervised.agent.SupervisedAgent"

    model: BaseModelConfig = field(
        default_factory=BaseModelConfig,
        metadata={"help": "Model configuration."},
    )
    expert_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional checkpoint for an external expert policy."},
    )
    rollout_actor: RolloutActor = field(
        default=RolloutActor.STUDENT,
        metadata={
            "help": "Policy used for collecting rollout actions."
        },
    )
    loss: SupervisionLossConfig = field(
        default_factory=SupervisionLossConfig,
        metadata={"help": "Supervised loss over model outputs and labels."},
    )
