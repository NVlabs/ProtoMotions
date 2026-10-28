# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for generic supervised rollout training."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Optional

from protomotions.agents.base_agent.config import BaseAgentConfig, BaseModelConfig
from protomotions.agents.common.supervision import SupervisionLossConfig


class RolloutActor(Enum):
    """Which model forward produces the rollout.

    STUDENT uses the model's forward. EXPERT uses the model's own
    ``collect_expert_rollout``. An optional external expert labels every state
    and drives a fraction of episodes through ``expert_rollout_probability``.
    """

    STUDENT = "student"
    EXPERT = "expert"


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
        metadata={
            "help": "External expert checkpoint that labels rollouts; requires rollout_actor=student."
        },
    )
    rollout_actor: RolloutActor = field(
        default=RolloutActor.STUDENT,
        metadata={"help": "Model forward used for collecting rollout actions."},
    )
    loss: SupervisionLossConfig = field(
        default_factory=SupervisionLossConfig,
        metadata={"help": "Supervised loss over model outputs and labels."},
    )
    expert_rollout_probability: float = field(
        default=0.0,
        metadata={
            "help": "Per-episode probability that the external expert drives the episode."
        },
    )

    def __post_init__(self):
        if not isinstance(self.rollout_actor, RolloutActor):
            raise TypeError("rollout_actor must be a RolloutActor")
        if (
            not math.isfinite(self.expert_rollout_probability)
            or not 0 <= self.expert_rollout_probability <= 1
        ):
            raise ValueError("expert_rollout_probability must be finite and in [0, 1]")
        if self.expert_model_path is None:
            if self.expert_rollout_probability > 0:
                raise ValueError(
                    "expert_rollout_probability > 0 requires expert_model_path"
                )
        elif self.rollout_actor is not RolloutActor.STUDENT:
            raise ValueError("expert_model_path requires rollout_actor=student")

    def apply_inference_defaults(self, env) -> None:
        """Run the trained student alone: drop the expert and its observations.

        The expert labels rollouts during training; at inference the student
        acts on its own, so the expert checkpoint, its rollout share and the
        expert-prefixed observations it added are removed.
        """
        from protomotions.agents.supervised.expert_utils import EXPERT_OBS_PREFIX

        for key in [
            key
            for key in env.observation_components
            if key.startswith(EXPERT_OBS_PREFIX)
        ]:
            del env.observation_components[key]
        self.expert_model_path = None
        self.rollout_actor = RolloutActor.STUDENT
        self.expert_rollout_probability = 0.0
