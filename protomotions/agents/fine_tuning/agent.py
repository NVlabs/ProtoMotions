# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PPO base class for agents that fine-tune frozen pretrained modules."""

from __future__ import annotations

from abc import abstractmethod

from torch import nn

from protomotions.agents.fine_tuning.pretrained_modules import PretrainedModulesMixin
from protomotions.agents.ppo.agent import PPO


class FineTuningAgent(PretrainedModulesMixin, PPO):
    """PPO base for models that capture frozen pretrained modules during setup.

    ``pretrained`` is populated before ``create_model()``, consumed by the
    concrete model, then dropped after model reset. Surviving references live
    through the model that captured them. Subclasses should put adapter setup or
    other initialization that depends on those captured modules in
    ``_post_create_model_hook``.
    """

    def _materialize_lazy_modules(self, dummy_obs_td) -> None:
        """Materialize LazyLinear/RunningMeanStd modules with a dummy forward."""
        if hasattr(self.model, "materialize"):
            self.model.materialize(dummy_obs_td)
        else:
            self.model(dummy_obs_td)

    @abstractmethod
    def create_model(self) -> nn.Module:
        """Build the trainable model. ``self.pretrained`` is already populated."""
