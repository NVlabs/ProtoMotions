# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Motion evaluation with explicit posterior/prior action selection."""

from protomotions.agents.evaluators.config import MaskedMimicActionSource
from protomotions.agents.evaluators.mimic_evaluator import MimicEvaluator


class MaskedMimicEvaluator(MimicEvaluator):
    """Use the configured student posterior or deployable prior.

    Both paths run with the model in eval mode. Never use ``model.training`` to
    choose the action source, and never substitute an external teacher's action.
    Motion coverage, measurements and restoration are inherited unchanged.
    """

    def _policy_action(self, obs_td):
        if self.config.action_source is MaskedMimicActionSource.POSTERIOR:
            return self.agent.model(obs_td)["privileged_action"]
        model = self.agent.model
        return model.forward_deployable(obs_td)[model.deployable_action_key]
