# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shipped config bundles load through the real constructors and use the evaluator API."""

from dataclasses import fields
from pathlib import Path

import pytest

from protomotions.agents.evaluators.config import (
    MaskedMimicActionSource,
    MaskedMimicEvaluatorConfig,
)
from protomotions.agents.supervised.masked_mimic_config import (
    MaskedMimicSupervisedAgentConfig,
)
from protomotions.tests.shipped_bundles import shipped_bundle_or_skip
from protomotions.utils.experiment_configs import load_experiment_configs


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATHS = sorted(
    (REPO_ROOT / "data/pretrained_models").rglob("resolved_configs.json")
)


@pytest.mark.parametrize(
    "path", CONFIG_PATHS, ids=lambda path: str(path.relative_to(REPO_ROOT))
)
def test_shipped_evaluator_config(path):
    shipped_bundle_or_skip(path)

    # Loading rebuilds every object through its real constructor, so a stale or
    # renamed field fails here naming it -- the JSON form has no pickle that can
    # smuggle a missing field past __init__.
    configs = load_experiment_configs(path)
    agent = configs.agent
    config = agent.evaluator
    config.validate()
    assert set(vars(config)) == {field.name for field in fields(config)}

    for name, rule in config.failure_components.items():
        assert rule.component is config.metric_components[name]
        assert rule.reduction == "any"
        assert rule.fail_above is True

    if isinstance(agent, MaskedMimicSupervisedAgentConfig):
        assert isinstance(config, MaskedMimicEvaluatorConfig)
        assert config.action_source is MaskedMimicActionSource.POSTERIOR
        assert config.failure_components["gt_error"].threshold == 0.25
