# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluator presets: one named evaluation config reused across recipes.

Every line binds a component, so the preset lives with stack A's kernel surface
rather than in the config bundle. ``overrides`` set ``MimicEvaluatorConfig``
fields by name through ``dataclasses.replace``, so the config's own validation
runs again and the dataclass refuses an unknown field.
"""

from dataclasses import replace

from protomotions.agents.evaluators.config import (
    MimicEvaluatorConfig,
    MotionWeightsRulesConfig,
)
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.terminations.tracking import (
    max_body_pos_error,
    mean_body_pos_error,
    mean_body_rot_error,
)
from protomotions.envs.terminations.rule import TerminationRule


def mimic_tracking_benchmark(
    threshold: float = 0.5, **overrides
) -> MimicEvaluatorConfig:
    """The standard mimic-tracking evaluation.

    Mean/max body-position and mean body-rotation error metrics, a gt-error
    failure cut at ``threshold`` (any frame), success-rate scoring and the motion
    sampling-weight discounts. ``overrides`` set ``MimicEvaluatorConfig`` fields
    by name (``dataclasses.replace``: the config validates again, an unknown
    field is a ``TypeError``).
    """
    config = MimicEvaluatorConfig(
        metric_components={
            "gt_error": MdpComponent.from_kernel(mean_body_pos_error),
            "gr_error": MdpComponent.from_kernel(mean_body_rot_error),
            "max_joint_error": MdpComponent.from_kernel(max_body_pos_error),
        },
        failure_components={
            "gt_error": TerminationRule(
                MdpComponent.from_kernel(mean_body_pos_error), threshold
            ),
        },
        score_key="eval/success_rate",
        motion_weights_rules=MotionWeightsRulesConfig(
            motion_weights_update_success_discount=0.999,
            motion_weights_update_failure_discount=0,
        ),
    )
    return replace(config, **overrides) if overrides else config


__all__ = ["mimic_tracking_benchmark"]
