# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration classes for evaluators."""

from typing import Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from protomotions.envs.mdp_component import MdpComponent, static_component_errors
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.agents.evaluators.components import ThresholdFailure, as_failure


@dataclass
class EvaluatorConfig:
    """Configuration for base evaluator."""

    _target_: str = "protomotions.agents.evaluators.base_evaluator.BaseEvaluator"
    metric_components: Dict[str, MdpComponent] = field(
        default_factory=dict,
        metadata={
            "help": "Per-environment numeric metrics; log mean/min/max over scored frames."
        },
    )
    failure_components: Dict[str, Union[TerminationRule, ThresholdFailure]] = field(
        default_factory=dict,
        metadata={
            "help": "A TerminationRule (any-frame failure) or ThresholdFailure(rule, "
            "reduction=...) over sum, mean, min, max or last value per trial."
        },
    )
    score_key: Optional[str] = field(
        default=None,
        metadata={
            "help": "'eval/success_rate' for checkpoint selection, None for metrics only."
        },
    )
    max_eval_steps: int = field(
        default=600,
        metadata={"help": "Maximum steps per evaluation episode.", "min": 1},
    )
    eval_metrics_every: Optional[int] = field(
        default=200,
        metadata={
            "help": "Evaluate metrics every N epochs. None = disabled.",
            "min": 1,
        },
    )

    def validate(self):
        if self.score_key not in (None, "eval/success_rate"):
            raise ValueError("score_key must be 'eval/success_rate' or None")
        if self.score_key is not None and not self.failure_components:
            raise ValueError("A success score requires explicit failure_components")
        failures = {}
        for name, entry in self.failure_components.items():
            # as_failure refuses a non-rule, a ThresholdFailure of a bad type and
            # a truncation rule; it normalises a bare rule to an any-frame failure.
            failure = as_failure(name, entry)
            # Unpickling bypasses dataclass initialization; validate resumed rules too.
            failure.rule.__post_init__()
            failures[name] = failure.component
        for group in (self.metric_components, failures):
            for name, component in group.items():
                if not isinstance(component, MdpComponent):
                    raise TypeError(
                        f"Evaluation component {name!r} must be an MdpComponent"
                    )
        errors = static_component_errors(
            self.metric_components, group="evaluation"
        ) + static_component_errors(failures, group="evaluation")
        if errors:
            raise ValueError("\n".join(errors))



@dataclass
class MotionWeightsRulesConfig:
    """Configuration for motion weights update rule."""

    motion_weights_update_success_discount: float = field(
        default=0.999,
        metadata={
            "help": "Discount factor for successful motion weights.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    motion_weights_update_failure_discount: float = field(
        default=0.999,
        metadata={
            "help": "Discount for failed motions. 0 = set weight straight to 1.",
            "min": 0.0,
            "max": 1.0,
        },
    )
    min_motion_weight: Union[float, str] = field(
        default="1/num_motions",
        metadata={
            "help": "Minimum weight for any motion. '1/num_motions' or float value."
        },
    )


@dataclass
class MimicEvaluatorConfig(EvaluatorConfig):
    """Configuration for Mimic evaluator."""

    _target_: str = "protomotions.agents.evaluators.mimic_evaluator.MimicEvaluator"
    save_predicted_motion_lib_every: Optional[int] = field(
        default=3,
        metadata={
            "help": "Save pred_motion_lib every M evals. None = disabled.",
            "min": 1,
        },
    )
    motion_weights_rules: MotionWeightsRulesConfig = field(
        default_factory=MotionWeightsRulesConfig,
        metadata={"help": "Rules for updating motion sampling weights."},
    )


class MaskedMimicActionSource(Enum):
    """Student action path used for MaskedMimic evaluation."""

    POSTERIOR = "posterior"
    PRIOR = "prior"


@dataclass
class MaskedMimicEvaluatorConfig(MimicEvaluatorConfig):
    """Mimic coverage with posterior training evaluation and prior inference."""

    _target_: str = "protomotions.agents.evaluators.masked_mimic_evaluator.MaskedMimicEvaluator"
    action_source: MaskedMimicActionSource = MaskedMimicActionSource.POSTERIOR

    def validate(self):
        super().validate()
        if not isinstance(self.action_source, MaskedMimicActionSource):
            raise ValueError("action_source must be a MaskedMimicActionSource enum member")
