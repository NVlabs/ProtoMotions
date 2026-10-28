# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation failures: a termination rule with a per-trial reduction.

A failure is a :class:`~protomotions.envs.terminations.rule.TerminationRule` —
the same rule a training termination uses — judged over a trial. A bare rule is
the any-frame failure; a :class:`ThresholdFailure` wraps it with another
reduction (sum, mean, min, max, last) judged after the whole scored window.
``as_failure`` normalises either shape to a ``ThresholdFailure``.
"""

from dataclasses import dataclass
from typing import Literal, Union

import torch

from protomotions.envs.terminations.rule import TerminationRule

_REDUCTIONS = {"any", "sum", "mean", "min", "max", "last"}


@dataclass(frozen=True)
class ThresholdFailure:
    """A termination rule judged over a trial with a per-trial reduction.

    ``any`` preserves historical any-frame failure. Other reductions are judged
    only after the complete scored window: an unmet goal is not an early failure.
    ``sum`` is a sum of samples, not a time integral. For seconds, the component
    must emit a per-step duration using dt. No complete trajectory is stored.
    """

    rule: TerminationRule
    reduction: Literal["any", "sum", "mean", "min", "max", "last"] = "any"

    def __post_init__(self):
        if not isinstance(self.rule, TerminationRule):
            raise TypeError(
                "ThresholdFailure.rule must be a TerminationRule, got "
                f"{type(self.rule).__name__}"
            )
        if self.rule.truncation:
            raise ValueError(
                "ThresholdFailure wraps a non-truncation rule; a truncation has "
                "no failure meaning"
            )
        if self.reduction not in _REDUCTIONS:
            raise ValueError(f"Unknown evaluation reduction: {self.reduction!r}")
        if self.reduction != "any" and self.rule.threshold is None:
            raise ValueError(
                f"ThresholdFailure(reduction={self.reduction!r}) needs a rule with "
                "a threshold: the reduced value is a number, which only a cut can "
                "turn into a failure"
            )

    @property
    def component(self):
        return self.rule.component

    @property
    def threshold(self):
        return self.rule.threshold

    @property
    def fail_above(self):
        return self.rule.fail_above

    @property
    def initial_value(self):
        return {"min": float("inf"), "max": float("-inf")}.get(self.reduction, 0.0)

    def compare(self, value):
        return self.rule.compare(value)

    def accumulate(self, previous, value):
        if self.reduction == "any":
            return (previous.bool() | self.compare(value)).float()
        value = value.float()  # a boolean kernel under a reduction counts frames
        if self.reduction in {"sum", "mean"}:
            return previous + value
        if self.reduction == "min":
            return torch.minimum(previous, value)
        if self.reduction == "max":
            return torch.maximum(previous, value)
        return value  # last scored frame, not the batch's final simulation frame

    def finish(self, value, count):
        if self.reduction == "any":
            failed = value.bool()
        else:
            if self.reduction == "mean":
                value = value / count.clamp(min=1)
            failed = self.compare(value)
        return failed & (count > 0)


def as_failure(name: str, entry: Union[TerminationRule, ThresholdFailure]):
    """Normalise a failure entry to a ``ThresholdFailure``.

    A bare ``TerminationRule`` becomes an any-frame failure; a ``ThresholdFailure``
    passes through. A truncation rule and anything else are refused.
    """
    if isinstance(entry, ThresholdFailure):
        return entry
    if isinstance(entry, TerminationRule):
        if entry.truncation:
            raise ValueError(
                f"failure '{name}' is a truncation; the evaluator has no "
                "truncation notion, pass a rule with truncation=False"
            )
        return ThresholdFailure(entry)
    raise TypeError(
        f"failure '{name}' must be a TerminationRule or "
        f"ThresholdFailure(rule, reduction=...), got {type(entry).__name__}"
    )


__all__ = ["ThresholdFailure", "as_failure"]
