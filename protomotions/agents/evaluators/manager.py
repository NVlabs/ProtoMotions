# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared evaluation measurements and bounded-memory trial reductions."""

import torch
from dataclasses import dataclass
from typing import Optional

from protomotions.agents.evaluators.components import as_failure
from protomotions.envs.body_selection import resolve_body_selections
from protomotions.envs.component_manager import ComponentManager


@dataclass
class EvaluationResult:
    """Scores and trial outcomes, independent of the restored environment."""

    logs: dict
    score: Optional[float]
    num_items: int
    failed: Optional[torch.Tensor] = None
    evaluated: Optional[torch.Tensor] = None


class EvaluationManager(ComponentManager):
    """Run configured MDP kernels and accumulate only scored env/trial pairs.

    Scheduling, policy actions and simulator state belong to the evaluator, not
    this manager. All tasks use the same reductions and result definitions.
    """

    def __init__(self, config, robot_config, num_trials, num_envs, device):
        super().__init__(device)
        self.config = config
        self.num_envs = num_envs
        # Normalise each failure entry (a bare TerminationRule or a ThresholdFailure)
        # to a ThresholdFailure once; the rest of the manager reads its threshold,
        # reduction and initial value from these. The kernels run from
        # ``self.components`` below.
        self._failures = {
            name: as_failure(name, entry)
            for name, entry in config.failure_components.items()
        }
        self.count = torch.zeros(num_trials, dtype=torch.long, device=device)
        self.failed = torch.zeros(num_trials, dtype=torch.bool, device=device)
        self.failure_values = {
            name: torch.full((num_trials,), rule.initial_value, device=device)
            for name, rule in self._failures.items()
        }
        self.value_sum = {name: torch.zeros(num_trials, device=device)
                          for name in config.metric_components}
        self.value_min = {name: torch.full((num_trials,), float("inf"), device=device)
                          for name in config.metric_components}
        self.value_max = {name: torch.full((num_trials,), float("-inf"), device=device)
                          for name in config.metric_components}

        # Build routing once, not every step. Reusing a component for logging and
        # a threshold executes it once; distinct components may share a log name.
        # The routing is keyed before the body selections are resolved, so a
        # component listed under both tables is resolved once, as one object
        # (resolving each table on its own would make two copies of it).
        sources, self._keys = {}, {}
        identity_keys = {}
        for kind, group in (("metric", config.metric_components),
                            ("failure", {name: failure.component for name, failure
                                         in self._failures.items()})):
            for name, component in group.items():
                key = identity_keys.setdefault(id(component), f"{kind}/{name}")
                sources.setdefault(key, component)
                self._keys[kind, name] = key
        # Resolve the body selections against the robot, as build_env does for
        # the env's tables; the configs keep their names.
        self.components = dict(resolve_body_selections(sources, robot_config))

    @property
    def evaluated(self):
        return self.count > 0

    def update(self, context, env_ids, trial_ids):
        """Accumulate one scored frame; trial IDs must be unique in this batch."""
        active = ~self.failed[trial_ids]
        env_ids, trial_ids = env_ids[active], trial_ids[active]
        if trial_ids.numel() == 0:
            return
        values = self.execute_all(self.components, context)
        for key, value in values.items():
            if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,):
                raise ValueError(f"Evaluation {key!r} must return shape [{self.num_envs}]")
            if value.is_complex():
                raise TypeError(f"Evaluation {key!r} must return real values")

        self.count[trial_ids] += 1
        # A failure rule reads the kernel's own output (a boolean kernel's
        # rule takes no threshold); the statistics read every value as a float.
        for name, rule in self._failures.items():
            value = values[self._keys["failure", name]][env_ids]
            self.failure_values[name][trial_ids] = rule.accumulate(
                self.failure_values[name][trial_ids], value
            )
            # Any-frame failure is final immediately. Window-level goals must
            # wait: e.g. not yet upright does not mean recovery has failed.
            if rule.reduction == "any":
                self.failed[trial_ids] |= self.failure_values[name][trial_ids].bool()
        for name in self.config.metric_components:
            value = values[self._keys["metric", name]][env_ids].float()
            self.value_sum[name][trial_ids] += value
            self.value_min[name][trial_ids] = torch.minimum(self.value_min[name][trial_ids], value)
            self.value_max[name][trial_ids] = torch.maximum(self.value_max[name][trial_ids], value)

    def results(self):
        """Judge completed windows, without treating empty trials as successes."""
        logs = {}
        evaluated = self.evaluated
        count = evaluated.sum().item()
        logs["eval/num_evaluated"] = count
        self.failed.zero_()
        for name, rule in self._failures.items():
            failed = rule.finish(self.failure_values[name], self.count)
            self.failed |= failed
            if count:
                logs[f"eval/{name}/failure_rate"] = failed[evaluated].float().mean().item()
        success_rate = None
        if count and self.config.failure_components:
            success_rate = 1.0 - self.failed[evaluated].float().mean().item()
            logs["eval/success_rate"] = success_rate
        if count:
            for name in self.config.metric_components:
                means = self.value_sum[name] / self.count.clamp(min=1)
                logs[f"eval/{name}/mean"] = means[evaluated].mean().item()
                logs[f"eval/{name}/min"] = self.value_min[name][evaluated].min().item()
                logs[f"eval/{name}/max"] = self.value_max[name][evaluated].max().item()
        score = success_rate if self.config.score_key is not None else None
        return logs, score, count
