# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``TerminationRule``: the one termination entry, shared with evaluation.

A termination is a rule over a bound kernel (a boolean kernel takes no
threshold, a scalar one takes the cut), ``truncation=True`` resets without
terminating, and the evaluator's failure is the same rule, bare or wrapped in a
``ThresholdFailure`` for a per-trial reduction. The task kernels are scalar
measurements whose rules reproduce the deleted boolean checks.
"""

import dataclasses
import pickle

import pytest
import torch

from protomotions.agents.evaluators.components import ThresholdFailure, as_failure
from protomotions.agents.evaluators.config import MimicEvaluatorConfig
from protomotions.agents.evaluators.presets import mimic_tracking_benchmark
from protomotions.envs.context_paths import FieldPath
from protomotions.envs.mdp_component import MdpComponent, eager_kernel
from protomotions.envs.terminations.rule import (
    TerminationRule,
    components_of,
    evaluate_terminations,
    or_terminations,
)
from protomotions.envs.terminations.task import (
    path_distance,
    path_height_error,
    steering_direction_alignment,
    steering_speed_error,
)


@eager_kernel
def read_value(value):
    return value


def metric(path="value"):
    return MdpComponent(read_value, {"value": FieldPath.from_path(path)})


# ---------- the rule ------------------------------------------------------------


def test_rule_refuses_an_unbound_kernel_and_bad_fields():
    with pytest.raises(TypeError, match="from_kernel"):
        TerminationRule(read_value)
    with pytest.raises(ValueError, match="finite"):
        TerminationRule(metric(), float("nan"))
    with pytest.raises(TypeError, match="bool"):
        TerminationRule(metric(), 0.5, fail_above=1)
    with pytest.raises(TypeError, match="bool"):
        TerminationRule(metric(), truncation="yes")


def test_rule_is_frozen_and_pickles():
    rule = TerminationRule(metric(), 0.5, fail_above=False, truncation=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.threshold = 0.7

    restored = pickle.loads(pickle.dumps(rule))

    assert (restored.threshold, restored.fail_above, restored.truncation) == (
        0.5,
        False,
        True,
    )
    assert restored.component.to_dict() == rule.component.to_dict()


def test_compare_cuts_a_scalar_and_passes_a_boolean_through():
    value = torch.tensor([0.4, 0.5, 0.6])
    above = TerminationRule(metric(), 0.5).compare(value)
    below = TerminationRule(metric(), 0.5, fail_above=False).compare(value)

    # Equality at the cut passes in both directions.
    assert above.tolist() == [False, False, True]
    assert below.tolist() == [True, False, False]

    flags = torch.tensor([True, False])
    assert TerminationRule(metric()).compare(flags) is flags
    with pytest.raises(ValueError, match="torch.bool"):
        TerminationRule(metric()).compare(value)


def test_components_of_unwraps_rules_and_refuses_other_shapes():
    component = metric()

    assert components_of({"a": TerminationRule(component)})["a"] is component
    with pytest.raises(TypeError, match="bare MdpComponent"):
        components_of({"a": component})
    with pytest.raises(TypeError, match="ThresholdFailure"):
        components_of({"a": ThresholdFailure(TerminationRule(component, 0.5))})
    with pytest.raises(TypeError, match="must be a TerminationRule"):
        components_of({"a": 0.5})


class _Manager:
    def __init__(self, values):
        self.values = values
        self.calls = []

    def execute_all(self, components, ctx):
        self.calls.append({name: component for name, component in components.items()})
        return {name: self.values[name] for name in components}


def test_evaluate_terminations_runs_kernels_under_their_entry_names():
    entries = {
        "error": TerminationRule(metric("error"), 0.5),
        "fell": TerminationRule(metric("fell")),
        "clip_done": TerminationRule(metric("clip_done"), truncation=True),
    }
    manager = _Manager(
        {
            "error": torch.tensor([0.1, 0.9]),
            "fell": torch.tensor([False, False]),
            "clip_done": torch.tensor([True, False]),
        }
    )

    flags = evaluate_terminations(manager, entries, ctx=None)

    assert [list(call) for call in manager.calls] == [["error", "fell", "clip_done"]]
    assert manager.calls[0]["error"] is entries["error"].component
    assert flags["error"].tolist() == [False, True]
    assert flags["fell"].tolist() == [False, False]
    assert flags["clip_done"].tolist() == [True, False]
    # A float under a thresholdless rule is refused with the entry's name.
    manager.values["fell"] = torch.tensor([0.0, 1.0])
    with pytest.raises(
        ValueError, match="termination component 'fell' returned torch.float32"
    ):
        evaluate_terminations(manager, entries, ctx=None)

    reset, terminated, logging = or_terminations(flags, entries, 2, torch.device("cpu"))

    assert reset.tolist() == [True, True]
    # The truncation resets without terminating: the agent bootstraps through it.
    assert terminated.tolist() == [False, True]
    assert set(logging) == {
        "termination/error",
        "termination/fell",
        "termination/clip_done",
    }
    assert logging["termination/clip_done"].dtype == torch.float32
    assert logging["termination/clip_done"].tolist() == [1.0, 0.0]


# ---------- the evaluator's failure ---------------------------------------------


def test_as_failure_normalises_a_bare_rule_and_refuses_a_truncation():
    rule = TerminationRule(metric(), 0.5)

    failure = as_failure("error", rule)

    assert isinstance(failure, ThresholdFailure)
    assert failure.rule is rule and failure.reduction == "any"
    assert (failure.component, failure.threshold, failure.fail_above) == (
        rule.component,
        0.5,
        True,
    )
    wrapped = ThresholdFailure(rule, "max")
    assert as_failure("error", wrapped) is wrapped
    with pytest.raises(ValueError, match="truncation"):
        as_failure("clip", TerminationRule(metric(), truncation=True))
    with pytest.raises(TypeError, match="TerminationRule or"):
        as_failure("error", rule.component)


def test_threshold_failure_refuses_a_non_rule_a_truncation_and_a_bad_reduction():
    with pytest.raises(TypeError, match="TerminationRule"):
        ThresholdFailure(metric(), 0.5)
    with pytest.raises(ValueError, match="truncation"):
        ThresholdFailure(TerminationRule(metric(), truncation=True))
    with pytest.raises(ValueError, match="reduction"):
        ThresholdFailure(TerminationRule(metric(), 0.5), "median")
    # A reduced value is a number; a boolean kernel's rule has no cut for it.
    with pytest.raises(ValueError, match="needs a rule with a threshold"):
        ThresholdFailure(TerminationRule(metric()), "max")


def test_evaluator_config_accepts_bare_rules_and_refuses_a_truncation():
    config = MimicEvaluatorConfig(
        failure_components={"error": TerminationRule(metric(), 0.5)},
        score_key="eval/success_rate",
    )
    config.validate()

    config.failure_components["clip"] = TerminationRule(metric(), truncation=True)
    with pytest.raises(ValueError, match="truncation"):
        config.validate()


def test_mimic_tracking_benchmark_preset_builds_the_standard_evaluation():
    config = mimic_tracking_benchmark(0.4, max_eval_steps=300)

    assert isinstance(config, MimicEvaluatorConfig)
    assert set(config.metric_components) == {"gt_error", "gr_error", "max_joint_error"}
    failure = config.failure_components["gt_error"]
    assert isinstance(failure, TerminationRule)
    assert (failure.threshold, failure.fail_above, failure.truncation) == (
        0.4,
        True,
        False,
    )
    assert (
        failure.component.compute_func
        is config.metric_components["gt_error"].compute_func
    )
    assert config.score_key == "eval/success_rate"
    assert config.max_eval_steps == 300
    assert config.motion_weights_rules.motion_weights_update_failure_discount == 0
    config.validate()
    # Overrides go through dataclasses.replace: the dataclass itself refuses a
    # name that is not a field (a method included).
    with pytest.raises(TypeError, match="unexpected keyword argument 'nope'"):
        mimic_tracking_benchmark(nope=1)
    with pytest.raises(TypeError, match="unexpected keyword argument 'validate'"):
        mimic_tracking_benchmark(validate=1)


# ---------- the task kernels against the deleted boolean checks -----------------


def test_task_kernels_bind_their_context_paths_by_default():
    component = MdpComponent.from_kernel(path_distance)
    assert {name: field.path for name, field in component.dynamic_vars.items()} == {
        "head_pos": "path.head_pos",
        "target_pos": "path.tar_pos",
        "progress_buf": "progress_buf",
    }
    assert component.static_params == {"min_progress": 10}

    component = MdpComponent.from_kernel(steering_direction_alignment)
    assert {name: field.path for name, field in component.dynamic_vars.items()} == {
        "root_pos": "current.anchor_pos",
        "prev_root_pos": "steering.prev_root_pos",
        "tar_dir": "steering.tar_dir",
        "dt": "dt",
    }
    assert component.static_params == {}


def _old_path_distance(head_pos, target_pos, fail_dist, progress_buf, min_progress=10):
    delta = target_pos - head_pos
    too_far = torch.sum(delta * delta, dim=-1) > fail_dist * fail_dist
    return too_far & (progress_buf > min_progress)


def _old_path_height(head_pos, target_pos, fail_height, progress_buf, min_progress=10):
    delta = target_pos[..., 2] - head_pos[..., 2]
    return (delta * delta > fail_height * fail_height) & (progress_buf > min_progress)


def _old_steering(root_pos, prev_root_pos, tar_dir, tar_speed, dt, speed_tol, dir_tol):
    vel_xy = (root_pos - prev_root_pos)[:, :2] / dt
    speed = torch.norm(vel_xy, dim=-1)
    speed_fail = torch.abs(speed - tar_speed) > speed_tol
    vel_dir = vel_xy / (speed.unsqueeze(-1) + 1e-6)
    dir_fail = ((vel_dir * tar_dir).sum(dim=-1) < dir_tol) & (speed > 0.1)
    return speed_fail | dir_fail


def test_scalar_task_kernels_cut_by_a_rule_reproduce_the_old_boolean_checks():
    generator = torch.Generator().manual_seed(0)
    n = 512
    head = torch.rand(n, 3, generator=generator) * 4 - 2
    target = torch.rand(n, 3, generator=generator) * 4 - 2
    progress = torch.randint(0, 25, (n,), generator=generator)
    for fail_dist in (0.5, 1.0, 2.0):
        rule = TerminationRule(MdpComponent.from_kernel(path_distance), fail_dist)
        assert torch.equal(
            rule.compare(path_distance(head, target, progress)),
            _old_path_distance(head, target, fail_dist, progress),
        )
        rule = TerminationRule(MdpComponent.from_kernel(path_height_error), fail_dist)
        assert torch.equal(
            rule.compare(path_height_error(head, target, progress)),
            _old_path_height(head, target, fail_dist, progress),
        )

    root = torch.rand(n, 3, generator=generator) * 0.4 - 0.2
    root[:64] *= 0.05  # near-stationary robots straddle the |v| > 0.1 guard
    prev = torch.zeros(n, 3)
    angle = torch.rand(n, generator=generator) * 6.283
    tar_dir = torch.stack([angle.cos(), angle.sin()], dim=-1)
    tar_speed = torch.rand(n, generator=generator) * 3
    dt = 0.1
    for speed_tol, dir_tol in ((0.5, 0.7), (0.25, 0.5), (1.0, 0.95)):
        too_fast = TerminationRule(
            MdpComponent.from_kernel(steering_speed_error), speed_tol
        )
        off_course = TerminationRule(
            MdpComponent.from_kernel(steering_direction_alignment),
            dir_tol,
            fail_above=False,
        )
        flags = too_fast.compare(
            steering_speed_error(root, prev, tar_speed, dt)
        ) | off_course.compare(steering_direction_alignment(root, prev, tar_dir, dt))
        assert torch.equal(
            flags,
            _old_steering(root, prev, tar_dir, tar_speed, dt, speed_tol, dir_tol),
        )
