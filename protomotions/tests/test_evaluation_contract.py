# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation rules and statistics through the final scored frame."""

from types import SimpleNamespace
import pickle

import pytest
import torch

from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
from protomotions.agents.evaluators.config import EvaluatorConfig
from protomotions.agents.evaluators.components import ThresholdFailure
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.agents.evaluators.manager import EvaluationResult
from protomotions.agents.utils.training import aggregate_evaluation_results
from protomotions.envs.context_paths import FieldPath
from protomotions.envs.mdp_component import MdpComponent, eager_kernel


@eager_kernel
def read_value(value):
    return value


def metric(path, **params):
    field = FieldPath()
    field.name = path
    return MdpComponent(read_value, {"value": field}, params)


def make_evaluator(config, num_envs=3):
    # No component here holds a BodySelection, so the robot is never consulted.
    env = SimpleNamespace(
        context=SimpleNamespace(), config=SimpleNamespace(), dt=0.02, robot_config=None
    )
    # Runtime contexts expose all bindings before evaluation buffers are built.
    # The tests fill their tensor values independently for each scored frame.
    components = list(config.metric_components.values()) + [
        rule.component for rule in config.failure_components.values()
    ]
    for component in components:
        for field in component.dynamic_vars.values():
            context = env.context
            parts = field.path.split(".")
            for part in parts[:-1]:
                if not hasattr(context, part):
                    setattr(context, part, SimpleNamespace())
                context = getattr(context, part)
            setattr(context, parts[-1], None)
    agent = SimpleNamespace(env=env, num_envs=num_envs)
    evaluator = BaseEvaluator(
        agent, SimpleNamespace(device=torch.device("cpu"), world_size=1), config
    )
    evaluator._init_eval_component_buffers(4)
    return evaluator


@pytest.mark.parametrize("kind", ["metric", "failure"])
def test_unavailable_binding_fails_when_evaluation_is_initialized(kind):
    component = metric("missing_sensor")
    config = EvaluatorConfig(
        metric_components={"sensor": component} if kind == "metric" else {},
        failure_components={"sensor": TerminationRule(component, 0.5)}
        if kind == "failure" else {},
    )
    agent = SimpleNamespace(
        env=SimpleNamespace(context=SimpleNamespace(), robot_config=None), num_envs=3
    )
    evaluator = BaseEvaluator(
        agent, SimpleNamespace(device=torch.device("cpu"), world_size=1), config
    )
    with pytest.raises(ValueError, match=f"{kind}/sensor.*missing_sensor"):
        evaluator._init_eval_component_buffers(4)


def test_failure_stops_statistics_without_changing_outcome_or_log_names():
    components = {name: metric(name) for name in ("error", "speed", "diagnostic")}
    evaluator = make_evaluator(
        EvaluatorConfig(
            metric_components=components,
            failure_components={
                "error": TerminationRule(components["error"], 0.5),
                "speed": TerminationRule(components["speed"], 0.25, fail_above=False),
            },
            score_key="eval/success_rate",
        )
    )
    # Env 1 is unscored. Motion 0 is never evaluated. Motion 3 has fewer frames.
    traces = [
        ([0.5, 99.0, 0.6], [0.25, -99.0, 0.4], [1.0, 99.0, 3.0], [0, 2], [1, 3]),
        ([0.9, 99.0, 0.1], [0.1, -99.0, 0.2], [5.0, 99.0, 7.0], [0], [1]),
        ([0.2, 99.0, 0.3], [0.3, -99.0, 0.3], [9.0, 99.0, 11.0], [0, 2], [1, 2]),
    ]
    for error, speed, diagnostic, env_ids, trial_ids in traces:
        evaluator.env.context = SimpleNamespace(
            error=torch.tensor(error),
            speed=torch.tensor(speed),
            diagnostic=torch.tensor(diagnostic),
        )
        evaluator._check_evaluation_failures(
            torch.tensor(env_ids), torch.tensor(trial_ids)
        )
    logs, score, count = evaluator.process_eval_results()
    assert count == 3
    assert score == pytest.approx(1 / 3)
    assert evaluator._evaluation.failed.tolist() == [False, True, False, True]
    assert evaluator._evaluation.evaluated.tolist() == [False, True, True, True]
    assert logs["eval/error/failure_rate"] == pytest.approx(2 / 3)
    assert logs["eval/speed/failure_rate"] == pytest.approx(1 / 3)
    assert logs["eval/error/mean"] == pytest.approx(
        ((0.5 + 0.9) / 2 + 0.3 + 0.6) / 3
    )
    assert logs["eval/diagnostic/mean"] == pytest.approx((3 + 11 + 3) / 3)
    assert logs["eval/diagnostic/max"] == 11
    assert logs["eval/diagnostic/min"] == 1
    assert "eval/diagnostic/failure_rate" not in logs
    # Include the failing frame, not later frames. Equality at the cut passes.
    assert evaluator._evaluation.count.tolist() == [0, 2, 1, 1]


def test_boolean_task_metric_keeps_numeric_mean_and_latched_failure():
    task = metric("task")
    evaluator = make_evaluator(EvaluatorConfig(
        metric_components={"task": task},
        failure_components={"task": TerminationRule(task, 0.5)},
        score_key="eval/success_rate",
    ))
    for values in ([False, True, False], [False, False, False]):
        evaluator.env.context = SimpleNamespace(task=torch.tensor(values))
        evaluator._check_evaluation_failures(torch.arange(3), torch.arange(3))
    logs, score, count = evaluator.process_eval_results()
    assert (count, score) == (3, pytest.approx(2 / 3))
    assert logs["eval/task/mean"] == pytest.approx(1 / 3)
    assert logs["eval/task/failure_rate"] == pytest.approx(1 / 3)


def test_bare_boolean_rule_is_the_any_frame_failure_and_keeps_the_numeric_mean():
    """A boolean kernel's rule takes no threshold; the evaluator reads its flag."""
    fell = metric("fell")
    evaluator = make_evaluator(EvaluatorConfig(
        metric_components={"fell": fell},
        failure_components={"fell": TerminationRule(fell)},
        score_key="eval/success_rate",
    ))
    for values in ([False, True, False], [False, False, False]):
        evaluator.env.context = SimpleNamespace(fell=torch.tensor(values))
        evaluator._check_evaluation_failures(torch.arange(3), torch.arange(3))
    logs, score, count = evaluator.process_eval_results()
    assert (count, score) == (3, pytest.approx(2 / 3))
    assert logs["eval/fell/mean"] == pytest.approx(1 / 3)
    assert logs["eval/fell/failure_rate"] == pytest.approx(1 / 3)


def test_shared_metric_is_executed_once_for_logging_and_failure():
    value = metric("value")
    evaluator = make_evaluator(
        EvaluatorConfig(
            metric_components={"same": value},
            failure_components={"same": TerminationRule(value, 0.5)},
            score_key="eval/success_rate",
        )
    )
    evaluator.env.context = SimpleNamespace(value=torch.tensor([0.2, 0.5, 0.8]))
    evaluator._check_evaluation_failures(torch.arange(3), torch.arange(3))
    assert evaluator.process_eval_results()[1] == pytest.approx(2 / 3)
    assert set(evaluator._evaluation._compiled) == {(read_value, True)}


def test_distinct_components_can_share_a_log_name():
    evaluator = make_evaluator(EvaluatorConfig(
        metric_components={"same": metric("diagnostic")},
        failure_components={"same": TerminationRule(metric("error"), 0.5)},
        score_key="eval/success_rate",
    ))
    evaluator.env.context = SimpleNamespace(diagnostic=torch.ones(3), error=torch.zeros(3))
    evaluator._check_evaluation_failures(torch.arange(3), torch.arange(3))
    logs, score, _ = evaluator.process_eval_results()
    assert logs["eval/same/mean"] == 1.0 and score == 1.0
    # Different context bindings share the kernel, not its resolved arguments.
    assert set(evaluator._evaluation._compiled) == {(read_value, True)}


@pytest.mark.parametrize("reduction,threshold,above,failed", [
    ("any", 0.5, True, True), ("any", 0.5, False, True),
    ("sum", 1.0, True, True), ("sum", 2.0, False, True),
    ("mean", 0.5, True, False), ("mean", 0.5, False, True),
    ("min", 0.5, True, False), ("min", 0.5, False, True),
    ("max", 0.5, True, True), ("max", 0.5, False, False),
    ("last", 0.5, True, False), ("last", 0.5, False, True),
])
def test_temporal_rules_reduce_only_the_valid_trial_window(reduction, threshold, above, failed):
    rule = ThresholdFailure(
        TerminationRule(metric("value"), threshold, above), reduction
    )
    evaluator = make_evaluator(EvaluatorConfig(
        failure_components={"rule": rule}, score_key="eval/success_rate",
    ))
    for value in (0.25, 0.75, 0.25):
        evaluator.env.context = SimpleNamespace(value=torch.tensor([value, 999., -999.]))
        evaluator._check_evaluation_failures(torch.tensor([0]), torch.tensor([2]))
    _, score, count = evaluator.process_eval_results()
    assert count == 1 and score == float(not failed)
    assert evaluator._evaluation.failed.tolist() == [False, False, failed, False]


@pytest.mark.parametrize("reduction", ["any", "sum", "mean", "min", "max", "last"])
@pytest.mark.parametrize("above", [True, False])
def test_temporal_threshold_equality_passes_and_empty_trials_are_unscored(reduction, above):
    rule = ThresholdFailure(TerminationRule(metric("value"), 0.5, above), reduction)
    state = torch.full((2,), rule.initial_value)
    assert not rule.finish(state, torch.zeros(2, dtype=torch.long)).any()
    state = rule.accumulate(state, torch.full((2,), 0.5))
    assert not rule.finish(state, torch.ones(2, dtype=torch.long)).any()


def test_goal_not_achieved_and_goal_not_maintained_are_different_rules():
    height = metric("height")
    evaluator = make_evaluator(EvaluatorConfig(
        metric_components={"height": height},
        failure_components={
            "never_upright": ThresholdFailure(
                TerminationRule(height, 0.8, fail_above=False), "max"
            ),
            "not_upright_at_end": ThresholdFailure(
                TerminationRule(height, 0.8, fail_above=False), "last"
            ),
        }, score_key="eval/success_rate",
    ))
    for values in ([0.2, 0.2, 0.2], [0.4, 0.9, 0.9], [0.5, 0.2, 0.9]):
        evaluator.env.context = SimpleNamespace(height=torch.tensor(values))
        evaluator._check_evaluation_failures(torch.arange(3), torch.arange(3))
    logs, score, count = evaluator.process_eval_results()
    assert count == 3 and score == pytest.approx(1 / 3)
    assert logs["eval/never_upright/failure_rate"] == pytest.approx(1 / 3)
    assert logs["eval/not_upright_at_end/failure_rate"] == pytest.approx(2 / 3)


def test_metrics_only_and_disabled_configs_have_no_success_score():
    evaluator = make_evaluator(
        EvaluatorConfig(metric_components={"value": metric("value")})
    )
    evaluator.env.context = SimpleNamespace(value=torch.ones(3))
    evaluator._check_evaluation_failures(torch.arange(3), torch.arange(3))
    logs, score, count = evaluator.process_eval_results()
    assert count == 3 and score is None and "eval/success_rate" not in logs
    empty = EvaluatorConfig()
    assert not empty.metric_components and not empty.failure_components
    assert empty.score_key is None
    assert make_evaluator(empty).evaluate() == EvaluationResult({}, None, 0)


@pytest.mark.parametrize(
    "config",
    [
        EvaluatorConfig(score_key="eval/success_rate"),
        EvaluatorConfig(score_key="eval/error/mean"),
    ],
)
def test_invalid_score_contract_fails_before_rollout(config):
    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "value, kind, error",
    [
        (torch.ones(3, 1), "failure", ValueError),
        (torch.ones(2), "metric", ValueError),
        (torch.ones(3, dtype=torch.complex64), "metric", TypeError),
    ],
)
def test_bad_component_output_names_the_component(value, kind, error):
    component = metric("value")
    if kind == "failure":
        component = TerminationRule(component, 0.5)
    evaluator = make_evaluator(
        EvaluatorConfig(**{f"{kind}_components": {"bad": component}})
    )
    evaluator.env.context = SimpleNamespace(value=value)
    with pytest.raises(error, match="bad"):
        evaluator._check_evaluation_failures(torch.arange(3), torch.arange(3))


def test_explicit_config_pickle_round_trip():
    component = metric("value")
    config = EvaluatorConfig(
        metric_components={"error": component},
        failure_components={"error": TerminationRule(component, 0.5)},
    )
    restored = pickle.loads(pickle.dumps(config))
    ctx = SimpleNamespace(value=torch.tensor([0.4, 0.5, 0.6]))
    rule = restored.failure_components["error"]
    assert rule.compare(rule.component.compute(ctx)).tolist() == [
        False,
        False,
        True,
    ]
    assert restored.metric_components["error"].compute(ctx).tolist() == pytest.approx(
        [0.4, 0.5, 0.6]
    )


def kernel_with_threshold(value, threshold):
    return value - threshold


def test_output_threshold_does_not_replace_a_kernels_own_threshold():
    component = metric("value", threshold=2.0)
    component.compute_func = kernel_with_threshold
    predicate = TerminationRule(component, 0.5)
    ctx = SimpleNamespace(value=torch.tensor([2.0, 2.5, 3.0]))
    assert predicate.compare(predicate.component.compute(ctx)).tolist() == [False, False, True]


def test_distributed_empty_rank_and_task_specific_keys(monkeypatch):
    records = [
        ({}, None, 0),
        (
            {"eval/success_rate": 0.5, "eval/a/mean": 2.0, "eval/num_evaluated": 2},
            0.5,
            2,
        ),
        (
            {"eval/success_rate": 1.0, "eval/b/mean": 4.0, "eval/num_evaluated": 6},
            1.0,
            6,
        ),
    ]

    def gather(output, local):
        output[:] = records

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    logs, score = aggregate_evaluation_results(
        {}, None, 0, SimpleNamespace(world_size=3)
    )
    assert score == 0.875
    assert logs == {
        "eval/success_rate": 0.875,
        "eval/a/mean": 2.0,
        "eval/b/mean": 4.0,
        "eval/num_evaluated": 8,
    }
    records[:] = [({}, None, 0)] * 3
    assert aggregate_evaluation_results({}, None, 0, SimpleNamespace(world_size=3)) == (
        {},
        None,
    )


def test_evaluator_checkpoint_restores_only_its_counter():
    original = make_evaluator(EvaluatorConfig(metric_components={"error": metric("value")}))
    original.eval_count = 7
    state = original.get_state_dict()
    identical = make_evaluator(pickle.loads(pickle.dumps(original.config)))
    identical.load_state_dict(state)
    assert identical.eval_count == 7
    assert state == {"eval_count": 7}
    identical.load_state_dict({"eval_count": 3})
    assert identical.eval_count == 3


def test_termination_rule_compiles_and_serializes_stably():
    one = TerminationRule(metric("value"), 0.5)
    two = pickle.loads(pickle.dumps(one))
    assert one.component.to_dict() == two.component.to_dict()
    assert one.threshold == two.threshold
    assert "0x" not in str(one.component.to_dict())
    ctx = SimpleNamespace(value=torch.tensor([0.4, 0.5, 0.6]))
    args, kwargs = one.component.resolve_args(ctx)
    compiled = torch.compile(one.component.compute_func, backend="eager")
    assert one.compare(compiled(**args, **kwargs)).tolist() == [False, False, True]


@pytest.mark.parametrize("task", ["path", "steering"])
def test_real_task_rules_preserve_saved_rules_and_numeric_logs(task):
    from protomotions.envs.terminations.task import (
        path_distance,
        steering_direction_alignment,
        steering_speed_error,
    )

    if task == "path":
        rules = {
            "path_distance": TerminationRule(
                MdpComponent.from_kernel(path_distance, min_progress=10), 0.5
            ),
        }
        contexts = [
            SimpleNamespace(
                path=SimpleNamespace(
                    head_pos=torch.tensor(
                        [[0.75, 0.0, 0.0], [0.5, 0.0, 0.0], [0.75, 0.0, 0.0]]
                    ),
                    tar_pos=torch.zeros(3, 3),
                ),
                progress_buf=torch.full((3,), step),
            )
            for step in (10, 11)
        ]
        expected = [[False, False, False], [True, False, True]]
    else:
        rules = {
            "steering_speed_error": TerminationRule(
                MdpComponent.from_kernel(steering_speed_error), 0.5
            ),
            "steering_direction_alignment": TerminationRule(
                MdpComponent.from_kernel(steering_direction_alignment),
                0.7,
                fail_above=False,
            ),
        }
        contexts = [
            SimpleNamespace(
                current=SimpleNamespace(anchor_pos=torch.tensor(positions)),
                steering=SimpleNamespace(
                    prev_root_pos=torch.zeros(3, 3),
                    tar_dir=torch.tensor([[1.0, 0.0]]).expand(3, -1),
                    tar_speed=torch.ones(3),
                ),
                dt=1.0,
            )
            for positions in (
                [[1.5, 0.0, 0.0], [1.6, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            )
        ]
        expected = [[False, True, True], [False, False, False]]
    new = make_evaluator(
        EvaluatorConfig(
            metric_components={name: rule.component for name, rule in rules.items()},
            failure_components=rules,
            score_key="eval/success_rate",
        )
    )
    for context, failures in zip(contexts, expected):
        flags = torch.zeros(3, dtype=torch.bool)
        for rule in rules.values():
            flags |= rule.compare(rule.component.compute(context))
        assert flags.tolist() == failures
        new.env.context = context
        new._check_evaluation_failures(torch.arange(3), torch.arange(3))
    assert new.process_eval_results()[1] == pytest.approx(1 / 3)
