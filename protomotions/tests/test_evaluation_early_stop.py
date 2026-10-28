# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Failed trials reset before another action and never resume collecting."""

import torch
from types import SimpleNamespace
import pytest

from protomotions.agents.evaluators.aggregate_metrics import ActionSmoothnessAggregateMetric
from protomotions.agents.evaluators.components import ThresholdFailure
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.agents.evaluators.mimic_evaluator import MimicEpisodeContext
from protomotions.tests.test_mimic_evaluator_helpers import (
    _StatefulModel,
    _evaluator,
    _metric,
)
from protomotions.tests.test_base_evaluator_helpers import (
    _EvalComponent,
    _evaluator as _base_evaluator,
)


@pytest.mark.parametrize("all_fail", [False, True])
def test_base_loop_resets_failed_trials_and_stops_when_all_fail(tmp_path, all_fail):
    evaluator = _base_evaluator(tmp_path, {"value": _EvalComponent(threshold=0.5)})
    evaluator._init_eval_component_buffers(2)
    evaluator._evaluation.execute_all = lambda components, context: {
        "metric/value": torch.tensor([1.0, float(all_fail)]),
    }

    evaluator.evaluate_episode(torch.arange(2), max_steps=4)

    assert len(evaluator.env.step_actions) == (1 if all_fail else 4)
    assert evaluator._evaluation.count.tolist() == ([1, 1] if all_fail else [1, 4])
    resets = [ids.tolist() for ids, _ in evaluator.env.reset_calls]
    assert resets == ([[0, 1]] if all_fail else [[0, 1], [0]])


def test_failed_trial_resets_before_next_action_and_stays_unscored(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator.agent.model = _StatefulModel()
    evaluator.agent.model.reset_rollout_context(num_envs=2, device=torch.device("cpu"))
    evaluator._episode_ctx = MimicEpisodeContext(
        motion_ids=torch.tensor([0, 1]), frame_limits=torch.tensor([8, 8]),
    )
    evaluator._metrics = {"actions": _metric(max_steps=8)}
    env = evaluator.env
    unsafe = torch.zeros(2, dtype=torch.bool)
    original_step, original_reset = env.step, env.reset

    def reset(env_ids, **kwargs):
        unsafe[env_ids] = False
        return original_reset(env_ids, **kwargs)

    def step(actions):
        # Without the reset, the next simulated step would diverge.
        assert not unsafe.any(), "A failed physical state reached another physics step"
        output = original_step(actions)
        if len(env.step_actions) == 1:
            unsafe[0] = True
        return output

    env.reset, env.step = reset, step
    scored_ids, collected_ids = [], []
    original_measure = evaluator._check_evaluation_failures
    original_collect = evaluator._on_episode_step

    def measure(env_ids, trial_ids):
        scored_ids.append((env_ids.tolist(), trial_ids.tolist()))
        original_measure(env_ids, trial_ids)

    def collect(env_ids, trial_ids, extras, actions):
        collected_ids.append((env_ids.tolist(), trial_ids.tolist()))
        original_collect(env_ids, trial_ids, extras, actions)

    evaluator._check_evaluation_failures = measure
    evaluator._on_episode_step = collect

    def measurements(components, context):
        frame = len(env.step_actions)
        # NaN after the failing frame must never enter this trial's statistics.
        return {"failure/failure": torch.tensor([
            1.0 if frame == 1 else float("nan"), float(frame == 3),
        ])}

    evaluator._evaluation.execute_all = measurements
    evaluator.evaluate_episode(torch.arange(2), max_steps=8)

    assert len(env.step_actions) == 3  # not the configured eight steps
    assert [ids.tolist() for ids, _ in env.reset_calls] == [[0, 1], [0]]
    assert evaluator._evaluation.count.tolist() == [1, 3, 0]
    assert evaluator._evaluation.failed.tolist() == [True, True, False]
    assert evaluator._metrics["actions"].frame_counts.tolist() == [1, 3, 0]
    assert scored_ids == collected_ids == [([0, 1], [0, 1]), ([1], [1]), ([1], [1])]
    active_envs, _ = evaluator._scored_trials(torch.arange(2), step_idx=3)
    assert active_envs.numel() == 0
    assert torch.isfinite(evaluator._evaluation.failure_values["failure"]).all()
    # The failed env's model-owned context was reset, but the healthy env's was not.
    assert evaluator.agent.model.episode_value[:, 0].tolist() == [3.0, 2.0]
    logs, score, count = evaluator._evaluation.results()
    assert (score, count) == (0.0, 2)
    assert logs["eval/failure/failure_rate"] == 1.0


def test_all_failed_stops_current_batch_but_visits_remaining_motions(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator._metrics = {"actions": _metric(max_steps=5)}
    evaluator._evaluation.execute_all = lambda components, context: {
        "failure/failure": torch.ones(2),
    }

    evaluator.run_evaluation()  # three motions, two environment slots

    assert len(evaluator.env.step_actions) == 2  # one step per motion batch
    assert evaluator._evaluation.count.tolist() == [1, 1, 1]
    assert evaluator._evaluation.failed.all()
    assert evaluator._metrics["actions"].frame_counts.tolist() == [1, 1, 1]


def test_never_upright_rule_waits_for_window_and_can_succeed_later(tmp_path):
    evaluator = _evaluator(tmp_path)
    component = evaluator.config.failure_components["failure"].component
    evaluator.config.failure_components = {
        "never_upright": ThresholdFailure(
            TerminationRule(component, 0.8, fail_above=False), "max"
        ),
    }
    evaluator._init_eval_component_buffers(3)
    evaluator._metrics = {}
    evaluator._episode_ctx = MimicEpisodeContext(
        motion_ids=torch.tensor([0, 1]), frame_limits=torch.tensor([3, 3]),
    )
    evaluator._evaluation.execute_all = lambda components, context: {
        "failure/never_upright": torch.tensor([
            1.0 if len(evaluator.env.step_actions) == 3 else 0.2, 0.2,
        ]),
    }

    evaluator.evaluate_episode(torch.arange(2), max_steps=3)

    assert len(evaluator.env.step_actions) == 3
    assert len(evaluator.env.reset_calls) == 1
    _, score, count = evaluator._evaluation.results()
    assert (score, count) == (0.5, 2)
    assert evaluator._evaluation.failed.tolist() == [False, True, False]


def test_finished_short_motion_is_not_recorded_while_long_motion_continues(tmp_path):
    evaluator = _evaluator(tmp_path)
    evaluator._metrics = {"actions": _metric(max_steps=5)}
    evaluator._episode_ctx = MimicEpisodeContext(
        motion_ids=torch.tensor([0, 1]), frame_limits=torch.tensor([1, 3]),
    )
    evaluator._evaluation.execute_all = lambda components, context: {
        "failure/failure": torch.zeros(2),
    }

    evaluator.evaluate_episode(torch.arange(2), max_steps=5)

    assert len(evaluator.env.step_actions) == 3
    assert evaluator._evaluation.count.tolist() == [1, 3, 0]
    assert evaluator._metrics["actions"].frame_counts.tolist() == [1, 3, 0]
    assert [ids.tolist() for ids, _ in evaluator.env.reset_calls] == [[0, 1], [0]]


def test_action_smoothness_ignores_unrecorded_frames():
    actions = _metric(num_motions=2, max_steps=8, features=1)
    actions.update(torch.tensor([0, 1]), torch.tensor([[10.], [20.]]))
    actions.update(torch.tensor([0]), torch.tensor([[12.]]))
    actions.data[:, 2:] = float("nan")
    plugin = ActionSmoothnessAggregateMetric(
        SimpleNamespace(device=torch.device("cpu"), env=SimpleNamespace(dt=0.5)),
    )
    result = plugin.compute({"actions": actions})
    assert result["eval/action_delta_mean_rad"] == 2.0
    assert result["eval/action_rate_mean_rad_s"] == 4.0
