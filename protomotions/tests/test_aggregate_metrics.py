# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for aggregate evaluator metrics."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.evaluators import aggregate_metrics
from protomotions.agents.evaluators.aggregate_metrics import (
    ActionSmoothnessAggregateMetric,
    AggregateMetric,
    SmoothnessAggregateMetric,
)
from protomotions.agents.evaluators.metrics import MotionMetrics


def _evaluator(dt=0.1, num_bodies=2):
    return SimpleNamespace(
        device=torch.device("cpu"),
        env=SimpleNamespace(
            dt=dt,
            robot_config=SimpleNamespace(
                kinematic_info=SimpleNamespace(num_bodies=num_bodies)
            ),
        ),
    )


def test_aggregate_metric_base_requires_compute_implementation():
    with pytest.raises(NotImplementedError):
        AggregateMetric().compute({})


def test_action_smoothness_returns_empty_when_actions_are_missing():
    metric = ActionSmoothnessAggregateMetric(_evaluator())

    assert metric.compute({}) == {}


def test_action_smoothness_computes_deltas_across_valid_motion_frames(capsys):
    actions = MotionMetrics(
        num_motions=3,
        motion_lens=torch.tensor([3, 1, 2]),
        max_motion_len=3,
        num_sub_features=2,
        device=torch.device("cpu"),
    )
    actions.data[0] = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, -2.0],
            [3.0, -1.0],
        ]
    )
    actions.data[1] = torch.tensor(
        [
            [10.0, 10.0],
            [99.0, 99.0],
            [99.0, 99.0],
        ]
    )
    actions.data[2] = torch.tensor(
        [
            [2.0, 2.0],
            [5.0, 6.0],
            [99.0, 99.0],
        ]
    )

    metric = ActionSmoothnessAggregateMetric(_evaluator(dt=0.1))
    result = metric.compute({"actions": actions})

    assert result["eval/action_delta_mean_rad"] == pytest.approx(2.5)
    assert result["eval/action_delta_max_rad"] == pytest.approx(3.0)
    assert result["eval/action_rate_mean_rad_s"] == pytest.approx(25.0)
    assert result["eval/action_delta_mean_deg"] == pytest.approx(2.5 * 180 / 3.14159)
    assert result["eval/action_delta_max_deg"] == pytest.approx(3.0 * 180 / 3.14159)
    assert "Action smoothness: mean_delta=2.5000" in capsys.readouterr().out


def test_smoothness_aggregate_metric_prefixes_calculator_results(
    monkeypatch,
    capsys,
):
    calls = {}

    class _Calculator:
        def __init__(self, device, dt, window_sec, high_jerk_threshold):
            calls["init"] = (device, dt, window_sec, high_jerk_threshold)

        def compute_smoothness_metrics(self, metrics, num_bodies):
            calls["compute"] = (metrics, num_bodies)
            return {"normalized_jerk": 1.5, "high_jerk_frame_pct": 0.25}

    monkeypatch.setattr(aggregate_metrics, "SmoothnessCalculator", _Calculator)
    evaluator = _evaluator(dt=0.02, num_bodies=5)
    metric = SmoothnessAggregateMetric(
        evaluator,
        window_sec=0.3,
        high_jerk_threshold=123.0,
    )
    metrics = {"rigid_body_pos": object()}

    result = metric.compute(metrics)

    assert calls["init"] == (torch.device("cpu"), 0.02, 0.3, 123.0)
    assert calls["compute"] == (metrics, 5)
    assert result == {
        "eval/normalized_jerk": 1.5,
        "eval/high_jerk_frame_pct": 0.25,
    }
    assert "Smoothness metric: normalized_jerk, value: 1.5" in capsys.readouterr().out
