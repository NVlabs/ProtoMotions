# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for smoothness metric tensor calculations."""

import logging
import re
import runpy
import sys

import pytest
import torch

from protomotions.agents.evaluators.metrics import MotionMetrics
from protomotions.agents.evaluators.smoothness_calculator import SmoothnessCalculator


def _cubic_position_metrics():
    metrics = MotionMetrics(
        num_motions=2,
        motion_lens=torch.tensor([4, 3]),
        max_motion_len=4,
        num_sub_features=3,
        device=torch.device("cpu"),
    )
    metrics.data[0] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
            [27.0, 0.0, 0.0],
        ]
    )
    metrics.data[1] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [99.0, 99.0, 99.0],
        ]
    )
    metrics.frame_counts[:] = torch.tensor([4, 3])
    return metrics


def test_smoothness_calculator_finite_differences_and_empty_windows():
    calculator = SmoothnessCalculator(
        device=torch.device("cpu"),
        dt=0.5,
        window_sec=2.0,
    )
    values = torch.tensor([[0.0], [1.0], [3.0]])

    assert torch.equal(calculator._diff(values, dt=0.5), torch.tensor([[2.0], [4.0]]))
    assert calculator._compute_windowed_normalized_jerk(
        torch.zeros(3, 2, 3),
        window_frames=4,
    ).shape == (0, 2)
    assert calculator._compute_windowed_normalized_jerk(
        torch.zeros(4, 2, 3),
        window_frames=3,
    ).shape == (0, 2)


def test_compute_normalized_jerk_handles_valid_and_short_motions():
    calculator = SmoothnessCalculator(
        device=torch.device("cpu"),
        dt=1.0,
        window_sec=4.0,
        high_jerk_threshold=5.0,
    )

    per_motion, per_body, windowed = calculator.compute_normalized_jerk_from_pos(
        _cubic_position_metrics(),
        num_bodies=1,
        window_sec=4.0,
        eps=0.1,
    )

    expected_nj = (3.0**5 * 36.0) / (27.0**2 + 0.1)
    assert per_motion[0].item() == pytest.approx(expected_nj)
    assert per_body[0, 0].item() == pytest.approx(expected_nj)
    assert per_motion[1].item() == 0.0
    assert windowed[0].shape == (1, 1)
    assert windowed[1].shape == (0, 1)


def test_high_jerk_percentage_uses_default_or_explicit_threshold():
    calculator = SmoothnessCalculator(
        device=torch.device("cpu"),
        dt=1.0,
        high_jerk_threshold=5.0,
    )
    windowed = torch.tensor([[1.0, 2.0], [6.0, 1.0], [4.0, 7.0]])

    assert calculator._compute_high_jerk_frame_percentage(torch.empty(0, 2)) == 0.0
    assert calculator._compute_high_jerk_frame_percentage(windowed) == pytest.approx(
        100.0 * 2 / 3
    )
    assert calculator._compute_high_jerk_frame_percentage(
        windowed,
        threshold=10.0,
    ) == 0.0


def test_compute_smoothness_metrics_reports_valid_and_missing_inputs():
    calculator = SmoothnessCalculator(
        device=torch.device("cpu"),
        dt=1.0,
        window_sec=4.0,
        high_jerk_threshold=5.0,
    )

    assert calculator.compute_smoothness_metrics({}, num_bodies=1) == {}

    result = calculator.compute_smoothness_metrics(
        {"rigid_body_pos": _cubic_position_metrics()},
        num_bodies=1,
        window_sec=4.0,
    )
    expected_nj = (3.0**5 * 36.0) / (27.0**2 + 0.1)

    assert result["normalized_jerk_mean"] == pytest.approx(expected_nj)
    assert result["high_jerk_frame_percentage_mean"] == pytest.approx(100.0)


def test_compute_smoothness_metrics_returns_zero_when_no_motion_has_jerk():
    calculator = SmoothnessCalculator(
        device=torch.device("cpu"),
        dt=1.0,
        window_sec=4.0,
    )
    linear = MotionMetrics(
        num_motions=1,
        motion_lens=torch.tensor([4]),
        max_motion_len=4,
        num_sub_features=3,
        device=torch.device("cpu"),
    )
    linear.data[0] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    linear.frame_counts[:] = 4

    assert calculator.compute_smoothness_metrics(
        {"rigid_body_pos": linear},
        num_bodies=1,
    ) == {
        "normalized_jerk_mean": 0.0,
        "high_jerk_frame_percentage_mean": 0.0,
    }


def test_compute_smoothness_metrics_handles_valid_motion_without_windowed_jerk(
    monkeypatch,
):
    calculator = SmoothnessCalculator(
        device=torch.device("cpu"),
        dt=1.0,
        window_sec=0.25,
    )
    metrics = MotionMetrics(
        num_motions=1,
        motion_lens=torch.tensor([4]),
        max_motion_len=4,
        num_sub_features=3,
        device=torch.device("cpu"),
    )

    def fake_compute(*args, **kwargs):
        assert kwargs["window_sec"] == 0.25
        return (
            torch.tensor([2.0]),
            torch.tensor([[2.0]]),
            [torch.empty(0, 1)],
        )

    monkeypatch.setattr(calculator, "compute_normalized_jerk_from_pos", fake_compute)

    assert calculator.compute_smoothness_metrics(
        {"rigid_body_pos": metrics},
        num_bodies=1,
    ) == {
        "normalized_jerk_mean": 2.0,
        "high_jerk_frame_percentage_mean": 0.0,
    }


def test_compute_smoothness_metrics_logs_and_suppresses_shape_errors(caplog):
    calculator = SmoothnessCalculator(
        device=torch.device("cpu"),
        dt=1.0,
    )
    bad_shape = MotionMetrics(
        num_motions=1,
        motion_lens=torch.tensor([4]),
        max_motion_len=4,
        num_sub_features=2,
        device=torch.device("cpu"),
    )
    bad_shape.frame_counts[:] = 4

    with caplog.at_level(logging.WARNING):
        result = calculator.compute_smoothness_metrics(
            {"rigid_body_pos": bad_shape},
            num_bodies=1,
        )

    assert result == {}
    assert "Failed to compute normalized jerk" in caplog.text


def test_smoothness_calculator_main_reports_demo_metric_values(capsys, monkeypatch):
    monkeypatch.delitem(
        sys.modules,
        "protomotions.agents.evaluators.smoothness_calculator",
        raising=False,
    )

    runpy.run_module(
        "protomotions.agents.evaluators.smoothness_calculator",
        run_name="__main__",
    )

    output = capsys.readouterr().out

    assert "Window size: 0.4s" in output
    assert "High jerk threshold: 6500.0" in output
    assert "Time step: 0.0333s (30.0 FPS)" in output
    assert re.search(r"normalized_jerk_mean\s+:\s+100519\.46", output)
    assert "high_jerk_frame_percentage_mean:        52.78%" in output
    assert "normalized_jerk_max" not in output
