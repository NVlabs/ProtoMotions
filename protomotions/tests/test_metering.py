# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for timing and tensor averaging utilities."""

import pytest
import torch

from protomotions.agents.utils import metering
from protomotions.agents.utils.metering import (
    TensorAverageMeter,
    TensorAverageMeterDict,
    TimeReport,
    Timer,
)


def test_timer_tracks_elapsed_time_reports_and_rejects_invalid_state(
    monkeypatch,
    capsys,
):
    times = iter([10.0, 10.25])
    monkeypatch.setattr(metering.time, "time", lambda: next(times))
    timer = Timer("rollout")

    timer.on()
    with pytest.raises(AssertionError, match="already turned on"):
        timer.on()
    timer.off()
    with pytest.raises(AssertionError, match="not started"):
        timer.off()
    timer.report()

    assert timer.num_ons == 1
    assert timer.time_total == pytest.approx(0.25)
    assert "Time report [rollout]: 0.25 0.2500 seconds" in capsys.readouterr().out

    timer.clear()

    assert timer.start_time is None
    assert timer.time_total == 0.0
    assert timer.num_ons == 1


def test_time_report_manages_named_timers_in_sorted_order(monkeypatch, capsys):
    times = iter([0.0, 0.2, 1.0, 1.1])
    monkeypatch.setattr(metering.time, "time", lambda: next(times))
    report = TimeReport()

    report.add_timer("slow")
    report.add_timer("fast")
    with pytest.raises(AssertionError, match="already exists"):
        report.add_timer("slow")
    report.start_timer("slow")
    report.end_timer("slow")
    report.start_timer("fast")
    report.end_timer("fast")
    report.report()
    output = capsys.readouterr().out

    assert output.index("Time report [fast]") < output.index("Time report [slow]")
    assert "------------Time Report------------" in output
    assert "-----------------------------------" in output

    report.report("slow")
    assert "Time report [slow]" in capsys.readouterr().out
    report.clear_timer("slow")
    assert report.timers["slow"].time_total == 0.0
    report.clear_timer()
    assert all(timer.time_total == 0.0 for timer in report.timers.values())
    report.pop_timer("fast")
    assert "fast" not in report.timers
    report.pop_timer()
    assert report.timers == {}
    with pytest.raises(AssertionError, match="does not exist"):
        report.start_timer("missing")


def test_tensor_average_meter_handles_scalars_empty_tensors_and_clear():
    meter = TensorAverageMeter(dtype=torch.float32, device="cpu")
    value = torch.tensor(1.0, requires_grad=True)

    assert meter.mean() == 0
    meter.add(value)
    meter.add(torch.tensor([3.0, 5.0]))

    assert meter.tensors[0].shape == (1,)
    assert meter.tensors[0].device.type == "cpu"
    assert not meter.tensors[0].requires_grad
    assert torch.allclose(meter.mean().cpu(), torch.tensor(3.0))
    assert torch.allclose(meter.mean_and_clear().cpu(), torch.tensor(3.0))
    assert meter.tensors == []

    meter.add(torch.tensor([]))
    assert meter.mean() == 0

    local_meter = TensorAverageMeter(dtype=torch.float32)
    local_meter.add(torch.tensor([2.0, 4.0]))
    assert torch.allclose(local_meter.mean(), torch.tensor(3.0))


def test_tensor_average_meter_dict_creates_meters_means_and_clears():
    meters = TensorAverageMeterDict(dtype=torch.float32, device="cpu")

    meters.add(
        {
            "loss": torch.tensor([1.0, 3.0]),
            "reward": torch.tensor(2.0),
        }
    )
    meters.add({"loss": torch.tensor([5.0])})
    means = meters.mean()

    assert torch.allclose(means["loss"].cpu(), torch.tensor(3.0))
    assert torch.allclose(means["reward"].cpu(), torch.tensor(2.0))
    assert set(meters.data) == {"loss", "reward"}
    cleared_means = meters.mean_and_clear()
    assert torch.allclose(cleared_means["loss"].cpu(), torch.tensor(3.0))
    assert meters.data == {}
