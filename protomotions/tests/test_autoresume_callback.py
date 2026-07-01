# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SLURM autoresume callback."""

from types import SimpleNamespace

from protomotions.agents.callbacks import slurm_autoresume_srun as autoresume
from protomotions.agents.callbacks.slurm_autoresume_srun import (
    AutoResumeCallbackSrun,
    wandb_run_exists,
)


class _Strategy:
    def __init__(self):
        self.barrier_calls = 0

    def barrier(self):
        self.barrier_calls += 1


class _Agent:
    def __init__(self):
        self.fabric = SimpleNamespace(strategy=_Strategy())
        self._should_stop = False
        self.save_calls = 0

    @property
    def should_stop(self):
        return self._should_stop

    def save(self):
        self.save_calls += 1


def test_autoresume_callback_initializes_start_time_without_stopping(
    monkeypatch,
    capsys,
):
    times = iter([100.0, 105.0])
    monkeypatch.setattr(autoresume.time, "time", lambda: next(times))
    callback = AutoResumeCallbackSrun(autoresume_after=10)
    agent = _Agent()
    capsys.readouterr()

    callback.before_play_steps(agent)

    assert callback.start_time == 100.0
    assert agent.fabric.strategy.barrier_calls == 1
    assert agent.save_calls == 0
    assert agent.should_stop is False


def test_autoresume_callback_saves_and_stops_after_threshold(monkeypatch):
    monkeypatch.setattr(autoresume.time, "time", lambda: 115.0)
    callback = AutoResumeCallbackSrun(autoresume_after=10)
    callback.start_time = 100.0
    agent = _Agent()

    callback.before_play_steps(agent)

    assert agent.fabric.strategy.barrier_calls == 1
    assert agent.save_calls == 1
    assert agent.should_stop is True


def test_autoresume_callback_fit_hooks_are_noops():
    callback = AutoResumeCallbackSrun(autoresume_after=10)
    agent = _Agent()

    assert callback.on_fit_start(agent) is None
    assert callback.on_fit_end(agent) is None


def test_wandb_run_exists_uses_wandb_run_type(monkeypatch):
    class _Run:
        pass

    monkeypatch.setattr(autoresume.wandb.sdk.wandb_run, "Run", _Run)
    monkeypatch.setattr(autoresume.wandb, "run", object())
    assert wandb_run_exists() is False

    monkeypatch.setattr(autoresume.wandb, "run", _Run())
    assert wandb_run_exists() is True
