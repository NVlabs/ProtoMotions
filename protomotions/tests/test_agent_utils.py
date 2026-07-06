# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for small agent utility modules that should not need live simulators."""

from types import SimpleNamespace

import torch
from torch import nn

from protomotions.agents.utils.normalization import (
    RewardRunningMeanStd,
    RunningMeanStd,
    combine_moments,
    materialize_lazy_running_stats_from_state_dict,
)
from protomotions.agents.utils.replay_buffer import ReplayBuffer
from protomotions.agents.utils.step_tracker import StepTracker


def _fabric():
    return SimpleNamespace(world_size=1, global_rank=0)


class _DistributedFabric:
    world_size = 2
    global_rank = 0

    def __init__(self):
        self.broadcasts = []
        self.gather_calls = 0

    def all_gather(self, value):
        self.gather_calls += 1
        if isinstance(value, torch.Tensor):
            if self.gather_calls == 1:
                return torch.stack([value, value + 2.0])
            return torch.stack([value, value])
        return torch.tensor([value, value])

    def broadcast(self, value, src):
        self.broadcasts.append((value.clone(), src))
        return value


def test_step_tracker_resets_advances_and_reports_done_indices():
    tracker = StepTracker(
        num_envs=3,
        min_steps=2,
        max_steps=3,
        device=torch.device("cpu"),
    )

    tracker.reset_steps()
    tracker.advance()
    assert tracker.done_indices().numel() == 0

    tracker.advance()
    assert torch.equal(tracker.done_indices(), torch.tensor([0, 1, 2]))

    tracker.shift_counter(torch.tensor([1]), torch.tensor([1]))
    assert torch.equal(tracker.steps, torch.tensor([2, 1, 2]))
    assert torch.equal(tracker.cur_max_steps, torch.tensor([2, 1, 2]))
    assert torch.equal(tracker.done_indices(), torch.tensor([0, 1, 2]))


def test_replay_buffer_wraps_oldest_entries_and_samples_registered_keys():
    buffer = ReplayBuffer(buffer_size=5, device=torch.device("cpu"))
    buffer.store({"obs": torch.arange(3).view(3, 1)})
    buffer.store({"obs": torch.arange(3, 6).view(3, 1)})

    assert len(buffer) == 5
    assert buffer._head == 1
    assert torch.equal(buffer.obs.squeeze(-1), torch.tensor([5, 1, 2, 3, 4]))

    torch.manual_seed(0)
    sample = buffer.sample(4)
    assert set(sample) == {"obs"}
    assert sample["obs"].shape == (4, 1)
    assert set(sample["obs"].squeeze(-1).tolist()).issubset({1, 2, 3, 4, 5})


def test_replay_buffer_reset_keeps_storage_but_marks_buffer_empty():
    buffer = ReplayBuffer(buffer_size=2, device=torch.device("cpu"))
    buffer.store({"obs": torch.ones(2, 1)})

    buffer.reset()

    assert len(buffer) == 0
    assert buffer._head == 0
    assert hasattr(buffer, "obs")


def test_combine_moments_matches_equivalent_concatenated_batch():
    mean, var, count = combine_moments(
        [torch.tensor([0.0]), torch.tensor([4.0])],
        [torch.tensor([1.0]), torch.tensor([1.0])],
        [torch.tensor(2), torch.tensor(2)],
    )

    assert torch.allclose(mean, torch.tensor([2.0]))
    assert torch.allclose(var, torch.tensor([5.0]))
    assert count == 4


def test_combine_moments_carries_odd_batch_through_pairwise_reduction():
    mean, var, count = combine_moments(
        [torch.tensor([0.0]), torch.tensor([4.0]), torch.tensor([10.0])],
        [torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([0.0])],
        [torch.tensor(2), torch.tensor(2), torch.tensor(1)],
    )

    values = torch.tensor([-1.0, 1.0, 3.0, 5.0, 10.0])
    assert torch.allclose(mean, values.mean().view(1))
    assert torch.allclose(var, values.var(unbiased=False).view(1))
    assert count == 5


def test_running_mean_std_lazy_init_records_and_round_trips_normalized_values():
    rms = RunningMeanStd(_fabric(), shape=None, device="cpu")
    values = torch.tensor([[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]])

    rms.record_moments(values)
    normalized = rms.normalize(values)
    restored = rms.normalize(normalized, un_norm=True)

    assert rms._initialized
    assert rms.mean.shape == (2,)
    assert torch.allclose(restored, values, atol=1e-5)
    assert rms.to(torch.device("cpu")) is rms
    assert rms.device == torch.device("cpu")


def test_running_mean_std_ema_updates_without_accumulating_count():
    rms = RunningMeanStd(
        _fabric(),
        shape=(1,),
        device="cpu",
        ema_decay=0.5,
    )

    rms.record_moments(torch.tensor([[2.0], [4.0]]))

    assert torch.allclose(rms.mean, torch.tensor([1.5], dtype=torch.float64))
    assert torch.allclose(rms.var, torch.tensor([1.0], dtype=torch.float64))
    assert rms.count == 1


def test_record_moments_records_local_only_with_distributed_fabric():
    fabric = _DistributedFabric()
    rms = RunningMeanStd(
        fabric,
        shape=(1,),
        device="cpu",
    )
    rms.count.zero_()

    rms.record_moments(torch.tensor([[1.0], [3.0]]))

    assert fabric.gather_calls == 0
    assert fabric.broadcasts == []
    assert torch.allclose(rms.mean, torch.tensor([2.0], dtype=torch.float64))
    assert torch.allclose(rms.var, torch.tensor([1.0], dtype=torch.float64))
    assert rms.count == 2


def test_record_moments_uses_no_hot_path_collectives(monkeypatch):
    """record_moments stays local; explicit training-loop sync owns collectives."""
    import protomotions.agents.utils.normalization as norm_mod

    fabric = _DistributedFabric()
    rms = RunningMeanStd(fabric, shape=(1,), device="cpu")
    rms.count.zero_()

    broadcast_tensor_calls = []
    object_broadcast_calls = []

    def fake_broadcast(tensor, src=0, group=None):
        # Emulate rank-0 as source: leave the (already rank-0-correct) tensor.
        broadcast_tensor_calls.append((tensor, src))
        return None

    def fake_broadcast_object_list(obj_list, src=0, group=None):
        object_broadcast_calls.append((list(obj_list), src))
        return None

    monkeypatch.setattr(norm_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(norm_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(norm_mod.dist, "broadcast", fake_broadcast)
    monkeypatch.setattr(
        norm_mod.dist, "broadcast_object_list", fake_broadcast_object_list
    )

    rms.record_moments(torch.tensor([[1.0], [3.0]]))

    assert broadcast_tensor_calls == []
    assert object_broadcast_calls == []
    assert fabric.broadcasts == []
    assert torch.allclose(rms.mean, torch.tensor([2.0], dtype=torch.float64))
    assert rms.count == 2


def test_materialize_lazy_running_stats_from_state_dict_creates_missing_buffers():
    model = nn.Module()
    model.obs_norm = RunningMeanStd(_fabric(), shape=None, device="cpu")
    state_dict = {
        "obs_norm.mean": torch.zeros(3, dtype=torch.float64),
        "obs_norm.var": torch.ones(3, dtype=torch.float64),
        "obs_norm.count": torch.tensor(10),
    }

    materialize_lazy_running_stats_from_state_dict(model, state_dict)
    model.load_state_dict(state_dict)

    assert model.obs_norm._initialized
    assert model.obs_norm.shape == torch.Size([3])
    assert torch.equal(model.obs_norm.count, torch.tensor(10))


def test_materialize_lazy_running_stats_skips_initialized_or_missing_state():
    model = nn.Module()
    model.initialized = RunningMeanStd(_fabric(), shape=(2,), device="cpu")
    model.missing = RunningMeanStd(_fabric(), shape=None, device="cpu")

    materialize_lazy_running_stats_from_state_dict(
        model,
        {
            "initialized.mean": torch.ones(3, dtype=torch.float64),
        },
    )

    assert model.initialized.shape == (2,)
    assert model.missing._initialized is False


def test_reward_running_mean_std_tracks_discounted_returns_with_termination_reset():
    rms = RewardRunningMeanStd(
        _fabric(),
        shape=(2,),
        gamma=0.5,
        device="cpu",
    )

    rms.record_reward(
        reward=torch.tensor([1.0, 2.0]),
        terminated=torch.tensor([False, False]),
    )
    rms.record_reward(
        reward=torch.tensor([1.0, 2.0]),
        terminated=torch.tensor([False, True]),
    )

    assert torch.allclose(rms.discounted_reward, torch.tensor([1.5, 2.0]))
    assert torch.all(torch.isfinite(rms.normalize(torch.tensor([1.0, 2.0]))))
    assert torch.all(torch.isfinite(rms.normalize(torch.tensor([1.0, 2.0]), un_norm=True)))
