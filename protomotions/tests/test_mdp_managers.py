# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three MDP managers (``base_env/mdp_managers.py``) on hand-made contexts."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.envs.base_env.mdp_managers import (
    ObservationManager,
    RewardManager,
    TerminationManager,
)
from protomotions.envs.rewards.meta import RewardMeta

CPU = torch.device("cpu")


def _reward_kernel():
    """A dummy kernel; the _Executor returns canned reward values instead."""
    return None


class _Executor:
    """Stands in for ``ComponentManager``: canned results, recorded calls."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def execute_all(self, components, ctx):
        self.calls.append((components, ctx))
        return self.results


def _observations(results, *, num_envs=3):
    manager = ObservationManager(
        _Executor(results), {name: object() for name in results}, num_envs, CPU
    )
    return manager, manager.buffer


def test_observation_manager_accepts_subset_and_multidimensional_outputs():
    env_ids = torch.tensor([0, 2])
    values = torch.arange(12, dtype=torch.float).reshape(2, 2, 3)
    manager, buffer = _observations({"grid": values})

    manager.compute(SimpleNamespace(env_ids=env_ids), env_ids)

    # The buffer grows to every env on first sight, the rows land in place.
    assert buffer["grid"].shape == (3, 2, 3)
    assert torch.equal(buffer["grid"][env_ids], values)
    assert torch.equal(buffer["grid"][1], torch.zeros(2, 3))
    assert manager.buffer is buffer


def test_observation_manager_preserves_global_order_for_reordered_full_batch():
    env_ids = torch.tensor([2, 0, 1])
    values = torch.tensor([[10.0], [20.0], [30.0]])
    manager, buffer = _observations({"ordered": values})

    # An all-env context (env_ids None) returns every row in global order; the
    # destination rows select from it.
    manager.compute(SimpleNamespace(env_ids=None), env_ids)

    assert torch.equal(buffer["ordered"], values)


def test_observation_manager_rejects_rows_that_do_not_match_the_context():
    env_ids = torch.tensor([2, 0])
    manager, _ = _observations({"bad": torch.zeros(3, 1)})

    with pytest.raises(ValueError, match="bad.*2 rows.*got 3"):
        manager.compute(SimpleNamespace(env_ids=env_ids), env_ids)
    with pytest.raises(ValueError, match="must match the destination"):
        manager.compute(SimpleNamespace(env_ids=torch.tensor([0, 2])), env_ids)
    manager, _ = _observations({"scalar": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="scalar.*3 rows.*got 0"):
        manager.compute(SimpleNamespace(env_ids=None), torch.arange(3))
    # A zero-width observation is refused: a module that reads it is a no-op.
    manager, _ = _observations({"empty": torch.zeros(3, 0)})
    with pytest.raises(ValueError, match="'empty' has a zero last dimension"):
        manager.compute(SimpleNamespace(env_ids=None), torch.arange(3))


def test_observation_manager_get_obs_copies_the_buffer():
    rows = torch.arange(3)
    values = torch.tensor([[1.0], [2.0], [3.0]])
    manager, buffer = _observations({"dynamic": values})
    manager.compute(SimpleNamespace(env_ids=None), rows)
    obs = manager.get_obs()
    assert set(obs) == {"dynamic"}  # every observation is a kernel of the table
    obs["dynamic"][0, 0] = -99.0
    assert buffer["dynamic"][0, 0] == 1.0


def test_reward_manager_combines_with_the_grace_mask_and_returns_logging():
    components = {
        "alive": MdpComponent(_reward_kernel, {}, meta=RewardMeta(weight=2.0)),
        "power": MdpComponent(
            _reward_kernel,
            {},
            meta=RewardMeta(weight=1.0, zero_during_grace_period=True),
        ),
    }
    results = {
        "alive": torch.tensor([1.0, 2.0, 3.0]),
        "power": torch.tensor([10.0, 10.0, 10.0]),
    }
    ctx = object()
    manager = RewardManager(_Executor(results), components, 3, CPU, 2)
    progress = torch.tensor([1, 2, 3])

    assert torch.equal(manager.grace_mask(progress), torch.tensor([True, True, False]))
    reward, logging = manager.compute(ctx, progress)

    assert torch.equal(reward, torch.tensor([2.0, 4.0, 16.0]))
    assert list(logging) == [
        "raw_r/alive",
        "scaled_r/alive",
        "raw_r/power",
        "scaled_r/power",
        "total_env_reward",
    ]
    assert torch.equal(logging["raw_r/power"], torch.tensor([0.0, 0.0, 10.0]))
    assert logging["total_env_reward"] is reward
    assert manager.component_manager.calls == [(components, ctx)]

    # No grace period: None mask, nothing zeroed.
    no_grace = RewardManager(_Executor(results), components, 3, CPU, 0)
    assert no_grace.grace_mask(progress) is None
    reward, _ = no_grace.compute(ctx, progress)
    assert torch.equal(reward, torch.tensor([12.0, 14.0, 16.0]))


def test_termination_manager_combines_timeout_requests_and_rules():
    def kernel():  # never called: the executor returns canned values
        return None

    def rule(**kwargs):
        return TerminationRule(MdpComponent(kernel, {}), **kwargs)

    # A boolean rule, a truncating boolean rule and a scalar rule with a cut.
    components = {
        "fell": rule(),
        "clip": rule(truncation=True),
        "far": rule(threshold=1.0),
    }
    results = {
        "fell": torch.tensor([False, True, False]),
        "clip": torch.tensor([True, False, False]),
        "far": torch.tensor([0.5, 0.5, 2.0]),
    }
    ctx = object()
    manager = TerminationManager(_Executor(results), components, 3, CPU)
    progress = torch.tensor([0, 0, 5])
    no_request = torch.zeros(3, dtype=torch.bool)

    reset, terminated, logging = manager.compute(ctx, progress, 5, no_request)

    # Timeout (env 2) | clip (env 0) | fell (env 1) | far (env 2); the timeout
    # and the truncating clip end reset without terminating.
    assert torch.equal(reset, torch.tensor([True, True, True]))
    assert torch.equal(terminated, torch.tensor([False, True, True]))
    assert list(logging) == ["termination/fell", "termination/clip", "termination/far"]
    assert torch.equal(logging["termination/fell"], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(logging["termination/clip"], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(logging["termination/far"], torch.tensor([0.0, 0.0, 1.0]))
    # The kernels run under their entry names, unwrapped from the rules.
    components_run, ctx_run = manager.component_manager.calls[0]
    assert components_run == {name: rule.component for name, rule in components.items()}
    assert ctx_run is ctx
    # The inputs are left alone: the manager is a pure function of them.
    assert torch.equal(progress, torch.tensor([0, 0, 5]))

    # A reset request resets without terminating, like the timeout.
    reset, terminated, logging = TerminationManager(_Executor({}), {}, 3, CPU).compute(
        ctx, progress, 5, torch.tensor([True, False, False])
    )
    assert torch.equal(reset, torch.tensor([True, False, True]))
    assert not terminated.any() and logging == {}
