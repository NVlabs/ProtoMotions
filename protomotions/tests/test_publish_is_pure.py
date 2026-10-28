# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publishing the context is a pure read.

``BaseEnv._publish(env_ids)`` asks every publish owner and every control
component for its names. None of them may change state or draw a random
number while answering: the all-env context is rebuilt whenever its cache was
dropped, a reset publishes a second context for just its rows, and the build
publishes once before any step, so a publisher with a side effect would run
it a different number of times than the step order promises. Two guards, on
the real environment over the fake sensor simulator:

- publishing twice in a row (every env, then a subset) leaves every owner's
  and control component's saved state and the global random stream
  unchanged, and the two all-env contexts agree value by value;
- resetting a subset of the rows leaves every other row's saved state and
  context values untouched.
"""

import pytest
import torch

from protomotions.envs.control.mimic_control import MimicControlConfig
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig


def _mimic_kwargs():
    from protomotions.tests.parity.chain_motions import build_chain_motion_lib

    return dict(
        motion_lib=build_chain_motion_lib(1),
        motion_manager=MimicMotionManagerConfig(),
        control_components={"mimic": MimicControlConfig()},
    )


def _tensor_leaves(obj, prefix=""):
    """Every tensor reachable through dicts, lists and tuples, with its path."""
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _tensor_leaves(value, f"{prefix}/{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            yield from _tensor_leaves(value, f"{prefix}/{index}")


def _saved_tensors(env):
    return {
        path: value.detach().clone() for path, value in _tensor_leaves(env.save_state())
    }


def _per_row_context(env):
    return {
        key: value.detach().clone()
        for key, value in env.context.items()
        if isinstance(value, torch.Tensor)
        and value.dim()
        and value.shape[0] == env.num_envs
    }


def _assert_rows_equal(name, before, after, rows=None):
    if rows is not None:
        before, after = before[rows], after[rows]
    assert before.shape == after.shape, name
    assert torch.equal(before, after), f"{name} changed"


@pytest.mark.parametrize("mimic", [False, True])
def test_publishing_twice_changes_neither_state_nor_rng(mimic):
    pytest.importorskip("mujoco")
    from protomotions.tests.support.sensor_simulator import make_sensor_test_env

    env = make_sensor_test_env(history_steps=3, **(_mimic_kwargs() if mimic else {}))
    try:
        env.step(torch.zeros(env.num_envs, 2))  # every owner holds real state
        before = _saved_tensors(env)
        rng_before = torch.get_rng_state().clone()

        first = env._publish()
        second = env._publish()
        subset = env._publish(torch.tensor([1]))

        assert torch.equal(torch.get_rng_state(), rng_before)
        after = _saved_tensors(env)
        assert after.keys() == before.keys()
        for path, value in before.items():
            _assert_rows_equal(path, value, after[path])
        assert set(first) == set(second) == set(subset)
        for key, value in first.items():
            other = second[key]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other), key
            else:
                assert value == other, key
    finally:
        env.close()


def test_partial_reset_leaves_other_rows_untouched():
    pytest.importorskip("mujoco")
    from protomotions.tests.support.sensor_simulator import make_sensor_test_env

    env = make_sensor_test_env(history_steps=3, **_mimic_kwargs())
    try:
        env.step(torch.zeros(env.num_envs, 2))
        others = torch.tensor([0, 2])
        saved_before = _saved_tensors(env)
        context_before = _per_row_context(env)

        env.reset(torch.tensor([1]))

        saved_after = _saved_tensors(env)
        for path, value in saved_before.items():
            if value.dim() and value.shape[0] == env.num_envs:
                _assert_rows_equal(path, value, saved_after[path], rows=others)
        context_after = _per_row_context(env)
        assert set(context_after) == set(context_before)
        for key, value in context_before.items():
            _assert_rows_equal(key, value, context_after[key], rows=others)
    finally:
        env.close()
