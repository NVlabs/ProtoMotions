# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for context-bound MDP component helpers."""

from __future__ import annotations

import logging
import pickle

import pytest
import torch

import protomotions.envs.component_manager as component_manager
from protomotions.envs.base_env.mdp_managers import (
    combine_rewards,
)
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.context import FrozenContext
from protomotions.envs.context_paths import FieldPath, NestedField, resolve_path
from protomotions.envs.mdp_component import MdpComponent, eager_kernel, is_mdp_component
from protomotions.envs.rewards.meta import RewardMeta
from protomotions.envs.terminations.rule import TerminationRule


class _InnerView:
    value: torch.Tensor = FieldPath()
    other: torch.Tensor = FieldPath()


class _NestedView:
    inner: _InnerView = NestedField(_InnerView)
    scale: torch.Tensor = FieldPath()


class _TestContext:
    current: _InnerView = NestedField(_InnerView)
    nested: _NestedView = NestedField(_NestedView)


def _make_context(
    value: torch.Tensor | None = None,
    other: torch.Tensor | None = None,
    env_ids: torch.Tensor | None = None,
) -> FrozenContext:
    """The context as the env publishes it: a flat map under the declared paths."""
    return FrozenContext(
        {
            "current.value": torch.tensor([1.0, 2.0, 3.0]) if value is None else value,
            "current.other": torch.tensor([4.0, 5.0, 6.0]) if other is None else other,
            "nested.inner.value": torch.tensor([7.0, 8.0, 9.0]),
            "nested.inner.other": torch.tensor([10.0, 11.0, 12.0]),
            "nested.scale": torch.tensor([0.5, 1.0, 1.5]),
        },
        env_ids=env_ids,
        dt=0.02,
    )


def _add_value(value: torch.Tensor, increment: float = 0.0) -> torch.Tensor:
    return value + increment


def _negate_value(value: torch.Tensor) -> torch.Tensor:
    return -value


@eager_kernel
def _eager_add(value: torch.Tensor, increment: float = 0.0) -> torch.Tensor:
    return value + increment


_RECORDED_CALLS: list = []


def _recording_add(value: torch.Tensor, increment: float = 0.0) -> torch.Tensor:
    _RECORDED_CALLS.append("eager")
    return value + increment


def _failing_kernel(value: torch.Tensor) -> torch.Tensor:
    raise ValueError("Input pose must be 2D")


def test_same_name_in_different_groups_keeps_its_own_kernel(monkeypatch):
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", True)

    def fake_compile(fn, mode):
        def compiled(**kwargs):
            return fn(**kwargs)

        return compiled

    monkeypatch.setattr(component_manager.torch, "compile", fake_compile)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context()
    observation = MdpComponent(
        compute_func=_add_value,
        dynamic_vars={"value": _TestContext.current.value},
    )
    reward = MdpComponent(
        compute_func=_negate_value,
        dynamic_vars={"value": _TestContext.current.value},
    )
    # The same manager executes observations, rewards and terminations.
    for _ in range(2):
        assert torch.equal(
            manager.execute_all({"shared": observation}, ctx)["shared"],
            ctx["current.value"],
        )
        assert torch.equal(
            manager.execute_all({"shared": reward}, ctx)["shared"],
            -ctx["current.value"],
        )
    subset = _make_context(env_ids=torch.tensor([0, 1, 2]))
    assert torch.equal(
        manager.execute_all({"shared": reward}, subset)["shared"],
        -subset["current.value"],
    )


def test_kernel_cache_keys_each_kernel_by_its_eager_marker(monkeypatch):
    """An @eager_kernel runs raw; an unmarked kernel is compiled; the cache keys them apart."""
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", True)
    compile_calls = []

    def fake_compile(fn, mode):
        compile_calls.append(fn)
        return lambda **kwargs: fn(**kwargs) + 10

    monkeypatch.setattr(component_manager.torch, "compile", fake_compile)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context()

    eager = MdpComponent(
        compute_func=_eager_add,
        dynamic_vars={"value": _TestContext.current.value},
    )
    compiled = MdpComponent(
        compute_func=_add_value,
        dynamic_vars={"value": _TestContext.current.value},
    )
    eager_result = manager.execute_all({"shared": eager}, ctx)["shared"]
    compiled_result = manager.execute_all({"shared": compiled}, ctx)["shared"]
    assert torch.equal(eager_result, ctx["current.value"])
    assert torch.equal(compiled_result, ctx["current.value"] + 10)
    assert compile_calls == [_add_value]


def _affine_value(
    value: torch.Tensor,
    bias: torch.Tensor,
    gain: float = 1.0,
) -> torch.Tensor:
    return value * gain + bias


def test_field_paths_resolve_class_paths_and_instance_values():
    same_path = FieldPath(parent_path="current")
    same_path.name = "value"
    nested = NestedField(_InnerView, parent_path="ctx")
    nested.name = "inner"

    assert _TestContext.current.value.path == "current.value"
    assert _TestContext.nested.inner.other.path == "nested.inner.other"
    assert str(_TestContext.current.value) == "current.value"
    assert repr(_TestContext.current.value) == "FieldPath('current.value')"
    assert _TestContext.current.value == "current.value"
    assert _TestContext.current.value == same_path
    assert _TestContext.current.value != object()
    assert len({_TestContext.current.value, same_path}) == 1
    assert str(nested) == "ctx.inner"
    assert repr(nested) == "NestedField('ctx.inner')"

    ctx = _make_context()
    assert torch.equal(resolve_path(ctx, "current.other"), ctx["current.other"])
    assert torch.equal(
        resolve_path(ctx, "nested.inner.value"),
        torch.tensor([7.0, 8.0, 9.0]),
    )

    with pytest.raises(AttributeError):
        resolve_path(ctx, "current.missing")


def test_the_deploy_slot_is_opaque_copied_with_the_path_and_shown_in_repr():
    # The slot holds whatever the declaration gives it; nothing here reads it.
    facts = {"host": "imu"}

    class _Deployed:
        value: torch.Tensor = FieldPath(deploy=facts)
        plain: torch.Tensor = FieldPath()

    class _Root:
        sensed: _Deployed = NestedField(_Deployed, deploy="whole namespace")

    assert _Root.sensed.value.deploy is facts
    assert _Root.sensed.plain.deploy is None
    assert vars(_Root)["sensed"].deploy == "whole namespace"
    assert repr(_Root.sensed.value) == (
        "FieldPath('sensed.value', deploy={'host': 'imu'})"
    )
    assert repr(_Root.sensed.plain) == "FieldPath('sensed.plain')"
    assert repr(vars(_Root)["sensed"]) == (
        "NestedField('sensed', deploy='whole namespace')"
    )
    # The slot never enters identity: a path is the same name with or without it.
    assert _Root.sensed.value == FieldPath.from_path("sensed.value")
    assert FieldPath.from_path("sensed.value", deploy=facts).deploy is facts


def test_mdp_component_resolves_dynamic_paths_and_passes_static_params_verbatim():
    # static_params reach the kernel unchanged: there are no reserved metadata
    # keys. Reward metadata lives on a typed RewardMeta, not in static_params.
    component = MdpComponent(
        compute_func=_affine_value,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={"bias": torch.tensor([0.5, 1.0, 1.5]), "gain": 2.0},
    )
    ctx = _make_context()

    resolved, func_params = component.resolve_args(ctx)
    result = component.compute(ctx)

    assert is_mdp_component(component)
    assert not is_mdp_component({"weight": 1.0})
    assert component.get_bindings_dict() == {"value": "current.value"}
    assert component.get_compute_func() is _affine_value
    assert resolved == {"value": ctx["current.value"]}
    assert func_params is component.static_params
    assert set(func_params) == {"bias", "gain"}
    assert component._device_ready is True
    assert torch.allclose(result, torch.tensor([2.5, 5.0, 7.5]))

    serialized = component.to_dict()
    assert serialized["compute_func"] == "_affine_value"
    assert serialized["dynamic_vars"] == {"value": "current.value"}
    assert serialized["bias"] == [0.5, 1.0, 1.5]
    assert serialized["gain"] == 2.0


def test_mdp_component_moves_static_tensor_params_to_runtime_device():
    component = MdpComponent(
        compute_func=_affine_value,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={"bias": torch.tensor([1.0, 2.0, 3.0]), "gain": 1.0},
    )
    ctx = _make_context(value=torch.empty(3, device="meta"))

    _, func_params = component.resolve_args(ctx)

    assert func_params["bias"].device.type == "meta"
    assert component.static_params["bias"].device.type == "meta"


def test_mdp_component_reinitializes_static_tensor_device_after_pickle():
    component = MdpComponent(
        compute_func=_affine_value,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={"bias": torch.tensor([1.0, 2.0, 3.0]), "gain": 1.0},
    )
    component.resolve_args(_make_context())
    assert component._device_ready is True
    assert "_device_ready" not in component.__getstate__()

    restored = pickle.loads(pickle.dumps(component))
    assert restored._device_ready is False

    _, func_params = restored.resolve_args(
        _make_context(value=torch.empty(3, device="meta"))
    )
    assert func_params["bias"].device.type == "meta"


def test_combine_rewards_applies_grace_mask_weighting_and_clamps():
    grace_mask = torch.tensor([False, True, False])
    raw_rewards = {
        "tracking": torch.tensor([1.0, 2.0, 3.0]),
        "energy": torch.tensor([-2.0, 0.5, 4.0]),
    }
    components = {
        "tracking": MdpComponent(
            compute_func=_add_value,
            dynamic_vars={},
            meta=RewardMeta(weight=2.0, zero_during_grace_period=True),
        ),
        "energy": MdpComponent(
            compute_func=_add_value,
            dynamic_vars={},
            meta=RewardMeta(weight=-1.0, min_value=-2.0, max_value=1.0),
        ),
    }

    reward, logs = combine_rewards(
        raw_rewards,
        components,
        grace_mask=grace_mask,
        num_envs=3,
        device=torch.device("cpu"),
    )

    assert torch.allclose(reward, torch.tensor([3.0, -0.5, 4.0]))
    assert torch.equal(logs["raw_r/tracking"], torch.tensor([1.0, 0.0, 3.0]))
    assert torch.allclose(
        logs["scaled_r/tracking"],
        torch.tensor([2.0, 0.0, 6.0]),
    )
    assert torch.allclose(logs["scaled_r/energy"], torch.tensor([1.0, -0.5, -2.0]))


def test_combine_rewards_adds_multiplicative_components_to_additive_sum():
    reward, logs = combine_rewards(
        raw_rewards={
            "alive": torch.tensor([0.5, 2.0]),
            "progress": torch.tensor([1.0, 3.0]),
        },
        components={
            "alive": MdpComponent(
                compute_func=_add_value,
                dynamic_vars={},
                meta=RewardMeta(multiplicative=True),
            ),
            "progress": MdpComponent(
                compute_func=_add_value,
                dynamic_vars={},
                meta=RewardMeta(weight=0.25),
            ),
        },
        num_envs=2,
        device=torch.device("cpu"),
    )

    assert torch.allclose(reward, torch.tensor([0.75, 2.75]))
    assert torch.allclose(logs["multiplicative_reward"], torch.tensor([0.5, 2.0]))
    assert torch.allclose(logs["additive_reward"], torch.tensor([0.25, 0.75]))


def test_combine_rewards_rejects_non_finite_values():
    with pytest.raises(AssertionError, match="Reward 'bad' not finite"):
        combine_rewards(
            raw_rewards={"bad": torch.tensor([1.0, float("nan")])},
            components={
                "bad": MdpComponent(
                    compute_func=_add_value,
                    dynamic_vars={},
                    meta=RewardMeta(weight=1.0),
                )
            },
            num_envs=2,
            device=torch.device("cpu"),
        )


def test_evaluation_predicates_compare_without_mutating_metrics():
    ctx = _make_context(
        value=torch.tensor([0.2, 0.8, 1.0]), other=torch.tensor([0.1, 0.5, 0.9])
    )
    height = MdpComponent(
        _add_value, {"value": _TestContext.current.value}, {"increment": 0.0}
    )
    clearance = MdpComponent(
        _add_value, {"value": _TestContext.current.other}, {"increment": 0.0}
    )
    above = TerminationRule(height, 0.7).compare(height.compute(ctx))
    below = TerminationRule(clearance, 0.2, fail_above=False).compare(
        clearance.compute(ctx)
    )
    assert torch.equal(above | below, torch.tensor([True, True, True]))
    assert torch.equal(above, torch.tensor([False, True, True]))
    assert torch.equal(below, torch.tensor([True, False, False]))
    assert torch.equal(height.compute(ctx), torch.tensor([0.2, 0.8, 1.0]))


def test_component_manager_executes_all_and_single_in_eager_mode(monkeypatch):
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", False)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context(
        value=torch.tensor([1.0, 2.0]),
        other=torch.tensor([3.0, 4.0]),
    )
    components = {
        "value": MdpComponent(
            compute_func=_add_value,
            dynamic_vars={"value": _TestContext.current.value},
            static_params={"increment": 2.0},
        ),
        "other": MdpComponent(
            compute_func=_add_value,
            dynamic_vars={"value": _TestContext.current.other},
            static_params={"increment": -1.0},
        ),
    }

    results = manager.execute_all(components, ctx)
    single = manager.execute_single("single", components["value"], ctx, compile=False)

    assert torch.equal(results["value"], torch.tensor([3.0, 4.0]))
    assert torch.equal(results["other"], torch.tensor([2.0, 3.0]))
    assert torch.equal(single, torch.tensor([3.0, 4.0]))
    assert set(manager._compiled) == {(_add_value, False)}


def test_component_manager_execute_all_uses_eager_for_subset_context(monkeypatch):
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", True)

    def fail_compile(fn, mode):
        raise AssertionError("subset contexts should not be compiled")

    monkeypatch.setattr(component_manager.torch, "compile", fail_compile)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context(value=torch.tensor([1.0, 2.0]), env_ids=torch.tensor([2, 5]))
    component = MdpComponent(
        compute_func=_add_value,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={"increment": 2.0},
    )

    results = manager.execute_all({"value": component}, ctx)

    assert torch.equal(results["value"], torch.tensor([3.0, 4.0]))
    assert manager._compiled == {}


@pytest.mark.parametrize("bulk", [False, True])
def test_component_manager_compiles_once_and_calls_the_compiled_kernel(
    monkeypatch, bulk
):
    """torch.compile runs once per kernel; every later call goes to its result."""
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", True)
    compile_calls = []
    compiled_calls = []

    def fake_compile(fn, mode):
        compile_calls.append((fn, mode))

        def compiled_fn(**kwargs):
            compiled_calls.append(kwargs)
            return fn(**kwargs) + 10.0

        return compiled_fn

    monkeypatch.setattr(component_manager.torch, "compile", fake_compile)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context(value=torch.tensor([1.0, 2.0]))
    component = MdpComponent(
        compute_func=_add_value,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={"increment": 2.0},
    )

    if bulk:
        first = manager.execute_all({"value": component}, ctx)["value"]
        second = manager.execute_all({"value": component}, ctx)["value"]
    else:
        first = manager.execute_single("value", component, ctx, compile=True)
        second = manager.execute_single("value", component, ctx, compile=True)

    assert torch.equal(first, torch.tensor([13.0, 14.0]))
    assert torch.equal(second, torch.tensor([13.0, 14.0]))
    assert compile_calls == [(_add_value, "default")]
    assert len(compiled_calls) == 2
    assert manager._compiled[(_add_value, False)] is not component.compute_func


def test_component_manager_compile_time_error_propagates(monkeypatch):
    """torch.compile() raising is not turned into a silent eager run."""
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", True)

    def failing_compile(fn, mode):
        raise RuntimeError("compile unavailable")

    monkeypatch.setattr(component_manager.torch, "compile", failing_compile)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context(value=torch.tensor([1.0, 2.0]))
    component = MdpComponent(
        compute_func=_add_value,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={"increment": 2.0},
    )

    with pytest.raises(RuntimeError, match="compile unavailable"):
        manager.execute_single("value", component, ctx, compile=True)
    assert manager._compiled == {}


@pytest.mark.parametrize("bulk", [False, True])
def test_component_manager_names_the_component_when_the_compiled_kernel_fails(
    monkeypatch, bulk
):
    """A failure under torch.compile propagates once, naming the component and
    the kernel and chained to the kernel's own exception. The kernel is not
    re-run eagerly, not demoted to eager, and the next call fails the same way
    instead of silently running eagerly."""
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", True)
    _RECORDED_CALLS.clear()

    def fake_compile(fn, mode):
        def compiled_fn(**kwargs):
            _RECORDED_CALLS.append("compiled")
            raise ValueError("Input pose must be 2D")

        return compiled_fn

    monkeypatch.setattr(component_manager.torch, "compile", fake_compile)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context(value=torch.tensor([1.0, 2.0]))
    component = MdpComponent(
        compute_func=_recording_add,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={"increment": 2.0},
    )

    def run():
        if bulk:
            return manager.execute_all({"value": component}, ctx)
        return manager.execute_single("value", component, ctx, compile=True)

    expected = (
        r"component 'value': kernel '_recording_add' raised under torch.compile "
        r"\(ValueError: Input pose must be 2D\).*@eager_kernel"
    )
    with pytest.raises(RuntimeError, match=expected) as info:
        run()
    assert isinstance(info.value.__cause__, ValueError)
    assert _RECORDED_CALLS == ["compiled"]
    assert manager._compiled[(_recording_add, False)] is not _recording_add
    with pytest.raises(RuntimeError, match=expected):
        run()
    assert _RECORDED_CALLS == ["compiled", "compiled"]


def test_component_manager_eager_kernel_error_propagates_unwrapped(monkeypatch, caplog):
    """Without torch.compile in play the kernel's own exception is the one the
    caller sees, once, and nothing is logged as a compile failure."""
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", False)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context(value=torch.tensor([1.0, 2.0]))
    component = MdpComponent(
        compute_func=_failing_kernel,
        dynamic_vars={"value": _TestContext.current.value},
    )

    with caplog.at_level(logging.WARNING, logger=component_manager.__name__):
        with pytest.raises(ValueError, match="Input pose must be 2D"):
            manager.execute_all({"value": component}, ctx)
        with pytest.raises(ValueError, match="Input pose must be 2D"):
            manager.execute_single("value", component, ctx, compile=True)

    assert "torch.compile" not in caplog.text
    assert manager._compiled[(_failing_kernel, False)] is _failing_kernel


@eager_kernel
def _failing_eager_kernel(value: torch.Tensor) -> torch.Tensor:
    raise ValueError("Input pose must be 2D")


def test_component_manager_eager_kernel_is_never_compiled_nor_wrapped(monkeypatch):
    """With torch.compile available, an @eager_kernel under an all-env context
    is called as itself: torch.compile is not invoked and its own exception is
    the one the caller sees."""
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", True)

    def must_not_compile(fn, mode):
        raise AssertionError("an @eager_kernel must not reach torch.compile")

    monkeypatch.setattr(component_manager.torch, "compile", must_not_compile)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context(value=torch.tensor([1.0, 2.0]))
    component = MdpComponent(
        compute_func=_failing_eager_kernel,
        dynamic_vars={"value": _TestContext.current.value},
    )

    with pytest.raises(ValueError, match="Input pose must be 2D"):
        manager.execute_all({"value": component}, ctx)
    with pytest.raises(ValueError, match="Input pose must be 2D"):
        manager.execute_single("value", component, ctx, compile=True)
    assert manager._compiled[(_failing_eager_kernel, True)] is _failing_eager_kernel


def test_component_manager_backend_failure_names_the_environment(monkeypatch):
    """A backend failure (triton or a C++ compiler missing, an inductor bug)
    surfaces on the first call as torch's BackendCompilerFailed; the hint says
    the environment is at fault, not the kernel, and does not suggest
    @eager_kernel."""
    from torch._dynamo.exc import BackendCompilerFailed

    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", True)

    def fake_compile(fn, mode):
        def compiled_fn(**kwargs):
            raise BackendCompilerFailed("inductor", RuntimeError("no triton"), None)

        return compiled_fn

    monkeypatch.setattr(component_manager.torch, "compile", fake_compile)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context(value=torch.tensor([1.0, 2.0]))
    component = MdpComponent(
        compute_func=_add_value,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={"increment": 2.0},
    )

    with pytest.raises(
        RuntimeError,
        match=r"component 'value': kernel '_add_value' raised under torch.compile "
        r"\(BackendCompilerFailed:",
    ) as info:
        manager.execute_all({"value": component}, ctx)
    message = str(info.value)
    assert "torch.compile's backend failed" in message
    assert "@eager_kernel" not in message
    assert isinstance(info.value.__cause__, BackendCompilerFailed)
