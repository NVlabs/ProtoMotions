# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for context-bound MDP component helpers."""
from __future__ import annotations

import logging

import pytest
import torch

import protomotions.envs.component_manager as component_manager
from protomotions.envs.base_env.utils import (
    combine_evaluation,
    combine_rewards,
    combine_terminations,
)
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.context_paths import FieldPath, NestedField, resolve_path
from protomotions.envs.mdp_component import MdpComponent, is_mdp_component


class _LeafView:
    value: torch.Tensor = FieldPath()
    other: torch.Tensor = FieldPath()


class _NestedView:
    leaf: _LeafView = NestedField(_LeafView)
    scale: torch.Tensor = FieldPath()


class _TestContext:
    current: _LeafView = NestedField(_LeafView)
    nested: _NestedView = NestedField(_NestedView)


def _make_context(
    value: torch.Tensor | None = None,
    other: torch.Tensor | None = None,
) -> _TestContext:
    current = _LeafView()
    current.value = (
        torch.tensor([1.0, 2.0, 3.0]) if value is None else value
    )
    current.other = (
        torch.tensor([4.0, 5.0, 6.0]) if other is None else other
    )

    leaf = _LeafView()
    leaf.value = torch.tensor([7.0, 8.0, 9.0])
    leaf.other = torch.tensor([10.0, 11.0, 12.0])

    nested = _NestedView()
    nested.leaf = leaf
    nested.scale = torch.tensor([0.5, 1.0, 1.5])

    ctx = _TestContext()
    ctx.current = current
    ctx.nested = nested
    return ctx


def _add_value(value: torch.Tensor, increment: float = 0.0) -> torch.Tensor:
    return value + increment


def _affine_value(
    value: torch.Tensor,
    bias: torch.Tensor,
    gain: float = 1.0,
) -> torch.Tensor:
    return value * gain + bias


def test_field_paths_resolve_class_paths_and_instance_values():
    same_path = FieldPath(parent_path="current")
    same_path.name = "value"
    nested = NestedField(_LeafView, parent_path="ctx")
    nested.name = "leaf"

    assert _TestContext.current.value.path == "current.value"
    assert _TestContext.nested.leaf.other.path == "nested.leaf.other"
    assert str(_TestContext.current.value) == "current.value"
    assert repr(_TestContext.current.value) == "FieldPath('current.value')"
    assert _TestContext.current.value == "current.value"
    assert _TestContext.current.value == same_path
    assert _TestContext.current.value != object()
    assert len({_TestContext.current.value, same_path}) == 1
    assert str(nested) == "ctx.leaf"
    assert repr(nested) == "NestedField('ctx.leaf')"

    ctx = _make_context()
    assert torch.equal(resolve_path(ctx, "current.other"), ctx.current.other)
    assert torch.equal(
        resolve_path(ctx, "nested.leaf.value"),
        torch.tensor([7.0, 8.0, 9.0]),
    )

    with pytest.raises(AttributeError):
        resolve_path(ctx, "current.missing")


def test_mdp_component_resolves_dynamic_paths_and_filters_metadata():
    component = MdpComponent(
        compute_func=_affine_value,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={
            "bias": torch.tensor([0.5, 1.0, 1.5]),
            "gain": 2.0,
            "weight": 3.0,
            "multiplicative": True,
            "zero_during_grace_period": True,
            "min_value": -1.0,
            "max_value": 10.0,
            "threshold": 4.0,
            "fail_above": False,
        },
    )
    ctx = _make_context()

    resolved, func_params = component.resolve_args(ctx)
    result = component.compute(ctx)

    assert is_mdp_component(component)
    assert not is_mdp_component({"weight": 1.0})
    assert component.get_bindings_dict() == {"value": "current.value"}
    assert component.get_compute_func() is _affine_value
    assert component.get_params()["weight"] == 3.0
    assert resolved == {"value": ctx.current.value}
    assert set(func_params) == {"bias", "gain"}
    assert component._device_ready is True
    assert torch.allclose(result, torch.tensor([2.5, 5.0, 7.5]))

    serialized = component.to_dict()
    assert serialized["compute_func"] == "_affine_value"
    assert serialized["dynamic_vars"] == {"value": "current.value"}
    assert serialized["bias"] == [0.5, 1.0, 1.5]
    assert serialized["weight"] == 3.0


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


def test_combine_rewards_applies_grace_mask_weighting_and_clamps():
    grace_mask = torch.tensor([False, True, False])
    raw_rewards = {
        "tracking": torch.tensor([1.0, 2.0, 3.0]),
        "energy": torch.tensor([-2.0, 0.5, 4.0]),
    }
    configs = {
        "tracking": MdpComponent(
            compute_func=_add_value,
            dynamic_vars={},
            static_params={
                "weight": 2.0,
                "zero_during_grace_period": True,
            },
        ),
        "energy": {"weight": -1.0, "min_value": -2.0, "max_value": 1.0},
    }

    reward, logs = combine_rewards(
        raw_rewards,
        configs,
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
        configs={
            "alive": {"multiplicative": True},
            "progress": {"weight": 0.25},
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
            configs={"bad": {"weight": 1.0}},
            num_envs=2,
            device=torch.device("cpu"),
        )


def test_combine_terminations_or_reduces_and_inverts_false_conditions():
    reset_buf, terminate_buf, logs = combine_terminations(
        raw_terms={
            "fell": torch.tensor([True, False, False]),
            "healthy": torch.tensor([True, False, True]),
        },
        configs={
            "fell": MdpComponent(
                compute_func=_add_value,
                dynamic_vars={},
                static_params={"terminate_on_true": True},
            ),
            "healthy": {"terminate_on_true": False},
        },
        num_envs=3,
        device=torch.device("cpu"),
    )

    expected = torch.tensor([True, True, False])
    assert torch.equal(reset_buf, expected)
    assert torch.equal(terminate_buf, expected)
    assert torch.equal(logs["termination/fell"], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(logs["termination/healthy"], torch.tensor([0.0, 1.0, 0.0]))


def test_combine_terminations_respects_per_component_settle_steps():
    reset_buf, terminate_buf, logs = combine_terminations(
        raw_terms={
            "tracking_error": torch.tensor([True, True, True, False]),
            "motion_end": torch.tensor([False, True, False, False]),
        },
        configs={
            "tracking_error": MdpComponent(
                compute_func=_add_value,
                dynamic_vars={},
                static_params={"settle_steps": 3},
            ),
            "motion_end": {"terminate_on_true": True},
        },
        num_envs=4,
        device=torch.device("cpu"),
        progress_buf=torch.tensor([1, 3, 4, 4]),
    )

    assert torch.equal(reset_buf, torch.tensor([False, True, True, False]))
    assert torch.equal(terminate_buf, torch.tensor([False, True, True, False]))
    assert torch.equal(
        logs["termination/tracking_error"], torch.tensor([0.0, 0.0, 1.0, 0.0])
    )
    assert torch.equal(
        logs["termination/motion_end"], torch.tensor([0.0, 1.0, 0.0, 0.0])
    )


def test_combine_evaluation_tracks_values_and_threshold_failures():
    failed_buf, values, failures = combine_evaluation(
        raw_values={
            "height": torch.tensor([0.2, 0.8, 1.0]),
            "clearance": torch.tensor([0.1, 0.5, 0.9]),
            "score": torch.tensor([1.0, 2.0, 3.0]),
        },
        configs={
            "height": {"threshold": 0.7},
            "clearance": MdpComponent(
                compute_func=_add_value,
                dynamic_vars={},
                static_params={"threshold": 0.2, "fail_above": False},
            ),
            "score": {"weight": 1.0},
        },
        num_envs=3,
        device=torch.device("cpu"),
    )

    assert torch.equal(failed_buf, torch.tensor([True, True, True]))
    assert torch.equal(values["score"], torch.tensor([1.0, 2.0, 3.0]))
    assert set(failures) == {"height", "clearance"}
    assert torch.equal(failures["height"], torch.tensor([False, True, True]))
    assert torch.equal(failures["clearance"], torch.tensor([True, False, False]))


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
    assert set(manager._compiled) == {"value_func", "other_func"}


def test_component_manager_promotes_successful_compiled_function(monkeypatch):
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

    first = manager.execute_single("value", component, ctx, compile=True)
    second = manager.execute_single("value", component, ctx, compile=True)

    assert torch.equal(first, torch.tensor([13.0, 14.0]))
    assert torch.equal(second, torch.tensor([13.0, 14.0]))
    assert compile_calls == [(_add_value, "default")]
    assert len(compiled_calls) == 2
    assert manager._compiled["value_func"] is not component.compute_func


def test_component_manager_falls_back_when_compiled_wrapper_fails(monkeypatch):
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", True)

    def fake_compile(fn, mode):
        def compiled_fn(**kwargs):
            raise RuntimeError("compile failed at first call")

        return compiled_fn

    monkeypatch.setattr(component_manager.torch, "compile", fake_compile)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context(value=torch.tensor([1.0, 2.0]))
    component = MdpComponent(
        compute_func=_add_value,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={"increment": 2.0},
    )

    result = manager.execute_single("value", component, ctx, compile=True)

    assert torch.equal(result, torch.tensor([3.0, 4.0]))
    assert manager._compiled["value_func"] is component.compute_func


def test_component_manager_execute_single_handles_cached_and_compile_time_failures(monkeypatch):
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", True)
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context(value=torch.tensor([1.0, 2.0]))
    component = MdpComponent(
        compute_func=_add_value,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={"increment": 2.0},
    )

    def failing_cached_fn(**kwargs):
        raise RuntimeError("cached function failed")

    manager._compiled["value_func"] = failing_cached_fn
    result = manager.execute_single("value", component, ctx, compile=True)
    assert torch.equal(result, torch.tensor([3.0, 4.0]))
    assert manager._compiled["value_func"] is component.compute_func

    def failing_compile(fn, mode):
        raise RuntimeError("compile unavailable")

    manager.clear_cache()
    monkeypatch.setattr(component_manager.torch, "compile", failing_compile)
    result = manager.execute_single("value", component, ctx, compile=True)

    assert torch.equal(result, torch.tensor([3.0, 4.0]))
    assert manager._compiled["value_func"] is component.compute_func


def test_component_manager_execute_all_falls_back_from_cached_failure(caplog):
    manager = ComponentManager(torch.device("cpu"))
    ctx = _make_context(value=torch.tensor([1.0, 2.0]))
    component = MdpComponent(
        compute_func=_add_value,
        dynamic_vars={"value": _TestContext.current.value},
        static_params={"increment": 2.0},
    )

    def failing_cached_fn(**kwargs):
        raise RuntimeError("cached compiled function failed")

    manager._compiled["value_func"] = failing_cached_fn

    with caplog.at_level(logging.WARNING, logger=component_manager.__name__):
        results = manager.execute_all({"value": component}, ctx)

    assert torch.equal(results["value"], torch.tensor([3.0, 4.0]))
    assert "torch.compile failed for 'value'" in caplog.text
    assert manager._compiled["value_func"] is component.compute_func
