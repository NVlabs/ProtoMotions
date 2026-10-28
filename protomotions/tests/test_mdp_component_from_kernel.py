# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The binding contract: a kernel's signature is how it is bound.

``MdpComponent.from_kernel`` reads the signature: a ``FieldPath`` default is a
context input, a plain default a constant, a missing default must be overridden.
An override that is a ``FieldPath`` rebinds; any other value (``None`` included)
sets a constant and unbinds a ``FieldPath`` default. ``static_params`` reach the
kernel verbatim (no reserved names); reward metadata is a typed ``RewardMeta``.
Eagerness is a marker on the kernel.
"""

import functools
import pickle

import pytest
import torch

import protomotions.envs.component_manager as component_manager
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import (
    MdpComponent,
    eager_kernel,
    static_component_errors,
)
from protomotions.envs.rewards.meta import RewardMeta


def _kernel(
    body_pos=EnvContext.current.rigid_body_pos,
    coefficient=-2.0,
    threshold=0.5,
):
    return body_pos * coefficient


# ---------- from_kernel: the signature is the binding ------------------------


def test_from_kernel_splits_fieldpath_defaults_and_constants():
    """A FieldPath default is a dynamic binding; a plain default is a constant."""
    component = MdpComponent.from_kernel(_kernel)
    assert component.get_bindings_dict() == {"body_pos": "current.rigid_body_pos"}
    assert component.get_params() == {"coefficient": -2.0, "threshold": 0.5}


def test_from_kernel_override_with_fieldpath_rebinds():
    """A FieldPath override repoints a dynamic binding."""
    component = MdpComponent.from_kernel(
        _kernel, body_pos=EnvContext.noisy.rigid_body_pos
    )
    assert component.get_bindings_dict() == {"body_pos": "noisy.rigid_body_pos"}


def test_from_kernel_override_with_value_unbinds_a_fieldpath_default():
    """Overriding a FieldPath default with any value (None included) makes it a constant."""
    component = MdpComponent.from_kernel(_kernel, body_pos=None)
    assert component.get_bindings_dict() == {}
    assert component.get_params()["body_pos"] is None


def test_from_kernel_override_sets_a_constant():
    """A non-FieldPath override replaces a constant default."""
    component = MdpComponent.from_kernel(_kernel, coefficient=5.0)
    assert component.get_params()["coefficient"] == 5.0


def test_from_kernel_missing_default_without_override_fails():
    """A parameter with neither default nor override names the kernel and parameter."""

    def kernel(required_value):
        return required_value

    with pytest.raises(TypeError, match="parameter required_value has no default"):
        MdpComponent.from_kernel(kernel)


def test_from_kernel_required_parameter_supplied_by_override():
    """A no-default parameter supplied by override binds without error."""

    def kernel(required_value):
        return required_value

    component = MdpComponent.from_kernel(required_value=EnvContext.dt, kernel=kernel)
    assert component.get_bindings_dict() == {"required_value": "dt"}


def test_from_kernel_unknown_override_lists_the_parameters():
    """A misspelled override names the kernel and its parameters."""
    with pytest.raises(TypeError, match=r"has no parameter zzz \(parameters: "):
        MdpComponent.from_kernel(_kernel, zzz=1.0)


def test_from_kernel_refuses_partial():
    """A functools.partial kernel is refused."""
    with pytest.raises(TypeError, match="refuses functools.partial"):
        MdpComponent.from_kernel(functools.partial(_kernel, coefficient=1.0))


def test_from_kernel_refuses_var_keyword_kernel():
    """A **kwargs kernel is refused; a kernel names every input."""

    def kernel(body_pos=EnvContext.current.rigid_body_pos, **kwargs):
        return body_pos

    with pytest.raises(TypeError, match="declares [*]args/[*][*]kwargs"):
        MdpComponent.from_kernel(kernel)


# ---------- constructor refusals ---------------------------------------------


def test_constructor_refuses_partial():
    """The constructor refuses a functools.partial compute_func."""
    with pytest.raises(TypeError, match="refuses functools.partial"):
        MdpComponent(functools.partial(_kernel), {})


# ---------- verbatim static_params, no reserved names ------------------------


def test_static_params_reach_the_kernel_verbatim():
    """Every static_param reaches the kernel, including names like weight/threshold."""

    def kernel(x, weight, threshold):
        return x * weight + threshold

    component = MdpComponent(
        compute_func=kernel,
        dynamic_vars={"x": EnvContext.dt},
        static_params={"weight": 3.0, "threshold": 0.7},
    )
    result = component.compute({"dt": torch.tensor([2.0])})
    assert torch.equal(result, torch.tensor([6.7]))


# ---------- RewardMeta rules --------------------------------------------------


def test_reward_meta_refuses_zero_weight():
    """An additive RewardMeta weight must be finite and nonzero."""
    with pytest.raises(ValueError, match="finite nonzero"):
        RewardMeta(weight=0.0)


def test_reward_meta_multiplicative_takes_no_weight_or_clamp():
    """A multiplicative RewardMeta carries no weight or clamp."""
    with pytest.raises(ValueError, match="multiplicative"):
        RewardMeta(multiplicative=True, weight=1.0)
    assert RewardMeta(multiplicative=True).multiplicative is True


def test_reward_meta_clamp_order():
    """When both clamps are given, min must be below max."""
    with pytest.raises(ValueError, match="min_value must be less"):
        RewardMeta(weight=1.0, min_value=1.0, max_value=0.0)


# ---------- static_component_errors ------------------------------------------


def test_static_component_errors_reward_requires_meta():
    """A reward component with no RewardMeta is an error."""
    component = MdpComponent.from_kernel(_kernel)
    errors = static_component_errors({"r": component}, group="reward")
    assert errors and "has no RewardMeta" in errors[0]


def test_static_component_errors_nonreward_forbids_meta():
    """A non-reward component that carries a RewardMeta is an error."""
    component = MdpComponent.from_kernel(_kernel, meta=RewardMeta(weight=1.0))
    errors = static_component_errors({"o": component}, group="observation")
    assert errors and "only reward components take meta" in errors[0]


def test_static_component_errors_clean_when_placed_right():
    """A reward with meta and an observation without meta both pass."""
    reward = MdpComponent.from_kernel(_kernel, meta=RewardMeta(weight=1.0))
    obs = MdpComponent.from_kernel(_kernel)
    assert static_component_errors({"r": reward}, group="reward") == []
    assert static_component_errors({"o": obs}, group="observation") == []


def _scaled(x=EnvContext.current.dof_pos, scale=1.0):
    return x * scale


def test_static_component_errors_flags_a_stale_reward_metadata_key():
    """A reward-metadata key left in static_params is refused and pointed at RewardMeta."""
    component = MdpComponent(
        _scaled,
        {"x": EnvContext.current.dof_pos},
        {"scale": 1.0, "weight": -0.5},
        meta=RewardMeta(weight=1.0),
    )
    errors = static_component_errors({"r": component}, group="reward")
    assert any("'weight' is not a parameter" in e and "RewardMeta" in e for e in errors)


def test_static_component_errors_flags_a_stale_rule_setting_key():
    """A stale threshold/fail_above key is refused and named as a rule setting."""
    component = MdpComponent(
        _scaled,
        {"x": EnvContext.current.dof_pos},
        {"scale": 1.0, "threshold": 0.5},
    )
    errors = static_component_errors({"o": component}, group="observation")
    assert any(
        "'threshold' is not a parameter" in e and "rule setting" in e for e in errors
    )


def test_static_component_errors_flags_an_unbound_required_parameter():
    """A kernel parameter with no default that nothing binds is refused at build."""

    def needs_two(a, b):
        return a + b

    component = MdpComponent(needs_two, {"a": EnvContext.dt})  # b unbound
    errors = static_component_errors({"o": component}, group="observation")
    assert any("'b' is required but not bound" in e for e in errors)


def test_static_component_errors_clean_for_a_fully_bound_kernel():
    """A component whose bindings exactly match the kernel signature has no errors."""
    assert (
        static_component_errors(
            {"o": MdpComponent.from_kernel(_scaled)}, group="observation"
        )
        == []
    )


# ---------- eager marker ------------------------------------------------------


def test_eager_kernel_is_never_compiled(monkeypatch):
    """A kernel marked @eager_kernel is returned uncompiled by the manager."""

    @eager_kernel
    def cast(x):
        return x.float()

    calls = []
    monkeypatch.setattr(
        component_manager.torch, "compile", lambda fn, **k: calls.append(fn) or fn
    )
    manager = ComponentManager(torch.device("cpu"))
    component = MdpComponent(compute_func=cast, dynamic_vars={"x": EnvContext.dt})
    assert manager._get_compiled_func(component) is cast
    assert calls == []


def test_non_eager_kernel_is_offered_to_compile(monkeypatch):
    """A kernel without the marker is handed to torch.compile."""

    def plain(x):
        return x.float()

    calls = []
    monkeypatch.setattr(
        component_manager.torch, "compile", lambda fn, **k: calls.append(fn) or fn
    )
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", True)
    manager = ComponentManager(torch.device("cpu"))
    component = MdpComponent(compute_func=plain, dynamic_vars={"x": EnvContext.dt})
    manager._get_compiled_func(component)
    assert calls == [plain]


# ---------- rebind ------------------------------------------------------------


def test_rebind_replaces_named_bindings_and_shares_the_rest():
    """rebind swaps a dynamic path and keeps the kernel, constants and meta."""
    component = MdpComponent.from_kernel(_kernel, meta=RewardMeta(weight=2.0))
    rebound = component.rebind(body_pos=EnvContext.noisy.rigid_body_pos)
    assert rebound.get_bindings_dict() == {"body_pos": "noisy.rigid_body_pos"}
    assert rebound.get_params() == component.get_params()
    assert rebound.meta == component.meta


def test_rebind_rejects_an_unknown_binding():
    """rebind refuses a name that is not a dynamic binding."""
    component = MdpComponent.from_kernel(_kernel)
    with pytest.raises(ValueError, match="not dynamic bindings"):
        component.rebind(coefficient=EnvContext.dt)


def test_rebind_copies_static_params_so_the_device_move_does_not_leak():
    """rebind's copy owns its static_params: a mutation on one does not reach the other."""
    component = MdpComponent(
        _scaled,
        {"x": EnvContext.current.rigid_body_pos},
        {"scale": torch.tensor([1.0])},
    )
    rebound = component.rebind(x=EnvContext.noisy.rigid_body_pos)
    assert rebound.static_params is not component.static_params
    rebound.static_params["scale"] = torch.tensor([2.0])
    assert torch.equal(component.static_params["scale"], torch.tensor([1.0]))


# ---------- equality and pickle ----------------------------------------------


def test_from_kernel_reconstructs_an_equal_component():
    """from_kernel(kernel, meta, **bindings, **params) round-trips by value."""
    component = MdpComponent.from_kernel(_kernel, meta=RewardMeta(weight=2.0))
    rebuilt = MdpComponent.from_kernel(
        _kernel,
        meta=RewardMeta(weight=2.0),
        body_pos=EnvContext.current.rigid_body_pos,
        coefficient=-2.0,
        threshold=0.5,
    )
    assert component == rebuilt


def test_pickle_round_trip_preserves_bindings_statics_and_meta():
    """A pickled component restores equal, and recomputes its device state."""
    component = MdpComponent.from_kernel(_kernel, meta=RewardMeta(weight=2.0))
    restored = pickle.loads(pickle.dumps(component))
    assert restored == component
    assert restored.get_bindings_dict() == component.get_bindings_dict()
