# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for which metadata keys in static_params reach a compute function.

Metadata keys such as ``threshold`` and ``weight`` are read by the combining
logic in ``base_env/utils.py``. A metadata key is also passed to the compute
function when, and only when, that function declares a parameter of the same
name. Termination kernels declare ``threshold``, so a termination's configured
threshold must reach its kernel. Evaluation kernels do not, because for them
``threshold`` is the failure cut that ``combine_evaluation`` applies.
"""

import inspect
import pickle
from types import SimpleNamespace

import pytest
import torch

import protomotions.envs.component_manager as component_manager
from protomotions.envs import component_factories as factories
from protomotions.envs import obs as obs_kernels
from protomotions.envs import rewards as reward_kernels
from protomotions.envs import terminations as termination_kernels
from protomotions.envs.base_env.utils import (
    combine_evaluation,
    combine_rewards,
    combine_terminations,
)
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.mdp_component import _METADATA_KEYS, MdpComponent
from protomotions.envs.terminations.task import check_path_distance_term

TERMINATION_FACTORIES = [
    factories.tracking_error_term_factory,
    factories.anchor_pos_error_term_factory,
    factories.anchor_ori_error_term_factory,
    factories.relative_body_pos_error_term_factory,
    factories.anchor_height_error_term_factory,
]

EVALUATION_FACTORIES = [
    factories.gt_error_factory,
    factories.max_joint_error_factory,
    factories.gr_error_factory,
    factories.anchor_pos_metric_factory,
    factories.anchor_ori_metric_factory,
    factories.relative_body_pos_metric_factory,
    factories.anchor_height_error_metric_factory,
]


class _AnyContext:
    """Context stand-in where every attribute path resolves to this object.

    ``resolve_args`` only walks the binding paths and filters static_params, so
    this is enough to see which parameters a compute function would receive.
    """

    def __getattr__(self, name):
        return self


def _func_params(component: MdpComponent) -> dict:
    _, func_params = component.resolve_args(_AnyContext())
    return func_params


def _body_pos_context(max_body_errors):
    """Context with one body per env offset along x by the given distance."""
    errors = torch.tensor(max_body_errors)
    ref = torch.zeros(len(errors), 2, 3)
    current = ref.clone()
    current[:, 1, 0] = errors
    return SimpleNamespace(
        current=SimpleNamespace(rigid_body_pos=current),
        mimic=SimpleNamespace(ref_state=SimpleNamespace(rigid_body_pos=ref)),
    )


@pytest.fixture
def eager_manager(monkeypatch):
    monkeypatch.setattr(component_manager, "TORCH_COMPILE_AVAILABLE", False)
    return ComponentManager(torch.device("cpu"))


# ---------- terminations ------------------------------------------------------


@pytest.mark.parametrize("factory", TERMINATION_FACTORIES, ids=lambda f: f.__name__)
def test_termination_factories_pass_configured_threshold_to_kernel(factory):
    component = factory(threshold=0.123)

    assert _func_params(component) == {"threshold": 0.123}


@pytest.mark.parametrize("factory", TERMINATION_FACTORIES, ids=lambda f: f.__name__)
def test_termination_factory_default_matches_kernel_default(factory):
    # If these differed, every config relying on the factory default would
    # change behaviour now that the threshold reaches the kernel.
    component = factory()
    kernel_default = (
        inspect.signature(component.compute_func).parameters["threshold"].default
    )

    assert component.get_params()["threshold"] == kernel_default


def test_tracking_error_termination_uses_configured_threshold_in_step_path(
    eager_manager,
):
    # 0.55 m is above the kernel default of 0.5 m but below the configured 0.6 m.
    components = {"tracking_error": factories.tracking_error_term_factory(0.6)}
    ctx = _body_pos_context([0.55, 0.65])

    raw_terms = eager_manager.execute_all(components, ctx)
    reset_buf, terminate_buf, logs = combine_terminations(
        raw_terms, components, num_envs=2, device=torch.device("cpu")
    )

    assert torch.equal(reset_buf, torch.tensor([False, True]))
    assert torch.equal(terminate_buf, torch.tensor([False, True]))
    assert torch.equal(logs["termination/tracking_error"], torch.tensor([0.0, 1.0]))

    single = eager_manager.execute_single(
        "tracking_error", components["tracking_error"], ctx
    )
    assert torch.equal(single, torch.tensor([False, True]))


# ---------- evaluation --------------------------------------------------------


@pytest.mark.parametrize("factory", EVALUATION_FACTORIES, ids=lambda f: f.__name__)
def test_evaluation_factories_keep_threshold_out_of_kernel(factory):
    component = factory(threshold=0.123)

    assert "threshold" not in _func_params(component)
    assert component.get_params()["threshold"] == 0.123
    assert "threshold" not in inspect.signature(component.compute_func).parameters


def test_boolean_evaluation_wrappers_do_not_receive_threshold_through_kwargs():
    # These wrappers forward **kwargs into TorchScript kernels that would reject
    # an unexpected ``threshold`` argument.
    path_metric = factories.path_distance_error_factory(threshold=1.5, min_progress=3)
    steering_metric = factories.steering_velocity_error_factory(
        speed_tolerance=0.4, direction_tolerance=0.6
    )

    assert _func_params(path_metric) == {"fail_dist": 1.5, "min_progress": 3}
    assert _func_params(steering_metric) == {
        "speed_tolerance": 0.4,
        "direction_tolerance": 0.6,
    }


def test_evaluation_failure_cut_is_applied_to_kernel_output(eager_manager):
    components = {"gt_error": factories.gt_error_factory(threshold=0.3)}
    # Two bodies, one of them offset, so the mean error is half the offset.
    ctx = _body_pos_context([0.4, 0.8])

    raw_values = eager_manager.execute_all(components, ctx)
    failed_buf, values, failures = combine_evaluation(
        raw_values, components, num_envs=2, device=torch.device("cpu")
    )

    assert torch.allclose(values["gt_error"], torch.tensor([0.2, 0.4]))
    assert torch.equal(failures["gt_error"], torch.tensor([False, True]))
    assert torch.equal(failed_buf, torch.tensor([False, True]))


# ---------- rewards -----------------------------------------------------------


def test_reward_metadata_stays_out_of_reward_kernels():
    components = {
        "gt_rew": factories.gt_rew_factory(weight=0.5, coefficient=-100.0),
        "power": factories.pow_rew_factory(weight=-1e-5, min_value=-0.5),
        "contact_match": factories.contact_match_rew_factory(
            weight=-0.1, zero_during_grace_period=True
        ),
    }

    assert _func_params(components["gt_rew"]) == {"coefficient": -100.0}
    assert _func_params(components["power"]) == {"use_torque_squared": False}
    assert _func_params(components["contact_match"]) == {}

    reward, logs = combine_rewards(
        raw_rewards={
            "gt_rew": torch.tensor([1.0, 2.0]),
            "power": torch.tensor([1e5, 1e4]),
            "contact_match": torch.tensor([3.0, 4.0]),
        },
        configs=components,
        grace_mask=torch.tensor([True, False]),
        num_envs=2,
        device=torch.device("cpu"),
    )

    assert torch.allclose(logs["scaled_r/power"], torch.tensor([-0.5, -0.1]))
    assert torch.allclose(logs["scaled_r/contact_match"], torch.tensor([0.0, -0.4]))
    assert torch.allclose(reward, torch.tensor([0.0, 0.5]))


def test_contact_force_change_reward_receives_its_threshold():
    # The kernel declares ``threshold`` (the force change it ignores), so it
    # now receives the configured value. Its default matches the factory's.
    component = factories.contact_force_change_rew_factory(
        weight=-2e-4, min_value=-0.5, threshold=45.0
    )
    kernel_default = (
        inspect.signature(component.compute_func).parameters["threshold"].default
    )

    assert _func_params(component) == {"threshold": 45.0}
    assert factories.contact_force_change_rew_factory().get_params()["threshold"] == (
        kernel_default
    )


def test_no_kernel_declares_a_reward_combining_metadata_key():
    # A kernel that declared ``weight`` or a clamp bound would receive the value
    # that combine_rewards also applies, so the reward would be scaled twice.
    combining_keys = _METADATA_KEYS - {"threshold"}
    offenders = []
    for module in (termination_kernels, reward_kernels, obs_kernels):
        for name in module.__all__:
            func = getattr(module, name)
            if not callable(func):
                continue
            try:
                parameters = inspect.signature(func).parameters
            except (TypeError, ValueError):
                continue
            offenders += [f"{name}({key})" for key in combining_keys & set(parameters)]

    assert offenders == []


# ---------- general forwarding rules ------------------------------------------


def _value_and_threshold(value, threshold=0.5):
    return value > threshold


def _value_and_kwargs(value, **kwargs):
    return value


class _UninspectableKernel:
    """Callable whose signature cannot be read, like a TorchScript function."""

    __signature__ = "not a signature"

    def __call__(self, value, threshold=0.5):
        return value > threshold


@pytest.mark.parametrize(
    ("compute_func", "expected"),
    [
        (_value_and_threshold, {"scale": 2.0, "threshold": 0.7}),
        (_value_and_kwargs, {"scale": 2.0}),
        (_UninspectableKernel(), {"scale": 2.0}),
        (check_path_distance_term, {"scale": 2.0}),
    ],
    ids=["declared", "kwargs-only", "uninspectable", "torchscript"],
)
def test_metadata_reaches_only_kernels_that_declare_it(compute_func, expected):
    component = MdpComponent(
        compute_func=compute_func,
        dynamic_vars={},
        static_params={"scale": 2.0, "threshold": 0.7, "weight": 3.0},
    )

    assert _func_params(component) == expected


def test_metadata_forwarding_survives_pickle_round_trip():
    component = factories.tracking_error_term_factory(threshold=0.6)
    assert _func_params(component) == {"threshold": 0.6}

    restored = pickle.loads(pickle.dumps(component))

    assert _func_params(restored) == {"threshold": 0.6}


def test_component_pickled_before_forwarding_existed_still_forwards():
    component = factories.tracking_error_term_factory(threshold=0.6)
    state = component.__getstate__()
    # Pickles written by earlier code carry only these attributes.
    state = {
        key: state[key]
        for key in ("compute_func", "dynamic_vars", "static_params", "compile")
    }
    restored = MdpComponent.__new__(MdpComponent)
    restored.__setstate__(state)

    assert _func_params(restored) == {"threshold": 0.6}
