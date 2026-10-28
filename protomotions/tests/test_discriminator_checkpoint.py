# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional discriminator weights must not weaken policy checkpoint checks."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.peft.utils.model_state import load_compatible_peft_model_state
from protomotions.agents.utils.checkpoint import (
    expected_checkpoint_state,
    validate_checkpoint_state,
)
from protomotions.agents.utils.normalization import RunningMeanStd


@pytest.mark.parametrize("bad_field", [None, "var", "count"])
def test_lazy_normalization_preflight_does_not_create_live_buffers(bad_field):
    model = nn.Module()
    model.norm = RunningMeanStd(fabric=None, device="cpu")
    state = {
        "norm.mean": torch.zeros(3),
        "norm.var": torch.ones(3),
        "norm.count": torch.ones((), dtype=torch.long),
    }
    if bad_field:
        state["norm." + bad_field] = torch.ones(2)
        with pytest.raises(RuntimeError, match="shape_mismatches"):
            validate_checkpoint_state(expected_checkpoint_state(model, state), state)
    else:
        validate_checkpoint_state(expected_checkpoint_state(model, state), state)
    assert not model.norm._initialized
    assert not model.state_dict()


class _Model(nn.Module):
    def __init__(self, discriminator, critic=True):
        super().__init__()
        self._actor = nn.Linear(2, 2)
        self._critic = nn.Linear(2, 1) if critic else None
        if discriminator:
            self._discriminator = nn.Linear(2, 1)
            self._discriminator.register_buffer("stats", torch.randn(2))
            self._disc_critic = nn.Linear(2, 1)

    def materialize_from_state_dict(self, state_dict):
        pass

    def optional_checkpoint_modules(self):
        return ("_discriminator", "_disc_critic")


@pytest.fixture(params=["base", "peft"])
def load(request):
    if request.param == "peft":
        return load_compatible_peft_model_state
    return lambda model, state: BaseAgent._load_model_state_dict(
        SimpleNamespace(model=model), state
    )


@pytest.mark.parametrize("source_amp", [False, True])
@pytest.mark.parametrize("target_amp", [False, True])
def test_loads_shared_networks_retains_new_and_ignores_unused(
    load, source_amp, target_amp
):
    source = _Model(source_amp)
    target = _Model(target_amp)
    before = deepcopy(target.state_dict())
    saved = deepcopy(source.state_dict())
    original = deepcopy(saved)

    load(target, saved)

    for key, value in target.state_dict().items():
        assert torch.equal(value, saved[key] if key in saved else before[key])
    assert saved.keys() == original.keys()
    for key in original:
        assert torch.equal(saved[key], original[key])


@pytest.mark.parametrize(
    "network", ["_actor", "_critic", "_discriminator", "_disc_critic"]
)
@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_present_networks_still_require_valid_weights(load, network, corruption):
    saved = _Model(True).state_dict()
    if corruption == "missing":
        del saved[network + ".weight"]
    elif corruption == "unexpected":
        saved[network + ".extra"] = torch.ones(1)
    else:
        saved[network + ".weight"] = torch.ones(8, 8)
    with pytest.raises(RuntimeError, match=network):
        load(_Model(True), saved)


@pytest.mark.parametrize("source_amp", [False, True])
def test_absent_policy_is_never_treated_as_optional(load, source_amp):
    saved = {
        key: value
        for key, value in _Model(source_amp).state_dict().items()
        if not key.startswith("_actor.")
    }
    with pytest.raises(RuntimeError, match="_actor"):
        load(_Model(True), saved)


@pytest.mark.parametrize("target_attribute", ["none", "absent"])
def test_actor_only_target_ignores_saved_critic(load, target_attribute):
    source = _Model(True)
    target = _Model(False, critic=False)
    if target_attribute == "absent":
        del target._critic
    load(target, source.state_dict())
    for key, value in target.state_dict().items():
        assert torch.equal(value, source.state_dict()[key])


def test_new_critic_requires_explicit_target_permission(load):
    source = _Model(False, critic=False)
    target = _Model(False)
    with pytest.raises(RuntimeError, match="_critic"):
        load(target, source.state_dict())
    target.optional_checkpoint_modules = lambda: ("_critic",)
    before = deepcopy(target._critic.state_dict())
    load(target, source.state_dict())
    for key, value in before.items():
        assert torch.equal(target._critic.state_dict()[key], value)


def test_actor_cannot_be_declared_optional(load):
    target = _Model(False)
    target.optional_checkpoint_modules = lambda: ("_actor",)
    with pytest.raises(ValueError, match="_actor.*cannot be optional"):
        load(target, target.state_dict())
