# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for discrete-prior PEFT model state loading helpers."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from protomotions.agents.peft.utils import model_state as model_state_module
from protomotions.agents.peft.prior_agent import DiscretePriorPEFTRLFTAgent
from protomotions.agents.peft.prior_setup import DiscretePriorPEFTSetupMixin
from protomotions.agents.peft.sft_agent import DiscretePriorPEFTSFTAgent
from protomotions.agents.peft.utils.model_state import (
    load_compatible_peft_model_state,
)


class _AdapterActor:
    def __init__(self):
        self.loaded = None

    def load_adapter_state_dict(self, state_dict, strict=True):
        self.loaded = (dict(state_dict), strict)
        return {"missing_keys": [], "unexpected_keys": []}


class _OptimizerRecorder:
    def __init__(self):
        self.loaded_state = None

    def load_state_dict(self, state):
        self.loaded_state = state


def test_optional_full_checkpoint_state_prefixes_are_module_owned_and_qualified():
    class _PriorWithPEFT(nn.Module):
        def optional_full_checkpoint_state_prefixes(self):
            return (
                "_anchor_",
                "film_input_norm.",
                "reference_prior.",
                "reference_film_input_norm.",
            )

    class _Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.prior_with_peft = _PriorWithPEFT()

        def optional_full_checkpoint_state_prefixes(self):
            return ("target_latent_encoder.",)

    class _PEFTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = _Actor()

        def optional_full_checkpoint_state_prefixes(self):
            return ("_critic.",)

    class _PEFTAMPModel(_PEFTModel):
        def __init__(self):
            super().__init__()
            self._discriminator = nn.Module()
            self._disc_critic = nn.Module()

        def optional_full_checkpoint_state_prefixes(self):
            return super().optional_full_checkpoint_state_prefixes() + (
                "_discriminator.",
                "_disc_critic.",
            )

    non_amp_prefixes = model_state_module.optional_full_checkpoint_state_prefixes(
        _PEFTModel()
    )
    assert non_amp_prefixes == (
        "_critic.",
        "_actor.target_latent_encoder.",
        "_actor.prior_with_peft._anchor_",
        "_actor.prior_with_peft.film_input_norm.",
        "_actor.prior_with_peft.reference_prior.",
        "_actor.prior_with_peft.reference_film_input_norm.",
    )

    assert set(
        model_state_module.optional_full_checkpoint_state_prefixes(_PEFTAMPModel())
    ) == {
        *non_amp_prefixes,
        "_discriminator.",
        "_disc_critic.",
    }


def test_load_compatible_peft_model_state_uses_strict_adapter_loader_for_adapter_only_state():
    actor = _AdapterActor()

    class _Model:
        _actor = actor

        def load_state_dict(self, state_dict, strict=True):
            raise AssertionError("adapter-only state should not load the full model")

    state = {
        "actor_peft_model.0.weight": torch.tensor([1.0]),
        "prior_with_peft.base_prior._transformer.layers.0.lora.A": torch.tensor([2.0]),
        "prior_with_peft.film_input_norm.running_obs_norm.mean": torch.tensor([3.0]),
    }

    load_compatible_peft_model_state(_Model(), state)

    assert actor.loaded == (state, True)


def test_load_compatible_peft_model_state_allows_peft_warm_start_differences(monkeypatch):
    calls = []

    class _PriorWithPEFT(nn.Module):
        def optional_full_checkpoint_state_prefixes(self):
            return (
                "_anchor_",
                "film_input_norm.",
                "reference_prior.",
                "reference_film_input_norm.",
            )

    class _Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.prior_with_peft = _PriorWithPEFT()

        def optional_full_checkpoint_state_prefixes(self):
            return ("target_latent_encoder.",)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = _Actor()

        def optional_full_checkpoint_state_prefixes(self):
            return ("_critic.",)

        def load_state_dict(self, state_dict, strict=True):
            calls.append(("load", strict, dict(state_dict)))
            return (
                [
                    "_critic.value.weight",
                    "_actor.target_latent_encoder.weight",
                    "_actor.prior_with_peft.reference_prior._pos_emb",
                    "_actor.prior_with_peft.film_input_norm.running_obs_norm.mean",
                ],
                [
                    "_critic.old_value.weight",
                    "_actor.target_latent_encoder.old_weight",
                    "_actor.prior_with_peft.reference_film_input_norm.running_obs_norm.var",
                    "_actor.prior_with_peft.film_input_norm.running_obs_norm.var",
                ],
            )

    model = _Model()
    state = {"_actor.weight": torch.tensor([1.0])}
    monkeypatch.setattr(
        model_state_module,
        "materialize_lazy_running_stats_from_state_dict",
        lambda module, model_state: calls.append(
            ("materialize", module, dict(model_state))
        ),
    )

    load_compatible_peft_model_state(model, state)

    assert calls == [
        ("materialize", model, state),
        ("load", False, state),
    ]


def test_load_compatible_peft_model_state_requires_complete_reference_when_present(
    monkeypatch,
):
    calls = []

    class _PEFT:
        def ensure_reference_modules(self):
            calls.append("ensure_reference")

        def mark_reference_loaded(self):
            calls.append("mark_reference")

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = SimpleNamespace(prior_with_peft=_PEFT())

        def load_state_dict(self, state_dict, strict=True):
            calls.append(("load", strict, dict(state_dict)))
            return (
                ["_actor.prior_with_peft.reference_prior.required.weight"],
                [],
            )

    model = _Model()
    state = {
        "_actor.prior_with_peft.reference_prior._pos_emb": torch.tensor([1.0]),
    }
    monkeypatch.setattr(
        model_state_module,
        "materialize_lazy_running_stats_from_state_dict",
        lambda module, model_state: calls.append(("materialize", module)),
    )

    with pytest.raises(RuntimeError, match="reference_prior.required.weight"):
        load_compatible_peft_model_state(model, state)

    assert calls == [
        "ensure_reference",
        ("materialize", model),
        ("load", False, state),
    ]


def test_load_compatible_peft_model_state_allows_amp_component_differences(monkeypatch):
    calls = []

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = nn.Module()
            self._discriminator = nn.Module()
            self._disc_critic = nn.Module()

        def optional_full_checkpoint_state_prefixes(self):
            return ("_discriminator.", "_disc_critic.")

        def load_state_dict(self, state_dict, strict=True):
            calls.append(("load", strict, dict(state_dict)))
            return (
                ["_discriminator.weight", "_disc_critic.value.weight"],
                ["_discriminator.old_weight", "_disc_critic.old_value.weight"],
            )

    model = _Model()
    state = {"_actor.weight": torch.tensor([1.0])}
    monkeypatch.setattr(
        model_state_module,
        "materialize_lazy_running_stats_from_state_dict",
        lambda module, model_state: calls.append(
            ("materialize", module, dict(model_state))
        ),
    )

    load_compatible_peft_model_state(model, state)

    assert calls == [
        ("materialize", model, state),
        ("load", False, state),
    ]


def test_load_compatible_peft_model_state_rejects_unexpected_full_state_keys(monkeypatch):
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self._actor = nn.Module()

        def load_state_dict(self, state_dict, strict=True):
            return (
                ["_actor.prior_with_peft.base_prior.required.weight"],
                ["_actor.prior_with_peft.base_prior.extra.weight"],
            )

    monkeypatch.setattr(
        model_state_module,
        "materialize_lazy_running_stats_from_state_dict",
        lambda module, model_state: None,
    )

    with pytest.raises(RuntimeError, match="Unexpected PEFT model state_dict mismatch"):
        load_compatible_peft_model_state(
            _Model(),
            {"_actor.weight": torch.tensor([1.0])},
        )


def test_peft_agents_use_training_state_hooks_not_public_load_overrides():
    assert "load_parameters" not in DiscretePriorPEFTSFTAgent.__dict__
    assert "load_parameters" not in DiscretePriorPEFTRLFTAgent.__dict__


def test_peft_after_model_load_restores_configured_frozen_prior():
    agent = object.__new__(DiscretePriorPEFTSFTAgent)
    calls = []
    agent._restore_configured_frozen_prior = lambda: calls.append("restore")

    DiscretePriorPEFTSetupMixin._after_load_model_state_dict(agent, {})

    assert calls == ["restore"]


def test_rlft_sft_warm_start_loads_actor_optimizer_and_resets_counters():
    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    agent.actor_optimizer = _OptimizerRecorder()
    agent.current_epoch = 7
    agent.step_count = 11
    agent.fit_start_time = 13.0
    agent.best_evaluated_score = 17.0
    state = {"actor_optimizer": {"actor": 1}}

    DiscretePriorPEFTRLFTAgent._load_training_state(agent, state)

    assert agent.actor_optimizer.loaded_state == {"actor": 1}
    assert agent.current_epoch == 0
    assert agent.step_count == 0
    assert agent.fit_start_time is None
    assert agent.best_evaluated_score is None


def test_rlft_full_training_state_delegates_to_ppo_state_load(monkeypatch):
    calls = []

    def fake_training_state(self, state_dict):
        calls.append(state_dict)

    from protomotions.agents.fine_tuning.agent import FineTuningAgent

    monkeypatch.setattr(FineTuningAgent, "_load_training_state", fake_training_state)

    agent = object.__new__(DiscretePriorPEFTRLFTAgent)
    state = {"critic_optimizer": {"critic": 1}}

    DiscretePriorPEFTRLFTAgent._load_training_state(agent, state)

    assert calls == [state]
