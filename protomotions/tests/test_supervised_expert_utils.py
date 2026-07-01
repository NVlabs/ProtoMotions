# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for supervised expert observation config helpers."""

from types import SimpleNamespace

import pytest

from protomotions.agents.supervised.expert_utils import (
    get_expert_actor_in_keys,
    get_expert_observation_components,
    get_expert_observation_keys,
)


def test_get_expert_actor_in_keys_prefers_actor_over_model_keys():
    config = SimpleNamespace(
        model=SimpleNamespace(
            actor=SimpleNamespace(in_keys=["actor_obs", "shared_obs"]),
            in_keys=["model_obs"],
        )
    )

    assert get_expert_actor_in_keys(config) == ["actor_obs", "shared_obs"]


def test_get_expert_actor_in_keys_falls_back_to_model_keys():
    config = SimpleNamespace(model=SimpleNamespace(in_keys=("model_obs", "other_obs")))

    assert get_expert_actor_in_keys(config) == ["model_obs", "other_obs"]


def test_get_expert_actor_in_keys_warns_when_unavailable(caplog):
    assert get_expert_actor_in_keys(SimpleNamespace()) == []
    assert "Could not determine expert actor in_keys" in caplog.text


def test_get_expert_observation_keys_filters_to_actor_inputs_with_prefix():
    env_config = SimpleNamespace(
        observation_components={
            "actor_obs": object(),
            "unused_obs": object(),
            "shared_obs": object(),
        }
    )
    agent_config = SimpleNamespace(
        model=SimpleNamespace(actor=SimpleNamespace(in_keys=["actor_obs", "shared_obs"]))
    )

    assert get_expert_observation_keys(env_config, agent_config) == [
        "expert_actor_obs",
        "expert_shared_obs",
    ]


def test_get_expert_observation_keys_returns_empty_without_observations():
    agent_config = SimpleNamespace(model=SimpleNamespace(in_keys=["obs"]))

    assert get_expert_observation_keys(SimpleNamespace(), agent_config) == []
    assert (
        get_expert_observation_keys(
            SimpleNamespace(observation_components=None),
            agent_config,
        )
        == []
    )


def test_get_expert_observation_keys_requires_actor_inputs():
    env_config = SimpleNamespace(
        observation_components={
            "obs_a": object(),
            "obs_b": object(),
        }
    )

    with pytest.raises(ValueError, match="model.actor.in_keys"):
        get_expert_observation_keys(env_config, SimpleNamespace(), prefix="teacher_")


def test_get_expert_observation_components_deepcopies_filtered_components():
    actor_component = {"history": []}
    unused_component = {"unused": []}
    env_config = SimpleNamespace(
        observation_components={
            "actor_obs": actor_component,
            "unused_obs": unused_component,
        }
    )
    agent_config = SimpleNamespace(
        model=SimpleNamespace(actor=SimpleNamespace(in_keys=["actor_obs"]))
    )

    components = get_expert_observation_components(env_config, agent_config)
    components["expert_actor_obs"]["history"].append("mutated")

    assert list(components) == ["expert_actor_obs"]
    assert actor_component == {"history": []}
    assert unused_component == {"unused": []}


def test_get_expert_observation_components_requires_actor_inputs():
    env_config = SimpleNamespace(
        observation_components={
            "obs_a": {"id": "a"},
            "obs_b": {"id": "b"},
        }
    )

    with pytest.raises(ValueError, match="model.actor.in_keys"):
        get_expert_observation_components(
            env_config,
            SimpleNamespace(),
            prefix="teacher_",
        )


def test_get_expert_observation_components_rejects_prefixed_key_conflicts():
    env_config = SimpleNamespace(observation_components={"obs": {"id": "obs"}})
    agent_config = SimpleNamespace(model=SimpleNamespace(in_keys=["obs"]))

    with pytest.raises(ValueError, match="conflicts"):
        get_expert_observation_components(
            env_config,
            agent_config,
            existing_obs_keys=["expert_obs"],
        )


def test_get_expert_observation_components_returns_empty_without_observations():
    agent_config = SimpleNamespace(model=SimpleNamespace(in_keys=["obs"]))

    assert get_expert_observation_components(SimpleNamespace(), agent_config) == {}
    assert (
        get_expert_observation_components(
            SimpleNamespace(observation_components=None),
            agent_config,
        )
        == {}
    )
