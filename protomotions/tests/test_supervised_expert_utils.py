# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for supervised expert observation config helpers."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.supervised.expert_utils import (
    get_expert_action_config,
    get_expert_actor_in_keys,
    get_expert_observation_components,
    get_expert_observation_keys,
    validate_expert_action_config,
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


def test_get_expert_action_config_deepcopies_valid_action_config():
    def action_fn():
        pass

    action_config = {
        "fn": action_fn,
        "pd_action_offset": torch.tensor([1.0, 2.0]),
        "action_scale": torch.tensor([0.1, 0.2]),
        "stiffness": torch.tensor([10.0, 20.0]),
        "damping": torch.tensor([1.0, 2.0]),
    }
    env_config = SimpleNamespace(action_config=action_config)
    robot_config = SimpleNamespace(number_of_actions=2)

    copied = get_expert_action_config(env_config, robot_config)
    copied["action_scale"][0] = 9.0

    assert copied["fn"] is action_config["fn"]
    assert action_config["action_scale"][0] == pytest.approx(0.1)


def test_validate_expert_action_config_rejects_vector_length_mismatch():
    robot_config = SimpleNamespace(number_of_actions=2)
    action_config = {
        "fn": object(),
        "action_scale": torch.tensor([0.1, 0.2, 0.3]),
    }

    with pytest.raises(ValueError, match="action_config.action_scale has length 3"):
        validate_expert_action_config(action_config, robot_config)


def test_validate_expert_action_config_rejects_missing_required_field():
    robot_config = SimpleNamespace(number_of_actions=2)
    action_config = {
        "fn": SimpleNamespace(__name__="bm_pd_action"),
        "pd_action_offset": torch.tensor([0.0, 0.0]),
        "stiffness": torch.tensor([10.0, 20.0]),
        "damping": torch.tensor([1.0, 2.0]),
    }

    with pytest.raises(ValueError, match="missing required field 'action_scale'"):
        validate_expert_action_config(action_config, robot_config)


def test_validate_expert_action_config_rejects_mismatched_dof_order():
    action_config = {
        "fn": object(),
        "action_scale": torch.tensor([0.1, 0.2]),
    }
    student_robot = SimpleNamespace(
        number_of_actions=2,
        kinematic_info=SimpleNamespace(dof_names=["hip", "knee"]),
    )
    expert_robot = SimpleNamespace(
        kinematic_info=SimpleNamespace(dof_names=["knee", "hip"]),
    )

    with pytest.raises(ValueError, match="DOF order"):
        validate_expert_action_config(action_config, student_robot, expert_robot)


def test_get_expert_action_config_requires_explicit_action_interface():
    with pytest.raises(ValueError, match="must define action_config"):
        get_expert_action_config(SimpleNamespace(), SimpleNamespace(number_of_actions=1))
