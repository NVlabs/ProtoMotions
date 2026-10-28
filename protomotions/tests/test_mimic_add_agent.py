# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the MimicADD observation additions."""

from types import SimpleNamespace

import torch
import pytest
from tensordict import TensorDict

from examples.experiments.add import mlp as add_experiment
from protomotions.agents.amp.model import AMPModel
from protomotions.agents.mimic import agent_add
from protomotions.agents.mimic.agent_add import MimicADD
from protomotions.agents.utils.data import ExperienceBuffer


def _state(body_pos):
    return SimpleNamespace(
        rigid_body_pos=body_pos,
        rigid_body_rot=torch.zeros(body_pos.shape[0], body_pos.shape[1], 4),
        rigid_body_vel=torch.zeros_like(body_pos),
        rigid_body_ang_vel=torch.zeros_like(body_pos),
    )


def test_mimic_add_adds_tracking_difference_observation(monkeypatch):
    num_envs = 2
    ref_pos = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ]
    )
    cur_pos = ref_pos - 0.5
    ground_queries = []

    def get_ground_heights(positions):
        ground_queries.append(positions.clone())
        return positions[:, 0]

    reference_queries = []

    def poses_at(env_ids, times):
        reference_queries.append((env_ids, times.clone()))
        return _state(ref_pos.clone())

    env = SimpleNamespace(
        # ADD reads the current reference pose, placed, through the query.
        reference=SimpleNamespace(times=torch.tensor([0.1, 0.2]), poses_at=poses_at),
        terrain=SimpleNamespace(get_ground_heights=get_ground_heights),
        simulator=SimpleNamespace(get_bodies_state=lambda: _state(cur_pos.clone())),
        robot_config=SimpleNamespace(anchor_body_index=1),
    )
    agent = MimicADD.__new__(MimicADD)
    agent.env = env
    agent.num_envs = num_envs

    monkeypatch.setattr(
        agent_add.AMP,
        "add_agent_info_to_obs",
        lambda self, obs: {**obs, "base_added": torch.ones(num_envs, 1)},
    )

    def fake_max_coords(body_pos, ground_height, **kwargs):
        assert kwargs["local_obs"] is False
        assert kwargs["root_height_obs"] is True
        assert kwargs["observe_contacts"] is False
        assert kwargs["body_contacts"].shape == (num_envs, 0)
        return body_pos.reshape(num_envs, -1)

    monkeypatch.setattr(
        agent_add,
        "compute_humanoid_max_coords_observations",
        fake_max_coords,
    )

    obs = agent.add_agent_info_to_obs({"obs": torch.zeros(num_envs, 1)})

    assert torch.equal(obs["base_added"], torch.ones(num_envs, 1))
    assert torch.equal(
        obs["mimic_target_poses_diff"],
        (ref_pos - cur_pos).reshape(num_envs, -1),
    )
    assert torch.equal(ground_queries[0], ref_pos[:, 1])
    assert torch.equal(ground_queries[1], cur_pos[:, 1])
    assert reference_queries[0][0] is None
    assert torch.equal(reference_queries[0][1], torch.tensor([0.1, 0.2]))


@pytest.mark.parametrize("width,num_samples", [(5, 3), (17, 9)])
def test_mimic_add_expert_obs_matches_rollout_without_reference_sampling(
    width, num_samples
):
    agent = MimicADD.__new__(MimicADD)
    agent.experience_buffer = ExperienceBuffer(4, 3, torch.device("cpu"))
    agent.experience_buffer.register_key(
        "mimic_target_poses_diff", shape=(width,), dtype=torch.float64
    )
    # No environment, reference sampler or observation manager is needed: an
    # expert's tracking difference is exactly zero, in the rollout feature space.
    obs = agent.get_expert_disc_obs(num_samples)
    assert set(obs) == {"mimic_target_poses_diff"}
    diff = obs["mimic_target_poses_diff"]
    assert diff.shape == (num_samples, width)
    assert diff.dtype == torch.float64
    assert diff.device == agent.experience_buffer.mimic_target_poses_diff.device
    assert torch.count_nonzero(diff) == 0


def test_add_example_materializes_both_critics_and_trains_discriminator():
    robot = SimpleNamespace(
        kinematic_info=SimpleNamespace(num_dofs=2), number_of_actions=2
    )
    config = add_experiment.agent_config(
        robot, None, SimpleNamespace(batch_size=4, training_max_steps=32)
    )
    assert config.amp_parameters.use_disc_critic
    assert config.reference_obs_components == {}
    assert config.model.critic is not config.model.disc_critic
    assert config.model.critic.out_keys == ["value"]
    assert config.model.disc_critic.out_keys == ["disc_value"]
    assert config.model.critic.layers[0] is not config.model.disc_critic.layers[0]
    disc_units = config.model.disc_critic.layers[0].units
    config.model.critic.layers[0].units += 1
    assert config.model.disc_critic.layers[0].units == disc_units
    # Exercise the recipe's actual routing with tiny networks for a CPU test.
    for module in (
        config.model.actor.mu_model,
        config.model.critic,
        config.model.disc_critic,
        *config.model.discriminator.models,
    ):
        module.layers = []
    model = AMPModel(config.model).eval()
    obs = TensorDict(
        {
            "max_coords_obs": torch.zeros(4, 7),
            "mimic_target_poses": torch.zeros(4, 9),
            "mimic_target_poses_diff": torch.zeros(4, 17),
        },
        batch_size=[4],
    )
    output = model(obs)
    for key in ("value", "disc_value", "disc_logits"):
        assert output[key].shape == (4, 1)
        assert torch.isfinite(output[key]).all()
    optimizer = torch.optim.Adam(model._discriminator.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model._discriminator.parameters()]
    loss = torch.nn.functional.softplus(-output["disc_logits"]).mean()
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model._discriminator.parameters())
    )
