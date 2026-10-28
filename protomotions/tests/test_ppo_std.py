# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded global exploration and exact rollout-to-update Gaussian KL."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.agents.ppo.agent import PPO
from protomotions.agents.ppo.config import StdMode
from protomotions.agents.ppo.model import PPOActor
from protomotions.tests.test_ppo_mean import make_actor, observations


@pytest.mark.parametrize("mode", [StdMode.FIXED, StdMode.GLOBAL])
def test_mode_controls_trainable_std_and_rollout_buffer(mode):
    config = make_actor().config
    config.std_mode = mode
    actor = PPOActor(config)
    assert actor.logstd.requires_grad == (mode == StdMode.GLOBAL)
    assert "action_std" in actor.experience_buffer_keys()
    output = actor(observations())
    assert output["action_std"].shape == output["action"].shape
    old_std = output["action_std"].detach().clone()
    with torch.no_grad():
        actor.logstd.add_(1)
    torch.testing.assert_close(output["action_std"], old_std)


def test_entropy_increases_exploration_until_upper_bound():
    config = make_actor().config
    config.std_mode = StdMode.GLOBAL
    actor = PPOActor(config)
    optimizer = torch.optim.SGD(actor.parameters(), lr=100)
    before = actor.logstd.detach().clone()
    output = actor(observations())
    dist = torch.distributions.Normal(output["mean_action"], output["action_std"])
    (-dist.entropy().sum(-1).mean()).backward()
    optimizer.step()
    actor.project_logstd()
    assert torch.all(actor.logstd > before)
    torch.testing.assert_close(
        actor.logstd, torch.full_like(actor.logstd, config.max_logstd)
    )
    with torch.no_grad():
        actor.logstd.fill_(-100)
    actor.project_logstd()
    torch.testing.assert_close(
        actor.logstd, torch.full_like(actor.logstd, config.min_logstd)
    )


def test_fixed_std_is_not_clamped():
    config = make_actor().config
    config.actor_logstd = 2.0
    actor = PPOActor(config)
    actor.project_logstd()
    torch.testing.assert_close(actor.logstd, torch.full_like(actor.logstd, 2.0))


def test_kl_detects_std_only_change():
    agent = PPO.__new__(PPO)
    agent.fabric = SimpleNamespace(world_size=1)
    mu = torch.zeros(4, 2)
    old_std = torch.full_like(mu, 0.2)
    new_std = torch.full_like(mu, 0.4)
    kl = agent._compute_kl(mu, mu, old_std, new_std)
    expected = (
        torch.distributions.kl_divergence(
            torch.distributions.Normal(mu, old_std),
            torch.distributions.Normal(mu, new_std),
        )
        .sum(-1)
        .mean()
    )
    torch.testing.assert_close(kl, expected)
    assert kl > 0
    assert agent._compute_kl(mu, mu, old_std, old_std) == 0
