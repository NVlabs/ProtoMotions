# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic teacher paths must not sample discarded actions."""

import torch
from tensordict import TensorDict

from protomotions.agents.common.config import MLPWithConcatConfig
from protomotions.agents.ppo.config import PPOActorConfig
from protomotions.agents.ppo.model import PPOActor
from protomotions.agents.supervised.agent import SupervisedAgent


def make_actor():
    actor = PPOActor(
        PPOActorConfig(
            mu_key="mu",
            in_keys=["obs"],
            num_out=2,
            mu_model=MLPWithConcatConfig(
                in_keys=["obs"],
                out_keys=["mu"],
                num_out=2,
                normalize_obs=False,
                layers=[],
            ),
        )
    ).eval()
    actor.forward_deployable(
        observations()
    )  # Materialize lazy parameters before testing RNG.
    return actor


def observations():
    return TensorDict({"obs": torch.ones(3, 4)}, batch_size=[3])


def test_teacher_and_deployable_use_the_same_mean_without_rng_changes():
    actor = make_actor()
    expected = actor(observations())["mean_action"]
    with torch.no_grad():
        actor.logstd.fill_(10)
    rng = torch.random.get_rng_state()
    for output in (
        actor.forward_deployable(observations()),
        SupervisedAgent._forward_expert(actor, observations()),
    ):
        torch.testing.assert_close(output["mean_action"], expected)
        assert "action" not in output
        assert "neglogp" not in output
    assert torch.equal(rng, torch.random.get_rng_state())


def test_non_ppo_teacher_labels_through_forward_deployable():
    class Teacher(torch.nn.Module):
        deployable_action_key = "action"

        def forward_deployable(self, td):
            td["action"] = td["obs"] * 2
            return td

    result = SupervisedAgent._forward_expert(Teacher(), observations())
    torch.testing.assert_close(result["action"], torch.full((3, 4), 2.0))
