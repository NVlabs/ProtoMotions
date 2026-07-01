# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared latent outputs and losses."""

import torch

from protomotions.agents.common.autoregressive import (
    prior_constrained_sampling_log_probs,
    sampling_log_probs,
)
from protomotions.agents.common.latent import (
    compute_discrete_latent_ppo_loss,
)


def test_discrete_latent_ppo_loss_matches_manual_ratio():
    logits = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 4.0]],
            [[-1.0, 5.0], [2.0, 0.0]],
        ]
    )
    selected = torch.tensor([[0, 1], [1, 0]])
    old_neglogp = torch.tensor([[0.2, 0.3], [0.4, 0.5]])
    advantages = torch.tensor([1.0, -0.5])

    loss, metrics = compute_discrete_latent_ppo_loss(
        logits=logits,
        selected=selected,
        old_neglogp=old_neglogp,
        advantages=advantages,
        e_clip=0.2,
        entropy_coef=0.01,
    )

    dist = torch.distributions.Categorical(logits=logits)
    logprob_sum = dist.log_prob(selected).sum(dim=-1)
    old_logprob_sum = (-old_neglogp).sum(dim=-1)
    ratio = torch.exp(logprob_sum - old_logprob_sum)
    unclipped = advantages * ratio
    clipped = advantages * torch.clamp(ratio, 0.8, 1.2)
    ppo_loss = -torch.min(unclipped, clipped).mean()
    entropy = dist.entropy().mean()

    assert torch.allclose(loss, ppo_loss - 0.01 * entropy)
    assert torch.allclose(metrics["actor/ppo_loss"], ppo_loss.detach())
    assert torch.allclose(metrics["actor/entropy"], entropy.detach())


def test_discrete_latent_ppo_loss_recomputes_top_p_sampling_logprob():
    logits = torch.tensor([[[4.0, 3.0, 0.0]]])
    selected = torch.tensor([[0]])
    logprob = sampling_log_probs(logits[:, 0], p=0.5).gather(
        -1, selected[:, 0:1]
    )

    _, metrics = compute_discrete_latent_ppo_loss(
        logits=logits,
        selected=selected,
        old_neglogp=-logprob,
        advantages=torch.ones(1),
        e_clip=0.2,
        entropy_coef=0.01,
        top_p=0.5,
    )

    assert torch.allclose(metrics["actor/ratio"], torch.ones(()))
    assert torch.allclose(metrics["actor/kl"], torch.zeros(()))
    assert torch.allclose(metrics["actor/entropy"], torch.zeros(()))


def test_discrete_latent_ppo_loss_recomputes_prior_constrained_logprob():
    logits = torch.tensor([[[0.0, 5.0, 5.0]]])
    prior_logits = torch.tensor([[[8.0, 0.0, 0.0]]])
    selected = torch.tensor([[0]])
    logprob = prior_constrained_sampling_log_probs(
        logits[:, 0],
        prior_logits[:, 0],
        p=0.5,
    ).gather(-1, selected[:, 0:1])

    _, metrics = compute_discrete_latent_ppo_loss(
        logits=logits,
        selected=selected,
        old_neglogp=-logprob,
        advantages=torch.ones(1),
        e_clip=0.2,
        entropy_coef=0.01,
        top_p=0.5,
        prior_logits=prior_logits,
    )

    assert torch.allclose(metrics["actor/ratio"], torch.ones(()))
    assert torch.allclose(metrics["actor/kl"], torch.zeros(()))
    assert torch.allclose(metrics["actor/entropy"], torch.zeros(()))
