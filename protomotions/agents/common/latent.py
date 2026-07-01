# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared latent-output keys and latent-specific loss helpers."""

import torch

from protomotions.agents.common.autoregressive import (
    prior_constrained_sampling_log_probs,
    sampling_log_probs,
)


LATENT_KEY = "latent"
LATENT_LOGITS_KEY = "latent_logits"
LATENT_LOGPROB_KEY = "latent_logprob"
LATENT_MU_KEY = "latent_mu"
LATENT_LOGVAR_KEY = "latent_logvar"
VAE_LATENT_KEY = "vae_latent"
VAE_NOISE_KEY = "vae_noise"
PRIVILEGED_LATENT_KEY = "privileged_latent"
PRIVILEGED_LATENT_MU_KEY = "privileged_latent_mu"
PRIVILEGED_LATENT_LOGVAR_KEY = "privileged_latent_logvar"
TARGET_LATENT_KEY = "target_latent"
TARGET_LATENT_LOGITS_KEY = "target_latent_logits"
TARGET_LATENT_MU_KEY = "target_latent_mu"
TARGET_LATENT_LOGVAR_KEY = "target_latent_logvar"


def compute_discrete_latent_ppo_loss(
    *,
    logits: torch.Tensor,
    selected: torch.Tensor,
    old_neglogp: torch.Tensor,
    advantages: torch.Tensor,
    e_clip: float,
    entropy_coef: float = 0.0,
    temperature: float = 1.0,
    top_p: float = 1.0,
    prior_logits: torch.Tensor = None,
    log_prefix: str = "actor",
):
    """Compute PPO loss under the same distribution used during token rollout."""
    if prior_logits is None:
        log_probs = sampling_log_probs(logits, p=top_p, temperature=temperature)
    else:
        log_probs = prior_constrained_sampling_log_probs(
            logits,
            prior_logits,
            p=top_p,
            temperature=temperature,
        )
    logprob = log_probs.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    old_logprob = -old_neglogp

    logprob_sum = logprob.sum(dim=-1)
    old_logprob_sum = old_logprob.sum(dim=-1)
    ratio = torch.exp(logprob_sum - old_logprob_sum)
    unclipped = advantages * ratio
    clipped = advantages * torch.clamp(ratio, 1.0 - e_clip, 1.0 + e_clip)
    ppo_loss = -torch.min(unclipped, clipped).mean()

    probs = log_probs.exp()
    finite_log_probs = torch.where(
        torch.isfinite(log_probs),
        log_probs,
        torch.zeros_like(log_probs),
    )
    entropy = -(probs * finite_log_probs).sum(dim=-1).mean()
    loss = ppo_loss - entropy_coef * entropy

    with torch.no_grad():
        kl = (old_logprob_sum - logprob_sum).mean()
        clip_frac = (torch.abs(ratio - 1.0) > e_clip).float().mean()

    metrics = {
        f"{log_prefix}/ppo_loss": ppo_loss.detach(),
        f"{log_prefix}/entropy": entropy.detach(),
        f"{log_prefix}/ratio": ratio.mean().detach(),
        f"{log_prefix}/clip_frac": clip_frac.detach(),
        f"{log_prefix}/kl": kl.detach(),
        f"{log_prefix}/loss": loss.detach(),
    }
    return loss, metrics
