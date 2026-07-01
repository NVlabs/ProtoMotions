# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SFT model for PEFT adapters on a frozen discrete-token GPC prior."""

import torch
from tensordict import TensorDict

from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGITS_KEY,
    TARGET_LATENT_KEY,
)
from protomotions.agents.peft.model import DiscretePriorPEFTModel


class DiscretePriorPEFTSFTModel(DiscretePriorPEFTModel):
    """Discrete-prior PEFT model used by the supervised SFT agent.

    Rollout uses the frozen target encoder as the expert: encode the target
    motion into prior tokens, decode those tokens to an action, and store the
    tokens as supervision labels. Optimization replays the batch with teacher
    forcing and writes ``latent_logits`` for the generic supervision loss.
    """

    def collect_expert_rollout(self, tensordict: TensorDict) -> TensorDict:
        target_prior_tokens = self._actor.predict_target_prior_tokens(tensordict)
        fsq_indices = self._actor.prior_tokens_to_fsq_indices(target_prior_tokens)
        fsq_codes = self._actor.fsq_indices_to_codes(fsq_indices)
        action = self._actor._decode(tensordict, fsq_codes)

        tensordict["action"] = action
        tensordict["mean_action"] = action
        tensordict["prior_tokens"] = target_prior_tokens
        tensordict[LATENT_KEY] = target_prior_tokens
        tensordict[TARGET_LATENT_KEY] = target_prior_tokens
        # The expert encoder is deterministic and SFT trains with
        # cross-entropy, so neglogp is an unused rollout-contract placeholder.
        if "neglogp" in self.out_keys:
            tensordict["neglogp"] = torch.zeros(
                action.shape[0],
                self._actor.num_prior_tokens,
                device=action.device,
                dtype=action.dtype,
            )
        return tensordict

    def materialize(self, tensordict: TensorDict) -> TensorDict:
        expert_td = self.collect_expert_rollout(tensordict.clone())
        return self.forward(expert_td)

    def forward(self, tensordict: TensorDict) -> TensorDict:
        if not isinstance(tensordict, TensorDict):
            raise TypeError(
                "DiscretePriorPEFTSFTModel.forward expects a TensorDict input."
            )
        if TARGET_LATENT_KEY not in tensordict:
            tensordict = self.collect_expert_rollout(tensordict)
        target_prior_tokens = tensordict[TARGET_LATENT_KEY].detach()

        teacher_tokens = self._actor.perturb_tokens(
            target_prior_tokens,
            rate=self.config.token_perturb_rate,
            mode=self.config.token_perturb_mode,
        )
        prior_dict = self._actor.build_prior_input(tensordict, tokens=teacher_tokens)
        tensordict[LATENT_LOGITS_KEY] = self._actor(prior_dict)
        return tensordict
