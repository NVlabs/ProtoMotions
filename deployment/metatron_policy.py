# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic export of the learned Metatron prior, without its encoder."""

from __future__ import annotations

from tensordict import TensorDict
from torch import nn

from protomotions.agents.common.latent import VAE_NOISE_KEY
from protomotions.agents.supervised.masked_mimic_model import MaskedMimicModel


class MetatronPriorExport(nn.Module):
    """Keep the trained latent geometry and decoder, with zero latent noise.

    This is the same prior path used in deterministic inference. The teacher
    encoder is loaded strictly with the checkpoint but is never executed.
    Noise has a dynamic batch axis rather than a carried batch-one rollout state.
    """

    def __init__(self, model: MaskedMimicModel) -> None:
        super().__init__()
        self.model = model
        self.in_keys = model.deployable_in_keys
        self.out_keys = [model.deployable_action_key]
        self.deployable_action_key = model.deployable_action_key

    def forward_deployable(self, inputs: TensorDict) -> TensorDict:
        reference = inputs[self.in_keys[0]]
        inputs[VAE_NOISE_KEY] = reference.new_zeros(
            (reference.shape[0], self.model.config.vae.vae_latent_dim)
        )
        return self.model.forward_deployable(inputs)
