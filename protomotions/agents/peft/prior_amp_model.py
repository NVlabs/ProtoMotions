# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discrete-prior PEFT RLFT model with AMP discriminator outputs."""

from __future__ import annotations

from tensordict import TensorDict

from protomotions.agents.amp.model import AMPModelComponentsMixin
from protomotions.agents.peft.model import DiscretePriorPEFTModel


class DiscretePriorPEFTRLFTAMPModel(AMPModelComponentsMixin, DiscretePriorPEFTModel):
    """Discrete-prior PEFT actor/critic with AMP discriminator outputs."""

    def __init__(
        self,
        config,
        pretrained_prior_model,
        mimic_target_poses_dim: int = 0,
    ):
        super().__init__(
            config=config,
            pretrained_prior_model=pretrained_prior_model,
            mimic_target_poses_dim=mimic_target_poses_dim,
        )
        self._build_amp_model_components(config)

    def optional_full_checkpoint_state_prefixes(self) -> tuple[str, ...]:
        """AMP discriminator state is optional for PEFT full-checkpoint loads."""
        return super().optional_full_checkpoint_state_prefixes() + (
            "_discriminator.",
            "_disc_critic.",
        )

    def experience_buffer_keys(self) -> list:
        base_keys = super().experience_buffer_keys()
        amp_keys = self._discriminator.out_keys + self._disc_critic.out_keys
        duplicate_keys = sorted(set(base_keys).intersection(amp_keys))
        if duplicate_keys:
            raise ValueError(
                "PEFT+AMP discriminator outputs must not reuse actor/critic "
                f"buffer keys: {duplicate_keys}"
            )
        return base_keys + amp_keys

    def forward_rollout(self, tensordict: TensorDict) -> TensorDict:
        tensordict = super().forward_rollout(tensordict)
        return self._forward_amp_model_components(tensordict)
