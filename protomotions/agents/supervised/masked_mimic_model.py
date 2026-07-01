# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MaskedMimic VAE student for the generic supervised agent.

MaskedMimic is trained by ``SupervisedAgent``, but the architecture itself is
not generic supervised-agent machinery. It is a VAE learned-prior student:

* ``prior`` predicts a deployable latent distribution from sparse observations.
* ``encoder`` predicts a privileged residual posterior during training.
* ``trunk`` decodes latent samples to actions.

Keeping this model named after MaskedMimic makes experiment files searchable and
keeps the VAE-specific KL/noise logic out of the generic supervised loop.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from tensordict import TensorDict

from protomotions.agents.base_agent.model import (
    BaseModel,
    ProtoMotionsTensorDictModule,
    RolloutStateSpec,
)
from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGVAR_KEY,
    LATENT_MU_KEY,
    PRIVILEGED_LATENT_KEY,
    PRIVILEGED_LATENT_LOGVAR_KEY,
    PRIVILEGED_LATENT_MU_KEY,
    VAE_LATENT_KEY,
    VAE_NOISE_KEY,
)
from protomotions.utils.hydra_replacement import get_class


class MaskedMimicModel(BaseModel):
    """MaskedMimic learned-prior VAE student.

    The non-privileged path is used at inference: prior -> latent sample ->
    trunk. During supervised training, the privileged encoder adds a residual
    posterior path that produces ``privileged_action`` for the imitation loss and
    a KL term against the prior distribution.
    """

    def __init__(self, config):
        super().__init__(config)

        encoder_class = get_class(self.config.encoder._target_)
        self._encoder = encoder_class(config=self.config.encoder)
        prior_class = get_class(self.config.prior._target_)
        self._prior = prior_class(config=self.config.prior)

        trunk_class = get_class(self.config.trunk._target_)
        self._trunk = trunk_class(config=self.config.trunk)

        trunk_in_keys = [key for key in self._trunk.in_keys if key != VAE_LATENT_KEY]
        self.in_keys = list(
            dict.fromkeys(self._prior.in_keys + self._encoder.in_keys + trunk_in_keys)
        )
        self.out_keys = [
            "action",
            "privileged_action",
            VAE_NOISE_KEY,
            LATENT_KEY,
            LATENT_MU_KEY,
            LATENT_LOGVAR_KEY,
            PRIVILEGED_LATENT_KEY,
            PRIVILEGED_LATENT_MU_KEY,
            PRIVILEGED_LATENT_LOGVAR_KEY,
        ]

    def _forward_module(
        self,
        module,
        tensordict: TensorDict,
        log_internals: bool,
    ) -> TensorDict:
        if isinstance(module, ProtoMotionsTensorDictModule):
            return module(tensordict, log_internals=log_internals)
        return module(tensordict)

    def rollout_state_specs(self) -> dict[str, RolloutStateSpec]:
        return {
            **super().rollout_state_specs(),
            VAE_NOISE_KEY: RolloutStateSpec(
                shape=(self.config.vae.vae_latent_dim,),
                init=self.config.vae.vae_noise_type,
                dtype=torch.float32,
            ),
        }

    @staticmethod
    def _sample_latent(
        mean: torch.Tensor,
        logvar: torch.Tensor,
        vae_noise: torch.Tensor,
    ) -> torch.Tensor:
        return mean + torch.exp(0.5 * logvar) * vae_noise

    def _vae_noise(self, tensordict: TensorDict) -> torch.Tensor:
        self.read_rollout_state(tensordict)
        return tensordict[VAE_NOISE_KEY]

    def _decode(
        self,
        tensordict: TensorDict,
        latent: torch.Tensor,
        log_internals: bool,
    ) -> torch.Tensor:
        tensordict[VAE_LATENT_KEY] = latent
        tensordict = self._forward_module(self._trunk, tensordict, log_internals)
        return tensordict[self._trunk.out_keys[0]]

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        tensordict = self._forward_module(self._prior, tensordict, log_internals)
        prior_mu = tensordict[self._prior.out_keys[0]]
        prior_logvar = tensordict[self._prior.out_keys[1]]
        vae_noise = self._vae_noise(tensordict)
        prior_latent = self._sample_latent(prior_mu, prior_logvar, vae_noise)

        tensordict[LATENT_MU_KEY] = prior_mu
        tensordict[LATENT_LOGVAR_KEY] = prior_logvar
        tensordict[LATENT_KEY] = prior_latent
        tensordict["action"] = self._decode(
            tensordict,
            prior_latent,
            log_internals=log_internals,
        )

        tensordict = self._forward_module(self._encoder, tensordict, log_internals)
        encoder_mu = tensordict[self._encoder.out_keys[0]]
        encoder_logvar = tensordict[self._encoder.out_keys[1]]
        privileged_mu = prior_mu + encoder_mu
        privileged_latent = self._sample_latent(
            privileged_mu,
            encoder_logvar,
            vae_noise,
        )

        tensordict[PRIVILEGED_LATENT_MU_KEY] = privileged_mu
        tensordict[PRIVILEGED_LATENT_LOGVAR_KEY] = encoder_logvar
        tensordict[PRIVILEGED_LATENT_KEY] = privileged_latent
        tensordict["privileged_action"] = self._decode(
            tensordict,
            privileged_latent,
            log_internals=log_internals,
        )
        return tensordict

    def forward_inference(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self._forward_module(
            self._prior,
            tensordict,
            log_internals=False,
        )
        prior_mu = tensordict[self._prior.out_keys[0]]
        prior_logvar = tensordict[self._prior.out_keys[1]]
        prior_latent = self._sample_latent(
            prior_mu,
            prior_logvar,
            self._vae_noise(tensordict),
        )
        tensordict[LATENT_MU_KEY] = prior_mu
        tensordict[LATENT_LOGVAR_KEY] = prior_logvar
        tensordict[LATENT_KEY] = prior_latent
        tensordict["action"] = self._decode(
            tensordict,
            prior_latent,
            log_internals=False,
        )
        return tensordict

    def get_inference_in_keys(self) -> list:
        trunk_in_keys = [key for key in self._trunk.in_keys if key != VAE_LATENT_KEY]
        return list(dict.fromkeys(self._prior.in_keys + trunk_in_keys))

    def kl_loss(self, tensordict: TensorDict) -> torch.Tensor:
        prior_mu_key, prior_logvar_key = self._prior.out_keys
        encoder_mu_key, encoder_logvar_key = self._encoder.out_keys
        return 0.5 * (
            tensordict[prior_logvar_key]
            - tensordict[encoder_logvar_key]
            + torch.exp(tensordict[encoder_logvar_key])
            / torch.exp(tensordict[prior_logvar_key])
            + tensordict[encoder_mu_key].square()
            / torch.exp(tensordict[prior_logvar_key])
            - 1
        )

    def _kld_coefficient(self, current_epoch: int) -> float:
        schedule = getattr(self.config.vae, "kld_schedule", None)
        if schedule is None:
            return 0.0

        if schedule.end_epoch <= schedule.start_epoch:
            progress = 0.0 if current_epoch < schedule.start_epoch else 1.0
        else:
            progress = min(
                max(0, current_epoch - schedule.start_epoch)
                / (schedule.end_epoch - schedule.start_epoch),
                1,
            )
        return (
            schedule.init_kld_coeff
            + progress * (schedule.end_kld_coeff - schedule.init_kld_coeff)
        )

    def compute_model_loss(
        self,
        tensordict: Optional[TensorDict],
        current_epoch: int,
        zero_loss: torch.Tensor,
        log_prefix: str = "model",
    ) -> Tuple[torch.Tensor, Dict]:
        loss, log_dict = super().compute_model_loss(
            tensordict,
            current_epoch=current_epoch,
            zero_loss=zero_loss,
            log_prefix=log_prefix,
        )
        if getattr(self.config.vae, "kld_schedule", None) is None:
            return loss, log_dict

        kld_coeff = self._kld_coefficient(current_epoch)
        if tensordict is None:
            kld_loss = zero_loss * 0.0
        else:
            kld_loss = torch.mean(torch.sum(self.kl_loss(tensordict), dim=-1))

        model_loss = kld_loss * kld_coeff
        log_dict.update(
            {
                f"{log_prefix}/kld_loss": model_loss.detach(),
                f"{log_prefix}/kld_coeff": torch.tensor(
                    kld_coeff,
                    device=model_loss.device,
                ),
            }
        )
        return loss + model_loss, log_dict
