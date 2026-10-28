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

from contextlib import contextmanager
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from protomotions.agents.common.temporal import adjacent_targets, masked_mse

from protomotions.agents.base_agent.model import (
    BaseModel,
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
from protomotions.agents.common.common import NormObsBase
from protomotions.agents.supervised.masked_mimic_config import LatentGeometry


class MaskedMimicModel(BaseModel):
    """MaskedMimic learned-prior VAE student.

    The non-privileged path is used at inference: prior -> latent sample ->
    trunk. During supervised training, the privileged encoder adds a residual
    posterior path that produces ``privileged_action`` for the imitation loss and
    a KL term against the prior distribution.
    """

    def __init__(self, config):
        super().__init__(config)

        self._encoder = self._build_submodule(self.config.encoder)
        self._prior = self._build_submodule(self.config.prior)
        self._trunk = self._build_submodule(self.config.trunk)

        self.in_keys = self._external_in_keys(include_encoder=True)
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
        declared = set(self.out_keys + self._prior.out_keys + self._trunk.out_keys)
        pairs = [(item.key, item.forward) for item in config.temporal_losses]
        if len(set(pairs)) != len(pairs):
            raise ValueError("Temporal loss (key, forward) pairs must be unique")
        required = []
        for item in config.temporal_losses:
            required.extend([item.key, item.prediction_key])
        for key in required:
            if key not in declared:
                raise ValueError(
                    f"Temporal loss key {key!r} is not a declared model output"
                )
        self.out_keys = list(dict.fromkeys(self.out_keys + required))

    def build_training_targets(self, rollout: dict) -> dict:
        targets = super().build_training_targets(rollout)
        # Frozen targets use the configured model output, not necessarily the
        # executed action. privileged_action is the raw posterior decoder output,
        # before env action processing and student/expert rollout selection.
        for item in self.config.temporal_losses:
            target, valid = adjacent_targets(
                rollout[item.key], rollout["dones"], forward=item.forward
            )
            targets[item.target_key] = target
            targets[item.valid_key] = valid
        return targets

    def rollout_state_specs(self) -> dict[str, RolloutStateSpec]:
        return {
            **super().rollout_state_specs(),
            VAE_NOISE_KEY: RolloutStateSpec(
                shape=(self.config.vae.vae_latent_dim,),
                init=self.config.vae.vae_noise_type,
                dtype=torch.float32,
            ),
        }

    def _sample_latent(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        vae_noise: torch.Tensor,
    ) -> torch.Tensor:
        latent = mean + torch.exp(0.5 * logvar) * vae_noise
        if self.config.vae.latent_geometry == LatentGeometry.SPHERE:
            # The origin has no direction. Use a deterministic unit vector,
            # including when deterministic inference starts from a zero mean.
            fallback = torch.zeros_like(latent)
            fallback[..., 0] = 1.0
            return torch.where(
                latent.norm(dim=-1, keepdim=True) <= 1e-8,
                fallback,
                F.normalize(latent, dim=-1),
            )
        return latent

    @contextmanager
    def _freeze_trunk_normalizer_updates(self):
        """Record the common decoder input batch once, not once per latent path."""
        normalizers = [
            module
            for module in self._trunk.modules()
            if isinstance(module, NormObsBase)
        ]
        previous = [module._freeze_running for module in normalizers]
        try:
            for module in normalizers:
                module._freeze_running = True
            yield
        finally:
            for module, frozen in zip(normalizers, previous):
                module._freeze_running = frozen

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
        tensordict = self._trunk(tensordict, log_internals=log_internals)
        return tensordict[self._trunk.out_keys[0]]

    def _forward_prior(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        """Use the same prior distribution and episode noise in both entry points."""
        tensordict = self._prior(tensordict, log_internals=log_internals)
        prior_mu = tensordict[self._prior.out_keys[0]]
        prior_logvar = tensordict[self._prior.out_keys[1]]
        vae_noise = self._vae_noise(tensordict)
        prior_latent = self._sample_latent(prior_mu, prior_logvar, vae_noise)

        tensordict[LATENT_MU_KEY] = prior_mu
        tensordict[LATENT_LOGVAR_KEY] = prior_logvar
        tensordict[LATENT_KEY] = prior_latent
        return tensordict

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        tensordict = self._forward_prior(tensordict, log_internals=log_internals)
        prior_mu = tensordict[LATENT_MU_KEY]
        vae_noise = tensordict[VAE_NOISE_KEY]
        with self._freeze_trunk_normalizer_updates():
            tensordict["action"] = self._decode(
                tensordict,
                tensordict[LATENT_KEY],
                log_internals=log_internals,
            )

        tensordict = self._encoder(tensordict, log_internals=log_internals)
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

    def forward_deployable(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self._forward_prior(tensordict)
        tensordict["action"] = self._decode(
            tensordict,
            tensordict[LATENT_KEY],
            log_internals=False,
        )
        return tensordict

    @property
    def deployable_in_keys(self) -> list:
        return self._external_in_keys(include_encoder=False)

    def _external_in_keys(self, *, include_encoder: bool) -> list[str]:
        """Prior outputs can condition the decoder; they are not observations.

        Only the prior runs on both paths. An encoder-only output must not
        satisfy a decoder input, because deployment cannot produce it.
        """
        produced = set(self._prior.out_keys) | {VAE_LATENT_KEY}
        keys = list(self._prior.in_keys)
        if include_encoder:
            keys.extend(key for key in self._encoder.in_keys if key not in produced)
        keys.extend(key for key in self._trunk.in_keys if key not in produced)
        return list(dict.fromkeys(keys))

    def materialize(self, tensordict: TensorDict) -> TensorDict:
        """Deployment initializes the prior; checkpoint loading supplies the posterior.

        Training supplies all observations and must initialize both paths before
        DDP. A partially supplied posterior is still an input-contract error.
        """
        privileged_keys = set(self.in_keys) - set(self.deployable_in_keys)
        if privileged_keys and privileged_keys.isdisjoint(tensordict.keys()):
            return self.forward_deployable(tensordict)
        return self(tensordict)

    def kl_loss(self, tensordict: TensorDict) -> torch.Tensor:
        prior_mu_key, prior_logvar_key = self._prior.out_keys[:2]
        encoder_mu_key, encoder_logvar_key = self._encoder.out_keys[:2]
        return 0.5 * (
            tensordict[prior_logvar_key]
            - tensordict[encoder_logvar_key]
            + torch.exp(tensordict[encoder_logvar_key] - tensordict[prior_logvar_key])
            + tensordict[encoder_mu_key].square()
            * torch.exp(-tensordict[prior_logvar_key])
            - 1
        )

    def _kld_coefficient(self, current_epoch: int) -> float:
        schedule = self.config.vae.kld_schedule
        if schedule.cyclical and current_epoch > schedule.end_epoch:
            period = schedule.end_epoch - schedule.start_epoch + 1
            current_epoch = (
                schedule.start_epoch + (current_epoch - schedule.start_epoch) % period
            )
        if schedule.end_epoch <= schedule.start_epoch:
            progress = 0.0 if current_epoch < schedule.start_epoch else 1.0
        else:
            progress = min(
                max(0, current_epoch - schedule.start_epoch)
                / (schedule.end_epoch - schedule.start_epoch),
                1,
            )
        return schedule.init_kld_coeff + progress * (
            schedule.end_kld_coeff - schedule.init_kld_coeff
        )

    def compute_model_loss(
        self,
        tensordict: TensorDict,
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
        for item in self.config.temporal_losses:
            term = item.coefficient * masked_mse(
                tensordict[item.prediction_key],
                tensordict[item.target_key],
                tensordict[item.valid_key],
            )
            loss = loss + term
            log_dict[
                f"{log_prefix}/temporal_{item.direction}_{item.prediction_key}"
            ] = term.detach()
        if self.config.vae.kld_schedule is not None:
            kld_coeff = self._kld_coefficient(current_epoch)
            kld_loss = kld_coeff * self.kl_loss(tensordict).sum(dim=-1).mean()
            loss = loss + kld_loss
            log_dict[f"{log_prefix}/kld_loss"] = kld_loss.detach()
            log_dict[f"{log_prefix}/kld_coeff"] = torch.tensor(
                kld_coeff, device=kld_loss.device
            )
        if self.config.vae.prior_mean_unit_norm_coeff:
            unit_norm_loss = (
                tensordict[LATENT_MU_KEY].norm(dim=-1) - 1.0
            ).square().mean() * self.config.vae.prior_mean_unit_norm_coeff
            loss = loss + unit_norm_loss
            log_dict[f"{log_prefix}/prior_mean_unit_norm_loss"] = (
                unit_norm_loss.detach()
            )
        return loss, log_dict
