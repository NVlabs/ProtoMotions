# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discrete-prior PEFT actor built around a pretrained discrete latent prior.

This module owns observation routing, FSQ encode/decode helpers, and actor
rollout outputs. Token-level adapter generation lives in ``prior_with_peft``;
the pretrained prior model itself lives in the Supervised latent-prior package.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from tensordict import TensorDict

from protomotions.agents.common.discrete_latent import (
    DiscreteLatentDecoder,
    DiscreteLatentTargetEncoder,
    FSQTokenization,
)
from protomotions.agents.common.latent import (
    LATENT_KEY,
    LATENT_LOGPROB_KEY,
)
from protomotions.agents.common.pretrained import freeze_module
from protomotions.agents.peft.utils.adapter_state import (
    build_adapter_state_dict,
    load_adapter_state_dict as load_adapter_state,
)
from protomotions.agents.peft.utils.frozen_prior_contract import (
    require_frozen_prior_attr,
    resolve_frozen_prior_input_keys,
)
from protomotions.agents.peft.prior_with_peft import DiscretePriorWithPEFT
from protomotions.utils.hydra_replacement import get_class

log = logging.getLogger(__name__)


class DiscretePriorPEFTActor(nn.Module):
    """Actor that wraps DiscretePriorWithPEFT and the frozen decoder for action generation.

    The actor owns the public observation contract. Task observations flow
    through ``config.peft.model`` into ``config.peft.condition_key``; frozen
    prior context keys are discovered from the pretrained prior and appended to
    ``in_keys`` internally. The target encoder is optional and is used only for
    SFT batches that include ``mimic_target_poses``.
    """

    def __init__(
        self,
        config,
        pretrained_prior_model,
        mimic_target_poses_dim: int = 0,
    ):
        super().__init__()
        peft_cfg = config.peft
        self.out_keys = list(config.out_keys)

        latent_decoder = require_frozen_prior_attr(
            pretrained_prior_model,
            "latent_decoder",
            DiscreteLatentDecoder,
        )
        latent_tokenization = require_frozen_prior_attr(
            pretrained_prior_model,
            "latent_tokenization",
            FSQTokenization,
        )
        self.frozen_prior_input_keys = resolve_frozen_prior_input_keys(
            pretrained_prior_model
        )
        prior_transformer = pretrained_prior_model.prior

        self.condition_key = peft_cfg.condition_key
        actor_model_config = peft_cfg.model
        if actor_model_config is None:
            raise AssertionError(
                "DiscretePriorPEFTActor requires config.peft.model. "
                "DiscretePriorPEFTActorConfig should resolve the default model."
            )
        ActorPEFTModelClass = get_class(actor_model_config._target_)
        self.actor_peft_model = ActorPEFTModelClass(config=actor_model_config)
        if self.condition_key not in self.actor_peft_model.out_keys:
            raise AssertionError(
                "DiscretePriorPEFTActor PEFT model must produce condition_key "
                f"{self.condition_key!r} for adapter "
                f"conditioning. out_keys={self.actor_peft_model.out_keys}"
            )
        missing_actor_inputs = [
            key for key in self.actor_peft_model.in_keys if key not in config.in_keys
        ]
        if missing_actor_inputs:
            raise AssertionError(
                "DiscretePriorPEFTActor in_keys must include actor model in_keys. "
                f"Missing: {missing_actor_inputs}"
            )
        self.in_keys = list(
            dict.fromkeys(
                [
                    *config.in_keys,
                    *self.actor_peft_model.in_keys,
                    *self.frozen_prior_input_keys,
                ]
            )
        )
        log.info(
            "[PEFT] Loaded frozen prior expects actor input keys=%s; "
            "actor.peft.model.in_keys=%s; actor.peft.model.out_keys=%s",
            tuple(self.frozen_prior_input_keys),
            tuple(self.actor_peft_model.in_keys),
            tuple(self.actor_peft_model.out_keys),
        )

        # Frozen decoder from the current discrete latent prior model.
        self.latent_decoder = latent_decoder
        self.latent_tokenization = latent_tokenization
        self.decoder_latent_key = latent_decoder.latent_key
        freeze_module(self.latent_decoder)

        # Frozen encoder from prior (only needed for SFT with mimic_target_poses).
        if mimic_target_poses_dim > 0:
            target_latent_encoder = require_frozen_prior_attr(
                pretrained_prior_model,
                "target_latent_encoder",
                DiscreteLatentTargetEncoder,
            )
            self.target_latent_encoder = target_latent_encoder
            freeze_module(self.target_latent_encoder)
        else:
            self.target_latent_encoder = None

        # Frozen prior transformer, wrapped with PEFT adapters below.
        freeze_module(prior_transformer)

        # FSQ scalar-code counts stay separate from autoregressive prior tokens.
        self.num_fsq_levels = latent_decoder.num_fsq_levels
        self.num_fsq_scalars = latent_decoder.num_fsq_scalars
        self.fsq_scalars_per_prior_token = (
            latent_tokenization.fsq_scalars_per_prior_token
        )
        self.num_prior_tokens = latent_tokenization.num_prior_tokens
        self.prior_token_vocab_size = latent_tokenization.prior_token_vocab_size
        self.L = latent_decoder.quantizer.L
        self.half_L = latent_decoder.quantizer.half_L
        self.half_width = latent_decoder.quantizer.half_width

        # PEFT-wrapped prior
        self.prior_with_peft = DiscretePriorWithPEFT(
            prior=prior_transformer,
            rank=peft_cfg.rank,
            alpha=peft_cfg.alpha,
            peft_type=peft_cfg.peft_type,
            temperature=peft_cfg.temperature,
            top_p=peft_cfg.top_p,
            sampling_mode=peft_cfg.sampling_mode,
            prior_top_p=peft_cfg.prior_top_p,
            condition_key=self.condition_key,
            film_input_norm=peft_cfg.film_input_norm,
            film_input_norm_clamp=peft_cfg.film_input_norm_clamp,
        )
        self.kl_coeff = peft_cfg.kl_coeff

    def optional_full_checkpoint_state_prefixes(self) -> tuple[str, ...]:
        """Frozen target encoder state is present only for SFT/checkpointing flows."""
        return ("target_latent_encoder.",)

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only PEFT adapter weights, excluding the frozen prior."""
        return build_adapter_state_dict(self)

    def load_adapter_state_dict(self, state_dict: dict, strict: bool = True):
        """Load adapter-only state, ignoring frozen-prior and critic entries."""
        return load_adapter_state(self, state_dict, strict=strict)

    def init_peft(self, warmup_obs: dict | None = None):
        if warmup_obs is not None:
            warmup_obs = self.build_prior_input(warmup_obs)
        self.prior_with_peft.init_peft(warmup_obs=warmup_obs)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.target_latent_encoder is not None:
            self.target_latent_encoder.eval()
            self.target_latent_encoder.encoder.eval()
        self.latent_decoder.eval()
        self.latent_decoder.decoder.eval()
        self.latent_decoder.quantizer.eval()
        self.prior_with_peft.base_prior.eval()
        self.prior_with_peft.train(mode)
        return self

    # ---- FSQ utilities ----

    def quantize(self, z):
        return self.latent_decoder.quantizer.quantize(z)

    def fsq_codes_to_fsq_indices(self, codes):
        return self.latent_decoder.quantizer.codes_to_indices(codes)

    def fsq_indices_to_codes(self, indices):
        return self.latent_decoder.quantizer.indices_to_codes(indices)

    def fsq_indices_to_prior_tokens(self, fsq_indices):
        return self.latent_tokenization.fsq_indices_to_prior_tokens(fsq_indices)

    def prior_tokens_to_fsq_indices(self, prior_tokens):
        return self.latent_tokenization.prior_tokens_to_fsq_indices(prior_tokens)

    def one_hot_prior_tokens(self, prior_tokens):
        return self.latent_tokenization.one_hot_prior_tokens(prior_tokens)

    def perturb_tokens(
        self,
        tokens: torch.Tensor,
        *,
        rate: float,
        mode: str,
    ) -> torch.Tensor:
        """Apply SFT token-noise augmentation to GPC prior tokens.

        Teacher forcing on perturbed target tokens reduces exact-sequence
        memorization and makes the adapter less brittle when RLFT samples drift
        off the expert trajectory.
        """
        if rate <= 0:
            return tokens

        b, n = tokens.shape
        device = tokens.device
        mask = torch.rand(b, n, device=device) < rate
        if not mask.any():
            return tokens

        result = tokens.clone()

        if mode == "replace" or mode == "mixed":
            rand_tokens = torch.randint(
                0,
                self.prior_token_vocab_size,
                (b, n),
                device=device,
                dtype=tokens.dtype,
            )
            if mode == "replace":
                result = torch.where(mask, rand_tokens, result)
            else:
                use_replace = torch.rand(b, n, device=device) < 0.2
                result = torch.where(mask & use_replace, rand_tokens, result)
                neighbor_mask = mask & ~use_replace
                if neighbor_mask.any():
                    result = self._neighbor_perturb(result, neighbor_mask, tokens)
        elif mode == "neighbor":
            result = self._neighbor_perturb(result, mask, tokens)
        else:
            raise ValueError(
                f"Unsupported token perturbation mode {mode!r}; expected "
                "'replace', 'mixed', or 'neighbor'."
            )

        return result

    def _neighbor_perturb(
        self,
        result: torch.Tensor,
        mask: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        b, _ = tokens.shape
        # Neighbor perturbation is easiest in flat FSQ-scalar space: each scalar
        # index can move by -1, 0, or +1 before packing back into prior tokens.
        fsq_indices = self.prior_tokens_to_fsq_indices(tokens)
        fsq_indices = fsq_indices.view(
            b,
            self.num_prior_tokens,
            self.fsq_scalars_per_prior_token,
        )
        offset = torch.randint(-1, 2, fsq_indices.shape, device=tokens.device)
        perturbed = (fsq_indices + offset).clamp(0, self.num_fsq_levels - 1)
        # Convert back to the categorical token vocabulary expected by the prior.
        perturbed = perturbed.view(b, -1)
        neighbor_prior_tokens = self.fsq_indices_to_prior_tokens(perturbed)
        return torch.where(mask, neighbor_prior_tokens, result)

    # ---- encode / decode ----

    def _encode(self, tensordict):
        """Encode observations to FSQ codes. Requires mimic_target_poses in tensordict."""
        if self.target_latent_encoder is None:
            raise RuntimeError(
                "DiscretePriorPEFTActor._encode requires mimic_target_poses_dim > 0 "
                "so the target_latent_encoder path is available."
            )
        encoder = self.target_latent_encoder.encoder
        td = encoder(tensordict)
        key = encoder.out_keys[0] if hasattr(encoder, "out_keys") else "latent"
        return self.quantize(td[key])

    def _decode(self, tensordict, fsq_codes):
        """Decode FSQ codes to actions. Only needs max_coords_obs + latent."""
        tensordict[self.decoder_latent_key] = fsq_codes
        decoder = self.latent_decoder.decoder
        td = decoder(tensordict)
        key = decoder.out_keys[0] if hasattr(decoder, "out_keys") else "mu"
        return td[key]

    def predict_target_prior_tokens(self, tensordict):
        """Encode target poses to GPC prior tokens for SFT."""
        with torch.no_grad():
            codes = self._encode(tensordict)
            fsq_indices = self.fsq_codes_to_fsq_indices(codes)
            return self.fsq_indices_to_prior_tokens(fsq_indices)

    # ---- prior input construction ----

    def _to_tensordict(self, d):
        if isinstance(d, TensorDict):
            return d.clone()
        first_tensor = next(value for value in d.values() if torch.is_tensor(value))
        return TensorDict(
            dict(d),
            batch_size=first_tensor.shape[0],
            device=first_tensor.device,
        )

    def _run_actor_peft_model(self, d):
        """Run the task-side conditioning network and validate its contract."""
        tensordict = self._to_tensordict(d)
        missing_keys = [
            key for key in self.actor_peft_model.in_keys if key not in tensordict
        ]
        if missing_keys:
            raise ValueError(
                "DiscretePriorPEFTActor actor PEFT model in_keys must be present in "
                f"the input TensorDict. Missing keys: {missing_keys}"
            )
        tensordict = self.actor_peft_model(tensordict)
        if self.condition_key not in tensordict:
            raise RuntimeError(
                "DiscretePriorPEFTActor actor PEFT model did not produce required "
                f"condition_key {self.condition_key!r}."
            )
        missing_context = [
            key for key in self.frozen_prior_input_keys if key not in tensordict
        ]
        if missing_context:
            raise RuntimeError(
                "DiscretePriorPEFTActor actor PEFT model did not produce frozen prior input "
                f"keys {missing_context}."
            )
        return tensordict

    def build_prior_input(self, tensordict, tokens: torch.Tensor | None = None):
        # Public PEFT config controls the task-conditioning network. The frozen
        # prior context keys come from the loaded prior checkpoint, and both are
        # merged here into the exact dictionary consumed by DiscretePriorWithPEFT.
        input_td = self._run_actor_peft_model(tensordict)
        token_one_hot = None
        if tokens is not None:
            token_one_hot = self.one_hot_prior_tokens(tokens)
        prior_dict = {key: input_td[key] for key in self.frozen_prior_input_keys}
        prior_dict[self.condition_key] = input_td[self.condition_key]
        if token_one_hot is not None:
            prior_dict["tokens"] = token_one_hot
        return prior_dict

    # ---- forward / rollout ----

    def forward(self, input_dict: dict):
        """Teacher-forced forward -> logits (B, num_prior_tokens, prior_token_vocab_size)."""
        return self.prior_with_peft(input_dict)

    def get_action_and_logp(self, tensordict):
        """Rollout step: generate action + per-token log-probs for PPO."""
        prior_dict = self.build_prior_input(tensordict)
        prior_tokens, logprob = self.prior_with_peft.generate(
            prior_dict,
            return_logits=False,
            return_logprob=True,
        )

        neglogp = -logprob

        fsq_indices = self.prior_tokens_to_fsq_indices(prior_tokens)
        fsq_codes = self.fsq_indices_to_codes(fsq_indices)
        action = self._decode(tensordict, fsq_codes)

        tensordict["action"] = action
        tensordict["mean_action"] = action
        tensordict["neglogp"] = neglogp
        tensordict["prior_tokens"] = prior_tokens
        tensordict[LATENT_KEY] = prior_tokens
        tensordict[LATENT_LOGPROB_KEY] = logprob
        return tensordict
