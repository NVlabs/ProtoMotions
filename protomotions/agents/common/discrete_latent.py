# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable helpers for discrete latent targets, decoders, and GPC priors."""

from typing import Protocol

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn

from protomotions.agents.base_agent.model import ProtoMotionsTensorDictModule
from protomotions.agents.common.latent import LATENT_KEY
from protomotions.agents.common.pretrained import (
    freeze_module,
    load_pretrained_model_module,
)


class DiscreteLatentQuantizer(Protocol):
    """Interface required by discrete latent adapters.

    FSQ-style quantizers expose scalar-code dimensions with FSQ terminology so
    callers do not confuse them with autoregressive prior tokens.
    """

    num_fsq_levels: int
    num_fsq_scalars: int

    def quantize(self, latent: torch.Tensor) -> torch.Tensor: ...

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor: ...

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor: ...


class FSQTokenization(nn.Module):
    """Pack flat FSQ scalar indices into categorical prior tokens.

    FSQ produces one small discrete index per latent scalar. The GPC prior
    consumes fewer categorical tokens by packing several FSQ scalar indices
    into a mixed-radix vocabulary entry.
    """

    def __init__(
        self,
        *,
        num_fsq_levels: int,
        num_fsq_scalars: int,
        fsq_scalars_per_prior_token: int,
    ):
        super().__init__()
        if fsq_scalars_per_prior_token <= 0:
            raise ValueError("fsq_scalars_per_prior_token must be positive")
        if num_fsq_scalars % fsq_scalars_per_prior_token != 0:
            raise ValueError(
                f"fsq_scalars_per_prior_token ({fsq_scalars_per_prior_token}) must evenly divide "
                f"num_fsq_scalars ({num_fsq_scalars})"
            )

        self.num_fsq_levels = num_fsq_levels
        self.num_fsq_scalars = num_fsq_scalars
        self.fsq_scalars_per_prior_token = fsq_scalars_per_prior_token
        self.num_prior_tokens = num_fsq_scalars // fsq_scalars_per_prior_token
        self.prior_token_vocab_size = num_fsq_levels**fsq_scalars_per_prior_token
        self.register_buffer(
            "basis",
            torch.tensor([num_fsq_levels**i for i in range(fsq_scalars_per_prior_token)]),
            persistent=False,
        )

    def fsq_indices_to_prior_tokens(self, fsq_indices: torch.Tensor) -> torch.Tensor:
        """Pack ``(batch, num_fsq_scalars)`` FSQ indices into prior tokens."""
        batch_size = fsq_indices.shape[0]
        fsq_groups = fsq_indices.view(
            batch_size,
            self.num_prior_tokens,
            self.fsq_scalars_per_prior_token,
        )
        basis = self.basis.to(fsq_indices.device)
        return (fsq_groups * basis[None, None, :]).sum(dim=-1)

    def prior_tokens_to_fsq_indices(self, prior_tokens: torch.Tensor) -> torch.Tensor:
        """Unpack prior tokens back to flat FSQ scalar indices."""
        batch_size = prior_tokens.shape[0]
        basis = self.basis.to(prior_tokens.device)

        fsq_indices = torch.zeros(
            batch_size,
            self.fsq_scalars_per_prior_token,
            self.num_prior_tokens,
            device=prior_tokens.device,
            dtype=torch.long,
        )
        remainder = prior_tokens.clone()
        for i in reversed(range(self.fsq_scalars_per_prior_token)):
            fsq_indices[:, i, :] = remainder // basis[i]
            remainder = remainder % basis[i]
        return fsq_indices.permute(0, 2, 1).reshape(batch_size, -1)

    def one_hot_prior_tokens(self, prior_tokens: torch.Tensor) -> torch.Tensor:
        """One-hot encode prior-token indices using the derived vocabulary."""
        return F.one_hot(prior_tokens, self.prior_token_vocab_size).float()


class DiscreteLatentDecoder(nn.Module):
    """Frozen FSQ-code-to-action decoder path."""

    def __init__(
        self,
        *,
        decoder: nn.Module,
        quantizer: DiscreteLatentQuantizer,
        latent_key: str = LATENT_KEY,
        decoder_out_key: str = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.decoder = decoder
        self.quantizer = quantizer
        self.latent_key = latent_key
        self.decoder_out_key = decoder_out_key or self.decoder.out_keys[0]
        if freeze:
            self.freeze()

    @property
    def num_fsq_levels(self) -> int:
        return self.quantizer.num_fsq_levels

    @property
    def num_fsq_scalars(self) -> int:
        return self.quantizer.num_fsq_scalars

    def freeze(self) -> None:
        freeze_module(self.decoder)
        freeze_module(self.quantizer)

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        self.quantizer.eval()
        return self

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.indices_to_codes(indices)

    def decode(self, tensordict: TensorDict, codes: torch.Tensor) -> torch.Tensor:
        tensordict[self.latent_key] = codes
        tensordict = self.decoder(tensordict)
        return tensordict[self.decoder_out_key]


class DiscreteLatentTargetEncoder(ProtoMotionsTensorDictModule):
    """Encode target observations to FSQ indices, then pack prior tokens."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        quantizer: DiscreteLatentQuantizer,
        tokenization: FSQTokenization,
        target_key: str,
        encoder_out_key: str = None,
        freeze: bool = True,
    ):
        ProtoMotionsTensorDictModule.__init__(self)
        self.encoder = encoder
        self.quantizer = quantizer
        self.tokenization = tokenization
        self.target_key = target_key
        self.encoder_out_key = encoder_out_key or self.encoder.out_keys[0]
        self.in_keys = list(self.encoder.in_keys)
        self.out_keys = [target_key]
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        freeze_module(self.encoder)
        freeze_module(self.quantizer)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        self.quantizer.eval()
        return self

    def compute_model_loss(
        self,
        tensordict: TensorDict,
        current_epoch: int,
        zero_loss: torch.Tensor,
        log_prefix: str = "model",
    ):
        if isinstance(self.encoder, ProtoMotionsTensorDictModule):
            return self.encoder.compute_model_loss(
                tensordict,
                current_epoch=current_epoch,
                zero_loss=zero_loss,
                log_prefix=log_prefix,
            )
        return zero_loss * 0.0, {}

    @torch.no_grad()
    def forward(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self.encoder(tensordict)
        codes = self.quantizer.quantize(tensordict[self.encoder_out_key])
        fsq_indices = self.quantizer.codes_to_indices(codes)
        tensordict[self.target_key] = self.tokenization.fsq_indices_to_prior_tokens(
            fsq_indices
        )
        return tensordict


def _require_discrete_latent_attribute(module: nn.Module, attribute: str):
    if not hasattr(module, attribute):
        raise TypeError(
            f"Expected discrete latent module to expose '{attribute}', "
            f"got {type(module).__name__}."
        )
    return getattr(module, attribute)


def make_discrete_latent_decoder(
    module: nn.Module,
    *,
    freeze: bool = True,
) -> DiscreteLatentDecoder:
    """Adapt a module exposing decoder/quantizer to ``DiscreteLatentDecoder``."""
    decoder = _require_discrete_latent_attribute(module, "decoder")
    quantizer = _require_discrete_latent_attribute(module, "quantizer")
    return DiscreteLatentDecoder(
        decoder=decoder,
        quantizer=quantizer,
        latent_key=getattr(module, "latent_key", LATENT_KEY),
        freeze=freeze,
    )


def load_pretrained_discrete_latent_decoder(config, device: torch.device):
    """Load the frozen latent-index-to-action path from a pretrained module."""
    module = load_pretrained_model_module(config, device=device)
    return make_discrete_latent_decoder(module, freeze=config.freeze)


def make_discrete_latent_target_encoder(
    module: nn.Module,
    *,
    tokenization: FSQTokenization,
    target_key: str,
    freeze: bool = True,
) -> DiscreteLatentTargetEncoder:
    """Adapt a module exposing encoder/quantizer to a target-label module."""
    encoder = _require_discrete_latent_attribute(module, "encoder")
    quantizer = _require_discrete_latent_attribute(module, "quantizer")
    return DiscreteLatentTargetEncoder(
        encoder=encoder,
        quantizer=quantizer,
        tokenization=tokenization,
        target_key=target_key,
        freeze=freeze,
    )


def load_pretrained_discrete_latent_target_encoder(
    config,
    *,
    tokenization: FSQTokenization,
    target_key: str,
    device: torch.device,
):
    """Load the frozen observation-to-target-latent path from a pretrained module."""
    module = load_pretrained_model_module(config, device=device)
    return make_discrete_latent_target_encoder(
        module,
        tokenization=tokenization,
        target_key=target_key,
        freeze=config.freeze,
    )
