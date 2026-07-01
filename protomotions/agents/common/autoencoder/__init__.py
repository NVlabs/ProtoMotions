# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared encoder-bottleneck-decoder TensorDict modules.

This file intentionally stays narrow. ``AutoEncoder`` is useful for models
that really are encoder -> bottleneck -> decoder, such as FSQ trackers and
small reconstruction-style students. Autoregressive GPC priors are not
autoencoders and should use their own ``BaseModel`` implementation instead.
"""

from typing import Dict

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase

from protomotions.agents.base_agent.model import BaseModel
from protomotions.agents.common.common import MODULE_INTERNALS_KEY
from protomotions.agents.common.autoencoder.config import AutoEncoderConfig
from protomotions.utils.hydra_replacement import get_class


class AutoEncoder(BaseModel):
    """Generic encoder-bottleneck-decoder module.

    Subclasses customize only the bottleneck behavior. For example, an FSQ
    tracker quantizes the encoder latent before decoding. Models with a
    different shape, such as causal token priors, should not inherit this class.
    """

    supports_log_internals = False
    config: AutoEncoderConfig

    def __init__(self, config: AutoEncoderConfig):
        super().__init__(config)
        self.config = config

        encoder_class = get_class(config.encoder._target_)
        decoder_class = get_class(config.decoder._target_)
        self.encoder: TensorDictModuleBase = encoder_class(config=config.encoder)
        self.decoder: TensorDictModuleBase = decoder_class(config=config.decoder)

        self.encoder_out_keys = list(config.encoder_out_keys or self.encoder.out_keys)
        self.decoder_out_keys = list(config.decoder_out_keys or self.decoder.out_keys)
        self.latent_key = config.latent_key

        latent_keys = set(self.encoder_out_keys + [self.latent_key])
        self.in_keys = list(
            dict.fromkeys(
                list(self.encoder.in_keys)
                + [key for key in self.decoder.in_keys if key not in latent_keys]
            )
        )
        self.out_keys = list(self.decoder_out_keys)

    def bottleneck(
        self,
        latent: torch.Tensor,
        tensordict: TensorDict,
    ) -> torch.Tensor:
        return latent

    def internal_logs(
        self,
        latent: torch.Tensor,
        tensordict: TensorDict,
    ) -> Dict[str, torch.Tensor]:
        return {}

    def predict_latent(self, tensordict: TensorDict) -> torch.Tensor:
        tensordict = self.encoder(tensordict)
        encoder_out = tensordict[self.encoder_out_keys[0]]
        return self.bottleneck(encoder_out, tensordict)

    def decode(
        self,
        tensordict: TensorDict,
        latent: torch.Tensor,
    ) -> TensorDict:
        tensordict[self.latent_key] = latent
        return self.decoder(tensordict)

    def forward(
        self,
        tensordict: TensorDict,
        log_internals: bool = False,
    ) -> TensorDict:
        latent = self.predict_latent(tensordict)
        tensordict = self.decode(tensordict, latent)

        if log_internals:
            logs = self.internal_logs(latent.detach(), tensordict)
            if logs:
                tensordict[MODULE_INTERNALS_KEY] = TensorDict(
                    logs,
                    batch_size=tensordict.batch_size,
                )

        return tensordict
