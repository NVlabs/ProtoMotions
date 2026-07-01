# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite scalar quantization modules."""

import torch
from tensordict import TensorDict
from torch import nn

from protomotions.agents.common.autoencoder import AutoEncoder
from protomotions.agents.common.fsq_config import FSQAutoEncoderConfig


class FiniteScalarQuantizer(nn.Module):
    """Finite scalar quantizer for continuous latent vectors.

    This implements the fixed-codebook scalar quantizer from
    "Finite Scalar Quantization: VQ-VAE Made Simple"
    (Mentzer et al., 2023, arXiv:2309.15505). Each latent dimension is
    independently bounded and rounded to one of ``num_fsq_levels`` integer codes,
    so the implicit codebook is the Cartesian product of the per-dimension
    scalar levels.

    The quantizer intentionally stays a plain ``nn.Module``: it owns buffers
    and tensor transforms only. The TensorDict/model contract is provided by
    ``FSQAutoEncoder``, which wraps this module inside an autoencoder
    bottleneck.
    """

    def __init__(self, num_fsq_levels: int, num_fsq_scalars: int, eps: float = 1e-4):
        """Create a scalar quantizer with one shared level count per scalar.

        Args:
            num_fsq_levels: Number of discrete scalar values available to each
                latent scalar. This implementation requires an odd value so
                zero is one of the quantization levels.
            num_fsq_scalars: Number of scalar code dimensions in the flattened latent.
            eps: Small shrink factor used by the tanh bounding transform to
                avoid saturating exactly at the outermost level before
                straight-through rounding.
        """
        super().__init__()
        if num_fsq_levels % 2 == 0:
            raise ValueError("FSQ requires an odd number of quantization levels")

        levels = torch.full((num_fsq_scalars,), num_fsq_levels, dtype=torch.float32)
        half_l = (levels - 1) * (1 - eps) / 2
        half_width = (levels.long() // 2).to(torch.float32)

        self.num_fsq_levels = num_fsq_levels
        self.num_fsq_scalars = num_fsq_scalars
        self.register_buffer("L", levels * (half_l / half_width), persistent=False)
        self.register_buffer("half_width", half_width, persistent=False)
        self.register_buffer("half_L", self.L / 2, persistent=False)

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Map unbounded latent values into the relaxed FSQ code range."""
        return z.tanh() * self.half_L.unsqueeze(0).to(z.device)

    @staticmethod
    def round_ste(z: torch.Tensor) -> torch.Tensor:
        """Round with the straight-through estimator used for FSQ training."""
        zhat = z.round()
        return z + (zhat - z).detach()

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Bound and straight-through round latent values to scalar codes."""
        return self.round_ste(self.bound(z))

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert centered scalar codes to non-negative FSQ indices."""
        codes = codes + self.half_width.unsqueeze(0).to(codes.device)
        return torch.round(codes).long()

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert non-negative FSQ indices back to centered scalar codes."""
        return indices.float() - self.half_width.unsqueeze(0).to(indices.device)

    def calculate_perplexity(self, codes: torch.Tensor, skip: bool = False):
        """Return mean per-FSQ-scalar perplexity for utilization logging."""
        if skip:
            return torch.tensor(0.0, device=codes.device)

        indices = self.codes_to_indices(codes.view(codes.shape[0], self.num_fsq_scalars))
        perplexities = []
        for scalar_idx in range(self.num_fsq_scalars):
            scalar_indices = indices[:, scalar_idx].flatten().long()
            counts = torch.bincount(scalar_indices, minlength=self.num_fsq_levels).float()
            probs = counts / counts.sum().clamp_min(1.0)
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            perplexities.append(torch.exp(entropy))
        return torch.stack(perplexities).mean()


class FSQAutoEncoder(AutoEncoder):
    """Autoencoder with a finite scalar quantization bottleneck."""

    supports_log_internals = True
    config: FSQAutoEncoderConfig

    def __init__(self, config: FSQAutoEncoderConfig):
        super().__init__(config)
        self._validate_encoder_output_dim(config.num_fsq_scalars)
        self.quantizer = FiniteScalarQuantizer(
            config.num_fsq_levels, config.num_fsq_scalars
        )

    def _validate_encoder_output_dim(self, num_fsq_scalars: int):
        """Fail early when the encoder cannot feed the FSQ bottleneck.

        FSQ quantizes one scalar per flattened latent dimension. If the encoder produces a
        different width, the error would otherwise surface later as an opaque
        broadcast failure inside the quantizer.
        """
        encoder_output_dim = self._configured_output_dim(
            self.config.encoder, self.encoder_out_keys[0]
        )
        if encoder_output_dim is None:
            return
        if encoder_output_dim != num_fsq_scalars:
            raise ValueError(
                "FSQAutoEncoder encoder output dim "
                f"({encoder_output_dim}) must match num_fsq_scalars "
                f"({num_fsq_scalars})."
            )

    @classmethod
    def _configured_output_dim(cls, config, out_key: str):
        """Best-effort output-width lookup from module config objects."""
        if hasattr(config, "num_out") and out_key in getattr(config, "out_keys", []):
            return config.num_out

        models = getattr(config, "models", None)
        if not models:
            return None

        for model_config in reversed(models):
            model_out_keys = getattr(model_config, "out_keys", [])
            if out_key in model_out_keys:
                return cls._configured_output_dim(model_config, out_key)
        return cls._configured_output_dim(models[-1], out_key)

    @property
    def num_fsq_levels(self) -> int:
        return self.quantizer.num_fsq_levels

    @property
    def num_fsq_scalars(self) -> int:
        return self.quantizer.num_fsq_scalars

    @property
    def L(self) -> torch.Tensor:
        return self.quantizer.L

    @property
    def half_width(self) -> torch.Tensor:
        return self.quantizer.half_width

    @property
    def half_L(self) -> torch.Tensor:
        return self.quantizer.half_L

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        return self.quantizer.bound(z)

    @staticmethod
    def round_ste(z: torch.Tensor) -> torch.Tensor:
        return FiniteScalarQuantizer.round_ste(z)

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        return self.quantizer.codes_to_indices(codes)

    def indices_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        return self.quantizer.indices_to_codes(indices)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        return self.quantizer.quantize(z)

    def calculate_perplexity(self, codes: torch.Tensor, skip: bool = False):
        return self.quantizer.calculate_perplexity(codes, skip=skip)

    def bottleneck(
        self,
        latent: torch.Tensor,
        tensordict: TensorDict,
    ) -> torch.Tensor:
        return self.quantize(latent)

    def internal_logs(
        self,
        latent: torch.Tensor,
        tensordict: TensorDict,
    ):
        perplexity = self.calculate_perplexity(latent)
        return {
            "perplexity": perplexity.expand(tensordict.batch_size).clone(),
        }
