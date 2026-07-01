# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conditioned LoRA/DoRA adapters for TransformerEncoder stacks.

This module is model-agnostic: it only knows how to wrap transformer layers,
freeze base weights, and expose adapter parameters for training.
"""

import torch
from torch import nn


# ---------------------------------------------------------------------------
# LoRA / DoRA layer
# ---------------------------------------------------------------------------


class LoRALayer(nn.Module):
    """Low-rank adaptation with FiLM-style gating from a conditioning vector."""

    def __init__(
        self,
        c_dim: int,
        in_dim: int,
        out_dim: int,
        rank: int,
        alpha: float,
    ):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) / (in_dim ** 0.5))
        self.B = nn.Parameter(torch.randn(rank, out_dim) * 1e-4)
        self.scaling = alpha / rank

    def forward(self, x, c_mul, c_add):
        xa = x @ self.A  # (B, T, rank)
        xa_gated = xa * c_mul.unsqueeze(1)  # FiLM multiplicative
        xa_gated = xa_gated + c_add.unsqueeze(1)  # FiLM additive
        return self.scaling * (xa_gated @ self.B)  # (B, T, out_dim)


class TransformerLayerWithLoRA(nn.Module):
    def __init__(self, transformer_layer, c_dim: int, rank: int, alpha: float):
        super().__init__()
        self.transformer_layer = transformer_layer
        d_model = transformer_layer.self_attn.embed_dim
        self.lora = LoRALayer(c_dim, d_model, d_model, rank, alpha)
        self.gamma = nn.Linear(c_dim, rank)
        self.beta = nn.Linear(c_dim, rank)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)

        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, c, x, **kwargs):
        c_mul = self.gamma(c)
        c_add = self.beta(c)
        return self.transformer_layer(x, **kwargs) + self.lora(x, c_mul, c_add)


class TransformerLayerWithDoRA(nn.Module):
    def __init__(self, transformer_layer, c_dim: int, rank: int, alpha: float):
        super().__init__()
        self.transformer_layer = transformer_layer
        d_model = transformer_layer.self_attn.embed_dim
        self.lora = LoRALayer(c_dim, d_model, d_model, rank, alpha)
        self.m = nn.Parameter(torch.empty(1, d_model).uniform_(-1e-5, 1e-5))
        self.gamma = nn.Linear(c_dim, rank)
        self.beta = nn.Linear(c_dim, rank)
        nn.init.zeros_(self.gamma.weight)
        nn.init.ones_(self.gamma.bias)

        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, c, x, **kwargs):
        transformer_output = self.transformer_layer(x, **kwargs)
        c_mul = self.gamma(c)
        c_add = self.beta(c)
        lora_output = self.lora(x, c_mul, c_add)
        lora_output_norm = lora_output / (
            lora_output.norm(p=2, dim=-1, keepdim=True) + 1e-6
        )
        return transformer_output + self.m * lora_output_norm


# ---------------------------------------------------------------------------
# Conditioning-aware transformer encoder
# ---------------------------------------------------------------------------


class TransformerEncoderWithConditioning(nn.Module):
    """Passes task conditioning to PEFT layers, standard forward to others."""

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x, task_c=None, mask=None, src_key_padding_mask=None):
        for layer in self.layers:
            if hasattr(layer, "lora"):
                x = layer(
                    task_c,
                    x,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                )
            else:
                x = layer(x, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return x


def freeze_base_and_enable_peft(module: nn.Module) -> None:
    """Freeze base weights and leave adapter/conditioning parameters trainable."""
    for parameter in module.parameters():
        parameter.requires_grad = False

    for submodule in module.modules():
        if hasattr(submodule, "lora"):
            for parameter in submodule.lora.parameters():
                parameter.requires_grad = True
            for parameter in submodule.gamma.parameters():
                parameter.requires_grad = True
            for parameter in submodule.beta.parameters():
                parameter.requires_grad = True
        if hasattr(submodule, "m"):
            submodule.m.requires_grad = True
        if hasattr(submodule, "transformer_layer"):
            submodule.transformer_layer.eval()


def set_peft_layers_train_mode(module: nn.Module, mode: bool) -> None:
    """Set PEFT wrappers to train/eval while keeping wrapped base layers eval."""
    for submodule in module.modules():
        if not hasattr(submodule, "lora"):
            continue
        submodule.train(mode)
        if hasattr(submodule, "transformer_layer"):
            submodule.transformer_layer.eval()


def inject_transformer_peft(
    transformer: nn.TransformerEncoder,
    conditioning_dim: int,
    rank: int,
    alpha: float,
    peft_type: str,
) -> TransformerEncoderWithConditioning:
    """Wrap TransformerEncoder layers with conditioned LoRA/DoRA adapters."""
    layer_cls = (
        TransformerLayerWithDoRA
        if peft_type == "dora"
        else TransformerLayerWithLoRA
    )

    for index, layer in enumerate(transformer.layers):
        peft_layer = layer_cls(layer, conditioning_dim, rank, alpha)
        peft_layer = peft_layer.to(next(layer.parameters()).device)
        transformer.layers[index] = peft_layer

    wrapped = TransformerEncoderWithConditioning(transformer.layers)
    freeze_base_and_enable_peft(wrapped)
    return wrapped
