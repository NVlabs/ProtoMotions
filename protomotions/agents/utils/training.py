# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training utility functions for agent training.

This module provides helper functions used during agent training including
gradient clipping, bounds loss, distributed metrics aggregation, and model utilities.

Key Functions:
    - bounds_loss: Penalize actions near joint limits
    - handle_model_grad_clipping: Clip gradients and handle bad gradients
    - aggregate_scalar_metrics: Aggregate metrics across distributed processes
    - get_activation_func: Get activation function by name
"""

import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict

from protomotions.utils import torch_utils
from lightning.fabric import Fabric

from protomotions.agents.common.common import get_params


def _numeric_metric_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.float().item()
        return value.mean().float().item()
    return None


def _distributed_metric_key_union(local_keys, fabric: Fabric):
    payload = "\n".join(sorted(local_keys)).encode("utf-8")
    length = torch.tensor([len(payload)], device=fabric.device, dtype=torch.int64)
    gathered_lengths = [torch.empty_like(length) for _ in range(fabric.world_size)]
    dist.all_gather(gathered_lengths, length)
    lengths = [int(item.cpu().item()) for item in gathered_lengths]
    max_length = max(lengths) if lengths else 0

    local_bytes = torch.zeros(max_length, device=fabric.device, dtype=torch.uint8)
    if payload:
        local_bytes[: len(payload)] = torch.tensor(
            list(payload), device=fabric.device, dtype=torch.uint8
        )
    gathered_bytes = [torch.empty_like(local_bytes) for _ in range(fabric.world_size)]
    dist.all_gather(gathered_bytes, local_bytes)

    union_keys = set()
    for byte_tensor, payload_length in zip(gathered_bytes, lengths):
        if payload_length == 0:
            continue
        encoded = bytes(byte_tensor[:payload_length].cpu().tolist())
        union_keys.update(key for key in encoded.decode("utf-8").split("\n") if key)
    return sorted(union_keys)


def _aggregate_scalar_metrics_tensor_only(
    log_dict: Dict, fabric: Fabric, weight: int = 1
) -> Dict:
    union_keys = _distributed_metric_key_union(log_dict.keys(), fabric)
    aggregated_dict = {}

    value_sums = torch.zeros(len(union_keys), device=fabric.device, dtype=torch.float32)
    weight_sums = torch.zeros_like(value_sums)
    for index, key in enumerate(union_keys):
        if key not in log_dict:
            continue
        numeric_value = _numeric_metric_value(log_dict[key])
        if numeric_value is None:
            aggregated_dict[key] = log_dict[key]
            continue
        value_sums[index] = numeric_value * float(weight)
        weight_sums[index] = float(weight)

    if len(union_keys) > 0:
        dist.all_reduce(value_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(weight_sums, op=dist.ReduceOp.SUM)

    for index, key in enumerate(union_keys):
        if weight_sums[index].item() > 0:
            aggregated_dict[key] = (value_sums[index] / weight_sums[index]).item()

    return aggregated_dict


def bounds_loss(mu: Tensor) -> Tensor:
    """Compute soft bounds loss for actions near limits.

    Penalizes actions that exceed soft bounds (±1.0) to keep actions within
    reasonable ranges and prevent extreme joint angles.

    Args:
        mu: Action means from policy (batch_size, action_dim).

    Returns:
        Bounds loss for each sample (batch_size,). Zero if within bounds,
        quadratic penalty beyond soft bounds.

    Example:
        >>> actions = policy(obs)
        >>> loss = bounds_loss(actions)  # Penalize extreme actions
    """
    soft_bound = 1.0
    mu_loss_high = (
        torch.maximum(mu - soft_bound, torch.tensor(0, device=mu.device)) ** 2
    )
    mu_loss_low = torch.minimum(mu + soft_bound, torch.tensor(0, device=mu.device)) ** 2
    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
    return b_loss


def handle_model_grad_clipping(config, fabric, model, optimizer, model_name):
    """Handle gradient clipping and detect bad gradients.

    Computes gradient norm, clips if configured, and checks for NaN or
    extremely large gradients. Optionally zeros gradients if bad.

    Args:
        config: Agent config with grad clipping settings.
        fabric: Fabric instance for distributed operations, or None for models
            that use their own process group (e.g., per-group DDP). When None,
            gradient clipping uses ``torch.nn.utils.clip_grad_norm_`` directly.
        model: Neural network model.
        optimizer: Optimizer for the model.
        model_name: Name for logging (e.g., "actor", "critic").

    Returns:
        Dictionary with gradient norm metrics for logging.

    Note:
        If bad gradients detected and fail_on_bad_grads=True, raises assertion error.
    """
    params = get_params(list(model.parameters()))
    grad_norm_before_clip = torch_utils.grad_norm(params)
    if config.check_grad_mag:
        bad_grads = (
            torch.isnan(grad_norm_before_clip) or grad_norm_before_clip > 1000000.0
        )
    else:
        bad_grads = torch.isnan(grad_norm_before_clip)

    bad_grads_count = 0
    # fix: bad-grad agreement is LOCAL-ONLY. The explicit
    # dist.all_reduce(MAX) that used to sit here (added 5433ca3, made
    # unconditional 7a366a0) is redundant under DDP: gradients are already
    # bucket-all-reduced to identical values on every rank by the time
    # backward returns, so torch.isnan(grad_norm) is rank-uniform by
    # construction. Worse, the collective sat in the most dangerous window
    # (the first CPU-blocking sync after backward): py-spy captures
    # (gpu2255_stall1_pyspy_20260707.txt, ddp7_stall_pyspy_20260707.txt)
    # show every healthy rank spinning at exactly this point while a
    # reducer-starved rank was parked in backward. Removing the collective
    # removes a mid-step cross-rank dependency without changing behavior.
    # (For fabric=None / per-group-DDP callers this was already local.)

    if bad_grads:
        if config.fail_on_bad_grads:
            all_params = torch.cat(
                [p.grad.view(-1) for p in params if p.grad is not None],
                dim=0,
            )
            raise ValueError(
                f"NaN gradient in {model_name}"
                + f" {all_params.isfinite().logical_not().float().mean().item()}"
                + f" {all_params.abs().min().item()}"
                + f" {all_params.abs().max().item()}"
                + f" {grad_norm_before_clip.item()}"
            )
        else:
            bad_grads_count = 1
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

    if config.gradient_clip_val > 0:
        torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=config.gradient_clip_val,
            error_if_nonfinite=True,
        )
    grad_norm_after_clip = torch_utils.grad_norm(params)
    clip_dict = {
        f"{model_name}/grad_norm_before_clip": grad_norm_before_clip.detach(),
        f"{model_name}/grad_norm_after_clip": grad_norm_after_clip.detach(),
        f"{model_name}/bad_grads_count": bad_grads_count,
    }
    return clip_dict


def aggregate_scalar_metrics(log_dict: Dict, fabric: Fabric, weight: int = 1) -> Dict:
    """
    Aggregate scalar metrics across all devices using weighted mean reduction.

    Each rank contributes ``weight`` (typically ``num_envs``) to the weighted
    average.  When all ranks have the same weight this reduces to a simple mean.

    All ranks compute the same averaged metrics. Then fabric.log_dict() only uploads
    from rank 0 (via Lightning's rank_zero_only pattern), so wandb logs the average
    across all ranks rather than just rank 0's local metrics.
    """
    aggregated_dict = {}

    if fabric.world_size == 1:
        # Single-GPU: no collectives, behavior-preserving.
        for key, value in log_dict.items():
            numeric_value = _numeric_metric_value(value)
            if numeric_value is not None:
                aggregated_dict[key] = numeric_value
            else:
                aggregated_dict[key] = value
        return aggregated_dict

    return _aggregate_scalar_metrics_tensor_only(log_dict, fabric, weight=weight)


def get_activation_func(activation_name, return_type="nn"):
    """Get activation function by name.

    Returns either nn.Module or functional version of the activation.
    Supports common activations: tanh, relu, elu, gelu, silu, mish, identity.

    Args:
        activation_name: Name of activation function (case-insensitive).
        return_type: Either "nn" for nn.Module or "functional" for functional version.

    Returns:
        Activation function (nn.Module if return_type="nn", function if return_type="functional").

    Raises:
        NotImplementedError: If activation name or return type not recognized.

    Example:
        >>> act_module = get_activation_func("relu", return_type="nn")
        >>> act_func = get_activation_func("relu", return_type="functional")
    """
    if activation_name.lower() == "tanh":
        activation = (nn.Tanh(), F.tanh)
    elif activation_name.lower() == "relu":
        activation = (nn.ReLU(), F.relu)
    elif activation_name.lower() == "elu":
        activation = (nn.ELU(), F.elu)
    elif activation_name.lower() == "gelu":
        activation = (nn.GELU(), F.gelu)
    elif activation_name.lower() == "identity":
        activation = (nn.Identity(), lambda x: x)
    elif activation_name.lower() == "silu":
        activation = (nn.SiLU(), F.silu)
    elif activation_name.lower() == "mish":
        activation = (nn.Mish(), F.mish)
    else:
        raise NotImplementedError(
            "Activation func {} not defined".format(activation_name)
        )

    if return_type == "nn":
        return activation[0]
    elif return_type == "functional":
        return activation[1]
    else:
        raise NotImplementedError("Return type {} not implemented".format(return_type))
