# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict

from protomotions.utils import torch_utils
from lightning.fabric import Fabric

from protomotions.agents.common.common import get_params


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
        fabric: Fabric instance for distributed operations.
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
        fabric.clip_gradients(
            model,
            optimizer,
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


def aggregate_scalar_metrics(log_dict: Dict, fabric: Fabric) -> Dict:
    """
    Aggregate scalar metrics across all devices using all_gather and mean reduction.

    All ranks compute the same averaged metrics. Then fabric.log_dict() only uploads
    from rank 0 (via Lightning's rank_zero_only pattern), so wandb logs the average
    across all ranks rather than just rank 0's local metrics.
    """
    aggregated_dict = {}

    for key, value in log_dict.items():
        if isinstance(value, (int, float)):
            # Convert to tensor for aggregation
            value_tensor = torch.tensor(
                value, device=fabric.device, dtype=torch.float32
            )
        elif isinstance(value, torch.Tensor):
            # Ensure it's a scalar tensor
            if value.numel() == 1:
                value_tensor = value.float().to(fabric.device)
            else:
                # For non-scalar tensors, take the mean and treat as scalar
                value_tensor = value.mean().float().to(fabric.device)
        else:
            # For non-numeric values, keep as is (no aggregation needed)
            aggregated_dict[key] = value
            continue

        if fabric.world_size > 1:
            # Gather values from all devices
            all_values = fabric.all_gather(value_tensor)
            # Take mean across all devices
            aggregated_value = all_values.mean().item()
        else:
            aggregated_value = value_tensor.item()

        aggregated_dict[key] = aggregated_value

    return aggregated_dict


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
