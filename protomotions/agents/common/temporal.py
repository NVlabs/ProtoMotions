# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time-aligned stop-gradient targets, formed before rollout flattening."""

import torch


def adjacent_targets(values: torch.Tensor, dones: torch.Tensor, *, forward: bool):
    """Pair adjacent frames of each environment, never adjacent flattened rows.

    Inputs are values [T, E, ...] and dones [T, E]. A done marks the reset
    after action t, including timeouts. Return detached targets of the same
    shape and a boolean [T, E] mask:

    * forward=False: target[t] = values[t-1], valid[t] = not dones[t-1].
      Used to penalize changes from the previous rollout output.
    * forward=True: target[t] = values[t+1], valid[t] = not dones[t].
      Used to supervise predictions of the next condition.

    The first/last row respectively is invalid: we do not pair across rollout
    buffers. Invalid targets are zeroed so cross-episode values never enter the
    batch.
    """
    if dones.ndim != 2 or values.ndim < 3 or values.shape[:2] != dones.shape:
        raise ValueError(
            "Expected values [time, env, features...] and dones [time, env]"
        )
    target = torch.zeros_like(values)
    valid = torch.zeros_like(dones, dtype=torch.bool)
    if forward:
        target[:-1] = values[1:].detach()
        valid[:-1] = ~dones[:-1].bool()
    else:
        target[1:] = values[:-1].detach()
        valid[1:] = ~dones[:-1].bool()
    mask = valid.reshape(*valid.shape, *([1] * (values.ndim - 2)))
    return target.masked_fill(~mask, 0), valid


def masked_mse(prediction, target, valid):
    """MSE for flattened/minibatched rows [N, ...] and validity [N].

    Average over features, then valid rows. Targets are stop-gradient rollout
    values. This is not a physical derivative: there is no dt scaling. An
    entirely invalid batch returns a differentiable zero with zero gradients.
    A non-finite prediction propagates; it is never masked away.
    """
    if prediction.shape != target.shape or valid.shape != prediction.shape[:1]:
        raise ValueError(
            "Temporal prediction/target shapes or transition mask do not match"
        )
    per_row = (prediction - target).square().flatten(start_dim=1).mean(dim=1)
    return (per_row * valid).sum() / valid.sum().clamp_min(1)
