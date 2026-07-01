# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muon optimizer variants."""

import fnmatch
from collections import defaultdict
from types import SimpleNamespace
from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn


def _matches_any(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def make_muon_param_groups(
    hidden_params,
    adam_params,
    config=None,
    empty_error: str = "Muon optimizer has no trainable parameters",
    *,
    lr: float = 0.02,
    weight_decay: float = 0.0,
    momentum: float = 0.95,
    adam_lr: float = 3e-4,
    adam_betas: Tuple[float, float] = (0.9, 0.95),
    adam_eps: float = 1e-10,
    adam_weight_decay: float = 0.0,
):
    """Build MuonWithAuxAdam parameter groups from split parameter lists."""
    if config is not None:
        lr = config.lr
        weight_decay = config.weight_decay
        momentum = config.momentum
        adam_lr = config.adam_lr
        adam_betas = config.adam_betas if config.adam_betas is not None else config.betas
        adam_eps = config.adam_eps if config.adam_eps is not None else config.eps
        adam_weight_decay = config.adam_weight_decay

    hidden_params = list(hidden_params)
    adam_params = list(adam_params)
    param_groups = []
    if hidden_params:
        param_groups.append(
            {
                "params": hidden_params,
                "use_muon": True,
                "lr": lr,
                "weight_decay": weight_decay,
                "momentum": momentum,
            }
        )
    if adam_params:
        param_groups.append(
            {
                "params": adam_params,
                "use_muon": False,
                "lr": adam_lr,
                "betas": adam_betas,
                "eps": adam_eps,
                "weight_decay": adam_weight_decay,
            }
        )
    if not param_groups:
        raise ValueError(empty_error)
    return param_groups


def split_muon_parameters(
    params,
    param_subset=None,
    adam_fallback_module_patterns=None,
    adam_fallback_parameter_patterns=None,
    use_adam_for_sequential_projections: bool = True,
):
    """Split trainable parameters into Muon-compatible and Adam fallback lists."""
    adam_fallback_module_patterns = adam_fallback_module_patterns or []
    adam_fallback_parameter_patterns = adam_fallback_parameter_patterns or []
    adam_param_ids = set()
    subset_param_ids = None
    if param_subset is not None:
        subset_param_ids = {id(param) for param in param_subset}

    if isinstance(params, nn.Module):
        module = params
        named_parameters = list(module.named_parameters())
        for module_name, child in module.named_modules():
            if (
                module_name
                and _matches_any(module_name, adam_fallback_module_patterns)
            ) or isinstance(child, nn.Embedding):
                adam_param_ids.update(
                    id(param) for param in child.parameters() if param.requires_grad
                )

            declared_modules = getattr(child, "muon_adam_fallback_modules", None)
            if callable(declared_modules):
                for fallback_module in declared_modules():
                    adam_param_ids.update(
                        id(param)
                        for param in fallback_module.parameters()
                        if param.requires_grad
                    )

            if use_adam_for_sequential_projections and isinstance(child, nn.Sequential):
                linears = [
                    submodule
                    for submodule in child.children()
                    if isinstance(submodule, (nn.Linear, nn.LazyLinear))
                ]
                for linear in {linears[0], linears[-1]} if linears else set():
                    adam_param_ids.update(
                        id(param)
                        for param in linear.parameters()
                        if param.requires_grad
                    )
    else:
        named_parameters = [
            (str(index), param) for index, param in enumerate(list(params))
        ]

    hidden_params = []
    adam_params = []
    hidden_names = []
    adam_names = []
    for name, param in named_parameters:
        if subset_param_ids is not None and id(param) not in subset_param_ids:
            continue
        if not param.requires_grad:
            continue
        use_aux = (
            id(param) in adam_param_ids
            or _matches_any(name, adam_fallback_parameter_patterns)
            or param.ndim not in (2, 4)
        )
        if use_aux:
            adam_params.append(param)
            adam_names.append(name)
        else:
            hidden_params.append(param)
            hidden_names.append(name)

    return hidden_params, adam_params, hidden_names, adam_names


def _config_from_kwargs(**kwargs):
    return SimpleNamespace(**kwargs)


def _distributed_rank_world() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _sort_muon_params_by_numel(params):
    """Put the largest Muon matrices first so distributed ownership is balanced."""
    return sorted(params, key=lambda x: x.numel(), reverse=True)


def _sync_updated_params(params, world_size: int) -> None:
    if world_size == 1:
        return
    owner_buckets = defaultdict(list)
    for param_idx, param in enumerate(params):
        owner = param_idx % world_size
        owner_buckets[(owner, param.device, param.dtype)].append(param)

    def bucket_key(item):
        owner, device, dtype = item[0]
        device_index = -1 if device.index is None else device.index
        return owner, device.type, device_index, str(dtype)

    for (owner, _, _), bucket_params in sorted(
        owner_buckets.items(),
        key=bucket_key,
    ):
        flat_params = [param.reshape(-1) for param in bucket_params]
        bucket = torch.cat(flat_params)
        dist.broadcast(bucket, src=owner)

        offset = 0
        for param in bucket_params:
            numel = param.numel()
            param.copy_(bucket[offset : offset + numel].view_as(param))
            offset += numel


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = _sort_muon_params_by_numel(params)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            rank, world_size = _distributed_rank_world()
            params = group["params"]
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            _sync_updated_params(params, world_size)

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Distributed Muon variant with an internal AdamW fallback.

    This optimizer can be used for all parameters in the network, since it runs
    an internal AdamW for the parameters that should not use Muon. When passed a
    module, trainable matrix weights use Muon, while embeddings, biases,
    normalization parameters, and declared input/output projections use Adam.
    Advanced users can still pass explicit parameter groups with the ``use_muon``
    flag set.

    This class lets callers use one optimizer instead of stepping separate Muon
    and Adam optimizers.

    Example:

        .. code-block:: python

           hidden_matrix_params = [
               p for n, p in model.blocks.named_parameters()
               if p.ndim >= 2 and "embed" not in n
           ]
           embed_params = [
               p for n, p in model.named_parameters()
               if "embed" in n
           ]
           scalar_params = [p for p in model.parameters() if p.ndim < 2]
           head_params = [model.lm_head.weight]

           adam_groups = [
               dict(params=head_params, lr=0.22),
               dict(params=embed_params, lr=0.6),
               dict(params=scalar_params, lr=0.04),
           ]
           adam_groups = [
               dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False)
               for g in adam_groups
           ]
           muon_group = dict(
               params=hidden_matrix_params,
               lr=0.05,
               momentum=0.95,
               use_muon=True,
           )
           optimizer = MuonWithAuxAdam([*adam_groups, muon_group])
    """
    accepts_module_params = True

    def __init__(
        self,
        param_groups=None,
        *,
        params=None,
        param_subset=None,
        lr=0.02,
        weight_decay=0.0,
        momentum=0.95,
        betas=(0.9, 0.95),
        eps=1e-10,
        adam_lr=3e-4,
        adam_betas=None,
        adam_eps=None,
        adam_weight_decay=0.0,
        adam_fallback_module_patterns=None,
        adam_fallback_parameter_patterns=None,
        use_adam_for_sequential_projections=True,
    ):
        adam_betas = betas if adam_betas is None else adam_betas
        adam_eps = eps if adam_eps is None else adam_eps

        if params is not None:
            hidden_params, adam_params, hidden_names, adam_names = split_muon_parameters(
                params,
                param_subset=param_subset,
                adam_fallback_module_patterns=adam_fallback_module_patterns,
                adam_fallback_parameter_patterns=adam_fallback_parameter_patterns,
                use_adam_for_sequential_projections=use_adam_for_sequential_projections,
            )
            config = _config_from_kwargs(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                betas=adam_betas,
                eps=adam_eps,
                adam_lr=adam_lr,
                adam_betas=adam_betas,
                adam_eps=adam_eps,
                adam_weight_decay=adam_weight_decay,
            )
            param_groups = make_muon_param_groups(
                hidden_params,
                adam_params,
                config=config,
            )
            self.hidden_param_names = hidden_names
            self.adam_param_names = adam_names
        elif param_groups is None:
            raise ValueError("MuonWithAuxAdam requires params or param_groups")

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = _sort_muon_params_by_numel(group["params"])
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                rank, world_size = _distributed_rank_world()
                params = group["params"]
                for base_i in range(0, len(params), world_size):
                    if base_i + rank < len(params):
                        p = params[base_i + rank]
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                _sync_updated_params(params, world_size)
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
