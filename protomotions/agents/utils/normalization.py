# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Running mean and standard deviation computation for normalization.

This module provides efficient online computation of mean and variance statistics
for observation and reward normalization in reinforcement learning. Uses Welford's
algorithm with distributed training support.

Key Classes:
    - RunningMeanStd: Computes running statistics with optional clamping
    - RewardRunningMeanStd: Specialized for reward normalization with discount factor

Key Features:
    - Online updates (no need to store all data)
    - Distributed training support (aggregates across processes)
    - Optional value clamping for stability
    - State dict support for checkpointing
"""

from typing import Optional, Tuple, List

import torch
import torch.distributed as dist
from torch import Tensor, nn
from lightning.fabric import Fabric


class RunningMeanStd(nn.Module):
    """Running mean and standard deviation computation.

    Computes and maintains running statistics (mean, variance, count) for data streams.
    Uses Welford's online algorithm extended for parallel/distributed computation.
    Commonly used for normalizing observations and rewards in RL.

    Args:
        fabric: Lightning Fabric instance for distributed aggregation.
        shape: Shape of the data being normalized.
        epsilon: Small constant for numerical stability.
        device: PyTorch device for tensors.
        clamp_value: Optional clipping value for normalized outputs.

    Attributes:
        mean: Running mean (float64 for precision).
        var: Running variance (float64 for precision).
        count: Number of samples seen.

    Example:
        >>> rms = RunningMeanStd(fabric, shape=(128,), device="cuda")
        >>> rms.record_moments(observations)
        >>> normalized_obs = rms.normalize(new_observations)

    References:
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self,
        fabric: Fabric,
        shape: Optional[Tuple[int, ...]] = None,
        epsilon: int = 1e-5,
        device="cuda:0",
        clamp_value: Optional[float] = None,
        ema_decay: Optional[float] = None,
    ):
        """Initialize running statistics tracker with optional lazy initialization.

        Args:
            fabric: Lightning Fabric for distributed training.
            shape: Shape of data to normalize. If None, will be inferred on first forward pass.
            epsilon: Numerical stability constant.
            device: PyTorch device.
            clamp_value: Optional value for clamping normalized outputs.
            ema_decay: If set, use EMA instead of Welford's algorithm.
                None (default) = Welford's (all-time statistics, count grows unbounded).
                Float in (0, 1) = EMA decay factor (e.g. 0.999 tracks ~1000-sample window).
                Checkpoint-compatible: same mean/var buffers, count is ignored in EMA mode.
        """
        super().__init__()
        self.fabric = fabric
        self.epsilon = epsilon
        self.clamp_value = clamp_value
        self.ema_decay = ema_decay
        self.shape = shape
        self.device = device
        self._initialized = False

        # If shape is provided, initialize buffers immediately
        if shape is not None:
            self._create_buffers(shape, device)
            self._initialized = True

    def _create_buffers(self, shape: Tuple[int, ...], device):
        """Create the buffers for mean, var, and count."""
        self.register_buffer(
            "mean", torch.zeros(shape, dtype=torch.float64, device=device)
        )
        self.register_buffer(
            "var", torch.ones(shape, dtype=torch.float64, device=device)
        )
        self.register_buffer("count", torch.ones((), dtype=torch.long, device=device))
        self.shape = shape

    def _lazy_init(self, x: Tensor):
        """Lazy initialization from first input tensor.

        Called on first forward pass if shape was not provided at construction.
        Also called after load_state_dict to mark as initialized.
        """
        if not self._initialized:
            # Infer shape from input (exclude batch dimension)
            inferred_shape = x.shape[1:]
            if self.shape is None:
                self.shape = inferred_shape
                self._create_buffers(inferred_shape, x.device)
            self._initialized = True

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Hook called when loading state dict - mark as initialized if buffers exist."""
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # If we loaded mean/var/count buffers, we're initialized
        if f"{prefix}mean" in state_dict:
            self._initialized = True
            # Update shape from loaded buffer
            if hasattr(self, "mean"):
                self.shape = self.mean.shape

    def to(self, device):
        # Call parent's .to() method to handle registered buffers properly
        super().to(device)
        self.device = device
        return self

    @torch.no_grad()
    def update_from_moments(
        self, batch_mean: torch.tensor, batch_var: torch.tensor, batch_count: int
    ) -> None:
        if self.ema_decay is not None:
            d = self.ema_decay
            self.mean[:] = d * self.mean + (1 - d) * batch_mean
            self.var[:] = d * self.var + (1 - d) * batch_var
            return

        new_mean, new_var, new_count = combine_moments(
            [self.mean, batch_mean], [self.var, batch_var], [self.count, batch_count]
        )

        self.mean[:] = new_mean
        self.var[:] = new_var
        self.count.fill_(new_count)

    def maybe_clamp(self, x: Tensor):
        if self.clamp_value is None:
            return x
        else:
            return torch.clamp(x, -self.clamp_value, self.clamp_value)

    def normalize(self, arr: torch.tensor, un_norm=False) -> torch.tensor:
        # Lazy initialization if needed
        self._lazy_init(arr)

        if not un_norm:
            result = (arr - self.mean.float()) / torch.sqrt(
                self.var.float() + self.epsilon
            )
            result = self.maybe_clamp(result)
        else:
            arr = self.maybe_clamp(arr)
            result = (
                arr * torch.sqrt(self.var.float() + self.epsilon) + self.mean.float()
            )

        return result

    @torch.no_grad()
    def record_moments(self, arr: torch.tensor) -> None:
        """Record moments from a batch of data during rollout collection."""
        # Lazy initialization if needed
        self._lazy_init(arr)

        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0, unbiased=False)
        batch_count = arr.shape[0]

        if self.fabric is not None and self.fabric.world_size > 1:
            self.update_from_moments(batch_mean, batch_var, batch_count)
            return

        self.update_from_moments(batch_mean, batch_var, batch_count)


@torch.no_grad()
def sync_running_mean_std_modules(
    named_modules: List[Tuple[str, RunningMeanStd]],
    fabric,
) -> None:
    """Synchronize RunningMeanStd buffers at a rank-uniform boundary.

    The caller owns ordering and must call this from the same training-loop
    boundary on every rank. The implementation uses tensor collectives only:
    no object collectives and no Fabric broadcast wrappers.
    """
    if fabric is None or getattr(fabric, "world_size", 1) <= 1:
        return
    if not (dist.is_available() and dist.is_initialized()):
        return

    world_size = dist.get_world_size()
    for name, module in named_modules:
        initialized = torch.tensor(
            int(module._initialized),
            device=fabric.device,
            dtype=torch.long,
        )
        initialized_count = initialized.clone()
        dist.all_reduce(initialized_count, op=dist.ReduceOp.SUM)
        initialized_total = int(initialized_count.item())
        if initialized_total == 0:
            continue
        if initialized_total != world_size:
            raise RuntimeError(
                f"RunningMeanStd '{name}' initialized on {initialized_total}/"
                f"{world_size} ranks; lazy normalizer materialization must be "
                "rank-consistent before synchronization."
            )

        mean = module.mean.to(dtype=torch.float64)
        var = module.var.to(dtype=torch.float64)
        count = module.count.to(device=module.mean.device, dtype=torch.float64)

        if module.ema_decay is not None:
            synced_mean = mean.clone()
            synced_var = var.clone()
            dist.all_reduce(synced_mean, op=dist.ReduceOp.SUM)
            dist.all_reduce(synced_var, op=dist.ReduceOp.SUM)
            synced_mean /= world_size
            synced_var /= world_size
            synced_count = count.clone()
            dist.all_reduce(synced_count, op=dist.ReduceOp.MAX)
        else:
            total_count = count.clone()
            total_sum = mean * count
            total_sumsq = (var + mean.square()) * count
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_sumsq, op=dist.ReduceOp.SUM)

            safe_count = torch.clamp(total_count, min=1.0)
            synced_mean = total_sum / safe_count
            synced_var = torch.clamp(
                total_sumsq / safe_count - synced_mean.square(), min=0.0
            )
            synced_count = total_count

        module.mean.copy_(
            synced_mean.to(device=module.mean.device, dtype=module.mean.dtype)
        )
        module.var.copy_(synced_var.to(device=module.var.device, dtype=module.var.dtype))
        module.count.copy_(
            synced_count.to(device=module.count.device, dtype=module.count.dtype)
        )


def materialize_lazy_running_stats_from_state_dict(
    model: nn.Module,
    state_dict: dict,
) -> None:
    """Initialize lazy RunningMeanStd modules before loading checkpoint buffers."""
    for module_name, module in model.named_modules():
        if not isinstance(module, RunningMeanStd):
            continue

        mean_key = f"{module_name}.mean" if module_name else "mean"
        if module._initialized or mean_key not in state_dict:
            continue

        mean = state_dict[mean_key]
        module._create_buffers(tuple(mean.shape), mean.device)
        module._initialized = True


def sync_record_moments_gates(model: nn.Module, fabric) -> None:
    """Rank-agree the normalizer record/freeze gates after a checkpoint load.

    ``NormObsBase.forward`` gates ``RunningMeanStd.record_moments`` — which
    issues all_gather/broadcast collectives — on the per-rank local flag
    ``_freeze_running``. If that flag diverges across DDP ranks (observed after
    ladder/checkpoint resume), ranks execute different collective schedules and
    the process group deadlocks. This one-shot sync runs at a symmetric point
    (checkpoint load, outside any forward) and forces every rank to the same
    decision: a module records unless EVERY rank wants it frozen (i.e. record
    if any rank would record — matching what an all_reduce(max) of the record
    flag would decide, but with zero collectives in the hot path).

    No-op for single-process runs and when torch.distributed is not initialized,
    so single-GPU behavior is unchanged. For fresh multi-rank runs the flags are
    already uniform, so this is behavior-preserving there too.
    """
    if fabric is None or getattr(fabric, "world_size", 1) <= 1:
        return
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return

    for name, module in model.named_modules():
        if not hasattr(module, "_freeze_running"):
            continue
        should_record = torch.tensor(
            int(not module._freeze_running),
            device=fabric.device,
            dtype=torch.long,
        )
        dist.all_reduce(should_record, op=dist.ReduceOp.MAX)
        agreed_freeze = int(should_record.item()) == 0
        if module._freeze_running != agreed_freeze:
            print(
                f"[sync_record_moments_gates] rank {fabric.global_rank}: "
                f"module '{name}' _freeze_running {module._freeze_running} -> "
                f"{agreed_freeze} (rank-agreed)"
            )
        module._freeze_running = agreed_freeze


def combine_moments(means: List[Tensor], vars: List[Tensor], counts: List[Tensor]):
    """
    Combine moments from multiple processes robustly using a pairwise algorithm.
    """
    if not isinstance(counts, torch.Tensor):
        counts = torch.tensor(counts)

    # Convert all inputs to a compatible type for accumulation
    counts = counts.float()

    while len(means) > 1:
        new_means, new_vars, new_counts = [], [], []

        # Iteratively combine pairs of means, variances, and counts
        # We use non-sequential pairwise combination to minimize combinations across different magnitudes
        for i in range(0, len(means), 2):
            if i + 1 < len(means):
                # Combine a pair of moments
                mean_a, var_a, count_a = means[i], vars[i], counts[i]
                mean_b, var_b, count_b = means[i + 1], vars[i + 1], counts[i + 1]

                total_count = count_a + count_b
                delta = mean_b - mean_a

                # Combine means
                combined_mean = mean_a + delta * (count_b / total_count)

                # Combine variances (numerically stable formula)
                m_2_a = var_a * count_a
                m_2_b = var_b * count_b
                m_2_combined = (
                    m_2_a + m_2_b + (delta**2) * (count_a * count_b / total_count)
                )
                combined_var = m_2_combined / total_count

                new_means.append(combined_mean)
                new_vars.append(combined_var)
                new_counts.append(total_count)
            else:
                # If there's an odd number of batches, just carry the last one over
                new_means.append(means[i])
                new_vars.append(vars[i])
                new_counts.append(counts[i])

        means = new_means
        vars = new_vars
        counts = new_counts

    combined_mean = means[0]
    combined_var = torch.clamp(vars[0], min=0.0)  # Ensure non-negative variance
    total_count = counts[0].long()

    return combined_mean, combined_var, total_count


class RewardRunningMeanStd(RunningMeanStd):
    """Running statistics for reward normalization.

    Supports two modes controlled by ``ema_decay``:

    * **Welford (default, ema_decay=None)** -- all-time running statistics.
      Variance estimate freezes after many updates because the sample count
      grows without bound.
    * **EMA (ema_decay in (0, 1))** -- exponential moving average of mean and
      variance.  Tracks non-stationary reward distributions (e.g. when
      discriminator reward magnitudes shift during adversarial training).

    Checkpoint compatibility: both modes store the same ``mean`` / ``var`` /
    ``count`` buffers.  Switching from Welford to EMA on resume is safe -- the
    loaded mean/var become the EMA starting point and ``count`` is ignored.

    Adopted from https://gymnasium.farama.org/_modules/gymnasium/wrappers/stateful_reward/#NormalizeReward
    """

    def __init__(
        self,
        fabric: Fabric,
        shape: Tuple[int, ...],
        gamma: float,
        epsilon: float = 1e-5,
        clamp_value: Optional[float] = None,
        device: str = "cuda:0",
        ema_decay: Optional[float] = None,
    ):
        super().__init__(fabric, shape, epsilon, device, clamp_value, ema_decay)
        self.gamma = gamma

        self.discounted_reward = None

    def record_reward(
        self, reward: torch.tensor, terminated: torch.tensor
    ) -> torch.tensor:
        if self.discounted_reward is None:
            self.discounted_reward = reward.clone()
        else:
            self.discounted_reward = (
                self.discounted_reward * self.gamma * (1 - terminated.float())
                + reward.clone()
            )
        self.record_moments(self.discounted_reward)

    def normalize(self, arr: torch.tensor, un_norm=False) -> torch.tensor:
        # Only normalize the magnitude, not the offset.
        if not un_norm:
            result = arr / torch.sqrt(self.var.float() + self.epsilon)
            result = self.maybe_clamp(result)
        else:
            arr = self.maybe_clamp(arr)
            result = arr * torch.sqrt(self.var.float() + self.epsilon)

        return result
