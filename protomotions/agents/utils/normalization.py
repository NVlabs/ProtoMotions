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
    ):
        """Initialize running statistics tracker with optional lazy initialization.

        Args:
            fabric: Lightning Fabric for distributed training.
            shape: Shape of data to normalize. If None, will be inferred on first forward pass.
            epsilon: Numerical stability constant.
            device: PyTorch device.
            clamp_value: Optional value for clamping normalized outputs.
        """
        super().__init__()
        self.fabric = fabric
        self.epsilon = epsilon
        self.clamp_value = clamp_value
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

        if self.fabric.world_size > 1:
            all_means = self.fabric.all_gather(batch_mean)
            all_vars = self.fabric.all_gather(batch_var)
            all_counts = self.fabric.all_gather(batch_count)

            if self.fabric.global_rank == 0:
                batch_mean, batch_var, batch_count = combine_moments(
                    all_means, all_vars, all_counts
                )

        if self.fabric.global_rank == 0:
            self.update_from_moments(batch_mean, batch_var, batch_count)

        # Broadcast updated parameters to all ranks
        updated_mean = self.fabric.broadcast(self.mean, src=0)
        updated_var = self.fabric.broadcast(self.var, src=0)
        updated_count = self.fabric.broadcast(self.count, src=0)

        self.mean.copy_(updated_mean)
        self.var.copy_(updated_var)
        self.count.fill_(updated_count.item())


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
    # Adopted from https://gymnasium.farama.org/_modules/gymnasium/wrappers/stateful_reward/#NormalizeReward
    def __init__(
        self,
        fabric: Fabric,
        shape: Tuple[int, ...],
        gamma: float,
        epsilon: float = 1e-5,
        clamp_value: Optional[float] = None,
        device: str = "cuda:0",
    ):
        super().__init__(fabric, shape, epsilon, device, clamp_value)
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
        # Override normalizer behavior for rewards. Only normalize the magnitude and not the offset.
        if not un_norm:
            result = arr / torch.sqrt(self.var.float() + self.epsilon)
            result = self.maybe_clamp(result)
        else:
            arr = self.maybe_clamp(arr)
            result = arr * torch.sqrt(self.var.float() + self.epsilon)

        return result
