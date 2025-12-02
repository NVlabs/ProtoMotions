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
"""Data utilities for experience management and batching.

This module provides utilities for managing experience buffers and creating
minibatch datasets for training. Handles efficient storage and retrieval of
rollout data collected during environment interaction.

Key Classes:
    - ExperienceBuffer: Buffer for storing rollout experience
    - DictDataset: Dataset for creating minibatches from experience

Key Functions:
    - swap_and_flatten01: Reshape tensors for batching
    - get_dict: Extract dictionary view of experience buffer
"""

import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from typing import Dict
import numpy as np


def swap_and_flatten01(arr: Tensor):
    """Swap and flatten first two dimensions of a tensor.

    Converts (num_steps, num_envs, ...) to (num_steps * num_envs, ...).
    Commonly used to batch experience from parallel environments.

    Args:
        arr: Tensor with at least 2 dimensions.

    Returns:
        Tensor with first two dimensions flattened.
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class ExperienceBuffer(nn.Module):
    """Buffer for storing rollout experience from parallel environments.

    Collects observations, actions, rewards, and other data during environment
    rollouts. Provides efficient storage and batching for on-policy algorithms.
    Uses PyTorch buffers for automatic device management.

    Args:
        num_envs: Number of parallel environments.
        num_steps: Number of steps per rollout.

    Attributes:
        store_dict: Dictionary tracking which keys have been populated.

    Example:
        >>> buffer = ExperienceBuffer(num_envs=1024, num_steps=16)
        >>> buffer.register_key("obs", shape=(128,))
        >>> buffer.update_data("obs", step=0, data=observations)
        >>> data_dict = buffer.get_dict()
    """

    def __init__(self, num_envs: int, num_steps: int, device: torch.device):
        super().__init__()
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.store_dict = {}
        self._device = device

    def register_key(self, key: str, shape=(), dtype=torch.float):
        assert not hasattr(self, key), key
        buffer = torch.zeros(
            (self.num_steps, self.num_envs) + shape, dtype=dtype, device=self._device
        )
        self.register_buffer(key, buffer, persistent=False)
        self.store_dict[key] = 0

    def update_data(self, key: str, index: int, data: Tensor):
        assert not data.requires_grad
        getattr(self, key)[index] = data
        self.store_dict[key] += index + 1

    def total_sum(self):
        return (self.num_steps + 1) * (self.num_steps / 2)

    def batch_update_data(self, key: str, data: Tensor):
        assert not data.requires_grad
        getattr(self, key)[:] = data
        self.store_dict[key] = self.total_sum()

    def make_dict(self):
        data = {k: swap_and_flatten01(v) for k, v in self.named_buffers()}
        for k, v in self.store_dict.items():
            assert v == self.total_sum(), f"Problem with '{k}', {v}, {self.total_sum()}"
            self.store_dict[k] = 0
        return data


class DictDataset(Dataset):
    """PyTorch Dataset for dictionary of tensors with minibatching.

    Creates minibatches from a dictionary of tensors. Supports shuffling and
    automatic batching for training. Used to create minibatch iterators from
    collected experience buffers.

    Args:
        batch_size: Size of each minibatch.
        tensor_dict: Dictionary of tensors to batch (all same length in dim 0).
        shuffle: Whether to shuffle indices before batching.

    Example:
        >>> data = {"obs": obs_tensor, "actions": action_tensor}
        >>> dataset = DictDataset(batch_size=256, tensor_dict=data, shuffle=True)
        >>> for batch in dataset:
        >>>     train_on_batch(batch)
    """

    def __init__(
        self,
        batch_size: int,
        tensor_dict: Dict[str, Tensor],
        shuffle=False,
    ):
        assert len(tensor_dict) > 0
        lengths_dict = {k: len(v) for k, v in tensor_dict.items()}
        assert (
            len(set(lengths_dict.values())) == 1
        ), f"All tensors must have the same length. Found: {lengths_dict}"

        self.num_tensors = next(iter(lengths_dict.values()))
        self.batch_size = batch_size
        assert (
            self.num_tensors % self.batch_size == 0
        ), f"{self.num_tensors} {self.batch_size}"
        self.tensor_dict = tensor_dict
        self.do_shuffle = shuffle
        self.shuffled_to_original = np.arange(self.num_tensors)

        if shuffle:
            self.shuffle()

    def shuffle(self):
        self.shuffled_to_original = np.random.permutation(self.num_tensors)

    def num_batches(self):
        return self.num_tensors // self.batch_size

    def __len__(self):
        return self.num_batches()

    def __getitem__(self, index):
        assert index < len(self), f"{index} {len(self)}"
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_tensors)
        return {
            k: v[self.shuffled_to_original[start_idx:end_idx]]
            for k, v in self.tensor_dict.items()
        }
