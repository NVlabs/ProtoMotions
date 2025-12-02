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
"""Replay buffer for off-policy learning.

This module provides a circular replay buffer used in AMP and ASE for storing
agent transitions. The discriminator trains on batches sampled from this buffer.

Key Classes:
    - ReplayBuffer: Circular buffer with random sampling
"""

import torch
from torch import nn


class ReplayBuffer(nn.Module):
    """Circular replay buffer for storing and sampling transitions.

    Stores agent transitions in a circular buffer and provides random sampling
    for discriminator training in AMP/ASE. Automatically handles buffer overflow
    by overwriting oldest data.

    Args:
        buffer_size: Maximum number of transitions to store.
        device: PyTorch device for tensors.

    Attributes:
        _head: Current write position in buffer.
        _is_full: Whether buffer has wrapped around.

    Example:
        >>> buffer = ReplayBuffer(buffer_size=10000, device=torch.device("cuda"))
        >>> buffer.store({"obs": observations, "actions": actions})
        >>> samples = buffer.sample(256)  # Sample 256 transitions
    """

    def __init__(self, buffer_size, device: torch.device):
        super().__init__()
        self._head = 0
        self._is_full = False
        self._buffer_size = buffer_size
        self._buffer_keys = []
        self._device = device

    def reset(self):
        self._head = 0
        self._is_full = False

    def get_buffer_size(self):
        return self._buffer_size

    def __len__(self) -> int:
        return self._buffer_size if self._is_full else self._head

    def store(self, data_dict):
        self._maybe_init_data_buf(data_dict)

        n = next(iter(data_dict.values())).shape[0]
        buffer_size = self.get_buffer_size()
        assert n <= buffer_size

        for key in self._buffer_keys:
            curr_buf = getattr(self, key)
            curr_n = data_dict[key].shape[0]
            assert n == curr_n

            end = self._head + n
            if end >= self._buffer_size:
                diff = self._buffer_size - self._head
                curr_buf[self._head :] = data_dict[key][:diff].clone()
                curr_buf[: n - diff] = data_dict[key][diff:].clone()
                self._is_full = True
            else:
                curr_buf[self._head : end] = data_dict[key].clone()

        self._head = (self._head + n) % buffer_size

    def sample(self, n):
        indices = torch.randint(0, len(self), (n,), device=self.device)

        samples = dict()
        for k in self._buffer_keys:
            v = getattr(self, k)
            samples[k] = v[indices].clone()

        return samples

    def _maybe_init_data_buf(self, data_dict):
        buffer_size = self.get_buffer_size()

        for k, v in data_dict.items():
            if not hasattr(self, k):
                v_shape = v.shape[1:]
                self.register_buffer(
                    k,
                    torch.zeros(
                        (buffer_size,) + v_shape, dtype=v.dtype, device=self.device
                    ),
                    persistent=False,
                )
                self._buffer_keys.append(k)

    @property
    def device(self) -> torch.device:
        """Get the current device."""
        return self._device
