# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch import nn

from protomotions.utils.device_dtype_mixin import DeviceDtypeModuleMixin


class ReplayBuffer(DeviceDtypeModuleMixin, nn.Module):
    def __init__(self, buffer_size):
        super().__init__()
        self._head = 0
        self._is_full = False
        self._buffer_size = buffer_size
        self._buffer_keys = []

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
