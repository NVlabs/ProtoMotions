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
from torch import Tensor

from phys_anim.utils.device_dtype_mixin import DeviceDtypeModuleMixin
from torch.utils.data import Dataset
from typing import Dict
import numpy as np


def swap_and_flatten01(arr: Tensor):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


class ExperienceBuffer(DeviceDtypeModuleMixin):
    def __init__(self, num_envs: int, num_steps: int):
        super().__init__()
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.store_dict = {}

    def register_key(self, key: str, shape=(), dtype=torch.float):
        assert not hasattr(self, key), key
        buffer = torch.zeros(
            (self.num_steps, self.num_envs) + shape, dtype=dtype, device=self.device
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
        self.store_dict[key] += self.total_sum()

    def make_dict(self):
        data = {k: swap_and_flatten01(v) for k, v in self.named_buffers()}
        for k, v in self.store_dict.items():
            assert v == self.total_sum(), f"Problem with '{k}', {v}, {self.total_sum()}"
            self.store_dict[k] = 0
        return data


class DictDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        tensor_dict: Dict[str, Tensor],
        shuffle=False,
    ):
        assert len(tensor_dict) > 0
        lengths = {len(t) for t in tensor_dict.values()}
        assert len(lengths) == 1

        self.num_tensors = next(iter(lengths))
        self.batch_size = batch_size
        assert (
            self.num_tensors % self.batch_size == 0
        ), f"{self.num_tensors} {self.batch_size}"
        self.tensor_dict = tensor_dict
        self.do_shuffle = shuffle

        if shuffle:
            self.shuffle()

    def shuffle(self):
        index = np.random.permutation(self.num_tensors)
        self.tensor_dict = {k: v[index] for k, v in self.tensor_dict.items()}

    def num_batches(self):
        return self.num_tensors // self.batch_size

    def __len__(self):
        return self.num_batches()

    def __getitem__(self, index):
        assert index < len(self), f"{index} {len(self)}"
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_tensors)
        return {k: v[start_idx:end_idx] for k, v in self.tensor_dict.items()}
