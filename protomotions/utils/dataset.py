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

import numpy as np
from torch.utils.data import Dataset


class GeneralizedDataset(Dataset):
    def __init__(self, batch_size, *tensors, shuffle=False):
        assert len(tensors) > 0
        lengths = {len(t) for t in tensors}
        assert len(lengths) == 1

        self.num_tensors = next(iter(lengths))
        self.batch_size = batch_size
        assert (
            self.num_tensors % self.batch_size == 0
        ), f"{self.num_tensors} {self.batch_size}"
        self.tensors = tensors
        self.do_shuffle = shuffle

        if shuffle:
            self.shuffle()

    def shuffle(self):
        index = np.random.permutation(self.tensors[0].shape[0])
        self.tensors = [t[index] for t in self.tensors]

    def __len__(self):
        return self.num_tensors // self.batch_size

    def __getitem__(self, index):
        assert index < len(self), f"{index} {len(self)}"
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, self.num_tensors)
        return [t[start_idx:end_idx] for t in self.tensors]
