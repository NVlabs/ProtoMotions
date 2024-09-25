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

from hydra.utils import instantiate
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from phys_anim.utils.motion_lib import MotionLib


class CountingDataset(Dataset):
    def __init__(self, config):
        self.config = config
        # assert config.num_envs * config.num_steps % config.batch_size == 0
        # self.length = (
        #     config.ngpu * config.num_envs * config.num_steps // config.batch_size
        # )
        self.length = config.num_batches * config.ngpu

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


class CountingDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ds = CountingDataset(config)

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=1, shuffle=False)


class ReferenceMotions(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.motion_lib: MotionLib = instantiate(config.motion_lib)

    def setup(self, stage=None):
        pass


MOTION_LIB: MotionLib = None


def global_motion_lib_wrapper(config):
    global MOTION_LIB
    if MOTION_LIB is None:
        MOTION_LIB = MotionLib(**config)
    return MOTION_LIB
