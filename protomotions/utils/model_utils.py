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

import torch.nn as nn
from torch.nn import functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def get_activation_func(activation_name, return_type="nn"):
    if activation_name.lower() == "tanh":
        activation = (nn.Tanh(), F.tanh)
    elif activation_name.lower() == "relu":
        activation = (nn.ReLU(), F.relu)
    elif activation_name.lower() == "elu":
        activation = (nn.ELU(), F.elu)
    elif activation_name.lower() == "gelu":
        activation = (nn.GELU(), F.gelu)
    elif activation_name.lower() == "identity":
        activation = (nn.Identity(), lambda x: x)
    elif activation_name.lower() == "silu":
        activation = (nn.SiLU(), F.silu)
    elif activation_name.lower() == "mish":
        activation = (nn.Mish(), F.mish)
    else:
        raise NotImplementedError(
            "Activation func {} not defined".format(activation_name)
        )

    if return_type == "nn":
        return activation[0]
    elif return_type == "functional":
        return activation[1]
    else:
        raise NotImplementedError("Return type {} not defined".format(return_type))
