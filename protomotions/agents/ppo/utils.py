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
from typing import Optional


def discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values, gamma, tau):
    lastgaelam = 0
    mb_advs = torch.zeros_like(mb_rewards)
    num_steps = mb_rewards.shape[0]

    for t in reversed(range(num_steps)):
        not_done = 1.0 - mb_fdones[t]
        not_done = not_done

        delta = mb_rewards[t] + gamma * mb_next_values[t] - mb_values[t]
        lastgaelam = delta + gamma * tau * not_done * lastgaelam
        mb_advs[t] = lastgaelam

    return mb_advs


def bounds_loss(mu: Tensor) -> Tensor:
    soft_bound = 1.0
    mu_loss_high = (
        torch.maximum(mu - soft_bound, torch.tensor(0, device=mu.device)) ** 2
    )
    mu_loss_low = (
        torch.minimum(mu + soft_bound, torch.tensor(0, device=mu.device)) ** 2
    )
    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
    return b_loss


def normalization_with_masks(values: Tensor, masks: Optional[Tensor]):
    if masks is None:
        return (values - values.mean()) / (values.std() + 1e-8)

    values_mean, values_var = get_mean_var_with_masks(values, masks)
    values_std = torch.sqrt(values_var)
    normalized_values = (values - values_mean) / (values_std + 1e-8)

    return normalized_values


def get_mean_var_with_masks(values: Tensor, masks: Tensor):
    sum_mask = masks.sum()
    values_mask = values * masks
    values_mean = values_mask.sum() / sum_mask
    min_sqr = (((values_mask) ** 2) / sum_mask).sum() - (
        (values_mask / sum_mask).sum()
    ) ** 2
    values_var = min_sqr * sum_mask / (sum_mask - 1)
    return values_mean, values_var
