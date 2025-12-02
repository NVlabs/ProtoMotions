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
"""Utility functions for PPO algorithm.

This module provides helper functions for PPO, including advantage computation
using Generalized Advantage Estimation (GAE).

Key Functions:
    - discount_values: Compute GAE advantages from rewards and values
"""

import torch


def discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values, gamma, tau):
    """Compute Generalized Advantage Estimation (GAE) advantages.

    Computes advantages using GAE-Lambda, which provides a bias-variance tradeoff
    for advantage estimation. Uses backwards iteration through the episode to
    compute bootstrapped advantages.

    Args:
        mb_fdones: Done flags (num_steps, num_envs). 1.0 = episode ended.
        mb_values: Value predictions at each timestep (num_steps, num_envs).
        mb_rewards: Rewards received at each timestep (num_steps, num_envs).
        mb_next_values: Value predictions for next states (num_steps, num_envs).
        gamma: Discount factor for future rewards (typically 0.99).
        tau: GAE lambda parameter for bias-variance tradeoff (typically 0.95).

    Returns:
        Tensor of advantages with shape (num_steps, num_envs).

    Example:
        >>> advantages = discount_values(dones, values, rewards, next_values, 0.99, 0.95)
        >>> returns = advantages + values  # Can compute returns from advantages

    Note:
        GAE-Lambda provides advantages that balance bias (low lambda) vs. variance (high lambda).
        Lambda=0 gives 1-step TD, lambda=1 gives Monte Carlo returns.

    Reference:
        Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015)
    """
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
