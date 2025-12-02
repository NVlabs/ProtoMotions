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
"""Adversarial Motion Priors (AMP) implementation.

This package implements the AMP algorithm which adds a discriminator network
to PPO for learning motion style rewards from reference data.

Key Components:
    - AMP: Main AMP agent
    - AMPModel: Actor-critic with discriminator
    - AMPAgentConfig: Configuration

Example:
    >>> from protomotions.agents.amp.agent import AMP
    >>> agent = AMP(fabric, env, config)
    >>> agent.train()

Reference:
    Peng et al. "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control" (2021)
"""
