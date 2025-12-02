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
"""Adversarial Skill Embeddings (ASE) implementation.

This package implements the ASE algorithm which extends AMP with learned skill
embeddings. The discriminator encodes motions into a latent space and the policy
is conditioned on these latent codes for diverse skill learning.

Key Components:
    - ASE: Main ASE agent
    - ASEModel: Actor-critic with skill encoder
    - ASEAgentConfig: Configuration

Example:
    >>> from protomotions.agents.ase.agent import ASE
    >>> agent = ASE(fabric, env, config)
    >>> agent.train()

Reference:
    Peng et al. "ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters" (2022)
"""
