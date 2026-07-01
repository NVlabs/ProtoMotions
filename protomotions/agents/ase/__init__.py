# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
