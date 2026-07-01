# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
