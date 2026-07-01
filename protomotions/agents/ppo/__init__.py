# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Proximal Policy Optimization (PPO) implementation.

This package implements the PPO algorithm for reinforcement learning, including
the actor-critic model architecture and training logic.

Key Components:
    - PPO: Main PPO agent
    - PPOModel: Actor-critic model
    - PPOActor: Policy network
    - PPOAgentConfig: Configuration

Example:
    >>> from protomotions.agents.ppo.agent import PPO
    >>> agent = PPO(fabric, env, config)
    >>> agent.train()
"""
