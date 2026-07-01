# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base agent implementation for reinforcement learning.

This package provides the foundational agent class that all RL agents inherit from.
It handles the core training loop, experience collection, optimization, checkpointing,
and evaluation. Specific algorithms (PPO, AMP, ASE, etc.) extend BaseAgent to implement
their own model creation and training logic.

Key Components:
    - BaseAgent: Core agent class with training loop
    - BaseModel: Abstract model interface
    - BaseAgentConfig: Configuration dataclass for agent parameters

Example:
    >>> from protomotions.agents.ppo.agent import PPO
    >>> agent = PPO(fabric, env, config)
    >>> agent.setup()
    >>> agent.train()
"""
