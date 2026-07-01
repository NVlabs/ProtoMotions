# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base environment implementation.

This package provides the foundational environment class that all tasks inherit from.
It integrates with multiple simulators, manages robot state, and provides modular
observation, reward, and control components.

Key Components:
    - BaseEnv: Core environment class
    - EnvConfig: Base environment configuration
    - Control components: MimicControl, PathFollowerControl, etc.
    - Reward/Termination components: Modular reward and termination functions

Example:
    >>> from protomotions.envs.base_env.env import BaseEnv
    >>> env = BaseEnv(config, device)
    >>> obs, info = env.reset()
"""
