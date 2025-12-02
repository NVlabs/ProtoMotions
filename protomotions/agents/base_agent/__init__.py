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
