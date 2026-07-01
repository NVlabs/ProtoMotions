# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial Motion Priors (AMP) PPO agent.

The AMP-specific training behavior lives in
`protomotions.agents.amp.component.AMPTrainingComponent` so other PPO-style
agents can reuse the same discriminator reward machinery.
"""

from pathlib import Path
from typing import Optional

from lightning.fabric import Fabric

from protomotions.agents.amp.component import AMPAgentMixin
from protomotions.agents.amp.config import AMPAgentConfig
from protomotions.agents.ppo.agent import PPO
from protomotions.envs.base_env.env import BaseEnv


class AMP(AMPAgentMixin, PPO):
    """PPO policy update with reusable AMP discriminator training."""

    config: AMPAgentConfig

    def __init__(
        self,
        fabric: Fabric,
        env: BaseEnv,
        config,
        root_dir: Optional[Path] = None,
    ):
        super().__init__(fabric, env, config, root_dir=root_dir)
