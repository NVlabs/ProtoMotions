# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Terrain observation component.

Computes height map observations around the agent.
"""

import torch

from protomotions.components.terrains.config import TerrainConfig
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv
else:
    BaseEnv = object


class TerrainObs:
    """Handles computation of terrain height map observations.

    Samples terrain heights around the agent's root position to provide local terrain info.

    Args:
        config: Configuration for terrain observations.
        env: Parent environment instance.
    """

    def __init__(self, config: TerrainConfig, env: BaseEnv):
        self.config = config
        self.env = env

        self.terrain_obs = torch.zeros(
            self.env.num_envs,
            self.env.terrain.num_height_points,
            device=self.env.device,
            dtype=torch.float,
        )

    def compute_observations(self, env_ids):
        """Compute terrain height map observations for specified environments.

        Args:
            env_ids: Environment indices to update
        """
        root_states = self.env.simulator.get_root_state(env_ids)
        self.terrain_obs[env_ids] = self.env.terrain.get_height_maps(
            root_states, env_ids
        )

    def get_obs(self):
        """Get terrain observations dictionary.

        Returns:
            Dictionary with 'terrain' key containing height maps
        """
        return {"terrain": self.terrain_obs.clone()}
