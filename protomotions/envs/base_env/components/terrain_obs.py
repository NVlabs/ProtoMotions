import torch

from protomotions.envs.base_env.components.base_component import BaseComponent


class TerrainObs(BaseComponent):
    def __init__(self, config, env):
        super().__init__(config, env)

        self.terrain_obs = torch.zeros(
            self.env.num_envs,
            self.env.terrain.num_height_points,
            device=self.env.device,
            dtype=torch.float,
        )

    def compute_observations(self, env_ids):
        root_states = self.env.simulator.get_root_state(env_ids)
        self.terrain_obs[env_ids] = self.env.terrain.get_height_maps(root_states, env_ids)

    def get_obs(self):
        return {"terrain": self.terrain_obs.clone()}
