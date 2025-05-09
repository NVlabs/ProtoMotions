from protomotions.envs.base_env.env_utils.terrains.terrain import Terrain
import math
from protomotions.envs.base_env.env_utils.terrains.terrain_config import TerrainConfig


class FlatTerrain(Terrain):
    def __init__(self, config: TerrainConfig, num_envs: int, device) -> None:
        config.load_terrain = False
        config.save_terrain = False

        env_rows = config.num_levels
        env_cols = config.num_terrains
        num_maps = env_rows * env_cols

        total_size = num_maps * config.map_length * config.map_width * 1.0
        space_between_humanoids = total_size / num_envs
        space_multiplier = max(
            1, config.minimal_humanoid_spacing / space_between_humanoids
        )
        # When creating a simple flat terrain, we need to make sure that there is enough space between the humanoids.
        # For irregular terrains the user will have to ensure that there is enough space between the humanoids.
        config.num_terrains = math.ceil(config.num_terrains * space_multiplier)

        super().__init__(config, num_envs, device)

    def generate_subterrains(self):
        self.flat_field_raw[:] = 0
        # Override to do nothing else
        pass
