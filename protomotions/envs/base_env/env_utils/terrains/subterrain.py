import numpy as np


class SubTerrain:
    def __init__(self, config_terrain, terrain_name="terrain", device="cuda:0"):
        self.terrain_name = terrain_name
        self.config_terrain = config_terrain
        self.device = device
        self.vertical_scale = config_terrain.vertical_scale
        self.horizontal_scale = config_terrain.horizontal_scale
        self.width = int(config_terrain.map_width / self.horizontal_scale)
        self.length = int(config_terrain.map_length / self.horizontal_scale)
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
        self.ceiling_field_raw = np.zeros((self.width, self.length), dtype=np.int16) + (
            3 / self.vertical_scale
        )
        self.walkable_field_raw = np.zeros((self.width, self.length), dtype=np.int16)
