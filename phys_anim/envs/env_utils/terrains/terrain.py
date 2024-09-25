# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import math
import torch
from scipy import ndimage

from phys_anim.envs.env_utils.terrains.subterrain import SubTerrain
from phys_anim.envs.env_utils.terrains.subterrain_generator import (
    discrete_obstacles_subterrain,
    poles_subterrain,
    pyramid_sloped_subterrain,
    pyramid_stairs_subterrain,
    random_uniform_subterrain,
    stepping_stones_subterrain,
)
from phys_anim.envs.env_utils.terrains.terrain_utils import (
    convert_heightfield_to_trimesh,
)
from phys_anim.utils.scene_lib import SceneLib

import matplotlib.pyplot as plt


class Terrain:
    def __init__(self, config, scene_lib: SceneLib, num_envs: int, device) -> None:
        self.config = config
        self.device = device
        self.num_scenes = 0
        self.spacing_between_scenes = config.spacing_between_scenes
        self.minimal_humanoid_spacing = config.minimal_humanoid_spacing

        # place scenes in the border region
        length = config.map_length * config.num_levels
        self.num_scenes_per_column = max(
            math.floor(length / self.spacing_between_scenes), 1
        )

        self.horizontal_scale = config.horizontal_scale
        self.vertical_scale = config.vertical_scale
        self.border_size = config.border_size
        self.env_length = config.map_length
        self.env_width = config.map_width
        self.proportions = [
            np.sum(config.terrain_proportions[: i + 1])
            for i in range(len(config.terrain_proportions))
        ]

        self.env_rows = config.num_levels
        self.env_cols = config.num_terrains
        self.num_maps = self.env_rows * self.env_cols
        self.border = int(self.border_size / self.horizontal_scale)

        if scene_lib is not None:
            scene_lib.call_when_terrain_init_scene_spacing(self)
            self.num_scenes = scene_lib.total_spawned_scenes

        scene_rows = (
            0
            if self.num_scenes == 0
            else math.ceil(self.num_scenes / self.num_scenes_per_column) + 2
        )
        self.object_playground_depth = scene_rows * self.spacing_between_scenes
        self.object_playground_buffer_size = int(
            5 / self.horizontal_scale
        )  # 5 meters buffer, adjust as needed

        total_size = self.num_maps * config.map_length * config.map_width * 1.0
        space_between_humanoids = total_size / num_envs
        assert (
            space_between_humanoids >= self.minimal_humanoid_spacing
        ), "Not enough space between humanoids, create a bigger terrain or reduce the number of envs."

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / self.horizontal_scale)

        self.object_playground_cols = math.ceil(
            self.object_playground_depth / self.horizontal_scale
        )
        self.tot_cols = (
            int(self.env_cols * self.width_per_env_pixels)
            + 2 * self.border
            + self.object_playground_cols
        )
        self.tot_rows = (
            int(self.env_rows * self.length_per_env_pixels) + 2 * self.border
        )

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.ceiling_field_raw = np.zeros(
            (self.tot_rows, self.tot_cols), dtype=np.int16
        ) + (3 / self.vertical_scale)

        self.walkable_field_raw = np.zeros(
            (self.tot_rows, self.tot_cols), dtype=np.int16
        )
        self.flat_field_raw = np.ones((self.tot_rows, self.tot_cols), dtype=np.int16)

        self.scene_map = torch.zeros(
            (self.tot_rows, self.tot_cols), dtype=torch.bool, device=self.device
        )

        if self.config.load_terrain:
            print("Loading a pre-generated terrain")
            params = torch.load(self.config.terrain_path)
            self.height_field_raw = params["height_field_raw"]
            self.walkable_field_raw = params["walkable_field_raw"]
        else:
            self.generate_subterrains()
        self.heightsamples = self.height_field_raw
        self.vertices, self.triangles = convert_heightfield_to_trimesh(
            self.height_field_raw,
            self.horizontal_scale,
            self.vertical_scale,
            self.config.slope_threshold,
        )
        self.compute_walkable_coords()
        self.compute_flat_coords()

        if self.config.save_terrain:
            print("Saving this generated terrain")
            torch.save(
                {
                    "height_field_raw": self.height_field_raw,
                    "walkable_field_raw": self.walkable_field_raw,
                    "vertices": self.vertices,
                    "triangles": self.triangles,
                    "border_size": self.border_size,
                },
                self.config.terrain_path,
            )

        if scene_lib is not None:
            # Push all scenes to spawn at the edge of the terrain
            scene_y_offset = (
                self.tot_cols - self.border - self.object_playground_cols
            ) * self.horizontal_scale
            scene_lib.call_at_terrain_done_init(scene_y_offset)

        # # Generate and show the plot
        # self.generate_terrain_plot(scene_lib)

    def generate_subterrains(self):
        if self.config.terrain_composition == "curriculum":
            self.curriculum(
                n_subterrains_per_level=self.env_cols, n_levels=self.env_rows
            )
        elif self.config.terrain_composition == "randomized_subterrains":
            self.randomized_subterrains()
        else:
            raise NotImplementedError(
                "Terrain composition configuration "
                + self.config.terrain_composition
                + " not implemented"
            )

    def compute_walkable_coords(self):
        self.walkable_field_raw[: self.border, :] = 1
        self.walkable_field_raw[
            :,
            -(
                self.border
                + self.object_playground_cols
                + self.object_playground_buffer_size
            ) :,
        ] = 1
        self.walkable_field_raw[:, : self.border] = 1
        self.walkable_field_raw[-self.border :, :] = 1

        self.walkable_field = torch.tensor(self.walkable_field_raw, device=self.device)

        walkable_x_indices, walkable_y_indices = torch.where(self.walkable_field == 0)
        self.walkable_x_coords = walkable_x_indices * self.horizontal_scale
        self.walkable_y_coords = walkable_y_indices * self.horizontal_scale

    def compute_flat_coords(self):
        self.flat_field_raw[: self.border, :] = 1
        self.flat_field_raw[
            :,
            -(
                self.border
                + self.object_playground_cols
                + self.object_playground_buffer_size
            ) :,
        ] = 1
        self.flat_field_raw[:, : self.border] = 1
        self.flat_field_raw[-self.border :, :] = 1

        self.flat_field_raw = torch.tensor(self.flat_field_raw, device=self.device)

        flat_x_indices, flat_y_indices = torch.where(self.flat_field_raw == 0)
        self.flat_x_coords = flat_x_indices * self.horizontal_scale
        self.flat_y_coords = flat_y_indices * self.horizontal_scale

    def sample_valid_locations(self, num_envs):
        x_loc = np.random.randint(0, self.walkable_x_coords.shape[0], size=num_envs)
        y_loc = np.random.randint(0, self.walkable_y_coords.shape[0], size=num_envs)
        valid_locs = torch.stack(
            [self.walkable_x_coords[x_loc], self.walkable_y_coords[y_loc]], dim=-1
        )

        # Raise an error if any position is invalid
        assert self.is_valid_spawn_location(
            valid_locs
        ).all(), "Invalid spawn locations detected"

        return valid_locs

    def sample_flat_locations(self, num_envs):
        x_loc = np.random.randint(0, self.flat_x_coords.shape[0], size=num_envs)
        y_loc = np.random.randint(0, self.flat_y_coords.shape[0], size=num_envs)
        flat_locs = torch.stack(
            [self.flat_x_coords[x_loc], self.flat_y_coords[y_loc]], dim=-1
        )

        # Raise an error if any position is invalid
        assert self.is_valid_spawn_location(
            flat_locs
        ).all(), "Invalid flat spawn locations detected"

        return flat_locs

    def randomized_subterrains(self):
        raise NotImplementedError("Randomized subterrains not properly implemented")
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            subterrain = SubTerrain(self.config, "terrain", device=self.device)
            choice = np.random.uniform(0, 1)
            if choice < 0.1:
                if np.random.choice([0, 1]):
                    pyramid_sloped_subterrain(
                        subterrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3])
                    )
                    random_uniform_subterrain(
                        subterrain,
                        min_height=-0.1,
                        max_height=0.1,
                        step=0.05,
                        downsampled_scale=0.2,
                    )
                else:
                    pyramid_sloped_subterrain(
                        subterrain, np.random.choice([-0.3, -0.2, 0, 0.2, 0.3])
                    )
            elif choice < 0.6:
                # step_height = np.random.choice([-0.18, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.18])
                step_height = np.random.choice([-0.15, 0.15])
                pyramid_stairs_subterrain(
                    subterrain,
                    step_width=0.31,
                    step_height=step_height,
                    platform_size=3.0,
                )
            elif choice < 1.0:
                discrete_obstacles_subterrain(
                    subterrain, 0.15, 1.0, 2.0, 40, platform_size=3.0
                )

            self.height_field_raw[start_x:end_x, start_y:end_y] = (
                subterrain.height_field_raw
            )

    def curriculum(self, n_subterrains_per_level, n_levels):
        for subterrain_idx in range(n_subterrains_per_level):
            for level_idx in range(n_levels):
                subterrain = SubTerrain(self.config, "terrain", device=self.device)
                difficulty = level_idx / n_levels
                choice = subterrain_idx / n_subterrains_per_level

                # Heightfield coordinate system
                start_x = self.border + level_idx * self.length_per_env_pixels
                end_x = self.border + (level_idx + 1) * self.length_per_env_pixels
                start_y = self.border + subterrain_idx * self.width_per_env_pixels
                end_y = self.border + (subterrain_idx + 1) * self.width_per_env_pixels

                slope = difficulty * 0.4
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty
                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_subterrain(
                        subterrain, slope=slope, platform_size=3.0
                    )
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_subterrain(
                        subterrain, slope=slope, platform_size=3.0
                    )
                    random_uniform_subterrain(
                        subterrain,
                        min_height=-0.1,
                        max_height=0.1,
                        step=0.025,
                        downsampled_scale=0.2,
                    )
                elif choice < self.proportions[3]:
                    if choice < self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_subterrain(
                        subterrain,
                        step_width=0.31,
                        step_height=step_height,
                        platform_size=3.0,
                    )
                elif choice < self.proportions[4]:
                    discrete_obstacles_subterrain(
                        subterrain,
                        discrete_obstacles_height,
                        1.0,
                        2.0,
                        40,
                        platform_size=3.0,
                    )
                elif choice < self.proportions[5]:
                    stepping_stones_subterrain(
                        subterrain,
                        stone_size=stepping_stones_size,
                        stone_distance=0.1,
                        max_height=0.0,
                        platform_size=3.0,
                    )
                elif choice < self.proportions[6]:
                    poles_subterrain(subterrain=subterrain, difficulty=difficulty)
                    self.walkable_field_raw[start_x:end_x, start_y:end_y] = (
                        subterrain.height_field_raw != 0
                    )
                elif choice < self.proportions[7]:
                    subterrain.terrain_name = "flat"

                    flat_border = int(4 / self.horizontal_scale)

                    self.flat_field_raw[
                        start_x + flat_border : end_x - flat_border,
                        start_y + flat_border : end_y - flat_border,
                    ] = 0
                    # plain walking terrain
                    pass
                self.height_field_raw[start_x:end_x, start_y:end_y] = (
                    subterrain.height_field_raw
                )

        self.walkable_field_raw = ndimage.binary_dilation(
            self.walkable_field_raw, iterations=3
        ).astype(int)

    def mark_scene_location(self, x, y):
        """
        Mark the location of an scene on the scene map.

        Args:
            x (int): X coordinate in terrain map coordinates.
            y (int): Y coordinate in terrain map coordinates.
            radius (int): Radius of the scene in terrain map coordinates.
        """
        radius = (
            math.floor(self.spacing_between_scenes * 1.0 / 2 / self.horizontal_scale)
            - 1
        )
        x_min = max(0, x - radius)
        x_max = min(self.tot_rows, x + radius + 1)
        y_min = max(0, y - radius)
        y_max = min(
            self.tot_cols - self.object_playground_buffer_size, y + radius + 1
        )  # Respect the buffer

        self.scene_map[x_min:x_max, y_min:y_max] = True

    def is_valid_spawn_location(self, locations: torch.Tensor) -> torch.Tensor:
        """
        Check if locations are valid for spawning scenes.

        Args:
            locations (torch.Tensor): Tensor of shape [B, 2] containing x, y coordinates in terrain map coordinates.

        Returns:
            torch.Tensor: Boolean tensor of shape [B] indicating valid (True) or invalid (False) spawn locations.
        """
        radius = (
            math.floor(self.spacing_between_scenes * 1.0 / 2 / self.horizontal_scale)
            - 1
        )
        batch_size = locations.shape[0]

        # Calculate boundaries
        x_min = torch.clamp(locations[:, 0] - radius, min=0)
        x_max = torch.clamp(locations[:, 0] + radius + 1, max=self.tot_rows)
        y_min = torch.clamp(locations[:, 1] - radius, min=0)
        y_max = torch.clamp(locations[:, 1] + radius + 1, max=self.tot_cols)

        # Check if the area is completely outside the valid range
        valid = (x_max > x_min) & (y_max > y_min)

        # Use advanced indexing to check all valid locations in a single operation
        for i in range(batch_size):
            if valid[i]:
                valid[i] = not self.scene_map[
                    int(x_min[i]) : int(x_max[i]), int(y_min[i]) : int(y_max[i])
                ].any()

        return valid

    def generate_terrain_plot(self, scene_lib):
        # Create the figure and subplots with fixed size and layout, arranged vertically
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(
            4, 1, figsize=(8, 24), constrained_layout=True
        )

        # 1. Plot showing the height of the terrain
        height_map = ax1.imshow(self.height_field_raw, cmap="terrain", aspect="auto")
        ax1.set_title("Terrain Height")
        fig.colorbar(height_map, ax=ax1, label="Height", shrink=0.8)

        # 2. Plot highlighting the object playground area
        object_playground_map = np.zeros_like(self.height_field_raw)
        object_playground_map[:, -(self.object_playground_cols + self.border) :] = (
            1  # Mark the entire object playground area, including the border
        )

        obj_playground_plot = ax2.imshow(
            object_playground_map, cmap="binary", interpolation="nearest", aspect="auto"
        )
        ax2.set_title("Object Playground Area")
        fig.colorbar(obj_playground_plot, ax=ax2, label="Object Playground", shrink=0.8)

        # 3. Plot marking the different regions
        region_map = np.zeros_like(self.height_field_raw)

        # Object playground
        region_map[:, -(self.object_playground_cols + self.border) :] = 1

        # Buffer region
        region_map[
            :,
            -(
                self.object_playground_cols
                + self.border
                + self.object_playground_buffer_size
            ) : -(self.object_playground_cols + self.border),
        ] = 2

        # Flat region
        flat_field_cpu = self.flat_field_raw.cpu().numpy()
        flat_region = np.where(flat_field_cpu == 0)
        region_map[flat_region] = 3

        # Irregular terrain (everything else)
        irregular_region = np.where((region_map == 0) & (self.height_field_raw != 0))
        region_map[irregular_region] = 4

        cmap = plt.cm.get_cmap("viridis", 5)
        region_plot = ax3.imshow(
            region_map,
            cmap=cmap,
            interpolation="nearest",
            aspect="auto",
            vmin=0,
            vmax=4,
        )
        ax3.set_title("Terrain Regions")

        # Add colorbar
        cbar = fig.colorbar(region_plot, ax=ax3, ticks=[0.5, 1.5, 2.5, 3.5], shrink=0.8)
        cbar.set_ticklabels(
            ["Object Playground", "Buffer", "Flat Region", "Irregular Terrain"]
        )

        # 4. Plot showing where objects are placed using scene_map
        scene_map_cpu = self.scene_map.cpu().numpy()
        object_plot = ax4.imshow(
            scene_map_cpu, cmap="hot", interpolation="nearest", aspect="auto"
        )
        ax4.set_title("Object Placement")
        fig.colorbar(object_plot, ax=ax4, label="Object Present", shrink=0.8)

        # Remove axis ticks for cleaner look
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Show the plot
        plt.show()
