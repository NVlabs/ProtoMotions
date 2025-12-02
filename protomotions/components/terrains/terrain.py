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
"""Terrain generation and management.

Handles the creation of procedural terrains, sub-terrains, and height fields.
Manages the layout of scenes within the terrain grid.


Terrain Layout (Top-Down View):
┌──────────────────────────────────────────────────────────────┐
│                     Border (flat, zero height)               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Terrain Grid (varied heights):  │ Object Playground    │  │
│  │ ┌──────┬──────┬──────┐          │ (FLAT, zero height)  │  │
│  │ │Stairs│Slope │ Flat │          │  ┌─────────────┐     │  │
│  │ ├──────┼──────┼──────┤          │  │ Scene 0     │     │  │
│  │ │Slope │Stones│Stairs│          │  │ (objects)   │     │  │
│  │ ├──────┼──────┼──────┤          │  ├─────────────┤     │  │ ← num_scenes_per_column
│  │ │Flat  │Steps │Poles │          │  │ Scene 1     │     │  │
│  │ └──────┴──────┴──────┘          │  │ (objects)   │     │  │
│  │                                 │  └─────────────┘     │  │
│  │  ← env_cols × env_rows →        │  ← always flat →     │  │
│  └────────────────────────────────────────────────────────┘  │
│                     Border (flat, zero height)               │
└──────────────────────────────────────────────────────────────┘
         ↑                                   ↑
    terrain grid                      Where SceneLib places
    (subterrains)                   Scene objects with offsets

"""

import numpy as np
import math
import torch
from scipy import ndimage

from protomotions.components.terrains.subterrain import SubTerrain
from protomotions.components.terrains.subterrain_generator import (
    discrete_obstacles_subterrain,
    poles_subterrain,
    pyramid_sloped_subterrain,
    pyramid_stairs_subterrain,
    random_uniform_subterrain,
    stepping_stones_subterrain,
)
from protomotions.components.terrains.terrain_utils import (
    convert_heightfield_to_trimesh,
    get_heights_jit,
    get_height_maps_jit,
)
from protomotions.components.terrains.config import TerrainConfig

import matplotlib.pyplot as plt


class Terrain:
    """Manages terrain generation and height field data.

    Generates a grid of sub-terrains (e.g., stairs, slopes, flat) based on configuration.
    Also allocates a flat "object playground" region for sceneLib placement (always at z=0).
    Provides utilities for querying terrain heights and sceneLib placement tracking.

    Layout:
        - Terrain grid: Variable height subterrains for curriculum learning
        - Object playground: Flat region (z=0) appended to the right, for scene objects
        - Border: Flat buffer regions around the edges

    Args:
        config: Terrain configuration object.
        num_envs: Number of environments to support (determines object playground size).
        device: Device for terrain tensors.
    """

    def __init__(self, config: TerrainConfig, num_envs: int, device) -> None:
        self.config = config
        self.device = device
        self.num_scene_slots = 0  # Number of scene slots to reserve space for
        self.spacing_between_scenes = config.spacing_between_scenes
        self.minimal_humanoid_spacing = config.minimal_humanoid_spacing

        # Expose sim_config for convenient access
        self.sim_config = config.sim_config

        # Place scenes in the object playground region (always flat at z=0)
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

        self.num_scene_slots = num_envs  # Reserve space for 1 scene per env

        scene_rows = (
            0
            if self.num_scene_slots == 0
            else math.ceil(self.num_scene_slots / self.num_scenes_per_column) + 2
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

        # Tracks which locations in object playground are occupied by scenes
        self.scene_placement_map = torch.zeros(
            (self.tot_rows, self.tot_cols), dtype=torch.bool, device=self.device
        )

        if self.config.load_terrain:
            print("Loading a pre-generated terrain")
            params = torch.load(self.config.terrain_path)
            self.height_field_raw = params["height_field_raw"]
            self.walkable_field_raw = params["walkable_field_raw"]
        else:
            self.generate_subterrains()

        # Normalize terrain heights so the lowest point is at z=0
        min_height = np.min(self.height_field_raw)
        if min_height < 0:
            print(
                f"Normalizing terrain: shifting all heights by {min_height * self.vertical_scale:.4f}m to set minimum to z=0"
            )
            self.height_field_raw = self.height_field_raw - min_height

        self.height_samples = (
            torch.tensor(self.height_field_raw, device=self.device, dtype=torch.float)
            * self.vertical_scale
        )
        self.num_height_points, self.height_points = self.init_height_points(num_envs)

        self.vertices, self.triangles = convert_heightfield_to_trimesh(
            self.height_field_raw,
            self.horizontal_scale,
            self.vertical_scale,
            self.config.slope_threshold,
            flat_tolerance=0.0001,
            max_triangle_size=50,
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

        self.scene_y_offset = (
            self.tot_cols - self.border - self.object_playground_cols
        ) * self.horizontal_scale

        # # Generate and show the plot
        # self.generate_terrain_plot()

    def generate_subterrains(self):
        self.curriculum(n_subterrains_per_level=self.env_cols, n_levels=self.env_rows)

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

    def is_flat(self) -> bool:
        """
        Check if the terrain is completely flat.

        Returns True only if terrain_proportions has flat=1.0 and all others are 0.0.
        This is used to skip expensive terrain height calculations when unnecessary.
        """
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping, poles, flat]
        # flat is at index 7

        # TODO: for now indexing is hardcoded.
        proportions = self.config.terrain_proportions
        assert len(proportions) == 8
        return proportions[-1] == 1.0 and sum(proportions[:-1]) == 0.0

    def find_terrain_height_for_max_below_body(self, respawned_rigid_body_pos):
        """
        Find the terrain height at the position of the rigid body that is furthest below the terrain.

        This identifies which body needs the maximum upward adjustment to place the character
        properly on the terrain, and returns the terrain height at that body's position.

        Args:
            respawned_rigid_body_pos: Rigid body positions [batch_size, num_bodies, 3]

        Returns:
            Terrain heights at the lowest body position for each env [batch_size]
        """

        # Get terrain heights at all body positions
        z_all_joints = self.get_ground_heights(
            respawned_rigid_body_pos
        )  # (batch_size, num_bodies)

        # Find body with maximum diff (furthest below terrain, needs most upward adjustment)
        z_diff = (
            z_all_joints - respawned_rigid_body_pos[:, :, 2]
        )  # (batch_size, num_bodies)

        z_indices = torch.max(z_diff, dim=1).indices  # (batch_size,)

        h = z_all_joints.gather(1, z_indices.unsqueeze(1)).squeeze(1)  # (batch_size,)

        # Assert only moving up (non-negative height adjustment)
        assert torch.all(
            h >= 0
        ), f"Invalid height adjustment: expected all >= 0, got min={h.min():.4f}"

        return h

    def sample_valid_locations(self, num_envs, sample_flat=False):
        if sample_flat:
            return self.sample_flat_locations(num_envs)

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
        Mark a location in the object playground as occupied by a scene.

        Args:
            x (int): X coordinate in terrain map coordinates.
            y (int): Y coordinate in terrain map coordinates.
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

        self.scene_placement_map[x_min:x_max, y_min:y_max] = True

    def is_valid_spawn_location(self, locations: torch.Tensor) -> torch.Tensor:
        """
        Check if locations in object playground are valid for spawning scenes.

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
                valid[i] = not self.scene_placement_map[
                    int(x_min[i]) : int(x_max[i]), int(y_min[i]) : int(y_max[i])
                ].any()

        return valid

    def init_height_points(self, num_envs):
        """
        Pre-defines the grid for the height-map observation.
        """
        y = torch.tensor(
            np.linspace(
                -self.config.sample_width,
                self.config.sample_width,
                self.config.num_samples_per_axis,
            ),
            device=self.device,
            requires_grad=False,
        )
        x = torch.tensor(
            np.linspace(
                -self.config.sample_width,
                self.config.sample_width,
                self.config.num_samples_per_axis,
            ),
            device=self.device,
            requires_grad=False,
        )
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        num_height_points = grid_x.numel()
        points = torch.zeros(
            num_envs,
            num_height_points,
            3,
            device=self.device,
            requires_grad=False,
        )
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return num_height_points, points

    def get_ground_heights(self, locations: torch.Tensor) -> torch.Tensor:
        """
        Get the height of the terrain at the specified locations.
        """
        return get_heights_jit(
            locations=locations,
            height_samples=self.height_samples,
            horizontal_scale=self.horizontal_scale,
        )

    def get_height_maps(self, root_states, env_ids=None, return_all_dims=False):
        """
        Generates a 2D heightmap grid observation rotated w.r.t. the character's heading.
        Each sample is the billinear interpolation between adjacent points.
        """
        if env_ids is not None:
            height_points = self.height_points[env_ids].clone()
        else:
            height_points = self.height_points.clone()

        return get_height_maps_jit(
            base_rot=root_states.root_rot,
            base_pos=root_states.root_pos,
            height_points=height_points,
            height_samples=self.height_samples,
            num_height_points=self.num_height_points,
            terrain_horizontal_scale=self.horizontal_scale,
            w_last=True,
            return_all_dims=return_all_dims,
        )

    def generate_terrain_plot(self):
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

        # 4. Plot showing where scenes are placed using scene_placement_map
        scene_placement_map_cpu = self.scene_placement_map.cpu().numpy()
        object_plot = ax4.imshow(
            scene_placement_map_cpu, cmap="hot", interpolation="nearest", aspect="auto"
        )
        ax4.set_title("Scene Placement")
        fig.colorbar(object_plot, ax=ax4, label="Scene Present", shrink=0.8)

        # Remove axis ticks for cleaner look
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Show the plot
        plt.show()
