# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Terrain class using small CPU-only maps."""

import os
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.components.terrains.terrain import Terrain  # noqa: E402


def _flat_config(**overrides) -> TerrainConfig:
    kwargs = {
        "map_length": 2.0,
        "map_width": 2.0,
        "border_size": 0.5,
        "num_levels": 1,
        "num_terrains": 1,
        "terrain_proportions": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "horizontal_scale": 0.5,
        "vertical_scale": 0.1,
        "spacing_between_scenes": 1.0,
        "minimal_humanoid_spacing": 0.0,
        "num_samples_per_axis": 2,
        "sample_width": 0.5,
        "slope_threshold": 100.0,
    }
    kwargs.update(overrides)
    return TerrainConfig(**kwargs)


def _identity_root_state(root_pos: torch.Tensor):
    root_rot = torch.zeros(root_pos.shape[0], 4)
    root_rot[:, 3] = 1.0
    return SimpleNamespace(root_pos=root_pos, root_rot=root_rot)


def test_flat_terrain_initialization_height_queries_and_scene_occupancy():
    terrain = Terrain(_flat_config(), num_envs=1, device=torch.device("cpu"))

    assert terrain.height_field_raw.shape == (6, 12)
    assert terrain.ceiling_field_raw.shape == terrain.height_field_raw.shape
    assert terrain.is_flat() is True
    assert terrain.num_height_points == 4
    assert terrain.scene_y_offset == 2.5

    locations = torch.tensor([[0.5, 0.5, 0.0], [1.0, 1.0, 0.0]])
    assert torch.equal(terrain.get_ground_heights(locations), torch.zeros(2, 1))
    body_pos = torch.tensor([[[0.5, 0.5, -1.0], [1.0, 1.0, 0.5]]])
    assert torch.equal(
        terrain.find_terrain_height_for_max_below_body(body_pos),
        torch.zeros(1),
    )

    root_state = _identity_root_state(torch.tensor([[1.0, 1.0, 1.0]]))
    height_map = terrain.get_height_maps(root_state)
    selected_height_map = terrain.get_height_maps(root_state, env_ids=torch.tensor([0]))
    height_points = terrain.get_height_maps(root_state, return_all_dims=True)

    assert height_map.shape == (1, 4)
    assert torch.equal(selected_height_map, height_map)
    assert height_points.shape == (1, 4, 3)

    np.random.seed(1)
    terrain.walkable_x_coords = torch.tensor([1.0, 2.0])
    terrain.walkable_y_coords = torch.tensor([1.0, 2.0])
    terrain.flat_x_coords = torch.tensor([1.0, 2.0])
    terrain.flat_y_coords = torch.tensor([1.0, 2.0])
    assert terrain.sample_valid_locations(2).shape == (2, 2)
    assert terrain.sample_valid_locations(2, sample_flat=True).shape == (2, 2)

    pixel = torch.tensor([[float(terrain.border), float(terrain.border)]])
    assert terrain.is_valid_spawn_location(pixel).item() is True
    terrain.mark_scene_location(terrain.border, terrain.border)
    assert terrain.is_valid_spawn_location(pixel).item() is False


def test_scene_marking_uses_scene_lib_offset_when_object_buffer_is_larger_than_tiny_map():
    terrain = Terrain(_flat_config(), num_envs=1, device=torch.device("cpu"))

    scene_x = int(
        (
            terrain.spacing_between_scenes
            + terrain.border * terrain.horizontal_scale
        )
        / terrain.horizontal_scale
    )
    scene_y = int(
        (terrain.spacing_between_scenes + terrain.scene_y_offset)
        / terrain.horizontal_scale
    )
    location = torch.tensor([[scene_x, scene_y]], device=terrain.device)

    assert scene_y >= terrain.tot_cols - terrain.object_playground_buffer_size
    assert terrain.is_valid_spawn_location(location).item() is True

    terrain.mark_scene_location(scene_x, scene_y)

    assert terrain.scene_placement_map[scene_x, scene_y].item() is True
    assert terrain.is_valid_spawn_location(location).item() is False


def test_spawn_validation_rejects_centers_outside_heightfield_before_clamping():
    terrain = Terrain(
        _flat_config(map_length=4.0, map_width=4.0, spacing_between_scenes=2.0),
        num_envs=1,
        device=torch.device("cpu"),
    )

    locations = torch.tensor(
        [
            [-1.0, float(terrain.border)],
            [float(terrain.border), -1.0],
            [float(terrain.tot_rows), float(terrain.border)],
            [float(terrain.border), float(terrain.tot_cols)],
        ],
        device=terrain.device,
    )

    assert torch.equal(
        terrain.is_valid_spawn_location(locations),
        torch.zeros(4, dtype=torch.bool, device=terrain.device),
    )


def test_curriculum_poles_branch_uses_generated_obstacles_for_walkability(monkeypatch):
    def fake_poles_subterrain(subterrain, difficulty):
        subterrain.height_field_raw[1, 2] = 7
        subterrain.terrain_name = "poles"
        return subterrain

    monkeypatch.setattr(
        "protomotions.components.terrains.terrain.poles_subterrain",
        fake_poles_subterrain,
    )

    terrain = Terrain(
        _flat_config(
            map_length=4.0,
            map_width=4.0,
            border_size=1.0,
            horizontal_scale=1.0,
            vertical_scale=0.1,
            spacing_between_scenes=4.0,
            terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
        num_envs=1,
        device=torch.device("cpu"),
    )

    obstacle_x = terrain.border + 1
    obstacle_y = terrain.border + 2
    assert terrain.height_field_raw[obstacle_x, obstacle_y] == 7
    assert terrain.walkable_field[obstacle_x, obstacle_y].item() == 1


def test_terrain_load_save_and_plot_paths(tmp_path, monkeypatch):
    saved_path = tmp_path / "saved.pt"
    saved_cfg = _flat_config(save_terrain=True, terrain_path=str(saved_path))
    saved_terrain = Terrain(saved_cfg, num_envs=1, device=torch.device("cpu"))

    assert saved_path.exists()
    saved_payload = torch.load(saved_path, weights_only=False)
    assert sorted(saved_payload) == [
        "border_size",
        "height_field_raw",
        "triangles",
        "vertices",
        "walkable_field_raw",
    ]

    load_path = tmp_path / "load.pt"
    monkeypatch.setattr(
        "protomotions.components.terrains.terrain.torch.load",
        lambda path: {
            "height_field_raw": np.zeros_like(saved_terrain.height_field_raw),
            "walkable_field_raw": np.zeros_like(saved_terrain.walkable_field_raw),
        },
    )
    loaded = Terrain(
        _flat_config(load_terrain=True, terrain_path=str(load_path)),
        num_envs=1,
        device=torch.device("cpu"),
    )
    assert loaded.height_field_raw.shape == saved_terrain.height_field_raw.shape


    monkeypatch.setattr(
        "protomotions.components.terrains.terrain.plt.show",
        lambda: None,
    )
    loaded.generate_terrain_plot()
