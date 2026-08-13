# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Terrain class using small CPU-only maps."""

import os
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import protomotions.components.terrains.terrain as terrain_module  # noqa: E402
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


def test_spawn_sampling_reuses_buffer_and_defers_validation_until_scene_placement():
    terrain = Terrain(_flat_config(), num_envs=2, device=torch.device("cpu"))
    terrain.walkable_x_coords = torch.tensor([1.0, 2.0, 3.0])
    terrain.walkable_y_coords = torch.tensor([11.0, 12.0, 13.0])
    initial_buffer = terrain._spawn_sample_index_buffer
    validation_calls = []

    def valid(locations):
        validation_calls.append(locations)
        return torch.ones(locations.shape[0], dtype=torch.bool)

    terrain.is_valid_spawn_location = valid
    sampled = terrain.sample_valid_locations(2)

    assert validation_calls == []
    assert terrain._spawn_sample_index_buffer.data_ptr() == initial_buffer.data_ptr()
    assert torch.equal(sampled[:, 1] - sampled[:, 0], torch.full((2,), 10.0))

    terrain.mark_scene_location(terrain.border, terrain.border)
    terrain.sample_valid_locations(2)
    assert len(validation_calls) == 1

    terrain.sample_valid_locations(3)
    assert terrain._spawn_sample_index_buffer.numel() == 3


def test_spawn_sampling_does_not_advance_global_torch_rng():
    torch.manual_seed(1234)
    terrain = Terrain(_flat_config(), num_envs=2, device=torch.device("cpu"))
    terrain.walkable_x_coords = torch.arange(16, dtype=torch.float)
    terrain.walkable_y_coords = terrain.walkable_x_coords + 100
    global_state_before = torch.random.get_rng_state()

    first = terrain.sample_valid_locations(2)
    second = terrain.sample_valid_locations(2)

    assert torch.equal(torch.random.get_rng_state(), global_state_before)
    assert torch.equal(first[:, 1] - first[:, 0], torch.full((2,), 100.0))
    assert torch.equal(second[:, 1] - second[:, 0], torch.full((2,), 100.0))


def test_get_height_maps_reuses_cached_height_points_without_full_env_clone(
    monkeypatch,
):
    terrain = Terrain(_flat_config(), num_envs=1, device=torch.device("cpu"))
    before = terrain.height_points.clone()
    seen = {}

    def fake_height_maps_jit(
        *,
        base_rot,
        base_pos,
        height_points,
        height_samples,
        num_height_points,
        terrain_horizontal_scale,
        w_last,
        return_all_dims,
    ):
        del (
            base_rot,
            base_pos,
            height_samples,
            num_height_points,
            terrain_horizontal_scale,
            w_last,
            return_all_dims,
        )
        seen["height_points_data_ptr"] = height_points.data_ptr()
        return height_points.new_zeros(height_points.shape[0], height_points.shape[1])

    monkeypatch.setattr(terrain_module, "get_height_maps_jit", fake_height_maps_jit)
    root_state = _identity_root_state(torch.tensor([[1.0, 1.0, 1.0]]))

    terrain.get_height_maps(root_state)

    assert seen["height_points_data_ptr"] == terrain.height_points.data_ptr()
    assert torch.equal(terrain.height_points, before)


def test_scene_marking_uses_scene_lib_offset_when_object_buffer_is_larger_than_tiny_map():
    terrain = Terrain(_flat_config(), num_envs=1, device=torch.device("cpu"))

    scene_x = int(
        (terrain.spacing_between_scenes + terrain.border * terrain.horizontal_scale)
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


def test_loaded_nonzero_heightfield_bypasses_generated_flat_fast_path(
    tmp_path, monkeypatch
):
    height_field_raw = np.zeros((6, 12), dtype=np.int16)
    height_field_raw[1:3, 1:3] = 2
    terrain_path = tmp_path / "nonflat.pt"
    torch.save(
        {
            "height_field_raw": height_field_raw,
            "walkable_field_raw": np.zeros_like(height_field_raw),
        },
        terrain_path,
    )
    torch_load = torch.load
    monkeypatch.setattr(
        terrain_module.torch,
        "load",
        lambda path: torch_load(path, weights_only=False),
    )

    terrain = Terrain(
        _flat_config(load_terrain=True, terrain_path=str(terrain_path)),
        num_envs=1,
        device=torch.device("cpu"),
    )
    heights = terrain.get_ground_heights(torch.tensor([[0.5, 0.5, 0.0]]))

    assert terrain.is_flat() is False, f"height query returned {heights}"
    assert heights.shape == (1, 1)
    torch.testing.assert_close(heights, torch.tensor([[0.2]]))


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
