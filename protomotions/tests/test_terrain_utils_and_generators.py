# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for terrain mesh utilities and sub-terrain generators."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from protomotions.components.terrains import subterrain_generator as generators
from protomotions.components.terrains import terrain_utils
from protomotions.components.terrains.subterrain import SubTerrain


def _subterrain(
    width: int = 16,
    length: int = 16,
    horizontal_scale: float = 0.25,
    vertical_scale: float = 0.1,
) -> SubTerrain:
    config = SimpleNamespace(
        vertical_scale=vertical_scale,
        horizontal_scale=horizontal_scale,
        map_width=width * horizontal_scale,
        map_length=length * horizontal_scale,
    )
    return SubTerrain(config_terrain=config, terrain_name="unit", device="cpu")


@pytest.fixture(scope="module")
def source_terrain_utils():
    terrain_utils_path = (
        Path(__file__).resolve().parents[1]
        / "components"
        / "terrains"
        / "terrain_utils.py"
    )
    module_name = "_terrain_utils_source_for_coverage"
    spec = importlib.util.spec_from_file_location(module_name, terrain_utils_path)
    module = importlib.util.module_from_spec(spec)
    original_script = torch.jit.script
    original_script_if_tracing = torch.jit.script_if_tracing
    torch.jit.script = lambda fn=None, *args, **kwargs: fn
    torch.jit.script_if_tracing = lambda fn=None, *args, **kwargs: fn
    try:
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        torch.jit.script = original_script
        torch.jit.script_if_tracing = original_script_if_tracing
        sys.modules.pop(module_name, None)
    return module


def _mesh_xy_area(vertices: np.ndarray, triangles: np.ndarray) -> float:
    pts = vertices[triangles, :2]
    edge_a = pts[:, 1] - pts[:, 0]
    edge_b = pts[:, 2] - pts[:, 0]
    area = 0.5 * np.abs(edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
    return float(area.sum())


def _manual_get_heights(locations, height_samples, horizontal_scale):
    num_envs = locations.shape[0]
    if len(locations.shape) == 2:
        locations = locations.unsqueeze(1)
    num_samples_per_env = locations.shape[1]
    points = locations[..., :2].clone().reshape(num_envs, num_samples_per_env, 2)
    points = points / horizontal_scale
    floored_points = points.long()
    px = floored_points[:, :, 0].view(-1).clip(0, height_samples.shape[0] - 2)
    py = floored_points[:, :, 1].view(-1).clip(0, height_samples.shape[1] - 2)
    fx = points[:, :, 0].view(-1) - px.float()
    fy = points[:, :, 1].view(-1) - py.float()
    h_tl = height_samples[px, py]
    h_tr = height_samples[px + 1, py]
    h_bl = height_samples[px, py + 1]
    h_br = height_samples[px + 1, py + 1]
    h_t = h_tl + (h_tr - h_tl) * fx
    h_b = h_bl + (h_br - h_bl) * fx
    return (h_t + (h_b - h_t) * fy).view(num_envs, -1)


def test_heightfield_to_basic_mesh_and_slope_correction():
    heightfield = np.array([[0, 1], [2, 3]], dtype=np.int16)

    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        heightfield,
        horizontal_scale=0.5,
        vertical_scale=0.1,
    )

    assert vertices.dtype == np.float32
    assert triangles.dtype == np.uint32
    assert np.allclose(
        vertices,
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.1],
                [0.5, 0.0, 0.2],
                [0.5, 0.5, 0.3],
            ],
            dtype=np.float32,
        ),
    )
    assert np.array_equal(triangles, np.array([[0, 3, 1], [0, 2, 3]]))

    corrected_vertices, corrected_triangles = terrain_utils.convert_heightfield_to_trimesh(
        heightfield.copy(),
        horizontal_scale=0.5,
        vertical_scale=0.1,
        slope_threshold=0.05,
    )

    assert corrected_vertices.shape == vertices.shape
    assert np.array_equal(corrected_triangles, triangles)
    assert not np.allclose(corrected_vertices[:, :2], vertices[:, :2])


def test_heightfield_slope_correction_moves_only_steep_edges():
    heightfield = np.array([[0, 2], [0, 0]], dtype=np.int16)

    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        heightfield,
        horizontal_scale=1.0,
        vertical_scale=1.0,
        slope_threshold=0.5,
    )

    assert np.array_equal(triangles, np.array([[0, 3, 1], [0, 2, 3]]))
    assert np.allclose(
        vertices,
        np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )

    uncorrected_vertices, _ = terrain_utils.convert_heightfield_to_trimesh(
        heightfield,
        horizontal_scale=1.0,
        vertical_scale=1.0,
        slope_threshold=3.0,
    )

    assert np.allclose(
        uncorrected_vertices[:, :2],
        np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_heightfield_to_optimized_mesh_merges_and_subdivides_flat_regions():
    heightfield = np.zeros((4, 4), dtype=np.int16)

    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        heightfield,
        horizontal_scale=0.5,
        vertical_scale=0.1,
        flat_tolerance=0.01,
    )

    assert vertices.shape == (4, 3)
    assert triangles.shape == (2, 3)
    assert np.all(vertices[:, 2] == 0.0)

    subdivided_vertices, subdivided_triangles = (
        terrain_utils.convert_heightfield_to_trimesh(
            heightfield,
            horizontal_scale=0.5,
            vertical_scale=0.1,
            flat_tolerance=0.01,
            max_triangle_size=0.75,
        )
    )

    assert subdivided_vertices.shape[0] > vertices.shape[0]
    assert subdivided_triangles.shape[0] > triangles.shape[0]


def test_optimized_mesh_stops_vertical_growth_near_steep_transition():
    heightfield = np.zeros((5, 4), dtype=np.int16)
    heightfield[3:, :] = 10

    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        heightfield,
        horizontal_scale=1.0,
        vertical_scale=0.1,
        flat_tolerance=0.01,
    )

    zero_height_faces = triangles[
        np.all(vertices[triangles, 2] == 0.0, axis=1)
    ]
    zero_face_vertices = vertices[zero_height_faces]
    zero_face_x_spans = np.ptp(zero_face_vertices[:, :, 0], axis=1)
    zero_face_y_spans = np.ptp(zero_face_vertices[:, :, 1], axis=1)

    assert triangles.shape[0] > 2
    assert np.max(zero_face_x_spans) <= 1.0
    assert np.max(zero_face_y_spans) == 3.0


def test_optimized_mesh_subdivision_does_not_emit_degenerate_triangles():
    heightfield = np.zeros((3, 3), dtype=np.int16)

    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        heightfield,
        horizontal_scale=1.0,
        vertical_scale=0.1,
        flat_tolerance=0.01,
        max_triangle_size=0.4,
    )

    assert vertices.shape == (9, 3)
    assert triangles.shape == (8, 3)
    assert np.isclose(_mesh_xy_area(vertices, triangles), 4.0)
    assert np.all(
        np.array([len(set(triangle.tolist())) for triangle in triangles]) == 3
    )


def test_optimized_mesh_subdivision_covers_uneven_regions():
    heightfield = np.zeros((5, 6), dtype=np.int16)

    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        heightfield,
        horizontal_scale=1.0,
        vertical_scale=0.1,
        flat_tolerance=0.01,
        max_triangle_size=2.1,
    )

    assert np.isclose(vertices[:, 0].min(), 0.0)
    assert np.isclose(vertices[:, 0].max(), 4.0)
    assert np.isclose(vertices[:, 1].min(), 0.0)
    assert np.isclose(vertices[:, 1].max(), 5.0)
    assert np.isclose(_mesh_xy_area(vertices, triangles), 20.0)
    assert np.all(
        np.array([len(set(triangle.tolist())) for triangle in triangles]) == 3
    )


def test_perlin_interpolation_helpers_are_numeric_and_seedable():
    assert terrain_utils.lerp(2.0, 6.0, 0.25) == 3.0
    assert terrain_utils.fade(0.0) == 0.0
    assert terrain_utils.fade(1.0) == 1.0

    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    y = np.array([[0.5, 0.6], [0.7, 0.8]])
    h = np.array([[0, 1], [2, 3]])

    gradient = terrain_utils.gradient(h, x, y)
    assert gradient.shape == x.shape
    assert np.isfinite(gradient).all()

    np.random.seed(123)
    first = terrain_utils.perlin(x, y)
    np.random.seed(123)
    second = terrain_utils.perlin(x, y)

    assert first.shape == x.shape
    assert np.allclose(first, second)


def test_jit_height_queries_interpolate_points_and_height_maps():
    height_samples = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    locations = torch.tensor([[0.25, 0.25, 0.0], [0.5, 0.0, 0.0]])

    heights = terrain_utils.get_heights_jit(
        locations,
        height_samples=height_samples,
        horizontal_scale=0.5,
    )

    assert torch.allclose(heights, torch.tensor([[1.5], [2.0]]))

    base_rot = torch.zeros(1, 4)
    base_rot[:, 3] = 1.0
    base_pos = torch.tensor([[0.0, 0.0, 4.0]])
    height_points = torch.tensor([[[0.25, 0.25, 0.0], [0.5, 0.0, 0.0]]])

    height_map = terrain_utils.get_height_maps_jit(
        base_rot=base_rot,
        base_pos=base_pos,
        height_points=height_points,
        height_samples=height_samples,
        num_height_points=2,
        terrain_horizontal_scale=0.5,
        w_last=True,
        return_all_dims=False,
    )
    all_dims = terrain_utils.get_height_maps_jit(
        base_rot=base_rot,
        base_pos=base_pos,
        height_points=height_points,
        height_samples=height_samples,
        num_height_points=2,
        terrain_horizontal_scale=0.5,
        w_last=True,
        return_all_dims=True,
    )

    assert torch.allclose(height_map, torch.tensor([[2.5, 2.0]]))
    assert all_dims.shape == (1, 2, 3)


def test_jit_height_queries_preserve_multiple_samples_per_env():
    height_samples = torch.tensor(
        [
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
        ]
    )
    locations = torch.tensor(
        [
            [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]],
            [[0.25, 1.0, 0.0], [1.5, 1.5, 0.0]],
        ]
    )

    heights = terrain_utils.get_heights_jit(
        locations,
        height_samples=height_samples,
        horizontal_scale=1.0,
    )

    assert heights.shape == (2, 2)
    assert torch.allclose(
        heights,
        torch.tensor([[5.5, 10.0], [3.5, 16.5]]),
    )


def test_get_heights_source_matches_scripted_export_for_2d_clipped_bilinear(
    source_terrain_utils,
):
    height_samples = torch.tensor(
        [
            [0.0, 2.0, 4.0],
            [10.0, 12.0, 14.0],
            [20.0, 22.0, 24.0],
        ]
    )
    locations = torch.tensor(
        [
            [0.25, 0.75, 99.0],
            [2.75, 2.75, -5.0],
        ]
    )

    source = source_terrain_utils.get_heights_jit(
        locations, height_samples=height_samples, horizontal_scale=1.0
    )
    scripted = terrain_utils.get_heights_jit(
        locations, height_samples=height_samples, horizontal_scale=1.0
    )

    torch.testing.assert_close(source, scripted)
    torch.testing.assert_close(
        source,
        _manual_get_heights(locations, height_samples, horizontal_scale=1.0),
    )


def test_get_heights_source_matches_scripted_export_for_batched_samples(
    source_terrain_utils,
):
    height_samples = torch.arange(16.0).reshape(4, 4)
    locations = torch.tensor(
        [
            [[0.5, 0.5, 0.0], [1.5, 0.25, 0.0], [2.8, 2.2, 0.0]],
            [[0.0, 1.0, 0.0], [1.25, 1.75, 0.0], [10.0, -1.0, 0.0]],
        ]
    )

    source = source_terrain_utils.get_heights_jit(
        locations, height_samples=height_samples, horizontal_scale=1.0
    )
    scripted = terrain_utils.get_heights_jit(
        locations, height_samples=height_samples, horizontal_scale=1.0
    )

    assert source.shape == (2, 3)
    torch.testing.assert_close(source, scripted)
    torch.testing.assert_close(
        source,
        _manual_get_heights(locations, height_samples, horizontal_scale=1.0),
    )


@pytest.mark.parametrize("w_last", [True, False])
@pytest.mark.parametrize("return_all_dims", [False, True])
def test_height_maps_source_matches_scripted_export_for_yaw_and_return_shapes(
    source_terrain_utils, w_last, return_all_dims
):
    height_samples = torch.arange(25.0).reshape(5, 5)
    angles = torch.tensor([0.0, torch.pi / 2])
    axis = torch.zeros(2, 3)
    axis[:, 2] = 1.0
    base_rot = terrain_utils.rotations.quat_from_angle_axis(
        angles, axis, w_last=w_last
    )
    base_pos = torch.tensor([[1.0, 1.0, 10.0], [2.0, 1.0, 20.0]])
    height_points = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
        ]
    )

    source = source_terrain_utils.get_height_maps_jit(
        base_rot=base_rot,
        base_pos=base_pos,
        height_points=height_points,
        height_samples=height_samples,
        num_height_points=3,
        terrain_horizontal_scale=1.0,
        w_last=w_last,
        return_all_dims=return_all_dims,
    )
    scripted = terrain_utils.get_height_maps_jit(
        base_rot=base_rot,
        base_pos=base_pos,
        height_points=height_points,
        height_samples=height_samples,
        num_height_points=3,
        terrain_horizontal_scale=1.0,
        w_last=w_last,
        return_all_dims=return_all_dims,
    )

    expected_shape = (2, 3, 3) if return_all_dims else (2, 3)
    assert source.shape == expected_shape
    torch.testing.assert_close(source, scripted, atol=1e-6, rtol=1e-6)


def test_subterrain_generators_mutate_heightfields_with_expected_names():
    np.random.seed(7)
    random_terrain = generators.random_uniform_subterrain(
        _subterrain(),
        min_height=0.0,
        max_height=0.2,
        step=0.1,
        downsampled_scale=0.5,
    )
    assert random_terrain.terrain_name == "random_uniform"
    assert random_terrain.height_field_raw.dtype == np.int16
    assert random_terrain.height_field_raw.max() <= 2

    sloped = generators.sloped_subterrain(_subterrain(), slope=0.25)
    assert sloped.terrain_name == "sloped"
    assert sloped.height_field_raw[-1].mean() > sloped.height_field_raw[0].mean()

    pyramid = generators.pyramid_sloped_subterrain(
        _subterrain(),
        slope=0.25,
        platform_size=0.5,
    )
    assert pyramid.terrain_name == "pyramid_sloped"
    assert pyramid.height_field_raw.max() >= 0

    flat_wave = _subterrain()
    generators.wave_subterrain(flat_wave, num_waves=0, amplitude=1.0)
    assert np.all(flat_wave.height_field_raw == 0)

    wave = generators.wave_subterrain(_subterrain(), num_waves=1, amplitude=0.4)
    assert wave.height_field_raw.max() > wave.height_field_raw.min()

    stairs = generators.stairs_subterrain(
        _subterrain(width=8, length=6),
        step_width=0.5,
        step_height=0.2,
    )
    assert np.all(stairs.height_field_raw[:2] == 2)
    assert np.all(stairs.height_field_raw[2:4] == 4)

    pyramid_stairs = generators.pyramid_stairs_subterrain(
        _subterrain(width=10, length=10),
        step_width=0.25,
        step_height=0.1,
        platform_size=0.5,
    )
    assert pyramid_stairs.height_field_raw[5, 5] > pyramid_stairs.height_field_raw[0, 0]


def test_stepping_stones_and_obstacles_cover_orientation_and_bounds():
    np.random.seed(11)
    long_stones = generators.stepping_stones_subterrain(
        _subterrain(width=6, length=10),
        stone_size=0.5,
        stone_distance=0.25,
        max_height=0.2,
        platform_size=0.5,
        depth=-1.0,
    )
    assert long_stones.terrain_name == "stepping_stones"
    assert long_stones.height_field_raw.min() <= -10

    np.random.seed(11)
    wide_stones = generators.stepping_stones_subterrain(
        _subterrain(width=10, length=6),
        stone_size=0.5,
        stone_distance=0.25,
        max_height=0.2,
        platform_size=0.5,
        depth=-1.0,
    )
    assert wide_stones.terrain_name == "stepping_stones"
    assert wide_stones.height_field_raw.min() <= -10

    np.random.seed(3)
    obstacles = generators.discrete_obstacles_subterrain(
        _subterrain(width=16, length=16),
        max_height=0.2,
        min_size=1.0,
        max_size=2.0,
        num_rects=2,
        platform_size=1.0,
    )
    center = obstacles.height_field_raw[6:10, 6:10]
    assert obstacles.terrain_name == "discrete_obstacles"
    assert np.all(center == 0)
    assert obstacles.height_field_raw.max() == 2


def test_obstacle_json_updates_static_top_dynamic_terrain_and_segmentation(tmp_path):
    terrain = _subterrain(width=10, length=10, horizontal_scale=0.1, vertical_scale=0.1)
    terrain.segmentation_field = {}

    map_description = {
        "segmentation": [
            {
                "name": "Default",
                "cx": 0.5,
                "cy": 0.5,
                "radius": 5.0,
                "goal_radius": 0.15,
                "color": "gray",
            }
        ],
        "terrain": [
            {
                "type": "gravel",
                "start_x": 0.0,
                "start_y": 0.0,
                "end_x": 0.5,
                "end_y": 0.5,
                "amplitude": 0.05,
            },
            {"type": "sloped"},
            {"type": "stairs"},
            {"type": "mixed"},
        ],
        "static_obstacles": [
            {
                "type": "box",
                "x": 0.2,
                "y": 0.2,
                "obs_size": 0.2,
                "obs_height": 0.4,
            }
        ],
        "top_obstacles": [
            {
                "cx": 0.5,
                "cy": 0.5,
                "length": 0.2,
                "width": 0.2,
                "z_bottom": 0.6,
            }
        ],
        "dynamic_obstacles": [
            {
                "start_x": 1.0,
                "start_y": 2.0,
                "cycle": 3.0,
                "velocity_x": 0.5,
                "velocity_y": -0.25,
            }
        ],
    }
    map_path = tmp_path / "terrain.json"
    map_path.write_text(json.dumps(map_description))

    np.random.seed(2)
    generators.obstacles_from_json(terrain, str(map_path))

    assert terrain.static_obstacles == map_description["static_obstacles"]
    assert terrain.top_obstacles == map_description["top_obstacles"]
    assert terrain.dynamic_obstacles[0]["end_x"] == 2.5
    assert terrain.dynamic_obstacles[0]["end_y"] == 1.25
    assert terrain.dynamic_obstacles[0]["cur_pos"] == [1.0, 2.0, 10]
    assert terrain.walkable_field_raw[2:4, 2:4].sum() == 4
    assert terrain.ceiling_field_raw.min() == int(
        map_description["top_obstacles"][0]["z_bottom"] / terrain.vertical_scale
    )
    assert terrain.seg_color == {"Default": "gray"}
    assert terrain.landmarks == ["Default"]
    assert terrain.segmentation_field[(0, 0)]["name"] == "default"


def test_update_top_obstacles_reads_map_description_directly():
    terrain = _subterrain(width=6, length=6, horizontal_scale=0.5, vertical_scale=0.1)
    map_description = {
        "top_obstacles": [
            {
                "cx": 1.0,
                "cy": 1.0,
                "length": 0.5,
                "width": 0.5,
                "z_bottom": 0.4,
            }
        ],
        "dynamic_obstacles": [],
    }

    generators.update_top_obstacles(map_description, terrain)

    assert terrain.top_obstacles == map_description["top_obstacles"]
    assert terrain.dynamic_obstacles == []
    assert terrain.ceiling_field_raw.min() == 4


def test_update_terrain_gravel_respects_configured_subregion(monkeypatch):
    terrain = _subterrain(width=6, length=6, horizontal_scale=0.1, vertical_scale=0.1)
    map_description = {
        "terrain": [
            {
                "type": "gravel",
                "start_x": 0.2,
                "start_y": 0.1,
                "end_x": 0.5,
                "end_y": 0.4,
                "amplitude": 0.2,
            }
        ]
    }
    monkeypatch.setattr(generators.np.random, "random", lambda shape: np.ones(shape))

    generators.update_terrain(map_description, terrain)

    expected = np.zeros((6, 6), dtype=np.int16)
    expected[2:5, 1:4] = 2
    assert np.array_equal(terrain.height_field_raw, expected)


def test_update_segmentation_assigns_nearest_region_and_goal_flags():
    terrain = _subterrain(width=5, length=5, horizontal_scale=1.0, vertical_scale=0.1)
    terrain.segmentation_field = {}
    map_description = {
        "segmentation": [
            {"name": "Default", "cx": 0.0, "cy": 0.0, "color": "gray"},
            {
                "name": "Target",
                "cx": 4.0,
                "cy": 4.0,
                "radius": 2.0,
                "goal_radius": 1.1,
                "color": "green",
            },
        ]
    }

    generators.update_segmentation(map_description, terrain)

    assert terrain.seg_color == {"Default": "gray", "Target": "green"}
    assert terrain.landmarks == ["Default", "Target"]
    assert terrain.segmentation_field[(4, 4)] == {"name": "target", "is_goal": True}
    assert terrain.segmentation_field[(3, 4)] == {"name": "target", "is_goal": True}
    assert terrain.segmentation_field[(2, 4)] == {"name": "default", "is_goal": True}


def test_poles_generator_and_wall_status_helpers():
    np.random.seed(13)
    poles = generators.poles_subterrain(
        _subterrain(width=96, length=96, horizontal_scale=0.1),
        difficulty=0.25,
    )

    assert poles.terrain_name == "poles"
    assert poles.height_field_raw.shape == (96, 96)

    assert generators.get_walls_status(0b1011) == {
        "N": 1,
        "E": 1,
        "S": 0,
        "W": 1,
    }


def test_small_poles_terrain_is_valid_noop_when_no_shape_batches_fit():
    np.random.seed(13)

    poles = generators.poles_subterrain(
        _subterrain(width=32, length=32, horizontal_scale=0.1),
        difficulty=1.0,
    )

    assert poles.terrain_name == "poles"
    assert np.all(poles.height_field_raw == 0)
