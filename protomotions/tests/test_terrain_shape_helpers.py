# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for terrain shape and sub-terrain helpers."""

from types import SimpleNamespace

import numpy as np

from protomotions.components.terrains import shape_utils
from protomotions.components.terrains.subterrain import SubTerrain


def _assert_binary_int16_image(image: np.ndarray, img_size: int):
    assert image.shape == (img_size, img_size)
    assert image.dtype == np.int16
    assert set(np.unique(image)).issubset({0, 1})
    assert image.sum() > 0


def test_shape_generators_return_binary_int16_images_with_seeded_randomness():
    cases = [
        (shape_utils.draw_disk, {"img_size": 32, "max_r": 8}),
        (shape_utils.draw_circle, {"img_size": 32, "max_r": 8}),
        (
            shape_utils.draw_curve,
            {"img_size": 32, "max_sides": 8, "iterations": 2},
        ),
        (shape_utils.draw_polygon, {"img_size": 32, "max_sides": 8}),
        (shape_utils.draw_ellipse, {"img_size": 32, "max_size": 8}),
    ]

    for draw_fn, kwargs in cases:
        np.random.seed(7)
        image = draw_fn(**kwargs)

        _assert_binary_int16_image(image, img_size=32)


def test_subterrain_initializes_scaled_fields_and_metadata():
    terrain_config = SimpleNamespace(
        vertical_scale=0.1,
        horizontal_scale=0.5,
        map_width=2.0,
        map_length=3.0,
    )

    terrain = SubTerrain(
        config_terrain=terrain_config,
        terrain_name="stairs",
        device="cpu",
    )

    assert terrain.terrain_name == "stairs"
    assert terrain.config_terrain is terrain_config
    assert terrain.device == "cpu"
    assert terrain.vertical_scale == 0.1
    assert terrain.horizontal_scale == 0.5
    assert terrain.width == 4
    assert terrain.length == 6
    assert terrain.height_field_raw.shape == (4, 6)
    assert terrain.ceiling_field_raw.shape == (4, 6)
    assert terrain.walkable_field_raw.shape == (4, 6)
    assert terrain.height_field_raw.dtype == np.int16
    assert terrain.walkable_field_raw.dtype == np.int16
    assert np.all(terrain.height_field_raw == 0)
    assert np.all(terrain.ceiling_field_raw == 30)
    assert np.all(terrain.walkable_field_raw == 0)
