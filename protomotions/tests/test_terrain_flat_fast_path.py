# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

import protomotions.components.terrains.terrain as terrain_module
from protomotions.components.terrains.terrain import Terrain


def _terrain_with_proportions(proportions: list[float]) -> Terrain:
    terrain = object.__new__(Terrain)
    terrain.config = SimpleNamespace(
        load_terrain=False, terrain_proportions=proportions
    )
    terrain.height_samples = torch.zeros(4, 4)
    terrain.horizontal_scale = 0.1
    return terrain


def test_flat_terrain_ground_heights_skip_kernel_and_preserve_shapes(monkeypatch):
    terrain = _terrain_with_proportions([0.0] * 7 + [1.0])

    def fail_kernel(**kwargs):
        raise AssertionError("height kernel should not run for flat terrain")

    monkeypatch.setattr(terrain_module, "get_heights_jit", fail_kernel)

    heights_2d = terrain.get_ground_heights(torch.zeros(3, 3))
    heights_3d = terrain.get_ground_heights(torch.zeros(3, 2, 3))

    assert heights_2d.shape == (3, 1)
    assert heights_3d.shape == (3, 2)
    assert torch.equal(heights_2d, torch.zeros(3, 1))
    assert torch.equal(heights_3d, torch.zeros(3, 2))


def test_flat_terrain_ground_heights_preserve_kernel_dtype_and_autograd_contract():
    terrain = _terrain_with_proportions([0.0] * 7 + [1.0])
    integer_locations = torch.zeros(3, 3, dtype=torch.int64)
    differentiable_locations = torch.zeros(
        3,
        3,
        dtype=torch.float64,
        requires_grad=True,
    )

    integer_heights = terrain.get_ground_heights(integer_locations)
    differentiable_heights = terrain.get_ground_heights(differentiable_locations)
    differentiable_heights.sum().backward()

    assert integer_heights.dtype == terrain.height_samples.dtype
    assert differentiable_heights.dtype == differentiable_locations.dtype
    assert differentiable_heights.requires_grad
    assert torch.equal(
        differentiable_locations.grad,
        torch.zeros_like(differentiable_locations),
    )


def test_cancelling_nonflat_proportions_do_not_enable_flat_fast_path():
    terrain = _terrain_with_proportions([0.25, -0.25] + [0.0] * 5 + [1.0])

    assert terrain.is_flat() is False


def test_nonflat_terrain_ground_heights_still_uses_kernel(monkeypatch):
    terrain = _terrain_with_proportions([1.0] + [0.0] * 7)
    expected = torch.ones(3, 2)
    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(terrain_module, "get_heights_jit", fake_kernel)

    assert terrain.get_ground_heights(torch.zeros(3, 2, 3)) is expected
    assert len(calls) == 1
