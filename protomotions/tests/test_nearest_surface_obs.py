# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nearest-surface observation wiring."""

import torch

from protomotions.envs.component_factories import nearest_surface_obs_factory
from protomotions.envs.context_views import EnvContext, SceneSurfaceContext


def test_nearest_surface_factory_binds_scene_object_surfaces():
    component = nearest_surface_obs_factory()

    assert component.dynamic_vars["object_pos"].path == "scene.object_pos"
    assert component.dynamic_vars["object_rot"].path == "scene.object_rot"
    assert (
        component.dynamic_vars["neutral_pointclouds"].path
        == "scene.neutral_pointclouds"
    )
    assert component.dynamic_vars["object_valid_mask"].path == "scene.object_valid_mask"


def test_scene_surface_context_exposes_object_surface_tensors():
    object_pos = torch.zeros(2, 3, 3)
    object_rot = torch.zeros(2, 3, 4)
    neutral_pointclouds = torch.zeros(2, 3, 5, 3)
    object_valid_mask = torch.ones(2, 3, dtype=torch.bool)

    context = SceneSurfaceContext(
        object_pos=object_pos,
        object_rot=object_rot,
        neutral_pointclouds=neutral_pointclouds,
        object_valid_mask=object_valid_mask,
    )

    assert context.object_pos is object_pos
    assert context.object_rot is object_rot
    assert context.neutral_pointclouds is neutral_pointclouds
    assert context.object_valid_mask is object_valid_mask
    assert EnvContext.scene.object_pos.path == "scene.object_pos"
