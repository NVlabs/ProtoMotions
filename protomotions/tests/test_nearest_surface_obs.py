# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nearest-surface observation wiring."""

from types import SimpleNamespace

import torch

from protomotions.envs.base_env.publishers import ScenePublisher
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
import protomotions.envs.obs.nearest_surface_obs as nearest_surface_obs_module
from protomotions.envs.obs.nearest_surface_obs import compute_nearest_surface_vectors


def test_nearest_surface_kernel_binds_scene_object_surfaces_unless_unbound():
    # The kernel binds the scene surfaces by default; a recipe whose run has no
    # scene file passes None for the four object inputs, so a configuration
    # without a scene binds no scene name.
    terrain_only = MdpComponent.from_kernel(
        compute_nearest_surface_vectors,
        object_pos=None,
        object_rot=None,
        neutral_pointclouds=None,
        object_valid_mask=None,
    )
    bound = [path.path for path in terrain_only.dynamic_vars.values()]
    assert not any(path.startswith("scene.") for path in bound)
    assert terrain_only.dynamic_vars["height_points"].path == "terrain.height_points"
    assert terrain_only.static_params["object_pos"] is None

    component = MdpComponent.from_kernel(compute_nearest_surface_vectors)
    assert component.dynamic_vars["object_pos"].path == "scene.object_pos"
    assert component.dynamic_vars["object_rot"].path == "scene.object_rot"
    assert (
        component.dynamic_vars["neutral_pointclouds"].path
        == "scene.neutral_pointclouds"
    )
    assert component.dynamic_vars["object_valid_mask"].path == "scene.object_valid_mask"


def test_scene_publisher_exposes_object_surface_tensors():
    object_pos = torch.zeros(2, 3, 3)
    object_rot = torch.zeros(2, 3, 4)
    neutral_pointclouds = torch.zeros(2, 3, 5, 3)
    object_valid_mask = torch.ones(2, 3, dtype=torch.bool)
    queried = []
    scene_lib = SimpleNamespace(
        num_scenes=lambda: 2,
        has_pointclouds=True,
        get_scene_neutral_pointcloud=lambda ids: queried.append(ids)
        or neutral_pointclouds,
        get_per_object_valid_mask=lambda ids: object_valid_mask,
    )
    simulator = SimpleNamespace(
        get_object_root_state=lambda env_ids: SimpleNamespace(
            root_pos=object_pos, root_rot=object_rot
        )
    )

    published = ScenePublisher(scene_lib, simulator, 2, torch.device("cpu")).publish(
        None
    )

    # The scene library's own tensors, not copies.
    assert published["scene.object_pos"] is object_pos
    assert published["scene.object_rot"] is object_rot
    assert published["scene.neutral_pointclouds"] is neutral_pointclouds
    assert published["scene.object_valid_mask"] is object_valid_mask
    assert torch.equal(queried[0], torch.arange(2))
    assert EnvContext.scene.object_pos.path == "scene.object_pos"

    # A subset context queries the selected scenes only.
    subset = ScenePublisher(scene_lib, simulator, 2, torch.device("cpu")).publish(
        torch.tensor([1])
    )
    assert torch.equal(queried[1], torch.tensor([1]))
    assert subset["scene.object_pos"] is object_pos

    # Objects without pointclouds publish the three object names only; no
    # scenes publish nothing (no zero-width stand-in either way).
    scene_lib.has_pointclouds = False
    partial = ScenePublisher(scene_lib, simulator, 2, torch.device("cpu")).publish(None)
    assert set(partial) == {
        "scene.object_pos",
        "scene.object_rot",
        "scene.object_valid_mask",
    }
    scene_lib.num_scenes = lambda: 0
    assert (
        ScenePublisher(scene_lib, simulator, 2, torch.device("cpu")).publish(None) == {}
    )


def test_nearest_surface_terrain_path_reuses_height_points_without_clone(monkeypatch):
    height_points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    before = height_points.clone()
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
        return height_points

    monkeypatch.setattr(
        nearest_surface_obs_module, "get_height_maps_jit", fake_height_maps_jit
    )

    vectors = nearest_surface_obs_module.compute_nearest_surface_vectors(
        rigid_body_pos=torch.tensor([[[0.2, 0.0, 0.0]]]),
        root_pos=torch.tensor([[0.0, 0.0, 0.0]]),
        root_rot=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        height_points=height_points,
        height_samples=torch.zeros(2, 2),
        object_pos=None,
        object_rot=None,
        neutral_pointclouds=None,
        object_valid_mask=None,
        terrain_horizontal_scale=1.0,
    )

    assert seen["height_points_data_ptr"] == height_points.data_ptr()
    assert torch.equal(height_points, before)
    assert vectors.shape == (1, 3)
