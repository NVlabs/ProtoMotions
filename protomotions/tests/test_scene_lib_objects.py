# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for protomotions.components.scene_lib object configuration helpers
and primitive scene object math (Box / Sphere / Cylinder).

Avoids file IO and trimesh by sticking to primitives and option dataclasses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import protomotions.components.scene_lib as scene_lib_module
from protomotions.components.scene_lib import (
    DEFAULT_PRIMITIVE_DENSITY,
    BoxSceneObject,
    CylinderSceneObject,
    ObjectOptions,
    PrimitiveSceneObject,
    SceneObject,
    SphereSceneObject,
)


# ---------- ObjectOptions ------------------------------------------------------


def test_object_options_defaults_density_when_neither_mass_nor_density_given():
    options = ObjectOptions()
    assert options.density == DEFAULT_PRIMITIVE_DENSITY
    assert options.mass is None


def test_object_options_accepts_density_only():
    options = ObjectOptions(density=2500.0)
    assert options.density == 2500.0
    assert options.mass is None


def test_object_options_accepts_mass_only():
    options = ObjectOptions(mass=12.5)
    assert options.mass == 12.5
    assert options.density is None


def test_object_options_rejects_mass_and_density_simultaneously():
    with pytest.raises(ValueError):
        ObjectOptions(mass=1.0, density=1000.0)


def test_object_options_to_dict_filters_none_top_level_fields():
    options = ObjectOptions(mass=4.0, color=(0.1, 0.2, 0.3))
    out = options.to_dict()

    assert out["mass"] == 4.0
    assert out["color"] == (0.1, 0.2, 0.3)
    # density and other unspecified fields should not appear.
    assert "density" not in out
    assert "fix_base_link" not in out
    assert "texture_path" not in out


def test_object_options_to_dict_recursively_filters_none_in_vhacd_params():
    """vhacd_params has all-None defaults — it should be omitted entirely."""
    options = ObjectOptions(mass=1.0)
    out = options.to_dict()
    assert "vhacd_params" not in out


def test_object_options_to_dict_keeps_partially_populated_nested_dict():
    options = ObjectOptions(
        mass=1.0,
        vhacd_params={"resolution": 100, "max_convex_hulls": None, "max_num_vertices_per_ch": None},
    )
    out = options.to_dict()
    assert out["vhacd_params"] == {"resolution": 100}


# ---------- Abstract SceneObject helpers --------------------------------------


def test_scene_object_base_methods_require_subclass_implementation():
    obj = SceneObject(
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )

    with pytest.raises(NotImplementedError, match="object_identifier"):
        _ = obj.object_identifier
    with pytest.raises(NotImplementedError, match="calculate_dimensions"):
        obj.calculate_dimensions()
    with pytest.raises(NotImplementedError, match="compute_pointcloud"):
        obj.compute_pointcloud(4)


def test_primitive_scene_object_keeps_identifier_abstract():
    primitive = PrimitiveSceneObject(
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        object_dims=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5),
    )

    with pytest.raises(NotImplementedError, match="object_identifier"):
        _ = primitive.object_identifier


# ---------- BoxSceneObject -----------------------------------------------------


def _box(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0), **kwargs):
    return BoxSceneObject(
        translation=translation,
        rotation=rotation,
        width=kwargs.get("width", 1.0),
        depth=kwargs.get("depth", 2.0),
        height=kwargs.get("height", 0.5),
    )


def test_box_requires_all_three_dimensions():
    with pytest.raises(ValueError):
        BoxSceneObject(
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            width=1.0,
            depth=None,
            height=0.5,
        )


def test_box_calculate_dimensions_is_centered_axis_aligned_extent():
    box = _box(width=4.0, depth=2.0, height=1.0)
    assert box.calculate_dimensions() == (-2.0, 2.0, -1.0, 1.0, -0.5, 0.5)


def test_box_post_init_populates_object_dims_and_normalizes_translation_shape():
    box = _box(translation=(1.0, 2.0, 3.0))
    assert box.object_dims is not None
    # SceneObject._convert_to_tensor reshapes static input to (1, 3) / (1, 4).
    assert box.translation.shape == (1, 3)
    assert box.rotation.shape == (1, 4)
    assert box.has_motion() is False
    assert box.fps == 1.0


def test_box_object_identifier_encodes_dimensions_with_underscores():
    box = _box(width=1.5, depth=2.0, height=0.25)
    assert box.object_identifier == "box_w1_5_d2_0_h0_25"


def test_box_compute_pointcloud_with_eight_returns_eight_corners():
    box = _box(width=2.0, depth=2.0, height=2.0)
    box.compute_pointcloud(8)

    assert box.object_pointcloud.shape == (8, 3)
    # All eight corners have |x|=|y|=|z|=1 for this unit cube.
    assert torch.allclose(box.object_pointcloud.abs(), torch.ones(8, 3))


def test_box_compute_pointcloud_surface_sampler_returns_requested_count_and_normals():
    box = _box(width=2.0, depth=4.0, height=1.0)
    box.compute_pointcloud(30)

    points = box.object_pointcloud
    normals = box.object_pointcloud_normals
    assert points.shape == (30, 3)
    assert normals.shape == (30, 3)
    assert torch.allclose(normals.norm(dim=-1), torch.ones(30))
    # Every sampled point lies on at least one face of the centered box.
    on_x = torch.isclose(points[:, 0].abs(), torch.tensor(1.0))
    on_y = torch.isclose(points[:, 1].abs(), torch.tensor(2.0))
    on_z = torch.isclose(points[:, 2].abs(), torch.tensor(0.5))
    assert torch.all(on_x | on_y | on_z)
    # Face normals are axis-aligned one-hot vectors for the grid sampler.
    assert torch.all(normals.abs().sum(dim=-1) == 1.0)


def test_box_compute_pointcloud_handles_zero_allocated_faces_for_small_samples():
    box = _box(width=1.0, depth=8.0, height=1.0)

    box.compute_pointcloud(9)

    assert box.object_pointcloud.shape == (9, 3)
    assert box.object_pointcloud_normals.shape == (9, 3)
    assert torch.allclose(box.object_pointcloud_normals.norm(dim=-1), torch.ones(9))


def test_box_compute_pointcloud_duplicates_underfilled_grids_to_requested_count():
    box = _box(width=1.0, depth=8.0, height=1.0)

    box.compute_pointcloud(38)

    assert box.object_pointcloud.shape == (38, 3)
    assert box.object_pointcloud_normals.shape == (38, 3)
    assert torch.unique(box.object_pointcloud, dim=0).shape[0] < 38
    assert torch.allclose(box.object_pointcloud_normals.norm(dim=-1), torch.ones(38))


def test_box_identifier_formats_none_dimension_if_state_is_partially_loaded():
    box = _box(width=1.0, depth=2.0, height=3.0)
    box.width = None

    assert box.object_identifier == "box_wnone_d2_0_h3_0"


# ---------- SphereSceneObject --------------------------------------------------


def _sphere(radius=1.0):
    return SphereSceneObject(
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        radius=radius,
    )


def test_sphere_requires_radius():
    with pytest.raises(ValueError):
        SphereSceneObject(
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )


def test_sphere_calculate_dimensions_returns_symmetric_axis_aligned_bbox():
    sphere = _sphere(radius=2.5)
    assert sphere.calculate_dimensions() == (-2.5, 2.5, -2.5, 2.5, -2.5, 2.5)


def test_sphere_object_identifier_encodes_radius():
    sphere = _sphere(radius=0.5)
    assert sphere.object_identifier == "sphere_r0_5"


def test_sphere_identifier_formats_none_radius_if_state_is_partially_loaded():
    sphere = _sphere(radius=0.5)
    sphere.radius = None

    assert sphere.object_identifier == "sphere_rnone"


def test_sphere_compute_pointcloud_returns_unit_norm_points_and_normals():
    sphere = _sphere(radius=1.0)
    sphere.compute_pointcloud(64)

    assert sphere.object_pointcloud.shape == (64, 3)
    # All points lie on unit sphere surface.
    assert torch.allclose(
        sphere.object_pointcloud.norm(dim=-1),
        torch.ones(64),
        atol=1e-5,
    )
    # Normals match position direction (radius=1).
    assert torch.allclose(
        sphere.object_pointcloud_normals.norm(dim=-1),
        torch.ones(64),
        atol=1e-5,
    )


def test_sphere_pointcloud_scales_with_radius():
    sphere = _sphere(radius=3.0)
    sphere.compute_pointcloud(32)
    assert torch.allclose(
        sphere.object_pointcloud.norm(dim=-1),
        torch.full((32,), 3.0),
        atol=1e-4,
    )


# ---------- CylinderSceneObject ------------------------------------------------


def _cylinder(radius=1.0, height=2.0):
    return CylinderSceneObject(
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        radius=radius,
        height=height,
    )


def test_cylinder_requires_radius_and_height():
    with pytest.raises(ValueError):
        CylinderSceneObject(
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            radius=1.0,
        )
    with pytest.raises(ValueError):
        CylinderSceneObject(
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            height=2.0,
        )


def test_cylinder_calculate_dimensions_uses_radius_for_xy_and_height_for_z():
    cyl = _cylinder(radius=1.5, height=4.0)
    assert cyl.calculate_dimensions() == (-1.5, 1.5, -1.5, 1.5, -2.0, 2.0)


def test_cylinder_object_identifier_encodes_radius_and_height():
    cyl = _cylinder(radius=0.25, height=1.5)
    assert cyl.object_identifier == "cylinder_r0_25_h1_5"


def test_cylinder_identifier_formats_none_radius_if_state_is_partially_loaded():
    cyl = _cylinder(radius=0.25, height=1.5)
    cyl.radius = None

    assert cyl.object_identifier == "cylinder_rnone_h1_5"


def test_cylinder_compute_pointcloud_returns_requested_count_and_unit_normals():
    cyl = _cylinder(radius=1.0, height=2.0)
    cyl.compute_pointcloud(25)

    assert cyl.object_pointcloud.shape == (25, 3)
    assert cyl.object_pointcloud_normals.shape == (25, 3)
    assert torch.allclose(
        cyl.object_pointcloud_normals.norm(dim=-1),
        torch.ones(25),
        atol=1e-6,
    )
    assert torch.all(cyl.object_pointcloud[:, 2] >= -1.0)
    assert torch.all(cyl.object_pointcloud[:, 2] <= 1.0)
    assert (cyl.object_pointcloud_normals[:, 2] == 1.0).any()
    assert (cyl.object_pointcloud_normals[:, 2] == -1.0).any()


def test_cylinder_compute_pointcloud_preserves_requested_count_for_small_samples():
    cyl = _cylinder(radius=1.0, height=2.0)

    cyl.compute_pointcloud(2)

    assert cyl.object_pointcloud.shape == (2, 3)
    assert cyl.object_pointcloud_normals.shape == (2, 3)


def test_cylinder_compute_pointcloud_single_sample_uses_cap_only():
    cyl = _cylinder(radius=1.0, height=2.0)

    cyl.compute_pointcloud(1)

    assert cyl.object_pointcloud.shape == (1, 3)
    assert cyl.object_pointcloud_normals.shape == (1, 3)
    assert torch.equal(cyl.object_pointcloud_normals, torch.tensor([[0.0, 0.0, -1.0]]))


def test_cylinder_cap_sampler_trims_overgenerated_grid(monkeypatch):
    original_sqrt = scene_lib_module.np.sqrt

    def inflated_sqrt(value):
        if value == 1:
            return 2.0
        return original_sqrt(value)

    monkeypatch.setattr(scene_lib_module.np, "sqrt", inflated_sqrt)
    cyl = _cylinder(radius=1.0, height=2.0)

    cyl.compute_pointcloud(1)

    assert cyl.object_pointcloud.shape == (1, 3)
    assert torch.equal(cyl.object_pointcloud_normals, torch.tensor([[0.0, 0.0, -1.0]]))


# ---------- SceneObject motion handling (via BoxSceneObject) -------------------


def test_box_with_motion_requires_fps_and_matches_translation_to_rotation_count():
    translations = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)]

    box = BoxSceneObject(
        translation=translations,
        rotation=rotations,
        width=1.0,
        depth=1.0,
        height=1.0,
        fps=30.0,
    )

    assert box.has_motion() is True
    assert box.translation.shape == (3, 3)
    assert box.rotation.shape == (3, 4)
    assert box.fps == 30.0


def test_box_with_motion_without_fps_raises():
    translations = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)]
    with pytest.raises(AssertionError):
        BoxSceneObject(
            translation=translations,
            rotation=rotations,
            width=1.0,
            depth=1.0,
            height=1.0,
        )


def test_box_motion_translation_rotation_count_mismatch_raises():
    translations = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)]
    with pytest.raises(AssertionError):
        BoxSceneObject(
            translation=translations,
            rotation=rotations,
            width=1.0,
            depth=1.0,
            height=1.0,
            fps=30.0,
        )


def test_box_start_pose_returns_first_frame_for_moving_object():
    translations = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    rotations = [(0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 0.0)]
    box = BoxSceneObject(
        translation=translations,
        rotation=rotations,
        width=1.0,
        depth=1.0,
        height=1.0,
        fps=30.0,
    )
    pose = box.start_pose
    # start_pose extracts the first frame (un-batched after .__getitem__).
    assert torch.equal(pose.root_pos, torch.tensor([0.0, 0.0, 0.0]))
    assert torch.equal(pose.root_rot, torch.tensor([0.0, 0.0, 0.0, 1.0]))


# ---------- _convert_to_tensor accepts numpy and tensor inputs -----------------


def test_box_translation_accepts_numpy_input():
    box = BoxSceneObject(
        translation=np.array([0.5, 1.5, 2.5], dtype=np.float32),
        rotation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        width=1.0,
        depth=1.0,
        height=1.0,
    )
    assert box.translation.shape == (1, 3)
    assert torch.allclose(box.translation[0], torch.tensor([0.5, 1.5, 2.5]))


def test_box_translation_accepts_torch_tensor_input_and_clones_it():
    src = torch.tensor([1.0, 2.0, 3.0])
    box = BoxSceneObject(
        translation=src,
        rotation=torch.tensor([0.0, 0.0, 0.0, 1.0]),
        width=1.0,
        depth=1.0,
        height=1.0,
    )
    assert box.translation.shape == (1, 3)
    # Underlying storage is independent — mutating the source doesn't affect
    # the SceneObject's tensor.
    src[0] = 99.0
    assert box.translation[0, 0].item() == 1.0
