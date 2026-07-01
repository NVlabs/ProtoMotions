# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for protomotions.components.scene_lib enum + dataclass helpers
that don't require file IO: Scene container, ReplicationMethod /
SubsetMethod from_str parsers, SceneLibConfig defaults.
"""

from __future__ import annotations

import pytest

from protomotions.components.scene_lib import (
    BoxSceneObject,
    ReplicationMethod,
    Scene,
    SceneLibConfig,
    SubsetMethod,
)


# ---------- Scene container ----------------------------------------------------


def _make_box(width=1.0, depth=1.0, height=1.0):
    return BoxSceneObject(
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        width=width,
        depth=depth,
        height=height,
    )


def test_scene_default_construction_is_empty_with_origin_offset():
    scene = Scene()
    assert scene.objects == []
    assert scene.offset == (0.0, 0.0)
    assert scene.humanoid_motion_id == -1


def test_scene_add_object_appends_in_order():
    scene = Scene()
    a = _make_box(width=1.0)
    b = _make_box(width=2.0)
    scene.add_object(a)
    scene.add_object(b)

    assert scene.objects == [a, b]


def test_scene_offset_overrides_default():
    scene = Scene(offset=(1.5, -2.5))
    assert scene.offset == (1.5, -2.5)


def test_scene_humanoid_motion_id_can_be_set():
    scene = Scene(humanoid_motion_id=7)
    assert scene.humanoid_motion_id == 7


# ---------- ReplicationMethod ---------------------------------------------------


def test_replication_method_known_values_round_trip_through_from_str():
    for member in ReplicationMethod:
        recovered = ReplicationMethod.from_str(member.value)
        assert recovered is member


def test_replication_method_from_str_is_case_insensitive():
    assert ReplicationMethod.from_str("WEIGHTED") is ReplicationMethod.WEIGHTED
    assert ReplicationMethod.from_str("Random") is ReplicationMethod.RANDOM
    assert ReplicationMethod.from_str("SEQUENTIAL") is ReplicationMethod.SEQUENTIAL


def test_replication_method_from_str_rejects_unknown_with_helpful_error():
    with pytest.raises(ValueError) as exc:
        ReplicationMethod.from_str("nope")
    assert "ReplicationMethod" in str(exc.value)
    # Error message should enumerate the valid values.
    for member in ReplicationMethod:
        assert member.value in str(exc.value)


# ---------- SubsetMethod -------------------------------------------------------


def test_subset_method_known_values_round_trip_through_from_str():
    for member in SubsetMethod:
        recovered = SubsetMethod.from_str(member.value)
        assert recovered is member


def test_subset_method_from_str_is_case_insensitive():
    assert SubsetMethod.from_str("FIRST") is SubsetMethod.FIRST
    assert SubsetMethod.from_str("Last") is SubsetMethod.LAST
    assert SubsetMethod.from_str("sequential") is SubsetMethod.SEQUENTIAL


def test_subset_method_from_str_rejects_unknown_with_helpful_error():
    with pytest.raises(ValueError) as exc:
        SubsetMethod.from_str("middle")
    assert "SubsetMethod" in str(exc.value)


# ---------- SceneLibConfig defaults --------------------------------------------


def test_scene_lib_config_defaults_to_first_subset_and_weighted_replication():
    config = SceneLibConfig()
    assert config.subset_method is SubsetMethod.FIRST
    assert config.replicate_method is ReplicationMethod.WEIGHTED


def test_scene_lib_config_optional_fields_default_to_none():
    config = SceneLibConfig()
    # Optional file-related fields default to None when not set.
    assert config.scene_file is None
    assert config.asset_root is None
    assert config.scene_indices is None
    assert config.inline_scenes is None
    assert config.pointcloud_samples_per_object is None
    assert config.mesh_collision_approximation is None
    assert config.mesh_collision_max_convex_hulls is None


def test_scene_lib_config_target_string_points_at_scene_lib_class():
    """The _target_ string must remain the canonical SceneLib import path so
    the config-driven instantiation in the agent stack continues to resolve."""
    config = SceneLibConfig()
    assert config._target_ == "protomotions.components.scene_lib.SceneLib"


def test_scene_lib_config_subset_method_accepts_explicit_index_list():
    """subset_method is Union[SubsetMethod, List[int]]; the dataclass must
    pass through a list unchanged."""
    config = SceneLibConfig(subset_method=[0, 3, 5])
    assert config.subset_method == [0, 3, 5]


def test_scene_lib_config_inline_scenes_accepts_programmatic_scene_list():
    s1 = Scene(offset=(0.0, 0.0))
    s2 = Scene(offset=(1.0, 0.0))
    config = SceneLibConfig(inline_scenes=[s1, s2])
    assert config.inline_scenes == [s1, s2]
    assert config.scene_file is None


def test_scene_lib_config_mesh_collision_overrides_propagate():
    config = SceneLibConfig(
        mesh_collision_approximation="convexDecomposition",
        mesh_collision_max_convex_hulls=16,
        mesh_collision_hull_vertex_limit=48,
        mesh_collision_voxel_resolution=200000,
    )
    assert config.mesh_collision_approximation == "convexDecomposition"
    assert config.mesh_collision_max_convex_hulls == 16
    assert config.mesh_collision_hull_vertex_limit == 48
    assert config.mesh_collision_voxel_resolution == 200000
