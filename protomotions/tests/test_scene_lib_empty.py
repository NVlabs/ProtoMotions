# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the empty / Null Object path of protomotions.components.scene_lib.SceneLib.

Stays env-free by exclusively constructing empty scene libraries and exercising
the small motion-id helpers that don't require file IO or a live terrain.
"""

from __future__ import annotations

import pytest
import torch

from protomotions.components.scene_lib import (
    Scene,
    SceneLib,
    SceneLibConfig,
)


# ---------- Empty SceneLib factory and direct constructor ----------------------


def test_scene_lib_empty_factory_zero_envs_creates_empty_library():
    sl = SceneLib.empty(num_envs=0, device="cpu")

    assert sl.num_objects_per_scene == 0
    assert sl.scenes == []
    assert sl._original_scenes == []
    assert sl._scene_offsets == []
    # Tensor placeholders are allocated as empty tensors on the requested device.
    assert sl._object_translations.shape == (0, 3)
    assert sl._object_rotations.shape == (0, 4)
    assert sl._motion_lengths.shape == (0,)
    assert sl._motion_starts.dtype == torch.long


def test_scene_lib_empty_factory_with_num_envs_does_not_replicate():
    """The empty factory takes num_envs but creates no scenes (so the env-shape
    plumbing remains a no-op)."""
    sl = SceneLib.empty(num_envs=4, device="cpu")
    assert sl.num_envs == 4
    assert sl.scenes == []  # no replication happened


def test_scene_lib_direct_construction_with_no_scene_file_and_no_scenes_is_empty():
    """SceneLibConfig(scene_file=None) + scenes=None falls into the Null Object
    branch identical to the factory."""
    sl = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=0,
        scenes=None,
        device="cpu",
    )
    assert sl.num_objects_per_scene == 0
    assert sl._is_static_object.shape == (0, 0)
    assert sl._scene_to_original_scene_id.shape == (0,)


def test_scene_lib_pointcloud_buffers_are_zero_sized_on_empty():
    sl = SceneLib.empty(num_envs=0, device="cpu")
    assert sl._object_pointclouds.shape == (0, 0, 0, 3)
    assert sl._object_pointcloud_normals.shape == (0, 0, 0, 3)
    assert sl._object_bbox_extents.shape == (0, 0, 3)
    assert sl._per_object_valid_mask.shape == (0, 0)
    assert sl._object_class_ids.shape == (0, 0)


def test_scene_lib_empty_indexed_geometry_accessors_return_empty_env_batches():
    sl = SceneLib.empty(num_envs=3, device="cpu")
    env_ids = torch.tensor([0, 2], dtype=torch.long)

    assert sl.get_scene_neutral_pointcloud(env_ids).shape == (2, 0, 0, 3)
    assert sl.get_scene_neutral_pointcloud_normals(env_ids).shape == (2, 0, 0, 3)
    assert sl.get_object_bbox_extents(env_ids).shape == (2, 0, 3)
    assert sl.get_object_scales(torch.device("cpu"), env_ids).shape == (2, 0, 1, 3)


def test_scene_lib_empty_indexed_masks_and_class_ids_return_empty_env_batches():
    sl = SceneLib.empty(num_envs=3, device="cpu")
    env_ids = torch.tensor([0, 2], dtype=torch.long)

    valid_mask = sl.get_per_object_valid_mask(env_ids)
    class_ids = sl.get_object_class_ids(env_ids)

    assert valid_mask.shape == (2, 0)
    assert valid_mask.dtype == torch.bool
    assert class_ids.shape == (2, 0)
    assert class_ids.dtype == torch.long


def test_scene_lib_empty_default_object_state_keeps_num_env_batch_dimension():
    sl = SceneLib.empty(num_envs=3, device="cpu")

    state = sl.get_default_object_state(device="cpu")

    assert state.root_pos.shape == (3, 0, 3)
    assert state.root_rot.shape == (3, 0, 4)
    assert state.root_vel.shape == (3, 0, 3)
    assert state.root_ang_vel.shape == (3, 0, 3)


def test_scene_lib_empty_scene_positions_keep_num_env_batch_dimension():
    sl = SceneLib.empty(num_envs=3, device="cpu")

    positions = sl.get_scene_positions(terrain=None, device="cpu")

    assert torch.equal(positions, torch.zeros(3, 3))


# ---------- num_envs validation when scenes are provided -----------------------


def test_scene_lib_with_scenes_requires_positive_num_envs():
    """Providing inline scenes but num_envs<=0 should fail loudly so an
    accidentally-misconfigured experiment doesn't silently produce an empty
    library."""
    scenes = [Scene()]
    with pytest.raises(ValueError):
        SceneLib(
            config=SceneLibConfig(scene_file=None),
            num_envs=0,
            scenes=scenes,
            device="cpu",
        )


# ---------- humanoid_motion_id helpers on empty / synthetic scenes -------------


def test_get_humanoid_motion_ids_returns_none_when_empty():
    sl = SceneLib.empty(num_envs=0, device="cpu")
    assert sl.get_humanoid_motion_ids() is None


def test_get_humanoid_motion_ids_returns_none_when_all_scenes_use_sentinel():
    sl = SceneLib.empty(num_envs=0, device="cpu")
    sl.scenes = [Scene(humanoid_motion_id=-1), Scene(humanoid_motion_id=-1)]
    assert sl.get_humanoid_motion_ids() is None


def test_get_humanoid_motion_ids_returns_list_when_any_scene_has_real_id():
    sl = SceneLib.empty(num_envs=0, device="cpu")
    sl.scenes = [
        Scene(humanoid_motion_id=-1),
        Scene(humanoid_motion_id=2),
        Scene(humanoid_motion_id=-1),
    ]
    assert sl.get_humanoid_motion_ids() == [-1, 2, -1]


def test_get_per_env_humanoid_motion_ids_tensor_returns_empty_on_empty():
    sl = SceneLib.empty(num_envs=0, device="cpu")
    tensor = sl.get_per_env_humanoid_motion_ids_tensor()
    assert tensor.shape == (0,)
    assert tensor.dtype == torch.long


def test_get_per_env_humanoid_motion_ids_tensor_caches_after_first_call():
    sl = SceneLib.empty(num_envs=0, device="cpu")
    sl.scenes = [Scene(humanoid_motion_id=0), Scene(humanoid_motion_id=3)]

    first = sl.get_per_env_humanoid_motion_ids_tensor()
    # Mutating the underlying scenes list should NOT change subsequent calls
    # because the tensor is cached on the first call.
    sl.scenes[0].humanoid_motion_id = 99
    second = sl.get_per_env_humanoid_motion_ids_tensor()
    assert torch.equal(first, second)
    assert first.tolist() == [0, 3]


# ---------- build_motion_to_original_scene_map --------------------------------


def test_build_motion_to_original_scene_map_returns_negative_one_for_empty():
    sl = SceneLib.empty(num_envs=0, device="cpu")
    mapping = sl.build_motion_to_original_scene_map(num_motions=4)
    assert torch.equal(mapping, torch.tensor([-1, -1, -1, -1], dtype=torch.long))


def test_build_motion_to_original_scene_map_assigns_each_scene_to_its_motion():
    sl = SceneLib.empty(num_envs=0, device="cpu")
    sl._original_scenes = [
        Scene(humanoid_motion_id=2),
        Scene(humanoid_motion_id=0),
        Scene(humanoid_motion_id=-1),  # universal — should not appear
    ]

    mapping = sl.build_motion_to_original_scene_map(num_motions=4)

    # motion 0 ← scene 1; motion 2 ← scene 0; motions 1 and 3 stay at -1.
    assert mapping.tolist() == [1, -1, 0, -1]


def test_build_motion_to_original_scene_map_skips_motion_ids_out_of_range():
    """Motion IDs beyond num_motions should be ignored (not raise / overflow)."""
    sl = SceneLib.empty(num_envs=0, device="cpu")
    sl._original_scenes = [
        Scene(humanoid_motion_id=10),  # out-of-range for num_motions=3
        Scene(humanoid_motion_id=1),
    ]

    mapping = sl.build_motion_to_original_scene_map(num_motions=3)

    # Only scene 1's mapping (motion 1 ← scene 1) should land; scene 0 dropped.
    assert mapping.tolist() == [-1, 1, -1]
