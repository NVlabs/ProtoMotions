# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SceneLib loading and asset bookkeeping tests that stay simulator-free."""

from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import protomotions.components.scene_lib as scene_lib_module
from protomotions.components.scene_lib import (
    BoxSceneObject,
    CylinderSceneObject,
    MeshSceneObject,
    ObjectOptions,
    ReplicationMethod,
    Scene,
    SceneLib,
    SceneLibConfig,
    SphereSceneObject,
    _sample_mesh_pointcloud,
)


def _box(
    width=1.0,
    depth=1.0,
    height=1.0,
    translation=(0.0, 0.0, 0.0),
    humanoid_motion_id=None,
):
    obj = BoxSceneObject(
        translation=translation,
        rotation=(0.0, 0.0, 0.0, 1.0),
        width=width,
        depth=depth,
        height=height,
        options=ObjectOptions(fix_base_link=True),
    )
    if humanoid_motion_id is None:
        return obj
    return Scene(objects=[obj], humanoid_motion_id=humanoid_motion_id)


def _mesh(path, scale=(1.0, 1.0, 1.0)):
    return MeshSceneObject(
        object_path=str(path),
        object_dims=(-0.5, 0.5, -1.0, 1.0, 0.0, 2.0),
        scale=scale,
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )


def test_inline_scenes_config_constructs_library_without_scenes_argument():
    scenes = [
        Scene(objects=[_box(width=1.0)], humanoid_motion_id=4),
        Scene(objects=[_box(width=2.0)], humanoid_motion_id=5),
    ]

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            inline_scenes=scenes,
            replicate_method=ReplicationMethod.SEQUENTIAL,
        ),
        num_envs=3,
        scenes=None,
        device="cpu",
    )

    assert scene_lib.num_scenes() == 3
    assert scene_lib._scene_to_original_scene_id.tolist() == [0, 1, 0]
    assert scene_lib.get_humanoid_motion_ids() == [4, 5, 4]


def test_inline_scenes_rejects_explicit_scenes_argument():
    scenes = [Scene(objects=[_box()])]

    with pytest.raises(ValueError, match="inline_scenes"):
        SceneLib(
            config=SceneLibConfig(scene_file=None, inline_scenes=scenes),
            num_envs=1,
            scenes=scenes,
            device="cpu",
        )


def test_inline_scenes_rejects_scene_file_configuration(tmp_path):
    scene_file = tmp_path / "scenes.pt"
    SceneLib.save_scenes_to_file([Scene(objects=[_box()])], str(scene_file))

    with pytest.raises(ValueError, match="inline_scenes"):
        SceneLib(
            config=SceneLibConfig(
                scene_file=str(scene_file),
                inline_scenes=[Scene(objects=[_box(width=2.0)])],
            ),
            num_envs=1,
            scenes=None,
            device="cpu",
        )


def test_scene_indices_filtering_loads_requested_file_scenes_in_order(tmp_path):
    scene_file = tmp_path / "scenes.pt"
    scenes = [
        Scene(objects=[_box(width=1.0)], humanoid_motion_id=10),
        Scene(objects=[_box(width=2.0)], humanoid_motion_id=20),
        Scene(objects=[_box(width=3.0)], humanoid_motion_id=30),
    ]
    SceneLib.save_scenes_to_file(scenes, str(scene_file))

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=str(scene_file),
            scene_indices=[2, 0],
        ),
        num_envs=2,
        scenes=None,
        device="cpu",
    )

    assert scene_lib.get_humanoid_motion_ids() == [30, 10]
    assert scene_lib._original_scenes[0].objects[0].width == 3.0
    assert scene_lib._original_scenes[1].objects[0].width == 1.0


def test_config_asset_root_resolves_relative_mesh_paths_when_loading(tmp_path):
    asset_root = tmp_path / "asset_root"
    scene_dir = tmp_path / "packs"
    mesh_path = asset_root / "meshes" / "crate.obj"
    scene_file = scene_dir / "scene_pack.pt"
    mesh_path.parent.mkdir(parents=True)
    scene_dir.mkdir(parents=True)

    scene = Scene(objects=[_mesh(mesh_path, scale=(2.0, 3.0, 4.0))])
    SceneLib.save_scenes_to_file([scene], str(scene_file), asset_root=str(asset_root))

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=str(scene_file),
            asset_root=str(asset_root),
        ),
        num_envs=1,
        scenes=None,
        device="cpu",
    )

    loaded_obj = scene_lib._original_scenes[0].objects[0]
    assert loaded_obj.object_path == str(mesh_path)
    assert loaded_obj.scale == (2.0, 3.0, 4.0)


def test_mesh_scene_object_requires_object_path():
    with pytest.raises(ValueError, match="object_path"):
        MeshSceneObject(
            object_path=None,
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )


def test_mesh_calculate_dimensions_uses_stl_when_obj_is_absent(tmp_path, monkeypatch):
    object_path = tmp_path / "prop.usd"
    stl_path = tmp_path / "prop.stl"
    fake_mesh = object()
    loaded_paths = []

    def fake_exists(path):
        return path == str(stl_path)

    monkeypatch.setattr(scene_lib_module.os.path, "exists", fake_exists)
    monkeypatch.setattr(
        scene_lib_module.trimesh,
        "load_mesh",
        lambda path: loaded_paths.append(path) or fake_mesh,
    )
    monkeypatch.setattr(scene_lib_module, "as_mesh", lambda mesh: mesh)
    monkeypatch.setattr(
        scene_lib_module,
        "compute_bounding_box",
        lambda mesh: (2.0, 3.0, 4.0, -1.0, -2.0, -3.0),
    )

    mesh = MeshSceneObject(
        object_path=str(object_path),
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )

    assert loaded_paths == [str(stl_path)]
    assert mesh.object_dims == (-1.0, 1.0, -2.0, 1.0, -3.0, 1.0)


def test_mesh_calculate_dimensions_prefers_obj_when_available(tmp_path, monkeypatch):
    object_path = tmp_path / "prop.usd"
    obj_path = tmp_path / "prop.obj"
    stl_path = tmp_path / "prop.stl"
    fake_mesh = object()
    loaded_paths = []

    def fake_exists(path):
        return path in {str(obj_path), str(stl_path)}

    monkeypatch.setattr(scene_lib_module.os.path, "exists", fake_exists)
    monkeypatch.setattr(
        scene_lib_module.trimesh,
        "load_mesh",
        lambda path: loaded_paths.append(path) or fake_mesh,
    )
    monkeypatch.setattr(scene_lib_module, "as_mesh", lambda mesh: mesh)
    monkeypatch.setattr(
        scene_lib_module,
        "compute_bounding_box",
        lambda mesh: (1.0, 2.0, 3.0, 0.5, 1.5, -0.5),
    )

    mesh = MeshSceneObject(
        object_path=str(object_path),
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )

    assert loaded_paths == [str(obj_path)]
    assert mesh.object_dims == (0.5, 1.5, 1.5, 3.5, -0.5, 2.5)


def test_mesh_calculate_dimensions_uses_ply_when_obj_and_stl_are_absent(
    tmp_path, monkeypatch
):
    object_path = tmp_path / "prop.usda"
    ply_path = tmp_path / "prop.ply"
    fake_mesh = object()
    loaded_paths = []

    def fake_exists(path):
        return path == str(ply_path)

    monkeypatch.setattr(scene_lib_module.os.path, "exists", fake_exists)
    monkeypatch.setattr(
        scene_lib_module.trimesh,
        "load_mesh",
        lambda path: loaded_paths.append(path) or fake_mesh,
    )
    monkeypatch.setattr(scene_lib_module, "as_mesh", lambda mesh: mesh)
    monkeypatch.setattr(
        scene_lib_module,
        "compute_bounding_box",
        lambda mesh: (1.5, 2.5, 3.5, -2.0, -3.0, 0.25),
    )

    mesh = MeshSceneObject(
        object_path=str(object_path),
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )

    assert loaded_paths == [str(ply_path)]
    assert mesh.object_dims == (-2.0, -0.5, -3.0, -0.5, 0.25, 3.75)


def test_mesh_scene_object_raises_when_no_resolved_mesh_file_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="Object file not found"):
        MeshSceneObject(
            object_path=str(tmp_path / "missing.usd"),
            translation=(0.0, 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
        )


def test_mesh_compute_pointcloud_stores_sampled_points_and_normals(monkeypatch):
    monkeypatch.setattr(
        scene_lib_module,
        "_sample_mesh_pointcloud",
        lambda path, n: (
            np.arange(n * 3, dtype=np.float32).reshape(n, 3),
            np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float32), (n, 1)),
        ),
    )
    mesh = MeshSceneObject(
        object_path="mesh.obj",
        object_dims=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )

    mesh.compute_pointcloud(3)

    assert torch.equal(
        mesh.object_pointcloud,
        torch.tensor(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
        ),
    )
    assert torch.equal(
        mesh.object_pointcloud_normals,
        torch.tensor([[0.0, 0.0, 1.0]]).expand(3, 3),
    )


def test_serialize_deserialize_preserves_primitive_type_specific_fields():
    sphere = SphereSceneObject(
        radius=0.75,
        translation=(1.0, 2.0, 3.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
        options=ObjectOptions(mass=3.0),
    )
    cylinder = CylinderSceneObject(
        radius=0.5,
        height=2.5,
        translation=[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        rotation=[(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)],
        fps=10.0,
        options=ObjectOptions(density=1200.0),
    )
    serialized = SceneLib._serialize_scenes_for_storage_static(
        [Scene(objects=[sphere, cylinder], offset=(4.0, 5.0), humanoid_motion_id=9)]
    )

    restored = SceneLib._deserialize_scenes_from_storage_static(serialized)

    assert len(restored) == 1
    assert restored[0].offset == (4.0, 5.0)
    assert restored[0].humanoid_motion_id == 9
    restored_sphere, restored_cylinder = restored[0].objects
    assert isinstance(restored_sphere, SphereSceneObject)
    assert restored_sphere.radius == 0.75
    assert restored_sphere.options.mass == 3.0
    assert isinstance(restored_cylinder, CylinderSceneObject)
    assert restored_cylinder.radius == 0.5
    assert restored_cylinder.height == 2.5
    assert restored_cylinder.has_motion() is True
    assert restored_cylinder.fps == 10.0
    assert restored_cylinder.options.density == 1200.0


def test_serialize_deserialize_preserves_mesh_scale_path_dims_and_options(tmp_path):
    mesh_path = tmp_path / "props" / "crate.obj"
    mesh = MeshSceneObject(
        object_path=str(mesh_path),
        object_dims=(-0.5, 0.5, -1.0, 1.0, 0.0, 2.0),
        scale=(2.0, 3.0, 4.0),
        translation=[(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
        rotation=[(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)],
        fps=12.0,
        options=ObjectOptions(
            mass=5.0,
            fix_base_link=False,
            vhacd_params={"resolution": 100, "max_convex_hulls": 4},
            texture_path="textures/crate.png",
            color=(0.2, 0.3, 0.4),
        ),
    )

    serialized = SceneLib._serialize_scenes_for_storage_static(
        [Scene(objects=[mesh], offset=(1.0, -1.0), humanoid_motion_id=3)]
    )
    restored = SceneLib._deserialize_scenes_from_storage_static(serialized)

    restored_mesh = restored[0].objects[0]
    assert isinstance(restored_mesh, MeshSceneObject)
    assert restored[0].offset == (1.0, -1.0)
    assert restored[0].humanoid_motion_id == 3
    assert restored_mesh.object_path == str(mesh_path)
    assert restored_mesh.object_dims == (-0.5, 0.5, -1.0, 1.0, 0.0, 2.0)
    assert restored_mesh.scale == (2.0, 3.0, 4.0)
    assert restored_mesh.has_motion() is True
    assert restored_mesh.fps == 12.0
    assert restored_mesh.options.mass == 5.0
    assert restored_mesh.options.fix_base_link is False
    assert restored_mesh.options.vhacd_params == {
        "resolution": 100,
        "max_convex_hulls": 4,
    }
    assert restored_mesh.options.texture_path == "textures/crate.png"
    assert restored_mesh.options.color == (0.2, 0.3, 0.4)


def test_scene_file_loading_deserializes_all_primitive_object_types(tmp_path):
    scene_file = tmp_path / "scene_pack.pt"
    scenes = [
        Scene(
            objects=[
                _box(width=1.5, depth=2.0, height=0.25),
                SphereSceneObject(
                    radius=0.75,
                    translation=(1.0, 0.0, 0.0),
                    rotation=(0.0, 0.0, 0.0, 1.0),
                    options=ObjectOptions(mass=2.0),
                ),
                CylinderSceneObject(
                    radius=0.5,
                    height=1.25,
                    translation=[(0.0, 0.0, 0.0), (0.0, 0.5, 0.0)],
                    rotation=[(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)],
                    fps=20.0,
                    options=ObjectOptions(fix_base_link=False),
                ),
            ],
            offset=(2.0, -3.0),
            humanoid_motion_id=42,
        )
    ]
    SceneLib.save_scenes_to_file(scenes, str(scene_file))

    loaded = SceneLib(
        config=SceneLibConfig(scene_file=str(scene_file)),
        num_envs=1,
        scenes=None,
        device="cpu",
    )

    loaded_box, loaded_sphere, loaded_cylinder = loaded._original_scenes[0].objects
    assert isinstance(loaded_box, BoxSceneObject)
    assert loaded_box.width == 1.5
    assert isinstance(loaded_sphere, SphereSceneObject)
    assert loaded_sphere.radius == 0.75
    assert loaded_sphere.options.mass == 2.0
    assert isinstance(loaded_cylinder, CylinderSceneObject)
    assert loaded_cylinder.height == 1.25
    assert loaded_cylinder.has_motion() is True
    assert loaded.get_humanoid_motion_ids() == [42]


def test_deserialize_rejects_unknown_object_type_with_helpful_error():
    serialized = [
        {
            "offset": (0.0, 0.0),
            "humanoid_motion_id": -1,
            "objects": [
                {
                    "type": "CapsuleSceneObject",
                    "translation": [[0.0, 0.0, 0.0]],
                    "rotation": [[0.0, 0.0, 0.0, 1.0]],
                    "fps": 1.0,
                    "object_dims": None,
                    "options": {},
                }
            ],
        }
    ]

    with pytest.raises(ValueError, match="Unsupported SceneObject type.*CapsuleSceneObject"):
        SceneLib._deserialize_scenes_from_storage_static(serialized)


def test_save_scenes_to_file_rejects_non_pt_suffix(tmp_path):
    with pytest.raises(AssertionError, match="File path must end with .pt"):
        SceneLib.save_scenes_to_file(
            [Scene(objects=[_box()])],
            str(tmp_path / "scenes.pkl"),
        )


def test_save_scenes_to_file_rejects_inconsistent_object_counts(tmp_path):
    scenes = [
        Scene(objects=[_box()]),
        Scene(objects=[_box(), _box(width=2.0)]),
    ]

    with pytest.raises(ValueError, match="same number of objects"):
        SceneLib.save_scenes_to_file(scenes, str(tmp_path / "scenes.pt"))


def test_load_scenes_from_file_raises_when_file_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="SceneLib file not found"):
        SceneLib._load_scenes_from_file(str(tmp_path / "missing.pt"), device="cpu")


def test_save_scenes_to_file_keeps_absolute_mesh_path_when_relpath_fails(
    tmp_path, monkeypatch
):
    mesh_path = tmp_path / "mesh.obj"
    scene_file = tmp_path / "scenes.pt"
    scene = Scene(objects=[_mesh(mesh_path)])

    def raise_value_error(path, start):
        raise ValueError("different drives")

    monkeypatch.setattr(scene_lib_module.os.path, "relpath", raise_value_error)

    SceneLib.save_scenes_to_file([scene], str(scene_file), asset_root=str(tmp_path))

    raw = torch.load(scene_file, map_location="cpu", weights_only=False)
    assert raw["original_scenes"][0]["objects"][0]["object_path"] == str(mesh_path)


def test_asset_tracking_marks_duplicate_object_types_across_scenes():
    duplicate_a = _box(width=1.0)
    duplicate_b = _box(width=1.0)
    unique = _box(width=2.0)
    scenes = [
        Scene(objects=[duplicate_a]),
        Scene(objects=[duplicate_b]),
        Scene(objects=[unique]),
    ]

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            replicate_method=ReplicationMethod.SEQUENTIAL,
        ),
        num_envs=3,
        scenes=scenes,
        device="cpu",
    )

    loaded_a = scene_lib._original_scenes[0].objects[0]
    loaded_b = scene_lib._original_scenes[1].objects[0]
    loaded_unique = scene_lib._original_scenes[2].objects[0]
    assert loaded_a.is_first_instance is True
    assert loaded_a.instance_id == 0
    assert loaded_a.first_instance_id == 0
    assert loaded_b.is_first_instance is False
    assert loaded_b.instance_id == 1
    assert loaded_b.first_instance_id == 0
    assert loaded_unique.is_first_instance is True
    assert loaded_unique.instance_id == 2
    assert loaded_unique.first_instance_id == 2


def test_get_object_scales_maps_replicated_scene_ids_to_original_mesh_scales(tmp_path):
    mesh_a = _mesh(tmp_path / "a.obj", scale=(2.0, 1.0, 0.5))
    mesh_b = _mesh(tmp_path / "b.obj", scale=(1.0, 3.0, 4.0))

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            replicate_method=ReplicationMethod.FIRST,
        ),
        num_envs=3,
        scenes=[Scene(objects=[mesh_a]), Scene(objects=[mesh_b])],
        device="cpu",
    )

    scales = scene_lib.get_object_scales(
        torch.device("cpu"), scene_indices=torch.tensor([0, 1, 2])
    )

    assert scales.shape == (3, 1, 1, 3)
    assert torch.equal(
        scales.squeeze(2),
        torch.tensor(
            [
                [[2.0, 1.0, 0.5]],
                [[1.0, 3.0, 4.0]],
                [[2.0, 1.0, 0.5]],
            ]
        ),
    )


def test_get_object_scales_without_indices_returns_all_original_scene_scales(tmp_path):
    mesh = _mesh(tmp_path / "mesh.obj", scale=(2.0, 3.0, 4.0))

    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=2,
        scenes=[
            Scene(objects=[_box(width=1.0)]),
            Scene(objects=[mesh]),
        ],
        device="cpu",
    )

    scales = scene_lib.get_object_scales(torch.device("cpu"))

    assert torch.equal(
        scales.squeeze(2),
        torch.tensor(
            [
                [[1.0, 1.0, 1.0]],
                [[2.0, 3.0, 4.0]],
            ]
        ),
    )


def test_sample_mesh_pointcloud_prefers_resolved_obj_and_fills_even_shortfall(
    tmp_path, monkeypatch
):
    object_path = tmp_path / "chair.urdf"
    obj_path = tmp_path / "chair.obj"
    obj_path.write_text("# placeholder")
    fake_mesh = SimpleNamespace(
        face_normals=np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
    )
    loaded_paths = []

    monkeypatch.setattr(
        scene_lib_module.trimesh,
        "load_mesh",
        lambda path: loaded_paths.append(path) or fake_mesh,
    )
    monkeypatch.setattr(scene_lib_module, "as_mesh", lambda mesh: mesh)
    monkeypatch.setattr(
        scene_lib_module.trimesh.sample,
        "sample_surface_even",
        lambda mesh, n: (
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            np.array([0], dtype=np.int64),
        ),
    )
    monkeypatch.setattr(
        scene_lib_module.trimesh.sample,
        "sample_surface",
        lambda mesh, n: (
            np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)[:n],
            np.array([1, 0], dtype=np.int64)[:n],
        ),
    )

    points, normals = _sample_mesh_pointcloud(str(object_path), 3)

    assert loaded_paths == [str(obj_path)]
    assert points.tolist() == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]
    assert normals.tolist() == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]


def test_sample_mesh_pointcloud_uses_ply_when_obj_and_stl_are_absent(
    tmp_path, monkeypatch
):
    object_path = tmp_path / "vase.usda"
    ply_path = tmp_path / "vase.ply"
    ply_path.write_text("ply placeholder")
    fake_mesh = SimpleNamespace(
        face_normals=np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    )
    loaded_paths = []

    monkeypatch.setattr(
        scene_lib_module.trimesh,
        "load_mesh",
        lambda path: loaded_paths.append(path) or fake_mesh,
    )
    monkeypatch.setattr(scene_lib_module, "as_mesh", lambda mesh: mesh)
    monkeypatch.setattr(
        scene_lib_module.trimesh.sample,
        "sample_surface_even",
        lambda mesh, n: (
            np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
            np.array([0], dtype=np.int64),
        ),
    )

    points, normals = _sample_mesh_pointcloud(str(object_path), 1)

    assert loaded_paths == [str(ply_path)]
    assert points.tolist() == [[0.0, 1.0, 2.0]]
    assert normals.tolist() == [[0.0, 0.0, 1.0]]


def test_sample_mesh_pointcloud_uses_stl_when_obj_is_absent(tmp_path, monkeypatch):
    object_path = tmp_path / "table.urdf"
    stl_path = tmp_path / "table.stl"
    stl_path.write_text("solid placeholder")
    fake_mesh = SimpleNamespace(
        face_normals=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )
    loaded_paths = []

    monkeypatch.setattr(
        scene_lib_module.trimesh,
        "load_mesh",
        lambda path: loaded_paths.append(path) or fake_mesh,
    )
    monkeypatch.setattr(scene_lib_module, "as_mesh", lambda mesh: mesh)
    monkeypatch.setattr(
        scene_lib_module.trimesh.sample,
        "sample_surface_even",
        lambda mesh, n: (
            np.array([[3.0, 2.0, 1.0]], dtype=np.float32),
            np.array([0], dtype=np.int64),
        ),
    )

    points, normals = _sample_mesh_pointcloud(str(object_path), 1)

    assert loaded_paths == [str(stl_path)]
    assert points.tolist() == [[3.0, 2.0, 1.0]]
    assert normals.tolist() == [[1.0, 0.0, 0.0]]


def test_sample_mesh_pointcloud_raises_when_no_resolved_mesh_file_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="Object file not found"):
        _sample_mesh_pointcloud(str(tmp_path / "missing.usd"), 4)


def test_pointcloud_parallel_deduplicates_mesh_sampling_and_copies_to_duplicates(
    tmp_path, monkeypatch
):
    calls = []

    class _Future:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

    class _ImmediatePool:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, func, path, num_samples):
            calls.append((path, num_samples))
            return _Future(
                (
                    np.full((num_samples, 3), 2.0, dtype=np.float32),
                    np.tile(
                        np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                        (num_samples, 1),
                    ),
                )
            )

    mesh_path = tmp_path / "crate.obj"
    mesh_a = _mesh(mesh_path)
    mesh_b = _mesh(mesh_path)
    monkeypatch.setattr(scene_lib_module, "ProcessPoolExecutor", _ImmediatePool)

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            pointcloud_samples_per_object=4,
        ),
        num_envs=2,
        scenes=[Scene(objects=[mesh_a]), Scene(objects=[mesh_b])],
        device="cpu",
    )

    assert calls == [(str(mesh_path), 4)]
    assert scene_lib._object_pointclouds.shape == (2, 1, 4, 3)
    assert torch.equal(
        scene_lib._object_pointclouds[0],
        scene_lib._object_pointclouds[1],
    )
    assert torch.equal(
        scene_lib._object_pointcloud_normals[0],
        scene_lib._object_pointcloud_normals[1],
    )


def test_scene_lib_main_example_runs_with_repo_assets():
    module_path = Path(scene_lib_module.__file__)

    namespace = runpy.run_path(str(module_path), run_name="__main__")

    scene_lib = namespace["scene_lib"]
    pose_obj0 = namespace["pose_obj0"]
    assert scene_lib.num_scenes() == 4
    assert scene_lib.num_objects_per_scene == 2
    assert pose_obj0.root_pos.shape == (1, 3)
    assert pose_obj0.root_rot.shape == (1, 4)
