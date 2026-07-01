# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime SceneLib behavior tests using lightweight inline primitive scenes."""

from __future__ import annotations

import pytest
import torch

import protomotions.components.scene_lib as scene_lib_module
from protomotions.components.scene_lib import (
    BoxSceneObject,
    MeshSceneObject,
    ObjectOptions,
    ReplicationMethod,
    Scene,
    SceneLib,
    SceneLibConfig,
    SubsetMethod,
)


def _box(
    translation=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0, 1.0),
    width=1.0,
    depth=1.0,
    height=1.0,
    options=None,
    fps=None,
):
    return BoxSceneObject(
        translation=translation,
        rotation=rotation,
        width=width,
        depth=depth,
        height=height,
        options=options or ObjectOptions(),
        fps=fps,
    )


def test_first_replication_reuses_first_scene_for_all_extra_envs():
    scenes = [
        Scene(objects=[_box(translation=(0.0, 0.0, 0.0))], humanoid_motion_id=10),
        Scene(objects=[_box(translation=(1.0, 0.0, 0.0))], humanoid_motion_id=20),
    ]

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            replicate_method=ReplicationMethod.FIRST,
        ),
        num_envs=5,
        scenes=scenes,
        device="cpu",
    )

    assert scene_lib._scene_to_original_scene_id.tolist() == [0, 1, 0, 0, 0]
    assert scene_lib.get_humanoid_motion_ids() == [10, 20, 10, 10, 10]


def test_sequential_subset_keeps_ordered_prefix_and_weights():
    scenes = [
        Scene(objects=[_box(width=1.0)], humanoid_motion_id=10),
        Scene(objects=[_box(width=2.0)], humanoid_motion_id=20),
        Scene(objects=[_box(width=3.0)], humanoid_motion_id=30),
        Scene(objects=[_box(width=4.0)], humanoid_motion_id=40),
    ]

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            subset_method=SubsetMethod.SEQUENTIAL,
        ),
        num_envs=2,
        scenes=scenes,
        device="cpu",
        scene_weights=[0.1, 0.2, 0.3, 0.4],
    )

    assert scene_lib.num_scenes() == 2
    assert scene_lib._scene_to_original_scene_id.tolist() == [0, 1]
    assert scene_lib.get_humanoid_motion_ids() == [10, 20]


def test_last_subset_keeps_tail_scenes_and_original_mapping():
    scenes = [
        Scene(objects=[_box(width=1.0)], humanoid_motion_id=10),
        Scene(objects=[_box(width=2.0)], humanoid_motion_id=20),
        Scene(objects=[_box(width=3.0)], humanoid_motion_id=30),
        Scene(objects=[_box(width=4.0)], humanoid_motion_id=40),
    ]

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            subset_method=SubsetMethod.LAST,
        ),
        num_envs=2,
        scenes=scenes,
        device="cpu",
        scene_weights=[0.1, 0.2, 0.3, 0.4],
    )

    assert scene_lib._scene_to_original_scene_id.tolist() == [2, 3]
    assert scene_lib.get_humanoid_motion_ids() == [30, 40]


def test_explicit_index_subset_preserves_requested_order_and_duplicates():
    scenes = [
        Scene(objects=[_box(width=1.0)], humanoid_motion_id=10),
        Scene(objects=[_box(width=2.0)], humanoid_motion_id=20),
        Scene(objects=[_box(width=3.0)], humanoid_motion_id=30),
        Scene(objects=[_box(width=4.0)], humanoid_motion_id=40),
    ]

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            subset_method=[2, 0, 2],
        ),
        num_envs=3,
        scenes=scenes,
        device="cpu",
        scene_weights=[0.1, 0.2, 0.3, 0.4],
    )

    assert scene_lib._scene_to_original_scene_id.tolist() == [2, 0, 2]
    assert scene_lib.get_humanoid_motion_ids() == [30, 10, 30]


def test_random_subset_uses_sampled_indices_for_scene_and_weight_order(monkeypatch):
    scenes = [
        Scene(objects=[_box(width=1.0)], humanoid_motion_id=10),
        Scene(objects=[_box(width=2.0)], humanoid_motion_id=20),
        Scene(objects=[_box(width=3.0)], humanoid_motion_id=30),
    ]
    sampled_args = []

    def fake_sample(population, k):
        sampled_args.append((list(population), k))
        return [2, 0]

    monkeypatch.setattr(scene_lib_module.random, "sample", fake_sample)

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            subset_method=SubsetMethod.RANDOM,
        ),
        num_envs=2,
        scenes=scenes,
        device="cpu",
        scene_weights=[0.1, 0.2, 0.7],
    )

    assert sampled_args == [([0, 1, 2], 2)]
    assert scene_lib._scene_to_original_scene_id.tolist() == [2, 0]
    assert scene_lib.get_humanoid_motion_ids() == [30, 10]


def test_weighted_replication_passes_weights_to_random_choices(monkeypatch):
    scenes = [
        Scene(objects=[_box(width=1.0)], humanoid_motion_id=10),
        Scene(objects=[_box(width=2.0)], humanoid_motion_id=20),
    ]
    choices = [1, 0]
    calls = []

    def fake_choices(population, weights, k):
        calls.append((list(population), list(weights), k))
        return [choices.pop(0)]

    monkeypatch.setattr(scene_lib_module.random, "choices", fake_choices)

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            replicate_method=ReplicationMethod.WEIGHTED,
        ),
        num_envs=4,
        scenes=scenes,
        device="cpu",
        scene_weights=[0.25, 0.75],
    )

    assert calls == [([0, 1], [0.25, 0.75], 1), ([0, 1], [0.25, 0.75], 1)]
    assert scene_lib._scene_to_original_scene_id.tolist() == [0, 1, 1, 0]
    assert scene_lib.get_humanoid_motion_ids() == [10, 20, 20, 10]


def test_random_replication_ignores_supplied_weights(monkeypatch):
    scenes = [
        Scene(objects=[_box(width=1.0)], humanoid_motion_id=10),
        Scene(objects=[_box(width=2.0)], humanoid_motion_id=20),
    ]
    calls = []

    def fake_choices(population, weights, k):
        calls.append((list(population), weights, k))
        return [1]

    monkeypatch.setattr(scene_lib_module.random, "choices", fake_choices)

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            replicate_method=ReplicationMethod.RANDOM,
        ),
        num_envs=3,
        scenes=scenes,
        device="cpu",
        scene_weights=[0.25, 0.75],
    )

    assert calls == [([0, 1], None, 1)]
    assert scene_lib._scene_to_original_scene_id.tolist() == [0, 1, 1]


def test_invalid_replication_method_raises_value_error():
    with pytest.raises(ValueError, match="Replicate method"):
        SceneLib(
            config=SceneLibConfig(
                scene_file=None,
                replicate_method="bogus",
            ),
            num_envs=2,
            scenes=[Scene(objects=[_box()])],
            device="cpu",
        )


def test_invalid_subset_method_raises_value_error():
    with pytest.raises(ValueError, match="Subset method"):
        SceneLib(
            config=SceneLibConfig(
                scene_file=None,
                subset_method="bogus",
            ),
            num_envs=1,
            scenes=[
                Scene(objects=[_box(width=1.0)]),
                Scene(objects=[_box(width=2.0)]),
            ],
            device="cpu",
        )


def test_scene_weights_must_match_original_scene_count():
    scenes = [
        Scene(objects=[_box(width=1.0)]),
        Scene(objects=[_box(width=2.0)]),
    ]

    with pytest.raises(ValueError, match="scene_weights"):
        SceneLib(
            config=SceneLibConfig(scene_file=None),
            num_envs=2,
            scenes=scenes,
            device="cpu",
            scene_weights=[1.0],
        )


def test_scene_file_and_scenes_argument_are_mutually_exclusive(tmp_path):
    scene_file = tmp_path / "scenes.pt"
    SceneLib.save_scenes_to_file([Scene(objects=[_box()])], str(scene_file))

    with pytest.raises(ValueError, match="both config.scene_file and scenes"):
        SceneLib(
            config=SceneLibConfig(scene_file=str(scene_file)),
            num_envs=1,
            scenes=[Scene(objects=[_box(width=2.0)])],
            device="cpu",
        )


def test_scene_lib_rejects_inconsistent_object_counts_during_construction():
    scenes = [
        Scene(objects=[_box(width=1.0)]),
        Scene(objects=[_box(width=2.0), _box(width=3.0)]),
    ]

    with pytest.raises(ValueError, match="inconsistent number of objects"):
        SceneLib(
            config=SceneLibConfig(scene_file=None),
            num_envs=2,
            scenes=scenes,
            device="cpu",
        )


def test_zero_object_scenes_create_empty_object_state_without_motion_frames():
    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=2,
        scenes=[Scene(), Scene()],
        device="cpu",
    )

    assert scene_lib.num_objects_per_scene == 0
    assert scene_lib._object_translations.shape == (0, 3)
    assert scene_lib._object_rotations.shape == (0, 4)
    assert scene_lib._motion_starts.shape == (0,)
    state = scene_lib.get_default_object_state(device="cpu")
    assert state.root_pos.shape == (2, 0, 3)
    assert state.root_rot.shape == (2, 0, 4)


def test_zero_object_scenes_accessors_preserve_env_and_original_scene_batches():
    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            replicate_method=ReplicationMethod.SEQUENTIAL,
        ),
        num_envs=3,
        scenes=[Scene(humanoid_motion_id=10), Scene(humanoid_motion_id=20)],
        device="cpu",
    )
    env_ids = torch.tensor([2, 1])

    assert scene_lib._scene_to_original_scene_id.tolist() == [0, 1, 0]
    assert scene_lib.get_per_object_valid_mask().shape == (3, 0)
    assert scene_lib.get_per_object_valid_mask(env_ids).shape == (2, 0)
    assert scene_lib.get_object_class_ids().shape == (3, 0)
    assert scene_lib.get_object_class_ids(env_ids).shape == (2, 0)
    assert scene_lib.get_object_scales(torch.device("cpu")).shape == (2, 0, 1, 3)
    assert scene_lib.get_object_scales(torch.device("cpu"), env_ids).shape == (
        2,
        0,
        1,
        3,
    )


def test_valid_masks_and_class_ids_are_mapped_from_original_scenes():
    padding = _box(width=0.001, depth=0.001, height=0.001)
    scenes = [
        Scene(objects=[_box(width=1.0, depth=1.0, height=1.0), padding]),
        Scene(
            objects=[
                _box(width=2.0, depth=2.0, height=2.0),
                _box(width=3.0, depth=3.0, height=3.0),
            ]
        ),
    ]

    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            replicate_method=ReplicationMethod.SEQUENTIAL,
            pointcloud_samples_per_object=8,
        ),
        num_envs=3,
        scenes=scenes,
        device="cpu",
    )

    assert scene_lib._scene_to_original_scene_id.tolist() == [0, 1, 0]
    env_ids = torch.tensor([2, 1])
    assert scene_lib.get_per_object_valid_mask().tolist() == [
        [True, False],
        [True, True],
        [True, False],
    ]
    assert scene_lib.get_per_object_valid_mask(env_ids).tolist() == [
        [True, False],
        [True, True],
    ]
    assert scene_lib.get_object_class_ids().tolist() == [
        [2, 0],
        [2, 2],
        [2, 0],
    ]
    assert scene_lib.get_object_class_ids(env_ids).tolist() == [
        [2, 0],
        [2, 2],
    ]
    assert scene_lib.get_scene_neutral_pointcloud(torch.tensor([2])).shape == (
        1,
        2,
        8,
        3,
    )
    assert scene_lib.get_object_bbox_extents(torch.tensor([1])).tolist() == [
        [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    ]


def test_get_scene_pose_interpolates_motion_and_offsets_only_dynamic_objects():
    static_obj = _box(translation=(0.0, 0.0, 1.0))
    moving_obj = _box(
        translation=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ],
        rotation=[
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
        ],
        fps=1.0,
    )
    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=1,
        scenes=[Scene(objects=[static_obj, moving_obj])],
        device="cpu",
    )

    pose = scene_lib.get_scene_pose(
        scene_indices=torch.tensor([0]),
        time=torch.tensor([1.5]),
        respawn_offset=0.25,
    )

    assert pose.root_pos.shape == (1, 2, 3)
    assert torch.allclose(pose.root_pos[0, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(pose.root_pos[0, 1], torch.tensor([1.5, 0.0, 0.25]))
    assert torch.allclose(
        pose.root_rot[0, 1], torch.tensor([0.0, 0.0, 0.0, 1.0])
    )

    near_end_pose = scene_lib.get_scene_pose(
        scene_indices=torch.tensor([0]),
        time=torch.tensor([2.5]),
        respawn_offset=0.25,
    )
    assert torch.allclose(
        near_end_pose.root_pos[0, 1],
        torch.tensor([2.0, 0.0, 0.25]),
    )


def test_get_scene_pose_maps_replicated_scene_indices_to_original_motion_buffers():
    scenes = [
        Scene(
            objects=[
                _box(
                    translation=[(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
                    rotation=[
                        (0.0, 0.0, 0.0, 1.0),
                        (0.0, 0.0, 0.0, 1.0),
                    ],
                    fps=1.0,
                )
            ],
            humanoid_motion_id=10,
        ),
        Scene(
            objects=[
                _box(
                    translation=[(10.0, 0.0, 0.0), (12.0, 0.0, 0.0)],
                    rotation=[
                        (0.0, 0.0, 0.0, 1.0),
                        (0.0, 0.0, 0.0, 1.0),
                    ],
                    fps=1.0,
                )
            ],
            humanoid_motion_id=20,
        ),
    ]
    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            replicate_method=ReplicationMethod.SEQUENTIAL,
        ),
        num_envs=4,
        scenes=scenes,
        device="cpu",
    )

    pose = scene_lib.get_scene_pose(
        scene_indices=torch.tensor([0, 1, 2, 3]),
        time=torch.full((4,), 0.5),
    )

    assert scene_lib._scene_to_original_scene_id.tolist() == [0, 1, 0, 1]
    assert torch.allclose(
        pose.root_pos.squeeze(1),
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [11.0, 0.0, 0.0],
            ]
        ),
    )


def test_get_object_pose_clamps_negative_time_to_first_frame():
    moving_obj = _box(
        translation=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        ],
        rotation=[
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, 0.0, 1.0),
        ],
        fps=1.0,
    )
    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=1,
        scenes=[Scene(objects=[moving_obj])],
        device="cpu",
    )

    pose = scene_lib.get_object_pose(
        object_indices=torch.tensor([0]),
        time=torch.tensor([-1.0]),
    )

    assert torch.allclose(pose.root_pos[0], torch.tensor([0.0, 0.0, 0.0]))


def test_calc_frame_blend_clamps_times_and_computes_indices_and_blends():
    scene_lib = SceneLib.empty(num_envs=0, device="cpu")

    frame_idx0, frame_idx1, blend = scene_lib._calc_frame_blend(
        time=torch.tensor([-1.0, 0.5, 3.0]),
        length=torch.tensor([2.0, 2.0, 2.0]),
        num_frames=torch.tensor([3, 3, 3]),
        dt=torch.tensor([1.0, 1.0, 1.0]),
    )

    assert torch.equal(frame_idx0, torch.tensor([0, 0, 2]))
    assert torch.equal(frame_idx1, torch.tensor([1, 1, 2]))
    assert torch.allclose(blend, torch.tensor([0.0, 0.5, 0.0]))


def test_get_object_pose_raises_when_motion_data_was_cleared():
    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=1,
        scenes=[Scene(objects=[_box()])],
        device="cpu",
    )
    scene_lib._motion_starts = None

    with pytest.raises(ValueError, match="Motion data not combined"):
        scene_lib.get_object_pose(torch.tensor([0]), torch.tensor([0.0]))


def test_get_scene_pose_raises_when_motion_data_was_cleared():
    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=1,
        scenes=[Scene(objects=[_box()])],
        device="cpu",
    )
    scene_lib._motion_starts = None

    with pytest.raises(ValueError, match="Motion data not combined"):
        scene_lib.get_scene_pose(torch.tensor([0]), torch.tensor([0.0]))


def test_default_object_state_uses_initial_pose_and_zero_velocities():
    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=2,
        scenes=[
            Scene(objects=[_box(translation=(1.0, 2.0, 3.0))]),
            Scene(objects=[_box(translation=(4.0, 5.0, 6.0))]),
        ],
        device="cpu",
    )

    state = scene_lib.get_default_object_state(device="cpu")

    assert state.root_pos.shape == (2, 1, 3)
    assert torch.equal(
        state.root_pos.squeeze(1),
        torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    )
    assert torch.equal(state.root_vel, torch.zeros_like(state.root_pos))
    assert torch.equal(state.root_ang_vel, torch.zeros_like(state.root_pos))


def test_default_object_state_uses_scene_lib_device_when_device_is_omitted():
    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=1,
        scenes=[Scene(objects=[_box(translation=(2.0, 4.0, 6.0))])],
        device="cpu",
    )

    state = scene_lib.get_default_object_state()

    assert state.root_pos.device.type == "cpu"
    assert torch.equal(state.root_pos, torch.tensor([[[2.0, 4.0, 6.0]]]))
    assert torch.equal(state.root_vel, torch.zeros_like(state.root_pos))


def test_pointcloud_normals_map_replicated_scene_indices_to_originals():
    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            replicate_method=ReplicationMethod.FIRST,
            pointcloud_samples_per_object=8,
        ),
        num_envs=3,
        scenes=[
            Scene(objects=[_box(width=1.0)]),
            Scene(objects=[_box(width=2.0)]),
        ],
        device="cpu",
    )

    normals = scene_lib.get_scene_neutral_pointcloud_normals(
        torch.tensor([0, 1, 2])
    )

    assert normals.shape == (3, 1, 8, 3)
    assert torch.equal(normals[0], scene_lib._object_pointcloud_normals[0])
    assert torch.equal(normals[1], scene_lib._object_pointcloud_normals[1])
    assert torch.equal(normals[2], scene_lib._object_pointcloud_normals[0])


def test_pointcloud_accessors_return_all_original_scenes_when_indices_omitted():
    scene_lib = SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            pointcloud_samples_per_object=8,
        ),
        num_envs=2,
        scenes=[
            Scene(objects=[_box(width=1.0)]),
            Scene(objects=[_box(width=2.0)]),
        ],
        device="cpu",
    )

    assert scene_lib.get_scene_neutral_pointcloud().shape == (2, 1, 8, 3)
    assert scene_lib.get_scene_neutral_pointcloud_normals().shape == (2, 1, 8, 3)
    assert scene_lib.get_object_bbox_extents().shape == (2, 1, 3)


def test_pointcloud_accessors_raise_when_sampling_was_not_enabled():
    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=1,
        scenes=[Scene(objects=[_box()])],
        device="cpu",
    )

    with pytest.raises(ValueError, match="pointclouds not initialized"):
        scene_lib.get_scene_neutral_pointcloud()
    with pytest.raises(ValueError, match="normals not initialized"):
        scene_lib.get_scene_neutral_pointcloud_normals()
    with pytest.raises(ValueError, match="Bbox extents not computed"):
        scene_lib.get_object_bbox_extents()


def test_combine_pointcloud_normals_raises_when_object_normals_are_missing():
    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=1,
        scenes=[Scene(objects=[_box()])],
        device="cpu",
    )

    with pytest.raises(ValueError, match="missing pointcloud normals"):
        scene_lib.combine_object_pointcloud_normals()


class _Terrain:
    num_scenes_per_column = 2
    spacing_between_scenes = 5.0
    border = 1.0
    horizontal_scale = 0.5
    scene_y_offset = 2.0
    device = "cpu"

    def __init__(self):
        self.validated = []
        self.marked = []

    def is_valid_spawn_location(self, locations):
        self.validated.append(locations.cpu().tolist())
        return torch.tensor([True])

    def mark_scene_location(self, x, y):
        self.marked.append((x, y))


def test_terrain_offsets_are_assigned_validated_and_returned_as_positions():
    terrain = _Terrain()
    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=3,
        scenes=[
            Scene(objects=[_box()]),
            Scene(objects=[_box()]),
            Scene(objects=[_box()]),
        ],
        device="cpu",
        terrain=terrain,
    )

    assert scene_lib.scene_offsets == [(5.5, 7.0), (10.5, 7.0), (5.5, 12.0)]
    assert terrain.validated == [[[11, 14]], [[21, 14]], [[11, 24]]]
    assert terrain.marked == [(11, 14), (21, 14), (11, 24)]
    assert torch.equal(
        scene_lib.get_scene_positions(terrain, device="cpu"),
        torch.tensor(
            [[5.5, 7.0, 0.0], [10.5, 7.0, 0.0], [5.5, 12.0, 0.0]]
        ),
    )
    assert torch.equal(
        scene_lib.get_scene_positions(terrain),
        torch.tensor(
            [[5.5, 7.0, 0.0], [10.5, 7.0, 0.0], [5.5, 12.0, 0.0]]
        ),
    )


def test_terrain_rejects_invalid_scene_spawn_location():
    class InvalidTerrain(_Terrain):
        def is_valid_spawn_location(self, locations):
            self.validated.append(locations.cpu().tolist())
            return torch.tensor([False])

    terrain = InvalidTerrain()

    with pytest.raises(AssertionError, match="not a valid spawn location"):
        SceneLib(
            config=SceneLibConfig(scene_file=None),
            num_envs=1,
            scenes=[Scene(objects=[_box()])],
            device="cpu",
            terrain=terrain,
        )


def test_none_scene_offset_defaults_to_origin_without_terrain():
    scene = Scene(objects=[_box()])
    scene.offset = None

    scene_lib = SceneLib(
        config=SceneLibConfig(scene_file=None),
        num_envs=1,
        scenes=[scene],
        device="cpu",
    )

    assert scene_lib.scene_offsets == [(0.0, 0.0)]
    assert scene_lib.scenes[0].offset == (0.0, 0.0)


def test_save_and_load_resolves_relative_mesh_paths_from_scene_file_grandparent(
    tmp_path,
):
    asset_root = tmp_path / "assets"
    scene_dir = asset_root / "scenes"
    mesh_path = asset_root / "meshes" / "chair.obj"
    scene_file = scene_dir / "scene_pack.pt"
    mesh_path.parent.mkdir(parents=True)
    scene_dir.mkdir(parents=True)

    scene = Scene(
        objects=[
            MeshSceneObject(
                object_path=str(mesh_path),
                object_dims=(-1.0, 1.0, -2.0, 2.0, 0.0, 3.0),
                translation=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 0.0, 1.0),
            )
        ],
        humanoid_motion_id=4,
    )

    SceneLib.save_scenes_to_file(
        [scene], str(scene_file), asset_root=str(asset_root)
    )
    loaded_scenes = SceneLib._load_scenes_from_file(str(scene_file), device="cpu")

    assert len(loaded_scenes) == 1
    loaded_obj = loaded_scenes[0].objects[0]
    assert loaded_scenes[0].humanoid_motion_id == 4
    assert loaded_obj.object_path == str(mesh_path)
    assert loaded_obj.object_dims == (-1.0, 1.0, -2.0, 2.0, 0.0, 3.0)
