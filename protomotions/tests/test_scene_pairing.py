# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SceneLib clip-pairing snapshot and curriculum weights by clip id."""

import pytest

from protomotions.components.scene_lib import (
    BoxSceneObject,
    ReplicationMethod,
    Scene,
    SceneLib,
    SceneLibConfig,
    ScenePairing,
    scene_weights_by_clip_id,
)


def _box() -> BoxSceneObject:
    return BoxSceneObject(
        width=1.0,
        depth=1.0,
        height=1.0,
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )


def _scene_lib(motion_ids, num_envs):
    scenes = [Scene(objects=[_box()], humanoid_motion_id=m) for m in motion_ids]
    return SceneLib(
        config=SceneLibConfig(
            scene_file=None,
            inline_scenes=scenes,
            replicate_method=ReplicationMethod.SEQUENTIAL,
        ),
        num_envs=num_envs,
        scenes=None,
        device="cpu",
    )


def test_pairing_snapshot_maps_per_env_ids_through_the_replica_index():
    # Two originals over three envs replicate sequentially to originals [0, 1, 0].
    sl = _scene_lib([4, 5], num_envs=3)
    assert sl.pairing.original_motion_ids == (4, 5)
    assert sl.pairing.per_env_motion_ids == (4, 5, 4)
    assert sl.pairing.is_paired is True
    assert sl.replica_to_original.tolist() == [0, 1, 0]
    assert sl.get_humanoid_motion_ids() == [4, 5, 4]


def test_pairing_not_paired_when_every_scene_is_a_sentinel():
    sl = _scene_lib([-1, -1], num_envs=2)
    assert sl.pairing.is_paired is False
    assert sl.pairing.original_motion_ids == (-1, -1)
    assert sl.get_humanoid_motion_ids() is None


def test_pairing_is_paired_when_any_scene_names_a_clip():
    sl = _scene_lib([-1, 2], num_envs=2)
    assert sl.pairing.is_paired is True
    assert sl.get_humanoid_motion_ids() == [-1, 2]


def test_scene_pairing_checks_its_own_fields():
    with pytest.raises(ValueError, match="integers >= -1"):
        ScenePairing(
            original_motion_ids=(-2,), per_env_motion_ids=(-2,), is_paired=False
        )
    with pytest.raises(ValueError, match="is_paired must be True iff"):
        ScenePairing(original_motion_ids=(3,), per_env_motion_ids=(3,), is_paired=False)
    valid = ScenePairing(
        original_motion_ids=(-1, 0), per_env_motion_ids=(0, -1, 0), is_paired=True
    )
    assert valid.is_paired is True


def test_scene_weights_by_clip_id_maps_through_humanoid_motion_id():
    scenes = [
        Scene(humanoid_motion_id=2),
        Scene(humanoid_motion_id=0),
        Scene(humanoid_motion_id=1),
    ]
    assert scene_weights_by_clip_id(scenes, [0.1, 0.2, 0.3]) == [0.3, 0.1, 0.2]


def test_scene_weights_by_clip_id_refuses_unpaired_or_out_of_range():
    with pytest.raises(ValueError, match="unpaired scene"):
        scene_weights_by_clip_id([Scene(humanoid_motion_id=-1)], [0.5])
    with pytest.raises(ValueError, match="beyond the 1 motion weights"):
        scene_weights_by_clip_id([Scene(humanoid_motion_id=3)], [0.5])
