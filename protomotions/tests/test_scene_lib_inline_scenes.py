# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SceneLib tests for programmatic scene configs."""

import pytest

from protomotions.components.scene_lib import (
    BoxSceneObject,
    ReplicationMethod,
    Scene,
    SceneLib,
    SceneLibConfig,
)


def _box(width: float = 1.0) -> BoxSceneObject:
    return BoxSceneObject(
        width=width,
        depth=1.0,
        height=1.0,
        translation=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0, 1.0),
    )


def test_inline_scenes_construct_scene_lib_without_scenes_argument():
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


def test_inline_scenes_reject_file_and_scenes_argument(tmp_path):
    scenes = [Scene(objects=[_box()])]
    scene_file = tmp_path / "scenes.pt"
    SceneLib.save_scenes_to_file(scenes, str(scene_file))

    with pytest.raises(ValueError, match="inline_scenes"):
        SceneLib(
            config=SceneLibConfig(
                scene_file=str(scene_file),
                inline_scenes=scenes,
            ),
            num_envs=1,
            scenes=None,
            device="cpu",
        )

    with pytest.raises(ValueError, match="inline_scenes"):
        SceneLib(
            config=SceneLibConfig(scene_file=None, inline_scenes=scenes),
            num_envs=1,
            scenes=scenes,
            device="cpu",
        )
