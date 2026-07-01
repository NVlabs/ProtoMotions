# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for scene observation and path generation utilities."""

from types import SimpleNamespace

import torch

from protomotions.envs.obs.scene_obs import SceneObs, SceneObsConfig
from protomotions.envs.utils import scene as scene_utils
from protomotions.envs.utils.path_generator import PathGenerator, PathGeneratorConfig


def _identity_quat(*shape: int) -> torch.Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def test_scene_geometry_helpers_compute_local_pointclouds_and_nearest_vectors():
    object_rot = _identity_quat(1, 1)
    object_pos = torch.tensor([[[1.0, 0.0, 0.0]]])
    neutral_points = torch.tensor([[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]])
    contact_pos = torch.tensor([[[1.2, 0.0, 0.0], [4.0, 0.0, 0.0]]])
    contact_rot = _identity_quat(1, 2)

    nearest, indices = scene_utils.closest_points_on_object_surface(
        current_object_rot=object_rot,
        current_object_pos=object_pos,
        scene_neutral_pointclouds=neutral_points,
        contact_bodies_pos=contact_pos,
    )
    contact_vectors = scene_utils.get_contact_bodies_to_object_pointcloud(
        current_object_rot=object_rot,
        current_object_pos=object_pos,
        scene_neutral_pointclouds=neutral_points,
        contact_bodies_pos=contact_pos,
        contact_bodies_rot=contact_rot,
    )
    pointcloud = scene_utils.get_object_pointcloud(
        current_object_rot=object_rot,
        current_object_pos=object_pos,
        scene_neutral_pointclouds=neutral_points,
        robot_root_pos=torch.tensor([[1.0, 0.0, 0.0]]),
        robot_root_rot=_identity_quat(1),
    )
    coordinates = scene_utils.get_local_object_coordinates(
        current_object_rot=object_rot,
        current_object_pos=object_pos,
        robot_root_pos=torch.tensor([[0.5, 0.0, 0.0]]),
        robot_root_rot=_identity_quat(1),
    )

    assert torch.equal(indices, torch.tensor([[[0, 1]]]))
    assert torch.allclose(
        nearest,
        torch.tensor([[[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]]]),
    )
    assert torch.allclose(contact_vectors, torch.tensor([[-0.2, 0.0, 0.0, -1.0, 0.0, 0.0]]))
    assert torch.allclose(pointcloud, torch.tensor([[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]]))
    assert torch.allclose(coordinates[:, :3], torch.tensor([[0.5, 0.0, 0.0]]))
    assert coordinates.shape == (1, 9)


def test_scene_obs_handles_disabled_and_enabled_fake_environments():
    disabled_env = SimpleNamespace(
        scene_lib=SimpleNamespace(num_scenes=lambda: 0),
        num_envs=2,
        device=torch.device("cpu"),
    )
    disabled = SceneObs(SceneObsConfig(enabled=True), disabled_env)
    disabled.compute_observations(torch.tensor([0]))
    assert disabled.scene_pointclouds is None
    assert disabled.contact_bodies_to_pointcloud is None

    class _SceneLib:
        num_objects_per_scene = 2
        pointcloud_samples_per_object = 2

        def num_scenes(self):
            return 1

        def get_scene_neutral_pointcloud(self, env_ids):
            return torch.tensor(
                [
                    [
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                    ]
                ]
            ).expand(len(env_ids), -1, -1, -1)

    class _Simulator:
        def get_object_root_state(self, env_ids):
            n = len(env_ids)
            return SimpleNamespace(
                root_pos=torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]).expand(
                    n, -1, -1
                ),
                root_rot=_identity_quat(n, 2),
            )

        def get_bodies_state(self, env_ids):
            n = len(env_ids)
            return SimpleNamespace(
                rigid_body_pos=torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]).expand(
                    n, -1, -1
                ),
                rigid_body_rot=_identity_quat(n, 2),
            )

    enabled_env = SimpleNamespace(
        scene_lib=_SceneLib(),
        robot_config=SimpleNamespace(contact_bodies=["foot"]),
        contact_body_ids=torch.tensor([1]),
        num_envs=2,
        device=torch.device("cpu"),
        simulator=_Simulator(),
    )
    enabled = SceneObs(SceneObsConfig(enabled=True, obs_object_index=[1]), enabled_env)
    enabled.compute_observations(torch.tensor([1]))
    obs = enabled.get_obs()
    obs["scene_pointclouds"][1, 0] = -99.0

    assert enabled.obs_object_index == [1]
    assert enabled.scene_pointclouds.shape == (2, 6)
    assert enabled.contact_bodies_to_pointcloud.shape == (2, 3)
    assert enabled.scene_object_coordinates.shape == (2, 9)
    assert enabled.scene_pointclouds[1, 0] != -99.0

    default_objects = SceneObs(SceneObsConfig(enabled=True), enabled_env)
    assert default_objects.obs_object_index == [0, 1]
    assert default_objects.scene_pointclouds.shape == (2, 12)
    assert default_objects.contact_bodies_to_pointcloud.shape == (2, 6)


def test_path_generator_resets_clips_heights_and_interpolates_positions():
    config = PathGeneratorConfig(
        num_verts=5,
        fixed_path=True,
        slow=True,
        height_conditioned=True,
        head_height_min=0.5,
        head_height_max=1.0,
        speed_min=0.4,
        speed_max=1.2,
        start_speed_max=0.8,
    )
    generator = PathGenerator(
        config=config,
        device=torch.device("cpu"),
        num_envs=2,
        episode_dur=4.0,
        height_conditioned=True,
    )
    initial_verts = generator.verts.clone()

    generator.reset(torch.tensor([], dtype=torch.long), torch.empty(0, 3))
    assert torch.equal(generator.verts, initial_verts)

    torch.manual_seed(0)
    generator.reset(
        torch.tensor([0, 1]),
        torch.tensor([[0.0, 0.0, 2.0], [10.0, 0.0, -1.0]]),
    )

    assert generator.get_num_verts() == 5
    assert generator.get_num_segs() == 4
    assert generator.get_num_envs() == 2
    assert generator.get_traj_duration() == 5.0
    assert torch.allclose(generator.verts[0, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(generator.verts[1, 0], torch.tensor([10.0, 0.0, 0.5]))
    assert torch.all(generator.verts[:, :, 2] >= 0.5)
    assert torch.all(generator.verts[:, :, 2] <= 1.0)
    assert torch.equal(generator.get_traj_verts(0), generator.verts[0])

    start = generator.calc_pos(torch.tensor([0]), torch.tensor([0.0]))
    half_between_seg_1_and_2 = generator.get_traj_duration() * (
        1.5 / generator.get_num_segs()
    )
    middle = generator.calc_pos(
        torch.tensor([0]),
        torch.tensor([half_between_seg_1_and_2]),
    )
    expected_middle = 0.5 * generator.verts[0, 1] + 0.5 * generator.verts[0, 2]

    assert torch.allclose(start, generator.verts[0, 0].unsqueeze(0))
    assert torch.allclose(middle, expected_middle.unsqueeze(0))


def test_path_generator_forward_only_and_naive_modes_are_deterministic_with_seed():
    config = PathGeneratorConfig(
        num_verts=4,
        use_forward_path_only=True,
        use_naive_path_generator=True,
        height_conditioned=False,
        head_height_min=0.4,
        head_height_max=1.5,
    )
    generator = PathGenerator(
        config=config,
        device=torch.device("cpu"),
        num_envs=1,
        episode_dur=3.0,
        height_conditioned=False,
    )

    torch.manual_seed(123)
    generator.reset(torch.tensor([0]), torch.tensor([[0.0, 0.0, 0.8]]))

    assert generator.use_naive_path_generator is True
    assert generator.verts.shape == (1, 4, 3)
    assert torch.isfinite(generator.verts).all()
