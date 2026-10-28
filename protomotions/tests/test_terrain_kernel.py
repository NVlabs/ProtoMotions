# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The terrain height-map kernel equals what the deleted TerrainObs callback
computed: ``Terrain.get_height_maps`` over the root states, per env."""

import os
from types import SimpleNamespace

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.components.terrains.terrain import Terrain  # noqa: E402
from protomotions.envs.base_env.publishers import TerrainPublisher  # noqa: E402
from protomotions.envs.context import FrozenContext  # noqa: E402
from protomotions.envs.mdp_component import (  # noqa: E402
    MdpComponent,
    is_exportable,
    non_exportable,
)
from protomotions.envs.obs.scene import compute_scene_pointclouds  # noqa: E402
from protomotions.envs.obs.terrain import compute_terrain_height_map  # noqa: E402


def _small_terrain(num_envs: int) -> Terrain:
    config = TerrainConfig(
        map_length=2.0,
        map_width=2.0,
        border_size=0.5,
        num_levels=1,
        num_terrains=1,
        terrain_proportions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        horizontal_scale=0.5,
        vertical_scale=0.1,
        spacing_between_scenes=1.0,
        minimal_humanoid_spacing=0.0,
        num_samples_per_axis=3,
        sample_width=0.5,
        slope_threshold=100.0,
    )
    terrain = Terrain(config, num_envs=num_envs, device=torch.device("cpu"))
    # A flat generator gives an all-zero field; the kernel must agree with the
    # terrain's own sampling on a real one, so the samples are replaced by a
    # fixed random field.
    generator = torch.Generator().manual_seed(7)
    terrain.height_samples = (
        torch.rand(terrain.height_samples.shape, generator=generator) * 0.4
    ).to(terrain.height_samples.dtype)
    return terrain


def _random_root_state(num_envs: int):
    generator = torch.Generator().manual_seed(11)
    root_pos = torch.rand(num_envs, 3, generator=generator) * 1.5 + 0.5
    root_rot = torch.randn(num_envs, 4, generator=generator)
    root_rot = root_rot / root_rot.norm(dim=-1, keepdim=True)
    return root_pos, root_rot


def _callback_height_map(terrain: Terrain, root_pos, root_rot) -> torch.Tensor:
    """What the deleted callback cached: the terrain's height maps for the root
    states of every env, in env order."""
    num_envs = root_pos.shape[0]
    env_ids = torch.arange(num_envs)
    root_states = SimpleNamespace(
        root_pos=root_pos[env_ids], root_rot=root_rot[env_ids]
    )
    return terrain.get_height_maps(root_states, env_ids)


def test_terrain_kernel_binds_the_root_pose_and_the_terrain_names():
    component = MdpComponent.from_kernel(compute_terrain_height_map)

    assert component.get_bindings_dict() == {
        "root_pos": "current.root_pos",
        "root_rot": "current.root_rot",
        "height_points": "terrain.height_points",
        "height_samples": "terrain.height_samples",
        "horizontal_scale": "terrain.horizontal_scale",
    }
    assert component.static_params == {}


def test_terrain_kernel_equals_the_callback_on_every_env():
    terrain = _small_terrain(num_envs=4)
    root_pos, root_rot = _random_root_state(4)
    expected = _callback_height_map(terrain, root_pos, root_rot)
    assert expected.abs().sum() > 0

    ctx = FrozenContext(
        {
            "current.root_pos": root_pos,
            "current.root_rot": root_rot,
            **TerrainPublisher(terrain).publish(None),
        },
        env_ids=None,
        dt=0.1,
    )
    actual = MdpComponent.from_kernel(compute_terrain_height_map).compute(ctx)

    assert actual.shape == (4, terrain.num_height_points)
    assert torch.equal(actual, expected)


def test_terrain_kernel_equals_the_callback_on_a_reset_subset():
    terrain = _small_terrain(num_envs=4)
    root_pos, root_rot = _random_root_state(4)
    expected = _callback_height_map(terrain, root_pos, root_rot)
    env_ids = torch.tensor([3, 1])

    ctx = FrozenContext(
        {
            "current.root_pos": root_pos[env_ids],
            "current.root_rot": root_rot[env_ids],
            **TerrainPublisher(terrain).publish(env_ids),
        },
        env_ids=env_ids,
        dt=0.1,
    )
    actual = MdpComponent.from_kernel(compute_terrain_height_map).compute(ctx)

    assert torch.equal(actual, expected[env_ids])


def test_the_terrain_kernel_is_the_one_non_exportable_kernel():
    assert not is_exportable(compute_terrain_height_map)
    assert is_exportable(compute_scene_pointclouds)

    def plain(x):
        return x

    assert is_exportable(plain)
    assert non_exportable(plain) is plain
    assert not is_exportable(plain)
