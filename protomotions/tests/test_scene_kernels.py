# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The scene kernels equal the three readings of the scene callback they replace,
reproduced here from the helpers that callback called with its exact arguments."""

from types import SimpleNamespace

import pytest
import torch

from protomotions.envs.context import FrozenContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.obs.scene import (
    compute_contact_bodies_to_pointcloud,
    compute_scene_object_coordinates,
    compute_scene_pointclouds,
)
from protomotions.envs.utils.scene import (
    get_contact_bodies_to_object_pointcloud,
    get_local_object_coordinates,
    get_object_pointcloud,
)

NUM_ENVS = 3
NUM_OBJECTS = 2
NUM_POINTS = 4
NUM_BODIES = 3


def _unit_quats(*shape: int, generator: torch.Generator) -> torch.Tensor:
    quats = torch.randn(*shape, 4, generator=generator)
    return quats / quats.norm(dim=-1, keepdim=True)


def _fixed_scene():
    """A scene library, a simulator and the published names over one fixed state."""
    generator = torch.Generator().manual_seed(3)
    neutral_pointclouds = torch.randn(
        NUM_ENVS, NUM_OBJECTS, NUM_POINTS, 3, generator=generator
    )
    object_pos = torch.randn(NUM_ENVS, NUM_OBJECTS, 3, generator=generator)
    object_rot = _unit_quats(NUM_ENVS, NUM_OBJECTS, generator=generator)
    body_pos = torch.randn(NUM_ENVS, NUM_BODIES, 3, generator=generator)
    body_rot = _unit_quats(NUM_ENVS, NUM_BODIES, generator=generator)

    scene_lib = SimpleNamespace(
        num_objects_per_scene=NUM_OBJECTS,
        pointcloud_samples_per_object=NUM_POINTS,
        num_scenes=lambda: 1,
        get_scene_neutral_pointcloud=lambda env_ids: neutral_pointclouds[env_ids],
    )
    simulator = SimpleNamespace(
        get_object_root_state=lambda env_ids: SimpleNamespace(
            root_pos=object_pos[env_ids], root_rot=object_rot[env_ids]
        ),
        get_bodies_state=lambda env_ids: SimpleNamespace(
            rigid_body_pos=body_pos[env_ids], rigid_body_rot=body_rot[env_ids]
        ),
    )
    published = {
        # The callback reads body 0 as the root; that is what current.root_*
        # publish.
        "current.root_pos": body_pos[:, 0],
        "current.root_rot": body_rot[:, 0],
        "current.rigid_body_pos": body_pos,
        "current.rigid_body_rot": body_rot,
        "scene.object_pos": object_pos,
        "scene.object_rot": object_rot,
        "scene.neutral_pointclouds": neutral_pointclouds,
    }
    return scene_lib, simulator, published


CONTACT_BODIES = torch.tensor([1, 2])


def _callback_readings(scene_lib, simulator, obs_object_index):
    """What the deleted ``SceneObs`` callback computed, reproduced from the three
    helpers it called with its exact arguments: every object (or the configured
    subset), body 0 as the root, the robot's contact bodies for the third
    reading, flattened per env."""
    env_ids = torch.arange(NUM_ENVS)
    index = slice(None) if obs_object_index is None else obs_object_index
    neutral = scene_lib.get_scene_neutral_pointcloud(env_ids)[:, index]
    objects = simulator.get_object_root_state(env_ids)
    object_pos, object_rot = objects.root_pos[:, index], objects.root_rot[:, index]
    bodies = simulator.get_bodies_state(env_ids)
    root_pos, root_rot = bodies.rigid_body_pos[:, 0], bodies.rigid_body_rot[:, 0]
    return {
        "scene_pointclouds": get_object_pointcloud(
            current_object_rot=object_rot,
            current_object_pos=object_pos,
            scene_neutral_pointclouds=neutral,
            robot_root_pos=root_pos,
            robot_root_rot=root_rot,
        ),
        "scene_object_coordinates": get_local_object_coordinates(
            current_object_rot=object_rot,
            current_object_pos=object_pos,
            robot_root_pos=root_pos,
            robot_root_rot=root_rot,
        ),
        "contact_bodies_to_pointcloud": get_contact_bodies_to_object_pointcloud(
            current_object_rot=object_rot,
            current_object_pos=object_pos,
            scene_neutral_pointclouds=neutral,
            contact_bodies_pos=bodies.rigid_body_pos[:, CONTACT_BODIES],
            contact_bodies_rot=bodies.rigid_body_rot[:, CONTACT_BODIES],
        ),
    }


def test_scene_kernels_bind_the_root_pose_and_the_scene_names():
    pointclouds = MdpComponent.from_kernel(compute_scene_pointclouds)
    coordinates = MdpComponent.from_kernel(
        compute_scene_object_coordinates, object_indices=(1,)
    )

    assert pointclouds.get_bindings_dict() == {
        "root_pos": "current.root_pos",
        "root_rot": "current.root_rot",
        "object_pos": "scene.object_pos",
        "object_rot": "scene.object_rot",
        "neutral_pointclouds": "scene.neutral_pointclouds",
    }
    assert pointclouds.static_params == {"object_indices": None}
    assert coordinates.get_bindings_dict() == {
        "root_pos": "current.root_pos",
        "root_rot": "current.root_rot",
        "object_pos": "scene.object_pos",
        "object_rot": "scene.object_rot",
    }
    assert coordinates.static_params == {"object_indices": (1,)}


@pytest.mark.parametrize(
    "object_indices, obs_object_index",
    [(None, None), ((1,), [1]), ((1, 0), [1, 0])],
)
def test_scene_kernels_equal_the_callback_readings(object_indices, obs_object_index):
    scene_lib, simulator, published = _fixed_scene()
    expected = _callback_readings(scene_lib, simulator, obs_object_index)
    ctx = FrozenContext(published, env_ids=None, dt=0.1)

    pointclouds = MdpComponent.from_kernel(
        compute_scene_pointclouds, object_indices=object_indices
    ).compute(ctx)
    coordinates = MdpComponent.from_kernel(
        compute_scene_object_coordinates, object_indices=object_indices
    ).compute(ctx)

    contacts = MdpComponent.from_kernel(
        compute_contact_bodies_to_pointcloud,
        contact_bodies=CONTACT_BODIES,
        object_indices=object_indices,
    ).compute(ctx)
    num_objects = NUM_OBJECTS if object_indices is None else len(object_indices)
    assert pointclouds.shape == (NUM_ENVS, num_objects * NUM_POINTS * 3)
    assert coordinates.shape == (NUM_ENVS, num_objects * 9)
    assert contacts.shape == (NUM_ENVS, num_objects * len(CONTACT_BODIES) * 3)
    assert torch.equal(pointclouds, expected["scene_pointclouds"])
    assert torch.equal(coordinates, expected["scene_object_coordinates"])
    assert torch.equal(contacts, expected["contact_bodies_to_pointcloud"])


def test_scene_kernels_compute_a_reset_subset_from_its_rows():
    scene_lib, simulator, published = _fixed_scene()
    expected = _callback_readings(scene_lib, simulator, None)
    env_ids = torch.tensor([2, 0])
    ctx = FrozenContext(
        {name: value[env_ids] for name, value in published.items()},
        env_ids=env_ids,
        dt=0.1,
    )

    pointclouds = MdpComponent.from_kernel(compute_scene_pointclouds).compute(ctx)
    coordinates = MdpComponent.from_kernel(compute_scene_object_coordinates).compute(
        ctx
    )

    assert torch.equal(pointclouds, expected["scene_pointclouds"][env_ids])
    assert torch.equal(coordinates, expected["scene_object_coordinates"][env_ids])
