# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene-object observations as kernels.

The surface points and the poses of the scene's objects in the robot's heading
frame, and the vectors from the recipe's contact bodies to the nearest surface
points, over the published ``scene.*`` names and the robot's state. The three
kernels call the helpers the built-in ``SceneObs`` callback called, with the
same arguments and the same flattening, so a recipe that binds them sees the
three readings the callback it replaces produced::

    "scene_pointclouds": MdpComponent.from_kernel(compute_scene_pointclouds),
    "scene_object_coordinates": MdpComponent.from_kernel(
        compute_scene_object_coordinates, object_indices=(0,)
    ),
    "contact_bodies_to_pointcloud": MdpComponent.from_kernel(
        compute_contact_bodies_to_pointcloud,
        contact_bodies=BodySelection.group(
            "all_left_foot_bodies", "all_right_foot_bodies"
        ),
    ),

``object_indices`` names which of the scene's objects a kernel reads (the
callback's ``obs_object_index``); ``None`` reads every object.
"""

from typing import Optional, Tuple

from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.utils.scene import (
    get_contact_bodies_to_object_pointcloud,
    get_local_object_coordinates,
    get_object_pointcloud,
)


def _select_objects(
    tensor: Tensor, object_indices: Optional[Tuple[int, ...]]
) -> Tensor:
    """The rows of the object axis (dim 1) a kernel reads; all of them for None."""
    if object_indices is None:
        return tensor
    return tensor[:, list(object_indices)]


def compute_scene_pointclouds(
    root_pos: Tensor = EnvContext.current.root_pos,
    root_rot: Tensor = EnvContext.current.root_rot,
    object_pos: Tensor = EnvContext.scene.object_pos,
    object_rot: Tensor = EnvContext.scene.object_rot,
    neutral_pointclouds: Tensor = EnvContext.scene.neutral_pointclouds,
    object_indices: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """The selected objects' surface points in the root's heading frame.

    ``[envs, objects * points * 3]``: each object's neutral point cloud
    (``neutral_pointclouds``: ``[envs, objects, points, 3]``) placed at the
    object's pose, then expressed relative to the root position and heading.
    """
    return get_object_pointcloud(
        current_object_rot=_select_objects(object_rot, object_indices),
        current_object_pos=_select_objects(object_pos, object_indices),
        scene_neutral_pointclouds=_select_objects(neutral_pointclouds, object_indices),
        robot_root_pos=root_pos,
        robot_root_rot=root_rot,
    )


def compute_scene_object_coordinates(
    root_pos: Tensor = EnvContext.current.root_pos,
    root_rot: Tensor = EnvContext.current.root_rot,
    object_pos: Tensor = EnvContext.scene.object_pos,
    object_rot: Tensor = EnvContext.scene.object_rot,
    object_indices: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """The selected objects' poses in the root's heading frame.

    ``[envs, objects * 9]``: per object its position (3) and its rotation as a
    6D tangent-normal pair.
    """
    return get_local_object_coordinates(
        current_object_rot=_select_objects(object_rot, object_indices),
        current_object_pos=_select_objects(object_pos, object_indices),
        robot_root_pos=root_pos,
        robot_root_rot=root_rot,
    )


def compute_contact_bodies_to_pointcloud(
    rigid_body_pos: Tensor = EnvContext.current.rigid_body_pos,
    rigid_body_rot: Tensor = EnvContext.current.rigid_body_rot,
    object_pos: Tensor = EnvContext.scene.object_pos,
    object_rot: Tensor = EnvContext.scene.object_rot,
    neutral_pointclouds: Tensor = EnvContext.scene.neutral_pointclouds,
    *,
    contact_bodies: Tensor,
    object_indices: Optional[Tuple[int, ...]] = None,
) -> Tensor:
    """From each contact body to the nearest surface point of each selected
    object, in that body's frame: ``[envs, objects * contact_bodies * 3]``.

    ``contact_bodies`` is the body selection the recipe names (a
    ``BodySelection``, resolved to indices at build), where the built-in scene
    callback this replaces read the robot's contact bodies.
    """
    return get_contact_bodies_to_object_pointcloud(
        current_object_rot=_select_objects(object_rot, object_indices),
        current_object_pos=_select_objects(object_pos, object_indices),
        scene_neutral_pointclouds=_select_objects(neutral_pointclouds, object_indices),
        contact_bodies_pos=rigid_body_pos[:, contact_bodies],
        contact_bodies_rot=rigid_body_rot[:, contact_bodies],
    )
