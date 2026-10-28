# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The terrain height map as an observation kernel.

``compute_terrain_height_map`` samples the height field around the robot's root
in the root's heading frame, exactly as ``Terrain.get_height_maps`` did for the
built-in ``TerrainObs`` callback it replaces. A recipe binds it like every
other observation::

    "terrain": MdpComponent.from_kernel(compute_terrain_height_map)

An actor on sensor estimates overrides ``root_pos`` and ``root_rot`` with the
``noisy.*`` names (or wraps the dict in ``sensed``). The kernel is marked
``@non_exportable``: it needs the whole height field, which the deployable
graph cannot carry; a deployed policy takes the sampled map as a passthrough
input.
"""

from torch import Tensor

from protomotions.components.terrains.terrain_utils import get_height_maps_jit
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import non_exportable


@non_exportable
def compute_terrain_height_map(
    root_pos: Tensor = EnvContext.current.root_pos,
    root_rot: Tensor = EnvContext.current.root_rot,
    height_points: Tensor = EnvContext.terrain.height_points,
    height_samples: Tensor = EnvContext.terrain.height_samples,
    horizontal_scale: float = EnvContext.terrain.horizontal_scale,
) -> Tensor:
    """Height of the root above the terrain at each sample point, ``[envs, points]``.

    ``height_points`` are the per-env sample offsets ``[envs, points, 3]`` in the
    heading frame, ``height_samples`` the whole height field in metres and
    ``horizontal_scale`` its cell size. Each sample is the bilinear interpolation
    between the four surrounding cells, as the terrain's own query computes it.
    """
    return get_height_maps_jit(
        base_rot=root_rot,
        base_pos=root_pos,
        height_points=height_points,
        height_samples=height_samples,
        num_height_points=height_points.shape[1],
        terrain_horizontal_scale=horizontal_scale,
        w_last=True,
        return_all_dims=False,
    )
