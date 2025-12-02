# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""General-purpose mesh utilities.

Provides functions for working with 3D mesh objects, used across the codebase.
"""

import trimesh


def as_mesh(scene_or_mesh):
    """Convert a Trimesh Scene or Mesh to a single Mesh object.

    Args:
        scene_or_mesh: Trimesh Scene or Mesh

    Returns:
        Trimesh Mesh (concatenated if Scene)
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [
                trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
                for m in scene_or_mesh.geometry.values()
            ]
        )
    else:
        mesh = scene_or_mesh
    return mesh


def compute_bounding_box(mesh):
    """Compute bounding box dimensions and min corner of a mesh.

    Args:
        mesh: Trimesh mesh object

    Returns:
        Tuple of (width, height, depth, min_x, min_y, min_z)
    """
    min_x, min_y, min_z = (
        mesh.vertices[:, 0].min(),
        mesh.vertices[:, 1].min(),
        mesh.vertices[:, 2].min(),
    )
    max_x, max_y, max_z = (
        mesh.vertices[:, 0].max(),
        mesh.vertices[:, 1].max(),
        mesh.vertices[:, 2].max(),
    )

    return max_x - min_x, max_y - min_y, max_z - min_z, min_x, min_y, min_z
