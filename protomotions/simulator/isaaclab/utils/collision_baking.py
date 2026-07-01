# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-bake collision approximation properties into USD asset files.

Applying collision APIs at runtime (per-clone, per-mesh) is O(num_envs × meshes)
and dominates co-training startup.  This module writes collision properties once
into a sibling USD file so subsequent runs load them directly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Short abbreviations for collision approximation types
_APPROX_ABBREV = {
    "convexDecomposition": "cd",
    "convexHull": "ch",
    "boundingCube": "bc",
    "boundingSphere": "bs",
}


def build_baked_collision_path(
    original_path: str | Path,
    approximation: str,
    max_convex_hulls: Optional[int] = None,
    hull_vertex_limit: Optional[int] = None,
    voxel_resolution: Optional[int] = None,
) -> Path:
    """Build the path for a baked collision USD file.

    Naming convention:
        {stem}.collision_{abbrev}[_h{hulls}][_v{vertices}][_r{resolution}]{ext}

    Examples:
        armchair.usda + convexDecomposition h=32 v=64 r=100000
            → armchair.collision_cd_h32_v64_r100000.usda
        chair.usd + convexHull v=64
            → chair.collision_ch_v64.usd
        table.usda + boundingCube
            → table.collision_bc.usda
    """
    p = Path(original_path)
    abbrev = _APPROX_ABBREV.get(approximation, approximation)
    suffix_parts = [f"collision_{abbrev}"]
    if max_convex_hulls is not None:
        suffix_parts.append(f"h{max_convex_hulls}")
    if hull_vertex_limit is not None:
        suffix_parts.append(f"v{hull_vertex_limit}")
    if voxel_resolution is not None:
        suffix_parts.append(f"r{voxel_resolution}")
    tag = "_".join(suffix_parts)
    # Always use .usd extension — the baked file is valid USD regardless of
    # the original format (obj, usda, urdf, etc.).
    return p.with_name(f"{p.stem}.{tag}.usd")


def ensure_baked_collision_usd(
    original_path: str | Path,
    approximation: str,
    max_convex_hulls: Optional[int] = None,
    hull_vertex_limit: Optional[int] = None,
    voxel_resolution: Optional[int] = None,
) -> Path:
    """Return path to a USD with collision APIs pre-baked.

    If the baked file already exists, returns immediately.  Otherwise opens the
    original USD, applies collision APIs to every Mesh prim, and writes an
    atomic sibling file (PID-suffixed temp → ``os.rename``).
    """
    baked = build_baked_collision_path(
        original_path,
        approximation,
        max_convex_hulls,
        hull_vertex_limit,
        voxel_resolution,
    )
    if baked.exists():
        log.debug("Baked collision USD already exists: %s", baked)
        return baked

    from pxr import Usd, UsdPhysics, PhysxSchema

    log.info("Baking collision '%s' into %s ...", approximation, baked.name)

    source_path = Path(original_path)
    supported = (".usd", ".usda", ".usdc")
    if source_path.suffix.lower() not in supported:
        raise ValueError(
            f"Cannot bake collision from '{source_path.suffix}' — only "
            f"{supported} are supported. Convert your meshes to USD first "
            f"(see scripts/convert_obj_scenes_to_usd.py)."
        )

    stage = Usd.Stage.Open(str(source_path))

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue

        mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_col.GetApproximationAttr().Set(approximation)

        if approximation == "convexDecomposition":
            cd_api = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
            if max_convex_hulls is not None:
                cd_api.GetMaxConvexHullsAttr().Set(max_convex_hulls)
            if hull_vertex_limit is not None:
                cd_api.GetHullVertexLimitAttr().Set(hull_vertex_limit)
            if voxel_resolution is not None:
                cd_api.GetVoxelResolutionAttr().Set(voxel_resolution)
        elif approximation == "convexHull":
            ch_api = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
            if hull_vertex_limit is not None:
                ch_api.GetHullVertexLimitAttr().Set(hull_vertex_limit)

    # Atomic write: export to PID-suffixed temp, then rename
    tmp_path = baked.with_suffix(f".tmp{os.getpid()}{baked.suffix}")
    stage.Export(str(tmp_path))
    os.rename(str(tmp_path), str(baked))
    log.info("Baked collision USD written: %s", baked)
    return baked
