# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-bake collision approximation properties into USD asset files.

Applying collision APIs at runtime (per-clone, per-mesh) is O(num_envs × meshes)
and dominates co-training startup.  This module writes collision properties once
into a USD file so subsequent runs load them directly.

The baked file goes next to the source mesh when that folder is writable.  On a
read-only dataset it goes to a per-user cache instead
(``~/.cache/protomotions/isaaclab_collision_usd``, or the folder named by
``PROTOMOTIONS_COLLISION_CACHE_DIR``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
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

CACHE_DIR_ENV = "PROTOMOTIONS_COLLISION_CACHE_DIR"
_DEFAULT_CACHE_DIR = "~/.cache/protomotions/isaaclab_collision_usd"


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

    The result is an absolute path next to the source mesh, and always ends in
    ``.usd`` whatever the source format.

    Examples:
        armchair.usda + convexDecomposition h=32 v=64 r=100000
            → armchair.collision_cd_h32_v64_r100000.usd
        chair.usd + convexHull v=64
            → chair.collision_ch_v64.usd
        table.usda + boundingCube
            → table.collision_bc.usd
    """
    p = Path(original_path).expanduser().resolve()
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


def cached_baked_collision_path(
    original_path: str | Path,
    approximation: str,
    max_convex_hulls: Optional[int] = None,
    hull_vertex_limit: Optional[int] = None,
    voxel_resolution: Optional[int] = None,
) -> Path:
    """Where the baked file goes when the source mesh's folder is read-only.

    The cache folder is keyed by the source file's absolute path, size and
    modification time, so editing or moving the source gets a fresh bake
    instead of silently reusing a stale one.
    """
    sibling = build_baked_collision_path(
        original_path,
        approximation,
        max_convex_hulls,
        hull_vertex_limit,
        voxel_resolution,
    )
    source = Path(original_path).expanduser().resolve()
    stat = source.stat()
    digest = hashlib.sha1(
        repr((str(source), stat.st_size, stat.st_mtime_ns)).encode("utf-8")
    ).hexdigest()[:16]
    root = Path(os.environ.get(CACHE_DIR_ENV) or _DEFAULT_CACHE_DIR)
    return root.expanduser().resolve() / f"{source.stem}_{digest}" / sibling.name


def ensure_baked_collision_usd(
    original_path: str | Path,
    approximation: str,
    max_convex_hulls: Optional[int] = None,
    hull_vertex_limit: Optional[int] = None,
    voxel_resolution: Optional[int] = None,
    *,
    cache_fallback: bool = True,
) -> Path:
    """Return path to a USD with collision APIs pre-baked.

    Lookup order:

    1. A baked file next to the source mesh, if one exists.
    2. A baked file in the cache (see :func:`cached_baked_collision_path`).
    3. Otherwise bake one: next to the mesh when that folder is writable, else
       into the cache.

    Baking opens the original USD, applies collision APIs to every Mesh prim,
    and writes atomically (uniquely named temp → ``os.replace``).  The export
    copies referenced layers in and makes asset paths absolute, so the baked
    file works from either location.

    Args:
        cache_fallback: When False, only the folder next to the mesh is used,
            and baking into a read-only folder raises.  The offline conversion
            script uses this, because its job is to ship baked files with the
            dataset.
    """
    args = (
        original_path,
        approximation,
        max_convex_hulls,
        hull_vertex_limit,
        voxel_resolution,
    )
    sibling = build_baked_collision_path(*args)
    if sibling.exists():
        log.debug("Baked collision USD already exists: %s", sibling)
        return sibling

    baked = sibling
    if cache_fallback:
        cached = cached_baked_collision_path(*args)
        if cached.exists():
            log.debug("Baked collision USD already exists: %s", cached)
            return cached
        if not os.access(sibling.parent, os.W_OK):
            log.info(
                "%s is not writable; baking collision into %s",
                sibling.parent,
                cached.parent,
            )
            baked = cached

    from pxr import Usd, UsdPhysics

    log.info("Baking collision '%s' into %s ...", approximation, baked.name)

    source_path = Path(original_path).expanduser().resolve()
    supported = (".usd", ".usda", ".usdc")
    if source_path.suffix.lower() not in supported:
        raise ValueError(
            f"Cannot bake collision from '{source_path.suffix}' — only "
            f"{supported} are supported. Convert your meshes to USD first "
            f"(see scripts/convert_obj_scenes_to_usd.py)."
        )

    stage = Usd.Stage.Open(str(source_path))
    if approximation in ("convexDecomposition", "convexHull"):
        # PhysxSchema only exists inside a running Isaac Sim app, so import it
        # only when a PhysX-specific API is needed.
        from pxr import PhysxSchema

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

    # Atomic write: export to a temp file, then rename. Ranks on different
    # machines (or pods) can share a cache folder and a PID, so the temp name
    # must not be just the PID; os.replace overwrites any copy another rank
    # wrote, on every platform.
    baked.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = baked.with_name(f"{baked.stem}.tmp{uuid.uuid4().hex}{baked.suffix}")
    stage.Export(str(tmp_path))
    os.replace(tmp_path, baked)
    log.info("Baked collision USD written: %s", baked)
    return baked
