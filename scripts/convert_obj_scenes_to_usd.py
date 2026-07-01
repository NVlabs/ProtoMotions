# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert OBJ/URDF scene meshes to USD and pre-bake collision approximations.

Scans a scene .pt file, finds all referenced mesh objects, converts any
non-USD meshes (OBJ, URDF) to USDA, and optionally bakes collision
approximation properties into sibling USD files.

This is a one-time data preparation step. Training code only accepts
USD-format meshes.

Usage (run inside IsaacLab container or with pxr available):

    # Convert + bake collision for Beyond dataset:
    python scripts/convert_obj_scenes_to_usd.py \\
        --scene-file /path/to/beyond_scenes_soma23.pt \\
        --asset-root /path/to/beyond/scenes/ \\
        --bake-collision \\
        --approximation convexDecomposition \\
        --max-convex-hulls 32 \\
        --hull-vertex-limit 64 \\
        --voxel-resolution 300000

    # Dry run (just list what would be converted):
    python scripts/convert_obj_scenes_to_usd.py \\
        --scene-file /path/to/beyond_scenes_soma23.pt \\
        --asset-root /path/to/beyond/scenes/ \\
        --dry-run
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import trimesh

log = logging.getLogger(__name__)


def obj_to_usda(obj_path: Path, usda_path: Path) -> None:
    """Convert an OBJ mesh to USDA with physics properties using trimesh + pxr."""
    from pxr import Usd, UsdGeom, UsdPhysics, Vt, Gf

    mesh = trimesh.load_mesh(str(obj_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())

    stage = Usd.Stage.CreateNew(str(usda_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, "/Root")
    usd_mesh = UsdGeom.Mesh.Define(stage, "/Root/Mesh")

    usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*v) for v in mesh.vertices]))
    usd_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(mesh.faces)))
    usd_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(mesh.faces.flatten().tolist()))

    # Add physics properties so IsaacLab can attach contact sensors and
    # simulate the object as a rigid body.
    mesh_prim = usd_mesh.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdPhysics.MassAPI.Apply(xform.GetPrim())
    UsdPhysics.CollisionAPI.Apply(mesh_prim)
    UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)

    stage.SetDefaultPrim(xform.GetPrim())
    stage.Save()
    log.info(
        "Converted %s -> %s (%d verts, %d faces)",
        obj_path.name,
        usda_path.name,
        len(mesh.vertices),
        len(mesh.faces),
    )


def collect_mesh_paths(scene_file: str, asset_root: str):
    """Extract unique mesh paths from a scene .pt file."""
    data = torch.load(scene_file, weights_only=False, map_location="cpu")
    scenes = data.get("original_scenes", data) if isinstance(data, dict) else data

    paths = set()
    for scene in scenes:
        objects = scene.get("objects", []) if isinstance(scene, dict) else []
        for obj in objects:
            if isinstance(obj, dict) and "object_path" in obj:
                p = obj["object_path"]
                if not os.path.isabs(p):
                    p = os.path.join(asset_root, p)
                paths.add(p)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene-file", required=True, help="Path to scene .pt file")
    parser.add_argument(
        "--asset-root",
        required=True,
        help="Root directory for resolving relative mesh paths",
    )
    parser.add_argument(
        "--bake-collision",
        action="store_true",
        help="Also bake collision approximation into USD files",
    )
    parser.add_argument(
        "--approximation",
        default="convexDecomposition",
        help="Collision approximation type (default: convexDecomposition)",
    )
    parser.add_argument("--max-convex-hulls", type=int, default=32)
    parser.add_argument("--hull-vertex-limit", type=int, default=64)
    parser.add_argument("--voxel-resolution", type=int, default=300000)
    parser.add_argument(
        "--update-scene-file",
        action="store_true",
        help="Rewrite scene .pt to reference .usda paths instead of .urdf/.obj",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be converted without doing it",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    mesh_paths = collect_mesh_paths(args.scene_file, args.asset_root)
    log.info("Found %d unique mesh references in %s", len(mesh_paths), args.scene_file)

    converted = 0
    skipped = 0
    for path_str in mesh_paths:
        p = Path(path_str)
        usda_sibling = p.with_suffix(".usda")

        # Already USD — nothing to convert
        if p.suffix.lower() in (".usd", ".usda", ".usdc"):
            log.debug("Already USD: %s", p.name)
            skipped += 1
            continue

        # Find the OBJ source
        if p.suffix.lower() == ".urdf":
            obj_path = p.with_suffix(".obj")
        elif p.suffix.lower() == ".obj":
            obj_path = p
        else:
            log.warning("Unknown format %s, skipping: %s", p.suffix, p)
            skipped += 1
            continue

        if not obj_path.exists():
            log.error("OBJ not found: %s", obj_path)
            skipped += 1
            continue

        if usda_sibling.exists():
            log.debug("USDA already exists: %s", usda_sibling.name)
            skipped += 1
            continue

        if args.dry_run:
            log.info(
                "[DRY RUN] Would convert: %s -> %s", obj_path.name, usda_sibling.name
            )
            converted += 1
            continue

        obj_to_usda(obj_path, usda_sibling)
        converted += 1

    log.info("Converted: %d, Skipped: %d", converted, skipped)

    # Optionally update scene file to reference .usda paths
    if args.update_scene_file and not args.dry_run:
        log.info("Updating scene file paths to .usda ...")
        data = torch.load(args.scene_file, weights_only=False, map_location="cpu")
        scenes_key = (
            "original_scenes"
            if isinstance(data, dict) and "original_scenes" in data
            else None
        )
        scenes = data[scenes_key] if scenes_key else data

        updated = 0
        for scene in scenes:
            objects = scene.get("objects", []) if isinstance(scene, dict) else []
            for obj in objects:
                if isinstance(obj, dict) and "object_path" in obj:
                    p = obj["object_path"]
                    if p.endswith(".urdf") or p.endswith(".obj"):
                        new_p = Path(p).with_suffix(".usda")
                        # Verify the USDA exists (resolve relative path)
                        abs_new = (
                            new_p
                            if new_p.is_absolute()
                            else Path(args.asset_root) / new_p
                        )
                        if abs_new.exists():
                            obj["object_path"] = str(new_p)
                            updated += 1

        out_path = args.scene_file  # overwrite in place
        torch.save(data, out_path)
        log.info("Updated %d object paths in %s", updated, out_path)

    # Optionally bake collision into the USDA files
    if args.bake_collision and not args.dry_run:
        from protomotions.simulator.isaaclab.utils.collision_baking import (
            ensure_baked_collision_usd,
        )

        log.info("Baking collision approximations...")
        for path_str in mesh_paths:
            p = Path(path_str)
            # Use the USDA we just created (or existing USD)
            if p.suffix.lower() in (".urdf", ".obj"):
                usda_path = p.with_suffix(".usda")
            else:
                usda_path = p

            if not usda_path.exists():
                log.warning("No USD found for baking: %s", usda_path)
                continue

            ensure_baked_collision_usd(
                usda_path,
                args.approximation,
                max_convex_hulls=args.max_convex_hulls,
                hull_vertex_limit=args.hull_vertex_limit,
                voxel_resolution=args.voxel_resolution,
            )
        log.info("Collision baking complete.")


if __name__ == "__main__":
    main()
