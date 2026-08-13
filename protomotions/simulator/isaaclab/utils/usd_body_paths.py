# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve body prim paths from USD files for contact sensor setup."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple


def relative_prim_path(full_path: str, default_path: str) -> str:
    """Strip the default prim prefix from a USD prim path."""
    if default_path and full_path == default_path:
        return ""
    if default_path and full_path.startswith(default_path + "/"):
        return full_path[len(default_path) + 1 :]
    return full_path.lstrip("/")


def resolve_body_prim_paths_from_records(
    body_records: Iterable[Mapping[str, object]],
    body_names: List[str],
    *,
    default_path: str = "",
) -> Dict[str, str]:
    """Resolve body names to relative prim paths from lightweight records.

    This is the Kit-free seam used by unit tests. Each record must provide
    ``name``, ``full_path``, and ``is_rigid_body``.
    """
    body_names_set = set(body_names)
    results: Dict[str, str] = {}
    for record in body_records:
        full_path = str(record["full_path"])
        if default_path and full_path != default_path and not full_path.startswith(
            default_path + "/"
        ):
            continue
        name = str(record["name"])
        if name not in body_names_set or name in results:
            continue
        if not bool(record.get("is_rigid_body", False)):
            continue
        results[name] = relative_prim_path(str(record["full_path"]), default_path)

    missing = [name for name in body_names if name not in results]
    if missing:
        raise ValueError(
            f"Could not find prim paths for bodies: {missing}. "
            f"Found: {list(results.keys())}"
        )
    return results


def resolve_articulation_root_prim_path_from_records(
    records: Iterable[Mapping[str, object]],
    *,
    default_path: str = "",
) -> str:
    """Resolve the sole articulation root relative to a USD default prim."""
    roots = []
    for record in records:
        full_path = str(record["full_path"])
        if not bool(record.get("is_articulation_root", False)):
            continue
        if default_path and full_path != default_path and not full_path.startswith(
            default_path + "/"
        ):
            continue
        roots.append(relative_prim_path(full_path, default_path))

    if len(roots) != 1:
        raise ValueError(
            "Expected exactly one prim with ArticulationRootAPI below the USD "
            f"default prim {default_path!r}, found {len(roots)}"
        )

    relative = roots[0]
    return "/" if not relative else f"/{relative}"


def _stage_records(usd_path: str, stage_factory=None):
    if stage_factory is not None:
        stage = stage_factory(usd_path)
        default_path = getattr(stage, "default_path", "") or ""
        records = getattr(stage, "body_records", None)
        if records is None:
            raise TypeError(
                "stage_factory must return an object with body_records and default_path"
            )
        return records, default_path

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise ValueError(f"Could not open USD stage: {usd_path}")
    default_prim = stage.GetDefaultPrim()
    default_path = default_prim.GetPath().pathString if default_prim else ""
    records = []
    for prim in stage.Traverse():
        records.append(
            {
                "name": prim.GetName(),
                "full_path": prim.GetPath().pathString,
                "is_rigid_body": prim.HasAPI(UsdPhysics.RigidBodyAPI),
                "is_articulation_root": prim.HasAPI(UsdPhysics.ArticulationRootAPI),
            }
        )
    return records, default_path


def resolve_body_prim_paths(
    usd_path: str,
    body_names: List[str],
    *,
    stage_factory=None,
) -> Dict[str, str]:
    """Find the relative prim path for each body name in a USD file.

    Traverses the USD stage to find prims with ``PhysicsRigidBodyAPI`` whose
    name matches a requested body. Returns paths relative to the default prim.
    Works with both flat and nested USD hierarchies.

    Args:
        usd_path: Path to the USD file.
        body_names: List of body names to resolve.
        stage_factory: Optional injectable ``(usd_path) -> stage-like``.
            Defaults to opening the file with ``pxr.Usd.Stage``.

    Returns:
        Dict mapping body_name -> relative prim path
        (e.g. ``"Geometry/pelvis/left_ankle_roll_link"``).

    Raises:
        ValueError: If any body name cannot be found in the USD.
    """
    records, default_path = _stage_records(usd_path, stage_factory)
    return resolve_body_prim_paths_from_records(
        records, body_names, default_path=default_path
    )


def resolve_articulation_root_prim_path(
    usd_path: str,
    *,
    stage_factory=None,
) -> str:
    """Find the articulation root path in a converted USD stage."""
    records, default_path = _stage_records(usd_path, stage_factory)
    return resolve_articulation_root_prim_path_from_records(
        records, default_path=default_path
    )


def resolve_robot_prim_paths(
    usd_path: str,
    body_names: List[str],
    *,
    stage_factory=None,
) -> Tuple[str, Dict[str, str]]:
    """Resolve articulation and rigid-body paths with one stage traversal."""
    records, default_path = _stage_records(usd_path, stage_factory)
    articulation_root = resolve_articulation_root_prim_path_from_records(
        records, default_path=default_path
    )
    body_paths = resolve_body_prim_paths_from_records(
        records, body_names, default_path=default_path
    )
    return articulation_root, body_paths


def contact_sensor_prim_path(
    body_name: str,
    body_prim_paths: MutableMapping[str, str],
    *,
    robot_prim_root: str = "/World/envs/env_.*/Robot",
) -> str:
    """Build a ContactSensorCfg prim path for a resolved body."""
    relative = body_prim_paths[body_name]
    return f"{robot_prim_root}/{relative}"
