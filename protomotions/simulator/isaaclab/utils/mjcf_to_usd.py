# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab-only MJCF → USD conversion at the scene construction boundary.

Humanoid articulations are authored as MJCF. IsaacLab consumes USD, so this
module invokes IsaacLab 3 ``MjcfConverter`` / ``MjcfConverterCfg`` lazily and
caches results by absolute MJCF path plus conversion options.

Kit is not required to import this module or to exercise path/config helpers.
Real conversion imports IsaacLab converters only inside
``default_mjcf_converter_factory``.
"""

from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple, Union

from filelock import FileLock

from protomotions.assets import resolve_asset_root
from protomotions.robot_configs.base import RobotAssetConfig

# Process-local cache: avoids reconverting when multiple SceneCfg builds share
# the same MJCF + options within one IsaacLab process.
_CONVERSION_CACHE: Dict[Tuple[Any, ...], str] = {}

ConverterFactory = Callable[..., str]
_MJCF_CONVERTER_CACHE_VERSION = "isaaclab3-d6-workaround-v2"


def _absolute_path(path: Union[os.PathLike, str]) -> str:
    """Expand user paths before resolving them against the current directory."""
    return str(Path(path).expanduser().resolve())


def resolve_robot_mjcf_path(asset: RobotAssetConfig) -> str:
    """Return the absolute MJCF path for a robot asset config."""
    if not asset.asset_file_name:
        raise ValueError("RobotAssetConfig.asset_file_name must be set to an MJCF path")
    asset_file_name = Path(asset.asset_file_name).expanduser()
    if not asset_file_name.is_absolute():
        asset_file_name = resolve_asset_root(asset.asset_root) / asset_file_name
    return _absolute_path(asset_file_name)


def predicted_converted_usd_path(mjcf_path: str, usd_dir: str) -> str:
    """Return the USD path IsaacLab 3 ``MjcfConverter`` will produce.

    The converter rewrites ``usd_file_name`` to ``{stem}/{stem}.usda`` under
    ``usd_dir``.
    """
    stem = Path(_absolute_path(mjcf_path)).stem
    return _absolute_path(Path(usd_dir).expanduser() / stem / f"{stem}.usda")


def default_usd_cache_dir(mjcf_path: str, options: Mapping[str, Any]) -> str:
    """Stable on-disk cache directory for a MJCF path and conversion options."""
    abs_mjcf = _absolute_path(mjcf_path)
    digest = hashlib.sha1(
        repr(
            (
                _MJCF_CONVERTER_CACHE_VERSION,
                abs_mjcf,
                _mjcf_fingerprint(abs_mjcf),
                _normalized_options(options),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    stem = Path(abs_mjcf).stem
    cache_root = Path("~/.cache/protomotions/isaaclab_mjcf_usd").expanduser()
    return _absolute_path(cache_root / f"{stem}_{digest}")


def build_mjcf_converter_cfg_kwargs(
    mjcf_path: str,
    *,
    usd_dir: Optional[str] = None,
    force_usd_conversion: bool = False,
    self_collision: bool = False,
    fix_base: bool = False,
    merge_mesh: bool = False,
    collision_from_visuals: bool = False,
) -> Dict[str, Any]:
    """Build kwargs for IsaacLab 3 ``MjcfConverterCfg`` without importing Kit.

    Only fields present on the public IsaacLab 3 develop API are included.
    Obsolete IsaacLab 2 fields (``import_sites``, ``make_instanceable``) are
    intentionally omitted.
    """
    abs_mjcf = _absolute_path(mjcf_path)
    options = {
        "self_collision": bool(self_collision),
        "fix_base": bool(fix_base),
        "merge_mesh": bool(merge_mesh),
        "collision_from_visuals": bool(collision_from_visuals),
    }
    resolved_usd_dir = (
        _absolute_path(usd_dir)
        if usd_dir is not None
        else default_usd_cache_dir(abs_mjcf, options)
    )
    return {
        "asset_path": abs_mjcf,
        "usd_dir": resolved_usd_dir,
        "force_usd_conversion": bool(force_usd_conversion),
        **options,
    }


def conversion_cache_key(
    cfg_kwargs: Mapping[str, Any],
    *,
    include_source_fingerprint: bool = True,
) -> Tuple[Any, ...]:
    """Stable in-memory cache key for converter cfg kwargs."""
    return (
        _MJCF_CONVERTER_CACHE_VERSION,
        _absolute_path(str(cfg_kwargs["asset_path"])),
        _absolute_path(str(cfg_kwargs["usd_dir"])),
        bool(cfg_kwargs.get("force_usd_conversion", False)),
        bool(cfg_kwargs.get("self_collision", False)),
        bool(cfg_kwargs.get("fix_base", False)),
        bool(cfg_kwargs.get("merge_mesh", False)),
        bool(cfg_kwargs.get("collision_from_visuals", False)),
        (
            _mjcf_fingerprint(str(cfg_kwargs["asset_path"]))
            if include_source_fingerprint
            else None
        ),
    )


def _mjcf_fingerprint(path: str) -> Optional[str]:
    """Hash an MJCF and its referenced assets for cache invalidation.

    IsaacLab's converter consumes mesh and texture files referenced by the
    MJCF, so hashing only the XML timestamp can leave a stale USD after an
    in-place mesh edit. Missing references are included in the digest too, so
    adding a previously missing asset invalidates the cache.
    """
    root_path = Path(_absolute_path(path))
    if not root_path.is_file():
        return None

    digest = hashlib.sha1()
    visited: set[Path] = set()

    def hash_file(file_path: Path) -> None:
        file_path = Path(_absolute_path(file_path))
        digest.update(b"\x00path:")
        digest.update(str(file_path).encode("utf-8"))
        if not file_path.is_file():
            digest.update(b"\x00missing")
            return
        try:
            data = file_path.read_bytes()
        except OSError:
            digest.update(b"\x00unreadable")
            return
        digest.update(b"\x00content:")
        digest.update(data)

    def visit_xml(xml_path: Path) -> None:
        xml_path = Path(_absolute_path(xml_path))
        if xml_path in visited:
            return
        visited.add(xml_path)
        try:
            data = xml_path.read_bytes()
        except OSError:
            hash_file(xml_path)
            return
        digest.update(b"\x00xml:")
        digest.update(str(xml_path).encode("utf-8"))
        digest.update(data)
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            return

        compiler = root.find("compiler")
        mesh_dir = (compiler.get("meshdir", "") if compiler is not None else "")
        texture_dir = (compiler.get("texturedir", "") if compiler is not None else "")
        for element in root.iter():
            reference = element.get("file")
            if not reference:
                continue
            if element.tag == "include":
                visit_xml(xml_path.parent / Path(reference).expanduser())
                continue
            directory = mesh_dir if element.tag == "mesh" else texture_dir
            reference_path = Path(reference).expanduser()
            if directory and not reference_path.is_absolute():
                reference_path = Path(directory).expanduser() / reference_path
            hash_file(xml_path.parent / reference_path)

    visit_xml(root_path)
    return digest.hexdigest()


def _normalized_options(options: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((str(k), options[k]) for k in options))


def _conversion_coordination_paths(
    usd_dir: str, cache_key: Tuple[Any, ...]
) -> Tuple[Path, Path]:
    output_dir = Path(_absolute_path(usd_dir))
    key_digest = hashlib.sha1(repr(cache_key).encode("utf-8")).hexdigest()[:16]
    lock_path = output_dir.with_name(f".{output_dir.name}.protomotions.lock")
    marker_path = output_dir.with_name(
        f".{output_dir.name}.{key_digest}.protomotions-complete"
    )
    return lock_path, marker_path


def _completed_conversion(marker_path: Path) -> Optional[str]:
    try:
        usd_path = _absolute_path(marker_path.read_text().strip())
    except (OSError, ValueError):
        return None
    return usd_path if Path(usd_path).is_file() else None


def _publish_completed_conversion(marker_path: Path, usd_path: str) -> None:
    temporary_marker = marker_path.with_name(
        f"{marker_path.name}.{os.getpid()}.tmp"
    )
    temporary_marker.write_text(usd_path)
    os.replace(temporary_marker, marker_path)


def default_mjcf_converter_factory(**cfg_kwargs: Any) -> str:
    """Run IsaacLab ``MjcfConverter`` and return the generated USD path.

    Requires an active Isaac Sim / Kit runtime.
    """
    import omni.kit.app

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    for extension_id in (
        "isaacsim.asset.importer.utils",
        "isaacsim.asset.importer.mjcf",
    ):
        extension_manager.set_extension_enabled_immediate(extension_id, True)

    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    from protomotions.simulator.isaaclab.utils.mjcf_d6_workaround import (
        install_isaaclab_mjcf_d6_workaround,
    )

    install_isaaclab_mjcf_d6_workaround()

    converter = MjcfConverter(MjcfConverterCfg(**cfg_kwargs))
    return _absolute_path(converter.usd_path)


def dry_run_mjcf_converter_factory(**cfg_kwargs: Any) -> str:
    """Deterministic Kit-free factory used by unit tests and dry-run mode."""
    return predicted_converted_usd_path(
        cfg_kwargs["asset_path"], cfg_kwargs["usd_dir"]
    )


def convert_mjcf_to_usd(
    mjcf_path: str,
    *,
    converter_factory: Optional[ConverterFactory] = None,
    cache: Optional[MutableMapping[Tuple[Any, ...], str]] = None,
    usd_dir: Optional[str] = None,
    force_usd_conversion: bool = False,
    self_collision: bool = False,
    fix_base: bool = False,
    merge_mesh: bool = False,
    collision_from_visuals: bool = False,
) -> str:
    """Convert MJCF to USD via IsaacLab 3 APIs, with process-local caching.

    Args:
        mjcf_path: Path to the MJCF file.
        converter_factory: Optional injectable factory ``(**cfg_kwargs) -> usd_path``.
            Defaults to ``default_mjcf_converter_factory``, or
            ``dry_run_mjcf_converter_factory`` when
            ``PROTOMOTIONS_ISAACLAB_MJCF_DRY_RUN`` is set.
        cache: Optional cache mapping; defaults to the module-level cache.
        usd_dir: Optional output directory; defaults to a stable user cache path.
        force_usd_conversion: Forwarded to ``MjcfConverterCfg``.
        self_collision: Forwarded to ``MjcfConverterCfg``.
        fix_base: Forwarded to ``MjcfConverterCfg``.
        merge_mesh: Forwarded to ``MjcfConverterCfg``.
        collision_from_visuals: Forwarded to ``MjcfConverterCfg``.

    Returns:
        Absolute path to the generated (or predicted) USD file.
    """
    uses_default_cache = usd_dir is None
    cfg_kwargs = build_mjcf_converter_cfg_kwargs(
        mjcf_path,
        usd_dir=usd_dir,
        force_usd_conversion=force_usd_conversion,
        self_collision=self_collision,
        fix_base=fix_base,
        merge_mesh=merge_mesh,
        collision_from_visuals=collision_from_visuals,
    )
    key = conversion_cache_key(
        cfg_kwargs,
        include_source_fingerprint=not uses_default_cache,
    )
    cache_store = _CONVERSION_CACHE if cache is None else cache
    if key in cache_store and not force_usd_conversion:
        return cache_store[key]

    lock_path, marker_path = _conversion_coordination_paths(
        cfg_kwargs["usd_dir"], key
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(lock_path):
        if key in cache_store and not force_usd_conversion:
            return cache_store[key]

        if not force_usd_conversion:
            completed_path = _completed_conversion(marker_path)
            if completed_path is not None:
                cache_store[key] = completed_path
                return completed_path

        if converter_factory is None:
            if os.environ.get("PROTOMOTIONS_ISAACLAB_MJCF_DRY_RUN"):
                converter_factory = dry_run_mjcf_converter_factory
            else:
                converter_factory = default_mjcf_converter_factory

        factory_kwargs = cfg_kwargs
        if not force_usd_conversion:
            existing_usd_path = Path(
                predicted_converted_usd_path(
                    cfg_kwargs["asset_path"], cfg_kwargs["usd_dir"]
                )
            )
            if existing_usd_path.is_file():
                # IsaacLab's .asset_hash does not include this workaround's
                # version, so an explicit output directory can otherwise
                # reuse a USD generated before the repair was installed.
                factory_kwargs = {**cfg_kwargs, "force_usd_conversion": True}

        usd_path = _absolute_path(converter_factory(**factory_kwargs))
        cache_store[key] = usd_path
        if Path(usd_path).is_file():
            _publish_completed_conversion(marker_path, usd_path)
        return usd_path


def convert_robot_mjcf_to_usd(
    asset: RobotAssetConfig,
    *,
    converter_factory: Optional[ConverterFactory] = None,
    cache: Optional[MutableMapping[Tuple[Any, ...], str]] = None,
    usd_dir: Optional[str] = None,
    force_usd_conversion: bool = False,
    fix_base: Optional[bool] = None,
    merge_mesh: bool = False,
    collision_from_visuals: bool = False,
) -> str:
    """Convert a robot asset's MJCF to USD for IsaacLab spawning."""
    return convert_mjcf_to_usd(
        resolve_robot_mjcf_path(asset),
        converter_factory=converter_factory,
        cache=cache,
        usd_dir=usd_dir,
        force_usd_conversion=force_usd_conversion,
        self_collision=bool(asset.self_collisions),
        fix_base=bool(asset.fix_base_link) if fix_base is None else fix_base,
        merge_mesh=merge_mesh,
        collision_from_visuals=collision_from_visuals,
    )


def clear_mjcf_usd_conversion_cache() -> None:
    """Clear the process-local conversion cache (tests only)."""
    _CONVERSION_CACHE.clear()
