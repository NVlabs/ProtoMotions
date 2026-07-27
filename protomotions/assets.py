# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve ProtoMotions runtime assets in source and installed distributions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union


ASSET_ROOT_ENV = "PROTOMOTIONS_ASSET_ROOT"

# Portable, checkout-relative default stored in configs. This value is what gets
# pickled into ``resolved_configs*.pt``, so it must never be machine-specific:
# an absolute path baked into a checkpoint stops resolving the moment the run is
# moved to another machine, venv, or container. Resolution to a concrete
# directory happens at read time via :func:`resolve_asset_root`.
DEFAULT_ASSET_ROOT = "protomotions/data/assets"


def get_asset_root() -> Path:
    """Return the directory containing the bundled simulator assets.

    ``PROTOMOTIONS_ASSET_ROOT`` can point at a separately managed asset tree.
    Otherwise assets are resolved relative to the installed ``protomotions``
    package, which also covers editable source checkouts.
    """

    configured_root = os.environ.get(ASSET_ROOT_ENV)
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"{ASSET_ROOT_ENV} points to a missing directory: {root}"
            )
        return root

    root = Path(__file__).resolve().parent / "data" / "assets"
    if not root.is_dir():
        raise FileNotFoundError(
            "ProtoMotions runtime assets are missing from the installed "
            f"distribution: {root}. Reinstall the package or set "
            f"{ASSET_ROOT_ENV} to a complete asset tree."
        )
    return root


def resolve_asset_root(asset_root: Optional[Union[str, os.PathLike]] = None) -> Path:
    """Resolve a configured asset root to a concrete directory.

    Configs store a portable, checkout-relative asset root (see
    :data:`DEFAULT_ASSET_ROOT`) rather than an absolute path, so that a
    ``resolved_configs*.pt`` written on one machine stays loadable on another.

    * The default (or an unset value) resolves through :func:`get_asset_root`,
      which honours ``PROTOMOTIONS_ASSET_ROOT`` and otherwise anchors to the
      installed package. For a source checkout that is the same
      ``<repo>/protomotions/data/assets`` directory as before, but it no longer
      depends on the process working directory.
    * An explicitly configured root is returned verbatim. It is deliberately
      NOT validated or substituted here: a wrong path must fail loudly naming
      the path the user configured, rather than silently loading a different
      robot from the packaged tree.
    """

    if asset_root and str(asset_root) != DEFAULT_ASSET_ROOT:
        return Path(asset_root)
    return get_asset_root()


def asset_path(*parts: Union[str, os.PathLike], must_exist: bool = True) -> Path:
    """Resolve a path below :func:`get_asset_root`.

    Args:
        *parts: Relative path components below the asset root.
        must_exist: Raise ``FileNotFoundError`` when the resolved path is absent.
    """

    path = get_asset_root().joinpath(*map(Path, parts))
    if must_exist and not path.exists():
        raise FileNotFoundError(
            f"ProtoMotions asset is missing: {path}. Published wheels omit the "
            "SMPL/SMPL-H body-model assets, which carry their own licence terms. "
            "Use a Git LFS source checkout or point "
            f"{ASSET_ROOT_ENV} at a complete asset tree."
        )
    return path


__all__ = [
    "ASSET_ROOT_ENV",
    "DEFAULT_ASSET_ROOT",
    "asset_path",
    "get_asset_root",
    "resolve_asset_root",
]
