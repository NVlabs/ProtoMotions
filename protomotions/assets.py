# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve ProtoMotions runtime assets in source and installed distributions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union


ASSET_ROOT_ENV = "PROTOMOTIONS_ASSET_ROOT"


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


def asset_path(*parts: Union[str, os.PathLike[str]], must_exist: bool = True) -> Path:
    """Resolve a path below :func:`get_asset_root`.

    Args:
        *parts: Relative path components below the asset root.
        must_exist: Raise ``FileNotFoundError`` when the resolved path is absent.
    """

    path = get_asset_root().joinpath(*map(Path, parts))
    if must_exist and not path.exists():
        raise FileNotFoundError(f"ProtoMotions asset is missing: {path}")
    return path


__all__ = ["ASSET_ROOT_ENV", "asset_path", "get_asset_root"]
