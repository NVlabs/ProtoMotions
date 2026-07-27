# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asset-root resolution: portable configs, explicit overrides, env override."""

from pathlib import Path

from protomotions.assets import (
    ASSET_ROOT_ENV,
    DEFAULT_ASSET_ROOT,
    get_asset_root,
    resolve_asset_root,
)


def test_default_asset_root_is_portable():
    """The value pickled into resolved_configs*.pt must not be machine-specific.

    An absolute path here stops resolving as soon as a checkpoint is moved to
    another machine, venv, or container.
    """

    assert DEFAULT_ASSET_ROOT == "protomotions/data/assets"
    assert not Path(DEFAULT_ASSET_ROOT).is_absolute()


def test_default_resolves_to_packaged_assets_regardless_of_cwd(tmp_path, monkeypatch):
    """The relative default must resolve from any working directory."""

    expected = get_asset_root()
    monkeypatch.chdir(tmp_path)
    assert resolve_asset_root(DEFAULT_ASSET_ROOT) == expected
    assert resolve_asset_root(None) == expected


def test_explicit_asset_root_is_returned_verbatim():
    """A configured root must not be validated away or silently substituted.

    Rewriting a wrong path to the packaged tree would load a *different* robot
    (e.g. mjcf/smpl_humanoid.xml exists in both trees) instead of failing.
    """

    assert resolve_asset_root("/assets") == Path("/assets")
    assert resolve_asset_root("/nonexistent/scratch/assets") == Path(
        "/nonexistent/scratch/assets"
    )


def test_env_override_wins_for_the_default(tmp_path, monkeypatch):
    """PROTOMOTIONS_ASSET_ROOT must override the default asset root."""

    external = tmp_path / "external_assets"
    external.mkdir()
    monkeypatch.setenv(ASSET_ROOT_ENV, str(external))
    assert resolve_asset_root(DEFAULT_ASSET_ROOT) == external.resolve()


def test_env_override_pointing_at_missing_directory_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(ASSET_ROOT_ENV, str(tmp_path / "missing"))
    try:
        resolve_asset_root(DEFAULT_ASSET_ROOT)
    except FileNotFoundError as exc:
        assert ASSET_ROOT_ENV in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for a missing override")
