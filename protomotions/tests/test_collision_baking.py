# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for where baked collision USD files are read from and written to."""

import os
import stat

import pytest

from protomotions.simulator.isaaclab.utils.collision_baking import (
    CACHE_DIR_ENV,
    build_baked_collision_path,
    cached_baked_collision_path,
    ensure_baked_collision_usd,
)

_TRIANGLE_USDA = """#usda 1.0
(
    defaultPrim = "root"
)

def Xform "root"
{
    def Mesh "tri"
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }
}
"""


def _skip_if_root():
    # Root ignores directory permissions, so a read-only folder stays writable.
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("directory permissions are not enforced for root")


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    monkeypatch.setenv(CACHE_DIR_ENV, str(root))
    return root


@pytest.fixture
def mesh(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    path = src / "m.usda"
    path.write_text(_TRIANGLE_USDA)
    return path


@pytest.fixture
def read_only(mesh):
    folder = mesh.parent
    mode = folder.stat().st_mode
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        yield folder
    finally:
        folder.chmod(mode)


def _approximation(path):
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/root/tri")
    return UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()


def test_existing_file_next_to_mesh_wins_even_when_folder_is_read_only(
    mesh, cache_root
):
    _skip_if_root()
    sibling = build_baked_collision_path(mesh, "boundingCube")
    sibling.write_text("already baked")
    folder = mesh.parent
    mode = folder.stat().st_mode
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        assert ensure_baked_collision_usd(mesh, "boundingCube") == sibling
    finally:
        folder.chmod(mode)
    assert not cache_root.exists()


def test_cache_path_is_per_source_file_and_under_the_cache_root(tmp_path, cache_root):
    first = tmp_path / "a" / "m.usda"
    second = tmp_path / "b" / "m.usda"
    for path in (first, second):
        path.parent.mkdir()
        path.write_text(_TRIANGLE_USDA)

    first_cached = cached_baked_collision_path(first, "boundingCube")
    second_cached = cached_baked_collision_path(second, "boundingCube")

    assert first_cached != second_cached
    assert first_cached.name == build_baked_collision_path(first, "boundingCube").name
    for cached in (first_cached, second_cached):
        assert cache_root.resolve() in cached.parents


def test_cache_path_changes_when_the_source_changes(mesh, cache_root):
    original = cached_baked_collision_path(mesh, "boundingCube")
    info = mesh.stat()

    # An in-place edit that keeps the size (0.5 -> 0.7) only moves the mtime.
    os.utime(mesh, ns=(info.st_atime_ns, info.st_mtime_ns + 10**9))
    assert mesh.stat().st_size == info.st_size
    touched = cached_baked_collision_path(mesh, "boundingCube")

    # A size change counts even when the mtime is put back.
    mesh.write_text(_TRIANGLE_USDA + "\n")
    os.utime(mesh, ns=(info.st_atime_ns, info.st_mtime_ns))
    resized = cached_baked_collision_path(mesh, "boundingCube")

    assert len({original, touched, resized}) == 3


def test_read_only_folder_bakes_into_the_cache(mesh, cache_root, read_only):
    pytest.importorskip("pxr")
    _skip_if_root()

    baked = ensure_baked_collision_usd(mesh, "boundingCube")

    assert baked.exists()
    assert cache_root.resolve() in baked.parents
    assert sorted(p.name for p in read_only.iterdir()) == ["m.usda"]
    assert sorted(p.name for p in baked.parent.iterdir()) == [baked.name]
    assert _approximation(baked) == "boundingCube"
    assert ensure_baked_collision_usd(mesh, "boundingCube") == baked


def test_writable_folder_still_bakes_next_to_the_mesh(mesh, cache_root):
    pytest.importorskip("pxr")

    baked = ensure_baked_collision_usd(mesh, "boundingCube")

    assert baked == build_baked_collision_path(mesh, "boundingCube")
    assert _approximation(baked) == "boundingCube"
    assert not cache_root.exists()


def test_without_cache_fallback_a_read_only_folder_still_raises(
    mesh, cache_root, read_only
):
    pytest.importorskip("pxr")
    _skip_if_root()

    with pytest.raises(Exception):
        ensure_baked_collision_usd(mesh, "boundingCube", cache_fallback=False)
    assert not cache_root.exists()


def test_bakes_that_share_a_pid_write_different_temp_files(
    mesh, cache_root, monkeypatch
):
    # Ranks in different pods can share both a cache folder and a PID.
    pytest.importorskip("pxr")
    from protomotions.simulator.isaaclab.utils import collision_baking

    temp_files = []
    replace = os.replace

    def recording_replace(src, dst):
        temp_files.append(src)
        replace(src, dst)

    monkeypatch.setattr(collision_baking.os, "getpid", lambda: 4242)
    monkeypatch.setattr(collision_baking.os, "replace", recording_replace)

    baked = ensure_baked_collision_usd(mesh, "boundingCube")
    baked.unlink()
    assert ensure_baked_collision_usd(mesh, "boundingCube") == baked

    assert len(temp_files) == 2
    assert temp_files[0] != temp_files[1]
