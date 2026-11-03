# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the robot guide complete without requiring simulator or LFS assets."""

from pathlib import Path

from protomotions.robot_configs.factory import registered_robot_names


ROOT = Path(__file__).resolve().parents[2]


def test_every_registered_robot_has_a_linked_guide():
    # The registry is the robot_configs/ directory, so the guide must cover
    # every module there without importing a simulator or loading LFS assets.
    names = set(registered_robot_names())
    assert names, "No robot names found in the registry"
    guide_dir = ROOT / "docs/source/robots"
    overview = (guide_dir / "index.rst").read_text()
    assert "   robots/index\n" in (ROOT / "docs/source/index.rst").read_text()
    for name in sorted(names):
        page = guide_dir / f"{name}.rst"
        assert page.is_file(), f"Missing robot guide: {name}"
        assert f"   {name}\n" in overview, f"Missing toctree entry: {name}"
        assert f":doc:`{name}`" in overview, f"Missing comparison row: {name}"
        assert f"--robot-name {name}" in page.read_text()
