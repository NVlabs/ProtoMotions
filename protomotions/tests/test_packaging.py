# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the installable ProtoMotions package."""

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 and earlier
    import tomli as tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_discovers_only_protomotions_code_packages():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    package_finder = pyproject["tool"]["setuptools"]["packages"]["find"]
    package_data = pyproject["tool"]["setuptools"]["package-data"]["protomotions"]

    assert package_finder["include"] == ["protomotions", "protomotions.*"]
    assert package_finder["namespaces"] is False
    assert "protomotions.tests" in package_finder["exclude"]
    assert "data/assets/**/*" in package_data


def test_setup_discovers_protomotions_subpackages():
    """Run the same package discovery used by setuptools."""

    from setuptools import find_packages

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    package_finder = pyproject["tool"]["setuptools"]["packages"]["find"]
    discovered = set(
        find_packages(
            where=str(REPO_ROOT),
            include=package_finder["include"],
            exclude=package_finder["exclude"],
        )
    )

    for expected in (
        "protomotions.agents",
        "protomotions.components",
        "protomotions.envs",
        "protomotions.robot_configs",
        "protomotions.simulator",
        "protomotions.simulator.isaacgym",
        "protomotions.simulator.isaaclab",
        "protomotions.simulator.mujoco",
        "protomotions.simulator.newton",
        "protomotions.utils",
    ):
        assert expected in discovered, f"{expected} would not ship in the wheel"

    assert "protomotions.tests" not in discovered


def test_every_protomotions_subpackage_has_init_so_it_ships():
    """Guard against a new module directory silently missing from the wheel."""

    from setuptools import find_packages

    code_dirs = set()
    for path in (REPO_ROOT / "protomotions").rglob("*.py"):
        relative = path.relative_to(REPO_ROOT)
        parts = relative.parts[:-1]
        if "tests" in parts or "data" in parts:
            continue
        if any(part.startswith(".") for part in parts):
            continue
        code_dirs.add(".".join(parts))

    discovered = set(
        find_packages(
            where=str(REPO_ROOT),
            include=["protomotions", "protomotions.*"],
            exclude=["protomotions.tests", "protomotions.tests.*"],
        )
    )
    missing = sorted(code_dirs - discovered)
    assert missing == [], (
        "these directories contain modules but would not ship in the wheel; "
        f"add an __init__.py: {missing}"
    )


def test_smpl_assets_are_excluded_from_built_distributions():
    """SMPL/SMPL-H assets must not ship under the Apache-2.0 package."""

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    excluded = pyproject["tool"]["setuptools"]["exclude-package-data"]["protomotions"]
    for pattern in (
        "data/assets/mesh/smpl/**/*",
        "data/assets/mjcf/smpl*.xml",
        "data/assets/usd/smpl*.usda",
    ):
        assert pattern in excluded, f"SMPL carve-out missing: {pattern}"


def test_core_dependency_bounds_preserve_preconfigured_environments():
    """Core bounds must not replace preselected NumPy or Torch builds."""

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]
    numpy_pin = next(dep for dep in dependencies if dep.startswith("numpy"))
    torch_pin = next(dep for dep in dependencies if dep.startswith("torch"))
    assert "<" not in numpy_pin
    assert torch_pin == "torch>=2.2"
