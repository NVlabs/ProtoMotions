# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify collision baking: naming, bake+load correctness, and speed.

Usage:
    python scripts/verify_collision_baking.py [--usd-dir <path>]

Defaults to checking examples/data/ for any .usd/.usda files.
If no USD files are found, runs naming-only tests.

Requires Isaac Sim runtime for bake/load tests (PhysxSchema needs SimulationApp).
"""

# Isaac Sim must be bootstrapped before any omni/pxr PhysX imports
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from protomotions.simulator.isaaclab.utils.collision_baking import (  # noqa: E402
    build_baked_collision_path,
    ensure_baked_collision_usd,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

APPROX = "convexDecomposition"
MAX_HULLS = 32
VERTEX_LIMIT = 64
VOXEL_RES = 100000


def test_naming():
    """Verify build_baked_collision_path produces correct names."""
    cases = [
        (
            ("armchair.usda", "convexDecomposition", 32, 64, 100000),
            "armchair.collision_cd_h32_v64_r100000.usda",
        ),
        (
            ("chair.usd", "convexHull", None, 64, None),
            "chair.collision_ch_v64.usd",
        ),
        (
            ("table.usda", "boundingCube", None, None, None),
            "table.collision_bc.usda",
        ),
        (
            ("sphere.usd", "boundingSphere", None, None, None),
            "sphere.collision_bs.usd",
        ),
        (
            ("dir/mesh.usda", "convexDecomposition", 10, None, 200000),
            "dir/mesh.collision_cd_h10_r200000.usda",
        ),
    ]
    passed = 0
    for (args, expected), i in zip(cases, range(len(cases))):
        result = build_baked_collision_path(*args)
        if str(result) == expected:
            passed += 1
            log.info("  naming test %d: PASS (%s)", i + 1, result)
        else:
            log.error(
                "  naming test %d: FAIL — expected %s, got %s", i + 1, expected, result
            )
    log.info("Naming tests: %d/%d passed", passed, len(cases))
    return passed == len(cases)


def test_bake_and_load(usd_path: Path):
    """Bake a USD, then verify reload is instant."""
    baked = build_baked_collision_path(
        usd_path, APPROX, MAX_HULLS, VERTEX_LIMIT, VOXEL_RES
    )
    # Clean up any prior baked file
    if baked.exists():
        baked.unlink()
        log.info("Removed stale baked file: %s", baked)

    # First call: bake (opens USD, applies APIs, exports)
    t0 = time.perf_counter()
    result1 = ensure_baked_collision_usd(
        usd_path, APPROX, MAX_HULLS, VERTEX_LIMIT, VOXEL_RES
    )
    t_bake = time.perf_counter() - t0
    assert result1 == baked, f"Expected {baked}, got {result1}"
    assert baked.exists(), f"Baked file not created: {baked}"
    log.info("  Bake time:  %.4f s — file: %s", t_bake, baked.name)

    # Second call: should find existing file (just a Path.exists() check)
    t0 = time.perf_counter()
    result2 = ensure_baked_collision_usd(
        usd_path, APPROX, MAX_HULLS, VERTEX_LIMIT, VOXEL_RES
    )
    t_load = time.perf_counter() - t0
    assert result2 == baked
    log.info("  Load time:  %.6f s (cached path lookup)", t_load)

    speedup = t_bake / t_load if t_load > 0 else float("inf")
    log.info("  Speedup:    %.0fx faster on reload", speedup)

    # Verify the baked file has collision APIs
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(baked))
    mesh_count = 0
    col_count = 0
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            mesh_count += 1
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                col_count += 1
    log.info(
        "  Verification: %d/%d mesh prims have MeshCollisionAPI",
        col_count,
        mesh_count,
    )
    assert (
        col_count == mesh_count
    ), f"Not all meshes have collision API: {col_count}/{mesh_count}"

    # Clean up
    baked.unlink()
    log.info("  Cleaned up: %s", baked.name)
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify collision baking")
    parser.add_argument(
        "--usd-dir",
        type=Path,
        default=Path("examples/data"),
        help="Directory with USD files to test baking on",
    )
    args = parser.parse_args()

    all_ok = True
    log.info("=== Naming tests ===")
    all_ok &= test_naming()

    # Find USD files for bake/load test
    usd_files = []
    if args.usd_dir.exists():
        usd_files = sorted(args.usd_dir.rglob("*.usda")) + sorted(
            args.usd_dir.rglob("*.usd")
        )

    if usd_files:
        test_files = usd_files[:3]
        log.info("\n=== Bake/load tests (%d files) ===", len(test_files))
        for f in test_files:
            log.info("Testing: %s", f)
            try:
                all_ok &= test_bake_and_load(f)
            except Exception as e:
                log.error("  FAIL: %s", e)
                all_ok = False
    else:
        log.info(
            "\nNo USD files found in %s — skipping bake/load tests. "
            "Run with --usd-dir pointing to scene object USDs.",
            args.usd_dir,
        )

    simulation_app.close()

    if all_ok:
        log.info("\nAll tests passed.")
    else:
        log.error("\nSome tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
