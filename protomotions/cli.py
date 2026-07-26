# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import-safe console entry points for installed ProtoMotions distributions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional, Sequence

from protomotions.assets import ASSET_ROOT_ENV, asset_path, get_asset_root


COMMAND_MODULES = {
    "train-agent": "protomotions.train_agent",
    "inference-agent": "protomotions.inference_agent",
    "train-slurm": "protomotions.train_slurm",
}


def _distribution_version() -> str:
    try:
        return version("protomotions")
    except PackageNotFoundError:
        return "source"


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _info_payload() -> dict:
    asset_root = get_asset_root()
    probes = {name: asset_path(name) for name in ("mjcf", "urdf", "usd")}
    return {
        "version": _distribution_version(),
        "package_root": str(Path(__file__).resolve().parent),
        "asset_root": str(asset_root),
        "asset_root_override": ASSET_ROOT_ENV in os.environ,
        "assets": {name: str(path) for name, path in probes.items()},
        "simulators": {
            name: _module_available(module)
            for name, module in {
                "genesis": "genesis",
                "isaacgym": "isaacgym",
                "isaaclab": "isaaclab",
                "mujoco": "mujoco",
                "newton": "newton",
            }.items()
        },
        "isaac_sim_eula_opt_in": os.environ.get(
            "OMNI_KIT_ACCEPT_EULA", ""
        ).lower()
        in {"1", "y", "yes"},
    }


def info(argv: Optional[Sequence[str]] = None) -> int:
    """Print installed distribution, resource, and simulator availability."""

    parser = argparse.ArgumentParser(
        prog="protomotions info",
        description="Inspect the installed ProtoMotions package and runtime assets.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)
    payload = _info_payload()

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"ProtoMotions {payload['version']}")
        print(f"Package root: {payload['package_root']}")
        print(f"Asset root: {payload['asset_root']}")
        print("Simulator modules:")
        for name, available in payload["simulators"].items():
            print(f"  {name}: {'available' if available else 'not installed'}")
        print(
            "Isaac Sim unattended EULA opt-in: "
            + ("set" if payload["isaac_sim_eula_opt_in"] else "not set")
        )
    return 0


def _run_module(command: str, argv: Optional[Sequence[str]] = None) -> int:
    module_name = COMMAND_MODULES[command]
    forwarded_args = list(sys.argv[1:] if argv is None else argv)
    original_argv = sys.argv
    try:
        sys.argv = [f"protomotions {command}", *forwarded_args]
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = original_argv
    return 0


def train_agent() -> int:
    return _run_module("train-agent")


def inference_agent() -> int:
    return _run_module("inference-agent")


def train_slurm() -> int:
    return _run_module("train-slurm")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch the import-safe top-level ``protomotions`` command."""

    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="protomotions",
        description="ProtoMotions command-line tools.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["info", *COMMAND_MODULES],
        help="Command to run (default: info).",
    )

    if not args or args[0] == "info":
        return info(args[1:] if args else [])
    if args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if args[0] not in COMMAND_MODULES:
        parser.error(f"unknown command: {args[0]}")
    return _run_module(args[0], args[1:])


if __name__ == "__main__":
    raise SystemExit(main())
