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


# NOTE: no "train-slurm" entry. train_slurm.py resolves the repository to upload
# from Path(__file__).parent.parent, which is site-packages for an installed
# distribution, so exposing it here would advertise a workflow that cannot work
# off a source checkout. See the commit message for the full rationale. Source
# checkouts continue to use `python protomotions/train_slurm.py`.
COMMAND_MODULES = {
    "train-agent": "protomotions.train_agent",
    "inference-agent": "protomotions.inference_agent",
}


def _distribution_version() -> str:
    try:
        return version("protomotions")
    except PackageNotFoundError:
        return "source"


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _info_payload() -> dict:
    # `info` is the command users reach for when an install is broken, so it
    # must never raise on a missing or partial asset tree.
    try:
        asset_root: Optional[str] = str(get_asset_root())
        asset_root_error = None
    except FileNotFoundError as exc:
        asset_root = None
        asset_root_error = str(exc)

    probes = {}
    for name in ("mjcf", "urdf", "usd", "mesh"):
        try:
            probes[name] = str(asset_path(name))
        except FileNotFoundError:
            probes[name] = None

    return {
        "version": _distribution_version(),
        "package_root": str(Path(__file__).resolve().parent),
        "asset_root": asset_root,
        "asset_root_error": asset_root_error,
        "asset_root_override": ASSET_ROOT_ENV in os.environ,
        "assets": probes,
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
        if payload["asset_root"] is None:
            print(f"Asset root: UNAVAILABLE ({payload['asset_root_error']})")
        else:
            print(f"Asset root: {payload['asset_root']}")
            missing = sorted(n for n, p in payload["assets"].items() if p is None)
            if missing:
                print(
                    "  missing asset trees: "
                    + ", ".join(missing)
                    + " (use a Git LFS checkout or set "
                    + f"{ASSET_ROOT_ENV} to a complete asset tree)"
                )
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
        # argv[0] must stay a real, resolvable target. Lightning Fabric relaunches
        # DDP workers by re-executing either `sys.argv[0]` or `-m __main__.__spec__.name`,
        # so a cosmetic argv[0] such as "protomotions train-agent" makes every
        # multi-GPU run die in the child process. `alter_sys=True` makes runpy set
        # argv[0] to the module file and populate __main__.__spec__, so both of
        # Lightning's relaunch strategies resolve correctly.
        sys.argv = [module_name, *forwarded_args]
        runpy.run_module(module_name, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = original_argv
    return 0


def train_agent() -> int:
    return _run_module("train-agent")


def inference_agent() -> int:
    return _run_module("inference-agent")


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
