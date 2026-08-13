# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility for conditionally importing simulator modules before torch.

IsaacGym and IsaacLab have a strict requirement that they must be imported
before torch. This module provides a utility to handle that import order correctly.
"""


_MINIMUM_ISAACLAB_VERSION = "12.0.0"
_ISAACLAB_PIN = "4ecd0b036da19ff6ad2bb4d621f886b63e9f6db8"


def _set_openblas_single_thread() -> None:
    """Prevent OpenBLAS thread shutdown crashes when Isaac Sim's Kit kernel forks.

    scipy can bundle OpenBLAS whose thread shutdown handler segfaults during
    fork(). Single-thread mode avoids that failure mode.
    """
    import os

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _validate_isaaclab_version() -> None:
    """Fail early when the installed IsaacLab predates the supported API."""
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import InvalidVersion, Version

    try:
        installed_version = version("isaaclab")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "IsaacLab is not installed. ProtoMotions requires "
            f"isaaclab>={_MINIMUM_ISAACLAB_VERSION} from commit {_ISAACLAB_PIN}."
        ) from exc

    try:
        is_supported = Version(installed_version) >= Version(
            _MINIMUM_ISAACLAB_VERSION
        )
    except InvalidVersion as exc:
        raise RuntimeError(
            f"Cannot validate unsupported IsaacLab version {installed_version!r}."
        ) from exc

    if not is_supported:
        raise RuntimeError(
            f"ProtoMotions requires isaaclab>={_MINIMUM_ISAACLAB_VERSION} "
            f"from commit {_ISAACLAB_PIN}; found {installed_version}."
        )


def import_simulator_before_torch(simulator_name):
    """
    Conditionally import isaacgym or isaaclab based on the simulator name.

    This must be called before any imports that might bring in torch (directly or transitively).
    Typically called right after parsing arguments.

    Args:
        simulator_name: Name of the simulator ('isaacgym', 'isaaclab', 'newton', 'genesis', etc.)

    Returns:
        AppLauncher class if simulator is 'isaaclab', None otherwise

    Example:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--simulator", type=str, required=True)
        args = parser.parse_args()

        from protomotions.utils.simulator_imports import import_simulator_before_torch
        AppLauncher = import_simulator_before_torch(args.simulator)

        # Now safe to import torch
        import torch
    """
    if simulator_name == "isaacgym":
        import isaacgym  # noqa: F401

        return None
    elif simulator_name == "isaaclab":
        _set_openblas_single_thread()
        _validate_isaaclab_version()
        # Import isaaclab base module to ensure it's loaded before torch
        from isaaclab.app import AppLauncher

        return AppLauncher
    else:
        return None
