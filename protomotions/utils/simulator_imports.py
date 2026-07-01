# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utility for conditionally importing simulator modules before torch.

IsaacGym and IsaacLab have a strict requirement that they must be imported
before torch. This module provides a utility to handle that import order correctly.
"""


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
        # Import isaaclab base module to ensure it's loaded before torch
        from isaaclab.app import AppLauncher

        return AppLauncher
    else:
        return None
