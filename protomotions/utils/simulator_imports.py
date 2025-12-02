# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
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
