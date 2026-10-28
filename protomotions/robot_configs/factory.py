# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from protomotions.robot_configs.base import RobotConfig

# Modules under robot_configs/ that are infrastructure, not robots.
_NON_ROBOT_MODULES = frozenset({"base", "factory", "motor_classes", "__init__"})


def registered_robot_names() -> Tuple[str, ...]:
    """The robots the registry knows about.

    The registry is the directory: every ``protomotions/robot_configs/*.py``
    module stem except the infrastructure modules is a robot. Returned sorted,
    without importing any robot module.
    """
    directory = Path(__file__).parent
    stems = {path.stem for path in directory.glob("*.py")} - _NON_ROBOT_MODULES
    return tuple(sorted(stems))


def robot_config(robot_name: str, **updates) -> "RobotConfig":
    """Build a robot configuration by its module stem.

    A robot is one file under ``protomotions/robot_configs/`` that defines
    exactly one class whose direct base is :class:`RobotConfig`. The name is
    checked against the registry before any import, then the module is imported
    and its one robot class is instantiated.

    Args:
        robot_name: Module stem of the robot (e.g. ``"g1"``, ``"smpl"``).
        **updates: Optional field updates applied through ``update_fields``.

    Returns:
        RobotConfig: Robot configuration object.

    Raises:
        ValueError: If ``robot_name`` is not a registered robot.
        TypeError: If the module does not define exactly one class whose direct
            base is RobotConfig.
    """
    # Imported lazily so registered_robot_names() (and importers that only need
    # the registry) do not pull in RobotConfig and torch.
    from protomotions.robot_configs.base import RobotConfig

    registered = registered_robot_names()
    if robot_name not in registered:
        raise ValueError(
            f"unknown robot {robot_name!r}; registered robots: {', '.join(registered)}"
        )

    module = importlib.import_module(f"protomotions.robot_configs.{robot_name}")
    classes = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and obj.__bases__ == (RobotConfig,)
    ]
    if len(classes) != 1:
        raise TypeError(
            f"protomotions/robot_configs/{robot_name}.py must define exactly one "
            f"class whose direct base is RobotConfig; found "
            f"{[c.__name__ for c in classes]}"
        )

    config = classes[0]()
    if updates:
        config.update_fields(**updates)
    return config
