# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every config dataclass that reaches the saved form resolves its annotations.

The B4 saved form and the B3 override funnel resolve each field's annotation by
eval-ing it against its own module's globals -- there is no fallback table of
``TYPE_CHECKING``-only names any more. So every type a config field names must be
importable at that config's module level. This test walks every dataclass
reachable from the seven root config classes (and every robot in the registry),
following base-typed fields into every loaded concrete subclass, and asserts
``typing.get_type_hints`` succeeds, so a field type left under ``TYPE_CHECKING``
fails the suite here instead of needing a per-module rider.
"""

import importlib
import pkgutil
import typing
from dataclasses import is_dataclass

import protomotions.agents
import protomotions.components.terrains
import protomotions.envs.control
import protomotions.envs.sensors
import protomotions.simulator
from protomotions.agents.base_agent.config import BaseAgentConfig, BaseModelConfig
from protomotions.components.motion_lib import MotionLibConfig
from protomotions.components.scene_lib import SceneLibConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.envs.base_env.config import EnvConfig
from protomotions.robot_configs.base import RobotConfig
from protomotions.robot_configs.factory import registered_robot_names, robot_config
from protomotions.simulator.base_simulator.config import SimulatorConfig

# The packages that define the concrete config subclasses a recipe can place in
# the saved tree under a base-typed field: the agent/model configs, the control
# components, the sensors, the terrains and the per-backend simulator configs.
# ``__subclasses__`` only sees imported classes, so every submodule of these is
# imported before the subclass walk.
_CONFIG_PACKAGES = (
    protomotions.agents,
    protomotions.components.terrains,
    protomotions.envs.control,
    protomotions.envs.sensors,
    protomotions.simulator,
)


def _import_config_modules() -> None:
    """Import every submodule of the config packages so ``__subclasses__`` sees
    every concrete subtype.

    A backend module whose optional package is not installed in this environment
    raises ``ImportError`` and is skipped -- the recipes that use it are covered by
    the build sweeps, not here. Anything else (a ``NameError`` from a type left
    under ``TYPE_CHECKING`` at module level, say) propagates and fails the test,
    which is the point.
    """
    for package in _CONFIG_PACKAGES:
        for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            try:
                importlib.import_module(info.name)
            except ImportError:
                continue


def _all_subclasses(cls) -> set:
    found = set()
    for sub in cls.__subclasses__():
        found.add(sub)
        found |= _all_subclasses(sub)
    return found


def _reachable_config_classes(roots) -> set:
    """Every dataclass reachable from ``roots`` through its field types.

    Following ``typing.get_type_hints`` resolves each class's annotations against
    its own module globals and raises if a type does not resolve -- which is the
    assertion. ``Dict``/``List``/``Tuple``/``Optional``/``Union`` args are followed
    recursively so a config nested in a container is reached too.
    """
    seen: set = set()
    stack = list(roots)
    while stack:
        hint = stack.pop()
        stack.extend(typing.get_args(hint))
        cls = typing.get_origin(hint) or hint
        if isinstance(cls, type) and is_dataclass(cls) and cls not in seen:
            seen.add(cls)
            stack.extend(typing.get_type_hints(cls).values())
    return seen


def reachable_config_classes() -> set:
    """Every config dataclass the saved form can reach, to a fixpoint.

    Resolving each class's annotations on the way is the assertion of the test
    below; ``test_config_annotations_py38_syntax.py`` reuses the set to know which
    modules define a config dataclass.
    """
    _import_config_modules()
    roots = [
        RobotConfig,
        SimulatorConfig,
        TerrainConfig,
        SceneLibConfig,
        MotionLibConfig,
        EnvConfig,
        BaseAgentConfig,
        BaseModelConfig,
    ]
    roots += [type(robot_config(name)) for name in registered_robot_names()]

    # A base-typed field (``EnvConfig.control_components: Dict[str,
    # ControlComponentConfig]``, an agent's ``model``, the simulator and terrain
    # bases, ...) holds a concrete subclass at runtime, so the saved tree reaches
    # types the static field-type walk alone does not. Grow to a fixpoint: add every
    # loaded subclass of each reached class, re-walk so the subclasses' own fields
    # resolve, and repeat until nothing new appears.
    classes = _reachable_config_classes(roots)
    while True:
        seeds = set(classes)
        for cls in classes:
            seeds |= _all_subclasses(cls)
        grown = _reachable_config_classes(list(seeds))
        if len(grown) == len(classes):
            break
        classes = grown
    return classes


def test_every_reachable_config_class_resolves_its_annotations():
    classes = reachable_config_classes()

    # Sanity: the walk reached the broad saved-form roster (well over a hundred
    # config dataclasses across agents, env, control, simulator and robots), not
    # just a collapsed handful -- a floor well below the true count still catches
    # the walk silently shrinking.
    assert len(classes) > 90, sorted(c.__name__ for c in classes)
