# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one home of the build sequence.

``build_configs`` turns an experiment module plus parsed args into a typed
:class:`~protomotions.utils.experiment_configs.ExperimentConfigs`; tools and
tests reach it through ``build_experiment(argv)``.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

from protomotions.utils.experiment_configs import ExperimentConfigs

log = logging.getLogger(__name__)

_NO_OVERRIDES: Mapping[str, Any] = MappingProxyType({})

# robot and simulator overrides apply after the robot hook and before the other
# roots are built, so those builders see the overridden robot; the rest apply
# after the build.
EARLY_OVERRIDE_ROOTS = ("robot", "simulator")
LATE_OVERRIDE_ROOTS = ("terrain", "scene_lib", "motion_lib", "env", "agent")


def load_experiment_module(experiment_path):
    """Load the experiment module from a given path.

    Args:
        experiment_path: Path to the experiment Python file

    Returns:
        Loaded experiment module
    """
    experiment_path = Path(experiment_path)

    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    log.info(f"Loading experiment module from: {experiment_path}")

    # A recipe imports shared recipe modules (``from examples.experiments.mimic
    # import mlp``). Those must come from the tree whose ``protomotions`` is
    # running, never from whichever checkout happens to hold the recipe file:
    # a run directory saved inside another checkout would otherwise import that
    # checkout's recipes against this tree's code. The running tree's root goes
    # first; the recipe's own project root (the first ancestor holding
    # ``examples/`` or a ``pyproject.toml``) and the working directory are
    # appended after it, so they only serve what the running tree cannot: an
    # installed distribution has no ``examples/``, and a recipe folder may
    # carry helper modules of its own. Only project roots are added
    # (``examples/`` and ``examples/experiments/`` are regular packages), and
    # nothing is placed ahead of entries already on ``sys.path``.
    running_root = Path(__file__).resolve().parent.parent.parent
    recipe_root = next(
        (
            ancestor
            for ancestor in experiment_path.resolve().parents
            if (ancestor / "examples").is_dir()
            or (ancestor / "pyproject.toml").is_file()
        ),
        None,
    )
    for root in (running_root, recipe_root, Path.cwd()):
        if root is not None and str(root) not in sys.path:
            sys.path.append(str(root))

    spec = importlib.util.spec_from_file_location("experiment_module", experiment_path)
    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)

    return experiment_module


def _required(experiment, name: str, params: str):
    """Return a required build function, or raise naming the missing hook."""
    fn = getattr(experiment, name, None)
    if fn is None:
        path = getattr(experiment, "__file__", "<experiment>")
        raise AttributeError(f"experiment {path} must define {name}({params})")
    return fn


def build_configs(
    experiment,
    args,
    overrides: Mapping[str, Any] = _NO_OVERRIDES,
    *,
    after_robot_hook: Optional[Callable[[Any, Any, Any], None]] = None,
) -> ExperimentConfigs:
    """Build the full config bundle from an experiment module and parsed args.

    The order is fixed: robot, simulator, the experiment's
    ``configure_robot_and_simulator``, the caller's ``after_robot_hook`` (the
    parity harness injects contact bodies here), then terrain, scene_lib,
    motion_lib, env and agent, followed by the iteration limit, the CLI
    overrides and the validations.
    """
    from protomotions.robot_configs.factory import robot_config
    from protomotions.simulator.factory import simulator_config as simulator_config_func

    robot = robot_config(args.robot_name)
    simulator = simulator_config_func(
        args.simulator, robot, args.headless, args.num_envs, args.experiment_name
    )

    configure = getattr(experiment, "configure_robot_and_simulator", None)
    if configure is not None:
        configure(robot, simulator, args)
    if after_robot_hook is not None:
        after_robot_hook(robot, simulator, args)

    from protomotions.utils.config_utils import apply_overrides, read_override

    # The robot and simulator overrides land before the remaining configs are
    # built, so terrain, scene, motion, env and agent see the overridden robot
    # (the #171 wrong-anchor case cannot recur). The rest land after the build.
    # After each builder step the early paths are re-read and the function that
    # undid one is named.
    experiment_path = getattr(experiment, "__file__", "<experiment>")
    configs = ExperimentConfigs(
        robot=robot,
        simulator=simulator,
        terrain=None,
        scene_lib=None,
        motion_lib=None,
        env=None,
        agent=None,
    )
    overrides = dict(overrides)
    early = (
        apply_overrides(configs, overrides, EARLY_OVERRIDE_ROOTS) if overrides else {}
    )

    def _refuse_if_undone(fn_name: str) -> None:
        for path, value in early.items():
            current = read_override(configs, path)
            if current != value:
                raise ValueError(
                    f"override {path}={value!r} was undone by {fn_name}() in "
                    f"{experiment_path}: the value is now {current!r}"
                )

    configs.terrain = _required(experiment, "terrain_config", "args")(args)
    if configs.terrain is None:
        raise TypeError(
            f"terrain_config() in {experiment_path} returned None; return "
            "TerrainConfig() for flat ground"
        )
    _refuse_if_undone("terrain_config")
    configs.scene_lib = _required(experiment, "scene_lib_config", "args")(args)
    _refuse_if_undone("scene_lib_config")
    configs.motion_lib = _required(experiment, "motion_lib_config", "args")(args)
    _refuse_if_undone("motion_lib_config")
    configs.env = _required(experiment, "env_config", "robot_config, args")(
        configs.robot, args
    )
    _refuse_if_undone("env_config")

    agent_fn = getattr(experiment, "agent_config", None)
    configs.agent = (
        agent_fn(configs.robot, configs.env, args) if agent_fn is not None else None
    )
    if (
        configs.agent is not None
        and getattr(args, "training_max_iterations", None) is not None
    ):
        configs.agent.training_max_iterations = args.training_max_iterations
    _refuse_if_undone("agent_config")

    if overrides:
        apply_overrides(configs, overrides, LATE_OVERRIDE_ROOTS)

    configs.motion_lib.validate()

    from protomotions.envs.context_check import run_config_binding_check

    run_config_binding_check(
        configs.env, configs.robot, configs.terrain, configs.agent, log
    )
    if configs.agent is not None:
        from protomotions.agents.common.config import validate_module_order

        validate_module_order(configs.agent)
    return configs


def build_experiment(
    argv: Sequence[str],
    *,
    after_robot_hook: Optional[Callable[[Any, Any, Any], None]] = None,
) -> ExperimentConfigs:
    """Build a config bundle from a training command line.

    This is the one door tests, sweeps and the parity harness use: give it the
    training argv and it returns exactly what training builds.
    """
    from protomotions.launch.training_cli import (
        experiment_path_from_argv,
        parse_training_args,
    )
    from protomotions.utils.config_utils import parse_cli_overrides

    experiment = load_experiment_module(experiment_path_from_argv(argv))
    args = parse_training_args(argv, experiment)
    overrides = parse_cli_overrides(args.overrides)
    return build_configs(experiment, args, overrides, after_robot_hook=after_robot_hook)
