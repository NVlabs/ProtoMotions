# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The inference entry-point body.

``protomotions/inference_agent.py`` is a thin shim that imports the simulator
backend and then calls :func:`run` here.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

from protomotions.utils.experiment_configs import (
    load_experiment_configs_from_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

log = logging.getLogger(__name__)


def create_inference_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for inference."""
    parser = argparse.ArgumentParser(
        description="Test trained reinforcement learning agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file to test"
    )
    # Optional arguments
    parser.add_argument(
        "--full-eval",
        action="store_true",
        default=False,
        help="Run full evaluation instead of simple inference",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton')",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments to run"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=False,
        default=None,
        help="Path to motion file for inference. If not provided, will use the motion file from the checkpoint.",
    )
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., env.max_episode_length=5000 simulator.headless=True)",
    )
    parser.add_argument(
        "--inference-config",
        default=None,
        help="Python file with apply_inference_overrides(); applied to the derived inference view; a hook that undoes a --overrides value is refused",
    )

    return parser


def scene_file_override(scene_lib_config, scenes_file_arg) -> None:
    """Point the scene library at ``--scenes-file`` and re-derive its asset root.

    A named helper, not an override entry, because it derives a second field:
    "none"/"null" disables scenes, and a path's asset root is recomputed (the
    experiment's may point at a different machine, e.g. lustre vs local).
    """
    if scenes_file_arg is None:
        return
    scenes_file = (
        None if scenes_file_arg.lower() in ("none", "null") else scenes_file_arg
    )
    log.info(f"CLI override: scenes_file = {scenes_file}")
    scene_lib_config.scene_file = scenes_file
    if scenes_file is None:
        scene_lib_config.asset_root = None
    elif scene_lib_config.asset_root is not None:
        scene_lib_config.asset_root = os.path.dirname(
            os.path.dirname(os.path.abspath(scenes_file))
        )


def run(argv: Sequence[str], app_launcher: Optional[type]) -> None:
    """Derive the inference view from the training bundle, build env+agent, evaluate."""
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.fabric_config import FabricConfig
    from lightning.fabric import Fabric

    parser = create_inference_parser()
    args = parser.parse_args(list(argv))

    checkpoint = Path(args.checkpoint)

    # Inference derives its view from the one (training) bundle at load; there
    # is no second file.
    log.info(f"Loading resolved configs from the bundle beside {checkpoint}")
    training = load_experiment_configs_from_checkpoint(checkpoint)

    # Switch the simulator before deriving, so the derivation and its binding
    # check run against the backend inference will use.
    current_simulator = training.simulator._target_.split(".")[-3]
    if args.simulator != current_simulator:
        log.info(
            f"Switching simulator from '{current_simulator}' (training) to "
            f"'{args.simulator}' (inference)"
        )
        from protomotions.simulator.factory import update_simulator_config_for_test

        training.simulator = update_simulator_config_for_test(
            current_simulator_config=training.simulator,
            new_simulator=args.simulator,
            robot_config=training.robot,
        )

    # The scalar inference flags are override entries through the same funnel;
    # the parsed --overrides win on a duplicate key.
    from protomotions.utils.config_utils import parse_cli_overrides

    overrides = {
        "simulator.num_envs": args.num_envs,
        "simulator.headless": args.headless,
    }
    if args.motion_file is not None:
        overrides["motion_lib.motion_file"] = args.motion_file
        overrides["motion_lib.motion_file_shard_indices"] = None
    if args.overrides:
        overrides.update(parse_cli_overrides(args.overrides))

    from protomotions.utils.inference_utils import (
        derive_inference_configs,
        recipe_hooks,
    )

    view = derive_inference_configs(
        training,
        args,
        overrides,
        recipe_hooks(checkpoint.parent, args.inference_config),
    )
    scene_file_override(view.scene_lib, args.scenes_file)

    agent_config = view.agent

    # Create fabric config for inference (simplified)
    # MuJoCo is CPU-only, so force CPU accelerator
    accelerator = "cpu" if args.simulator == "mujoco" else "gpu"
    fabric_config = FabricConfig(
        accelerator=accelerator,
        devices=1,
        num_nodes=1,
        loggers=[],  # No loggers needed for inference
        callbacks=[],  # No callbacks needed for inference
    )
    fabric: Fabric = Fabric(**fabric_config.as_kwargs())
    fabric.launch()

    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        if not args.headless:
            app_launcher_flags["visualizer"] = ["kit"]
        launcher = app_launcher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = launcher.app

    # Create the env over its components (the simulator initialises inside).
    # The friction conversion is written back into the view, which nothing saves.
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.utils.component_builder import build_env_from_configs

    env: BaseEnv = build_env_from_configs(
        view,
        fabric.device,
        **simulator_extra_params,  # simulation_app for IsaacLab
    )

    # Determine root_dir for agent based on checkpoint path
    agent_kwargs = {}
    checkpoint_path = Path(args.checkpoint)
    agent_kwargs["root_dir"] = checkpoint_path.parent

    # Create agent
    from protomotions.agents.base_agent.agent import BaseAgent

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(
        config=agent_config, env=env, fabric=fabric, **agent_kwargs
    )

    agent.setup(training=False)
    agent.load(
        args.checkpoint, load_env=False, load_training_state=False, for_inference=True
    )
    headless = getattr(env.simulator, "headless", True)
    ui = getattr(env.simulator, "user_interface", None)
    if not headless and ui is not None:
        help_text = ui.help_text()
        if help_text:
            log.info("Viewer keybinds:\n%s", help_text)

    try:
        if args.full_eval:
            agent.evaluator.eval_count = 0
            result = agent.evaluator.evaluate()

            # Print evaluation metrics
            print("\n" + "=" * 60)
            print("EVALUATION RESULTS")
            print("=" * 60)
            for key, value in sorted(result.logs.items()):
                print(f"  {key}: {value:.6f}")
            print(f"  Items Evaluated: {result.num_items}")
            print("=" * 60)
            if result.score is not None:
                print(f"  Overall Score: {result.score:.6f}")
            print("=" * 60 + "\n")
        else:
            agent.evaluator.simple_test_policy()
    finally:
        # Ensure simulator viewer is properly closed (prevents hangs)
        if hasattr(env.simulator, "shutdown"):
            env.simulator.shutdown()
