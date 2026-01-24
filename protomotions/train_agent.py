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
"""Train reinforcement learning agents for physics-based character animation.

This is the main training script for ProtoMotions. It handles configuration loading,
distributed training setup, agent initialization, and checkpoint management.

Configuration System
--------------------

When you train a model, all configurations are automatically saved to the experiment
directory for exact reproducibility::

    results/my_experiment/
    ├── config.yaml              # CLI arguments
    ├── resolved_configs.pt      # Full config objects (pickled)
    ├── resolved_configs.yaml    # Human-readable configs
    ├── experiment_config.py     # Copy of experiment file
    └── last.ckpt               # Model checkpoint

The system saves three types of configuration files:

1. **resolved_configs.pt** (Primary): Full Python objects saved with pickle.
   Handles ALL types (Union, nested dataclasses, torch.Tensor, etc.) for guaranteed
   exact reproducibility.

2. **resolved_configs.yaml** (Human Reference): Best-effort YAML conversion for
   easy inspection and diffing.

3. **experiment_config.py** (Context): Copy of your experiment file showing original
   logic and intent.

Config Building Process
-----------------------

At first run without checkpoint:

1. configure_robot_and_simulator() - customize robot & sim
2. env_config() - build environment config
3. agent_config() - build agent config
4. Apply CLI overrides (--overrides) if provided
5. Save all to resolved_configs.pt

Important
---------
CLI overrides during training are PERMANENT! They are saved to resolved_configs.pt
and used in future resumes. For temporary overrides, use a new experiment name.

Create Config Only Mode
-----------------------
Use ``--create-config-only`` to generate config files without training. This is useful
for migrating old policy checkpoints when the config system API changes:

Generate new configs compatible with current code::

    python protomotions/train_agent.py \\
        --robot-name g1 --simulator isaacgym \\
        --experiment-path examples/experiments/mimic/mlp.py \\
        --experiment-name my_migrated_experiment \\
        --motion-file /path/to/motion.pt \\
        --num-envs 4096 --batch-size 16384 \\
        --create-config-only

Example
-------
>>> # Training with custom configuration
>>> # PYTHON_PATH protomotions/train_agent.py \\
>>> #     +exp=full_body_tracker/transformer_flat_terrain \\
>>> #     +robot=smpl \\
>>> #     +simulator=isaacgym \\
>>> #     motion_file=data/motions/amass_train.pt \\
>>> #     num_envs=2048 \\
>>> #     agent.config.batch_size=4096 \\
>>> #     +experiment_name=my_tracker
"""

import os
import json

os.environ["WANDB_DISABLE_SENTRY"] = "true"  # Must be first environment variable
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLE_CODE"] = "true"

"""
## Quick Start

When you train a model, all configurations are automatically saved to the experiment directory for exact reproducibility:

```bash
# Training (automatic config saving)
python protomotions/train_agent.py \
    --robot-name g1 \
    --simulator isaacgym \
    --experiment-path examples/experiments/mimic/mlp.py \
    --experiment-name my_experiment \
    --motion-file /path/to/motion.pt \
    --num-envs 4096 \
    --batch-size 16384

# Results in:
results/my_experiment/
├── config.yaml              # CLI arguments
├── resolved_configs.pt      # Full config objects (pickled)
├── resolved_configs.yaml    # Human-readable configs (best-effort)
├── experiment_config.py     # Copy of mlp.py
└── last.ckpt               # Model checkpoint
```
## Why This Approach?

### Problem
Config dataclasses can have complex types (Union, nested dataclasses, torch.Tensors) that JSON/YAML can't handle, plus experiments often inherit from base configs that may change over time.

### Solution
**Three files for different purposes:**

1. **`resolved_configs.pt`** (Primary)
   - Full Python objects saved with pickle
   - Handles ALL types (Union, nested, torch.Tensor, etc.)
   - Guaranteed exact reproducibility
   - Not human-readable

2. **`resolved_configs.yaml`** (Human Reference)
   - Best-effort YAML conversion
   - Easy to inspect and diff
   - May fail for complex types (non-critical)
   - Human-readable

3. **`experiment_config.py`** (Context)
   - Copy of your experiment file
   - Shows original logic and intent
   - Useful for understanding decisions

Config System, at 1st run without ckpt

Config Building (from experiment file):
1. configure_robot_and_simulator() - customize robot & sim
2. env_config() - build environment config
3. agent_config() - build agent config
4. Apply CLI overrides (--overrides) if provided
5. Save all to resolved_configs.pt

IMPORTANT: CLI overrides during training are PERMANENT!
They are saved to resolved_configs.pt and used in future resumes.
For temporary overrides, use a new experiment name.
"""


def create_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Train reinforcement learning agent with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--robot-name",
        type=str,
        required=True,
        help="Name of the robot (e.g., 'g1', 'smpl')",
    )
    parser.add_argument(
        "--simulator",
        type=str,
        required=True,
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        required=True,
        help="Number of parallel environments to run",
    )
    parser.add_argument(
        "--batch-size", type=int, required=True, help="Batch size for training"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        required=True,
        help="Path to motion file for training",
    )
    parser.add_argument(
        "--experiment-path",
        type=str,
        required=True,
        help="File path to experiment configuration (e.g., 'examples/train/mimic/mimic_mlp.py')",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Name of the experiment for logging and checkpointing",
    )

    # Optional arguments
    parser.add_argument(
        "--scenes-file", type=str, default=None, help="Path to scenes file (optional)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--use-slurm",
        action="store_true",
        default=False,
        help="Enable SLURM autoresume functionality",
    )
    parser.add_argument(
        "--ngpu", type=int, default=1, help="Number of GPUs to use for training"
    )
    parser.add_argument(
        "--nodes", type=int, default=1, help="Number of nodes for distributed training"
    )
    parser.add_argument(
        "--headless", default=True, help="Run simulation in headless mode"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--torch-deterministic",
        action="store_true",
        default=False,
        help="Enable deterministic PyTorch operations",
    )
    parser.add_argument(
        "--training-max-steps",
        type=int,
        default=10000000000000,
        help="Maximum number of training steps. Default to 'loads of steps'.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Config overrides in format key=value (e.g., env.max_episode_length=1000 simulator.num_envs=4096)",
    )
    parser.add_argument(
        "--create-config-only",
        action="store_true",
        default=False,
        help="Only create and save config files without training. "
        "Useful for migrating old policy checkpoints when config system API changes - "
        "generate new configs that are compatible with current code, then load old weights.",
    )

    return parser


# Parse arguments first (argparse is safe, doesn't import torch)
import argparse  # noqa: E402

parser = create_parser()
args, unknown_args = parser.parse_known_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
from pathlib import Path  # noqa: E402
import logging  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
import importlib.util  # noqa: E402
import shutil  # noqa: E402
import wandb  # noqa: E402
from lightning.pytorch.loggers import WandbLogger  # noqa: E402
import torch  # noqa: E402
from utils.torch_utils import seeding  # noqa: E402
from dataclasses import asdict  # noqa: E402
from protomotions.utils.config_utils import clean_dict_for_storage, make_json_serializable  # noqa: E402


log = logging.getLogger(__name__)


def detect_checkpoint_mode(args, save_dir):
    """
    Detect checkpoint mode: resume, warm start, or fresh.

    Returns:
        tuple: (mode, checkpoint_path, wandb_id)
            mode: "resume", "warm_start", or "fresh"
            checkpoint_path: Path to checkpoint or None
            wandb_id: Wandb ID for resume or None
    """
    pre_existing_checkpoint = save_dir / "last.ckpt"
    checkpoint_config_path = save_dir / "config.yaml"

    # Priority 1: Resume - continuing same run
    if pre_existing_checkpoint.exists():
        log.info(f"RESUME: Found checkpoint in save_dir: {pre_existing_checkpoint}")

        # Load wandb_id
        wandb_id = None
        if checkpoint_config_path.exists():
            log.info(f"Loading saved args from {checkpoint_config_path}")
            with open(checkpoint_config_path, "r") as file:
                checkpoint_config = json.load(file)

            # Update args with checkpoint config
            for key, value in checkpoint_config.items():
                if key != "wandb_id":
                    setattr(args, key, value)
            wandb_id = checkpoint_config.get("wandb_id", None)
        else:
            raise FileNotFoundError(
                f"Config file not found at {checkpoint_config_path}"
            )

        return "resume", pre_existing_checkpoint, wandb_id

    # Priority 2: Warm Start - new run with pretrained weights
    elif args.checkpoint is not None:
        log.info(f"WARM START: Using checkpoint for initialization: {args.checkpoint}")
        return "warm_start", Path(args.checkpoint), None

    # No checkpoint - training from scratch
    else:
        log.info("FRESH START: Training from scratch")
        return "fresh", None, None


def load_experiment_module(experiment_path):
    """
    Load the experiment module from a given path.

    Args:
        experiment_path: Path to the experiment Python file

    Returns:
        Loaded experiment module
    """
    experiment_path = Path(experiment_path)

    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    log.info(f"Loading experiment module from: {experiment_path}")

    spec = importlib.util.spec_from_file_location("experiment_module", experiment_path)
    experiment_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_module)

    return experiment_module


def save_configs(
    save_dir,
    args,
    robot_config,
    simulator_config,
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    env_config,
    agent_config,
    fabric_config,
    experiment_source_path,
    file_name="resolved_configs",
):
    """
    Save all configuration files (first run only).

    Saves:
    - config.yaml (CLI args + wandb_id)
    - resolved_configs.pt (pickled config objects)
    - resolved_configs.yaml (human-readable, best-effort)
    - experiment_config.py (copy of experiment file)
    """
    checkpoint_config_path = save_dir / "config.yaml"

    # Convert args to dict and add wandb_id
    checkpoint_config = vars(args).copy()
    checkpoint_config["wandb_id"] = None

    # Try to get wandb_id from loggers
    if args.use_wandb:
        try:
            wandb_id = wandb.run.id
            log.info(f"wandb_id found: {wandb_id}")
            checkpoint_config["wandb_id"] = wandb_id
        except Exception:
            log.warning("Could not get wandb_id")

    # Save CLI args + wandb_id
    log.info(f"Saving config file to {save_dir}")
    checkpoint_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_config_path, "w") as file:
        json.dump(checkpoint_config, file, indent=2)

    # Save pickled configs (guaranteed reproducibility)
    resolved_configs_path = (save_dir / file_name).with_suffix(".pt")
    resolved_configs = {
        "robot": robot_config,
        "simulator": simulator_config,
        "terrain": terrain_config,
        "scene_lib": scene_lib_config,
        "motion_lib": motion_lib_config,
        "env": env_config,
        "agent": agent_config,
    }
    log.info(f"Saving resolved configs (pickled) to {resolved_configs_path}")
    torch.save(resolved_configs, resolved_configs_path)

    # Save YAML configs (human-readable, best-effort)
    try:
        resolved_configs_yaml_path = (save_dir / file_name).with_suffix(".yaml")
        resolved_configs_dict = {
            "robot": clean_dict_for_storage(asdict(robot_config)),
            "simulator": clean_dict_for_storage(asdict(simulator_config)),
            "terrain": clean_dict_for_storage(asdict(terrain_config)),
            "scene_lib": clean_dict_for_storage(asdict(scene_lib_config)),
            "motion_lib": clean_dict_for_storage(asdict(motion_lib_config)),
            "env": clean_dict_for_storage(asdict(env_config)),
            "agent": clean_dict_for_storage(asdict(agent_config)),
        }
        import yaml

        log.info(f"Saving resolved configs (YAML) to {resolved_configs_yaml_path}")
        with open(resolved_configs_yaml_path, "w") as file:
            yaml.dump(
                resolved_configs_dict, file, default_flow_style=False, sort_keys=False
            )
    except Exception as e:
        log.warning(f"Could not save YAML configs (non-critical): {e}")

    # Copy experiment Python file just for human reference, not used by code.
    experiment_copy_path = save_dir / "experiment_config.py"
    log.info(f"Copying experiment file to {experiment_copy_path}")
    shutil.copy(experiment_source_path, experiment_copy_path)


def try_log_hyperparams_to_wandb(
    fabric,
    robot_config,
    simulator_config,
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    env_config,
    agent_config,
    fabric_config,
):
    """Try to log hyperparameters to wandb (non-critical)."""
    for logger in fabric.loggers:
        if isinstance(logger, WandbLogger):
            try:
                hyper_params = {
                    "robot": clean_dict_for_storage(asdict(robot_config)),
                    "simulator": clean_dict_for_storage(asdict(simulator_config)),
                    "terrain": clean_dict_for_storage(asdict(terrain_config)),
                    "scene_lib": clean_dict_for_storage(asdict(scene_lib_config)),
                    "motion_lib": clean_dict_for_storage(asdict(motion_lib_config)),
                    "env": clean_dict_for_storage(asdict(env_config)),
                    "agent": clean_dict_for_storage(asdict(agent_config)),
                    "fabric": clean_dict_for_storage(asdict(fabric_config)),
                }

                log.info("Preparing configs for wandb logging...")
                serializable_params = make_json_serializable(hyper_params)
                logger.log_hyperparams(serializable_params)
                log.info("Successfully logged hyperparams to wandb")
            except Exception as e:
                log.warning(f"Could not log hyperparams to wandb (non-critical): {e}")


def main():
    global parser, args
    torch.set_float32_matmul_precision("high")

    # ===================================================================
    # 1. Setup: Detect Checkpoint Mode
    # ===================================================================
    save_dir = Path("results") / args.experiment_name
    resolved_configs_path = save_dir / "resolved_configs.pt"
    original_experiment_path = Path(args.experiment_path)

    # --create-config-only: Force fresh mode to just generate configs
    if args.create_config_only:
        log.info("CREATE CONFIG ONLY: Generating configs without training")
        mode, checkpoint_path, wandb_id = "fresh", None, None
    else:
        mode, checkpoint_path, wandb_id = detect_checkpoint_mode(args, save_dir)

    # ===================================================================
    # 2. Load Configs Based on Mode
    # ===================================================================

    if mode == "resume":
        # ===============================================================
        # RESUME: Continuing same run - load from pickle only
        # Does NOT load experiment module or rebuild configs
        # ===============================================================
        if not resolved_configs_path.exists():
            raise FileNotFoundError(
                f"Resume requires resolved_configs.pt but not found at {resolved_configs_path}\n"
                f"This may be an old checkpoint. Use --checkpoint flag for warm start instead."
            )

        log.info(f"Loading configs from {resolved_configs_path}")
        resolved_configs = torch.load(
            resolved_configs_path, map_location="cpu", weights_only=False
        )

        robot_config = resolved_configs["robot"]
        simulator_config = resolved_configs["simulator"]
        terrain_config = resolved_configs["terrain"]
        scene_lib_config = resolved_configs["scene_lib"]
        motion_lib_config = resolved_configs["motion_lib"]
        env_config = resolved_configs["env"]
        agent_config = resolved_configs["agent"]

        args.checkpoint = checkpoint_path
        experiment_module = (
            None  # Intentionally skip loading - frozen config from pickle
        )

        # Warn if user tried to use overrides during resume
        if args.overrides:
            log.warning(
                "CLI overrides provided during RESUME will be IGNORED.\n"
                "Resume uses exact configs from resolved_configs.pt.\n"
                "For a new run with modified configs, use --checkpoint for warm start instead."
            )

        log.info(
            "RESUME: Using exact configs from first run (no config building, no CLI overrides)"
        )

    elif mode in ["warm_start", "fresh"]:
        # ===============================================================
        # WARM START / FRESH: Build configs from experiment file
        # Calls: configure_robot_and_simulator() → env_config() → agent_config()
        # ===============================================================
        log.info(f"{mode.upper()}: Building configs from experiment file")

        experiment_path = original_experiment_path
        log.info(f"Using original experiment path: {experiment_path}")
        args.checkpoint = checkpoint_path if mode == "warm_start" else None

        experiment_module = load_experiment_module(experiment_path)
        
        # Allow experiment files to add custom CLI arguments
        additional_args_fn = getattr(experiment_module, "additional_experiment_arguments", None)
        if additional_args_fn:
            additional_args_fn(parser)
        
        args = parser.parse_args()

        # Get required config functions
        terrain_config_fn = getattr(experiment_module, "terrain_config")
        scene_lib_config_fn = getattr(experiment_module, "scene_lib_config")
        motion_lib_config_fn = getattr(experiment_module, "motion_lib_config")
        env_config_fn = getattr(experiment_module, "env_config")

        # Get optional config functions
        configure_robot_and_simulator_fn = getattr(experiment_module, "configure_robot_and_simulator", None)
        agent_config_fn = getattr(experiment_module, "agent_config", None)

        from protomotions.utils.config_builder import build_standard_configs

        configs = build_standard_configs(
            args=args,
            terrain_config_fn=terrain_config_fn,
            scene_lib_config_fn=scene_lib_config_fn,
            motion_lib_config_fn=motion_lib_config_fn,
            env_config_fn=env_config_fn,
            configure_robot_and_simulator_fn=configure_robot_and_simulator_fn,
            agent_config_fn=agent_config_fn,
        )
        robot_config = configs["robot"]
        simulator_config = configs["simulator"]
        terrain_config = configs["terrain"]
        scene_lib_config = configs["scene_lib"]
        motion_lib_config = configs["motion_lib"]
        env_config = configs["env"]
        agent_config = configs["agent"]

        # Apply CLI overrides (highest priority)
        # NOTE: These overrides are saved to resolved_configs.pt and become permanent!
        # True resume will use these overridden values.
        if args.overrides:
            from protomotions.utils.config_utils import (
                apply_config_overrides,
                parse_cli_overrides,
            )

            cli_overrides = parse_cli_overrides(args.overrides)
            if cli_overrides:
                log.info(
                    f"Applying {len(cli_overrides)} CLI override(s) - these will be saved to resolved_configs.pt"
                )
                apply_config_overrides(
                    cli_overrides,
                    env_config,
                    simulator_config,
                    robot_config,
                    agent_config,
                    terrain_config=terrain_config,
                    motion_lib_config=motion_lib_config,
                    scene_lib_config=scene_lib_config,
                )

    # ===================================================================
    # 2b. Create Config Only Mode: Save configs and exit early
    # ===================================================================
    if args.create_config_only:
        _handle_create_config_only(
            args,
            save_dir,
            original_experiment_path,
            experiment_module,
            robot_config,
            simulator_config,
            terrain_config,
            scene_lib_config,
            motion_lib_config,
            env_config,
            agent_config,
        )
        return

    # ===================================================================
    # 3. Fabric Configuration: Loggers, Callbacks, Distributed Setup
    # ===================================================================
    loggers = [
        {"_target_": "lightning.fabric.loggers.TensorBoardLogger", "root_dir": save_dir}
    ]

    if args.use_wandb:
        loggers.append(
            {
                "_target_": "lightning.pytorch.loggers.WandbLogger",
                "name": args.experiment_name,
                "save_dir": save_dir,
                "project": "physical_animation",
                "tags": None,
                "group": None,
                "id": wandb_id,
                "entity": None,
                "resume": "allow",
            }
        )

    callbacks = []
    if args.use_slurm:
        callbacks.append(
            {
                "_target_": "agents.callbacks.slurm_autoresume_srun.AutoResumeCallbackSrun",
                "autoresume_after": 12600,
            }
        )

    from protomotions.utils.fabric_config import FabricConfig
    from lightning.fabric import Fabric

    fabric_config = FabricConfig(
        devices=args.ngpu,
        num_nodes=args.nodes,
        loggers=loggers,
        callbacks=callbacks,
    )
    print(asdict(fabric_config))
    fabric: Fabric = Fabric(**asdict(fabric_config))
    fabric.launch()

    # ===================================================================
    # 4. Environment Setup: IsaacLab, Seeding
    # ===================================================================
    # Setup IsaacLab simulation_app if using IsaacLab simulator
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        if fabric.world_size > 1:
            # This is needed when running with SLURM.
            # When launching multi-GPU/node jobs without SLURM, or differently, maybe this needs to be adapted accordingly.
            app_launcher_flags["distributed"] = True
            os.environ["LOCAL_RANK"] = str(fabric.local_rank)
            os.environ["RANK"] = str(fabric.global_rank)
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    if args.seed is not None:
        rank = fabric.global_rank if fabric.global_rank is not None else 0
        fabric.seed_everything(args.seed + rank)
        seeding(args.seed + rank, torch_deterministic=args.torch_deterministic)

    # ===================================================================
    # 5. Create Environment and Agent
    # ===================================================================
    # Note: Configs are already loaded/built in section 2 based on mode
    fabric.call(
        "on_app_start",
        fabric,
        {
            "fabric_config": fabric_config,
            "robot_config": robot_config,
            "simulator_config": simulator_config,
            "env_config": env_config,
            "agent_config": agent_config,
        },
    )
    fabric.call("on_env_init_start")

    # ===================================================================
    # 5a. Convert Friction for Simulator Compatibility
    # ===================================================================
    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator

    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    # ===================================================================
    # 5b. Create Components
    # ===================================================================

    from protomotions.utils.component_builder import build_all_components

    save_dir_for_weights = (
        getattr(env_config, "save_dir", None)
        if hasattr(env_config, "save_dir")
        else None
    )
    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=fabric.device,
        save_dir=save_dir_for_weights,
        **simulator_extra_params,  # simulation_app for IsaacLab
    )

    terrain = components["terrain"]
    scene_lib = components["scene_lib"]
    motion_lib = components["motion_lib"]
    simulator = components["simulator"]

    # ===================================================================
    # 5c. Create Environment (auto-initializes simulator)
    # ===================================================================
    from protomotions.envs.base_env.env import BaseEnv

    EnvClass = get_class(env_config._target_)
    env: BaseEnv = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=fabric.device,
        terrain=terrain,
        scene_lib=scene_lib,
        motion_lib=motion_lib,
        simulator=simulator,
    )
    fabric.call("on_env_init_end")

    from protomotions.agents.base_agent.agent import BaseAgent

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(config=agent_config, env=env, fabric=fabric)

    agent.setup()
    agent.fabric.strategy.barrier()
    agent.load(args.checkpoint)

    # ===================================================================
    # 6. Save Configs (First Run Only - Warm Start or Fresh)
    # ===================================================================
    # Only save configs for warm_start or fresh modes (not resume)
    # Resume already has all configs saved from the original run
    is_first_run = mode in ["warm_start", "fresh"]

    if fabric.global_rank == 0 and is_first_run:
        if args.use_wandb:
            try_log_hyperparams_to_wandb(
                fabric,
                robot_config,
                simulator_config,
                terrain_config,
                scene_lib_config,
                motion_lib_config,
                env_config,
                agent_config,
                fabric_config,
            )

        save_configs(
            save_dir,
            args,
            robot_config,
            simulator_config,
            terrain_config,
            scene_lib_config,
            motion_lib_config,
            env_config,
            agent_config,
            fabric_config,
            experiment_source_path=original_experiment_path,
            file_name="resolved_configs",
        )

        from protomotions.utils.inference_utils import apply_all_inference_overrides
        from copy import deepcopy

        # Copy all configs to avoid eval parameters leaking into the training
        robot_config_inference = deepcopy(robot_config)
        simulator_config_inference = deepcopy(simulator_config)
        terrain_config_inference = deepcopy(terrain_config)
        scene_lib_config_inference = deepcopy(scene_lib_config)
        motion_lib_config_inference = deepcopy(motion_lib_config)
        env_config_inference = deepcopy(env_config)
        agent_config_inference = deepcopy(agent_config)
        apply_all_inference_overrides(
            robot_config_inference,
            simulator_config_inference,
            env_config_inference,
            agent_config_inference,
            terrain_config_inference,
            motion_lib_config_inference,
            scene_lib_config_inference,
            experiment_module=experiment_module,
            args=args,
        )
        save_configs(
            save_dir,
            args,
            robot_config_inference,
            simulator_config_inference,
            terrain_config_inference,
            scene_lib_config_inference,
            motion_lib_config_inference,
            env_config_inference,
            agent_config_inference,
            fabric_config,
            experiment_source_path=original_experiment_path,
            file_name="resolved_configs_inference",
        )

    agent.fabric.strategy.barrier()

    # Skip first policy update after resume to avoid training spike from full reset
    if mode == "resume":
        agent._skip_next_policy_update = True

    # ===================================================================
    # 7. Train
    # ===================================================================
    agent.fit()


def _handle_create_config_only(
    args,
    save_dir,
    experiment_source_path,
    experiment_module,
    robot_config,
    simulator_config,
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    env_config,
    agent_config,
):
    """
    Handle --create-config-only mode: save configs and exit without training.

    This is useful for migrating old policy checkpoints when the config system API changes.
    Generate new configs compatible with current code, then load old weights with --checkpoint.

    Workflow:
        1. Run with --create-config-only to generate configs
        2. Run again with --checkpoint /path/to/old_weights.ckpt to train with old weights
    """
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.inference_utils import apply_all_inference_overrides
    from copy import deepcopy

    # Create minimal fabric config (no loggers/callbacks needed for config-only mode)
    fabric_config = FabricConfig(
        devices=args.ngpu,
        num_nodes=args.nodes,
        loggers=[],
        callbacks=[],
    )

    # Save training configs
    save_configs(
        save_dir,
        args,
        robot_config,
        simulator_config,
        terrain_config,
        scene_lib_config,
        motion_lib_config,
        env_config,
        agent_config,
        fabric_config,
        experiment_source_path=experiment_source_path,
        file_name="resolved_configs",
    )

    # Save inference configs
    robot_config_inference = deepcopy(robot_config)
    simulator_config_inference = deepcopy(simulator_config)
    terrain_config_inference = deepcopy(terrain_config)
    scene_lib_config_inference = deepcopy(scene_lib_config)
    motion_lib_config_inference = deepcopy(motion_lib_config)
    env_config_inference = deepcopy(env_config)
    agent_config_inference = deepcopy(agent_config)
    apply_all_inference_overrides(
        robot_config_inference,
        simulator_config_inference,
        env_config_inference,
        agent_config_inference,
        terrain_config_inference,
        motion_lib_config_inference,
        scene_lib_config_inference,
        experiment_module=experiment_module,
        args=args,
    )
    save_configs(
        save_dir,
        args,
        robot_config_inference,
        simulator_config_inference,
        terrain_config_inference,
        scene_lib_config_inference,
        motion_lib_config_inference,
        env_config_inference,
        agent_config_inference,
        fabric_config,
        experiment_source_path=experiment_source_path,
        file_name="resolved_configs_inference",
    )

    log.info(f"CREATE CONFIG ONLY: Configs saved to {save_dir}")
    log.info("  - resolved_configs.pt / .yaml (training)")
    log.info("  - resolved_configs_inference.pt / .yaml (inference)")
    log.info(
        "Exiting without training. Use these configs with old policy checkpoints via --checkpoint."
    )


if __name__ == "__main__":
    main()
