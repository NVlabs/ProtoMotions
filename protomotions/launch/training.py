# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The training entry-point body.

``protomotions/train_agent.py`` is a thin shim that imports the simulator
backend and then calls :func:`run` here, so importing the shim stays free of
torch, wandb and the backends.
"""

from __future__ import annotations

import os

os.environ["WANDB_DISABLE_SENTRY"] = "true"  # Must be first environment variable
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_DISABLE_CODE"] = "true"

import faulthandler
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Sequence

import torch
import wandb
from lightning.pytorch.loggers import WandbLogger

from protomotions.utils.config_builder import build_configs, load_experiment_module
from protomotions.utils.config_utils import parse_cli_overrides
from protomotions.utils.experiment_configs import (
    BUNDLE_FILENAME,
    LAUNCHER_RUN_KEYS,
    ExperimentConfigs,
    is_tracking_metadata_key,
    load_experiment_configs,
    save_experiment_configs,
)

faulthandler.enable()

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

log = logging.getLogger(__name__)


def detect_checkpoint_mode(args, save_dir):
    """Detect checkpoint mode: resume, warm start, or fresh.

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

            # Update args with checkpoint config. --kit-thread-count is a launch-time
            # workaround for a startup crash, not a training setting, and adding it to
            # the resume command is how a run that starts aborting gets fixed; so a
            # value given on this command line wins over the saved one, which for any
            # run started without the flag is None.
            cli_kit_thread_count = getattr(args, "kit_thread_count", None)
            for key, value in checkpoint_config.items():
                if key != "wandb_id":
                    setattr(args, key, value)
            if cli_kit_thread_count is not None:
                saved = checkpoint_config.get("kit_thread_count")
                if saved != cli_kit_thread_count:
                    log.info(
                        f"RESUME: using --kit-thread-count {cli_kit_thread_count} from "
                        f"the command line instead of the saved value {saved}"
                    )
                args.kit_thread_count = cli_kit_thread_count
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


def save_configs(
    save_dir,
    args,
    configs: ExperimentConfigs,
    experiment_source_path,
    bundle_filename=BUNDLE_FILENAME,
    portable=False,
):
    """Save one resolved-config bundle plus CLI arguments and experiment source.

    Writes config.yaml (CLI args + wandb_id), the ``resolved_configs.json`` bundle
    (through :func:`save_experiment_configs`, a tagged tree rebuilt through the real
    constructors) and a copy of the experiment source.

    ``portable`` writes config.yaml without the keys a shipped (redistributed)
    directory must not carry: tracking metadata (:func:`is_tracking_metadata_key`)
    and a run's launcher arguments (:data:`LAUNCHER_RUN_KEYS`). Training runs leave
    it ``False`` and keep the full form.
    """
    save_dir = Path(save_dir)
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

    if portable:
        checkpoint_config = {
            key: value
            for key, value in checkpoint_config.items()
            if not (is_tracking_metadata_key(key) or key in LAUNCHER_RUN_KEYS)
        }

    # Save CLI args + wandb_id
    log.info(f"Saving config file to {save_dir}")
    checkpoint_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_config_path, "w") as file:
        json.dump(checkpoint_config, file, indent=2)

    # Save the resolved bundle as a validated JSON tree (reproducibility).
    resolved_configs_path = save_experiment_configs(
        configs, save_dir, filename=bundle_filename
    )
    log.info(f"Saving resolved configs to {resolved_configs_path}")

    # Copy experiment Python file just for human reference, not used by code.
    experiment_copy_path = save_dir / "experiment_config.py"
    log.info(f"Copying experiment file to {experiment_copy_path}")
    shutil.copy(experiment_source_path, experiment_copy_path)


def try_log_hyperparams_to_wandb(fabric, configs: ExperimentConfigs, fabric_config):
    """Try to log hyperparameters to wandb (non-critical)."""
    for logger in fabric.loggers:
        if isinstance(logger, WandbLogger):
            try:
                from protomotions.utils.config_schema import to_tree

                hyper_params = {
                    root: to_tree(getattr(configs, root), where=root)
                    for root in ExperimentConfigs.ROOTS
                }
                hyper_params["fabric"] = fabric_config.as_loggable_dict()

                log.info("Preparing configs for wandb logging...")
                logger.log_hyperparams(hyper_params)
                log.info("Successfully logged hyperparams to wandb")
            except Exception as e:
                log.warning(f"Could not log hyperparams to wandb (non-critical): {e}")


def build_wandb_logger_config(args, save_dir, wandb_id):
    """Build the WandB logger configuration from CLI arguments."""
    return {
        "_target_": "lightning.pytorch.loggers.WandbLogger",
        "name": args.experiment_name,
        "save_dir": save_dir,
        "project": args.wandb_project,
        "tags": None,
        "group": None,
        "id": wandb_id,
        "entity": None,
        "resume": "allow",
    }


def load_motion_shard_cycle(checkpoint_path: Path) -> int:
    """Read the required packaged-motion shard cycle from a training checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    motion_shard_cycle = state_dict["motion_shard_cycle"]
    if type(motion_shard_cycle) is not int:
        raise TypeError("motion_shard_cycle must be a plain Python int")
    return motion_shard_cycle


def resolve_resume_motion_shard_cycle(
    fabric, checkpoint_path: Path, motion_lib_config
) -> int:
    """Restore a switching run's shard cycle before constructing components."""
    from protomotions.agents.utils.distributed import raise_if_any_rank_failed
    from protomotions.components.motion_lib import MotionFileSwitchMode

    if motion_lib_config.motion_file_switch_mode is MotionFileSwitchMode.FIXED:
        return 0

    motion_shard_cycle = 0
    load_error = None
    if fabric.global_rank == 0:
        try:
            motion_shard_cycle = load_motion_shard_cycle(checkpoint_path)
        except Exception as error:
            load_error = error
    raise_if_any_rank_failed(
        load_error, "Motion shard cycle checkpoint load", fabric.device
    )
    return fabric.broadcast(motion_shard_cycle)


def save_training_configs(
    save_dir,
    args,
    configs: ExperimentConfigs,
    experiment_source_path,
):
    """Save the one (training) bundle and dry-run the inference derivation.

    Inference derives its view from this bundle at load, so there is no second
    file. The dry-run runs the derivation the way inference will (defaults, no
    overrides, the recipe's hook from the experiment_config.py just copied
    beside the checkpoint) and discards it, so a recipe hook that breaks fails
    here, during training, not at inference.
    """
    save_configs(
        save_dir,
        args,
        configs,
        experiment_source_path=experiment_source_path,
        bundle_filename=BUNDLE_FILENAME,
    )

    from protomotions.utils.inference_utils import (
        derive_inference_configs,
        recipe_hooks,
    )

    derive_inference_configs(configs, args, {}, recipe_hooks(save_dir, None))


def _handle_create_config_only(
    args,
    save_dir,
    experiment_source_path,
    configs: ExperimentConfigs,
):
    """Save the resolved config bundle without constructing an environment or agent."""
    save_training_configs(
        save_dir,
        args,
        configs,
        experiment_source_path=experiment_source_path,
    )
    log.info(f"CREATE CONFIG ONLY: Configs saved to {save_dir}")
    log.info("  - resolved_configs.json")
    log.info(
        "Exiting without training. Check model and observation compatibility before using existing weights."
    )


def run(argv: Sequence[str], app_launcher: Optional[type]) -> None:
    """Build or load configs, construct the env and agent, and train."""
    from protomotions.envs.context_check import run_config_binding_check
    from protomotions.launch.training_cli import (
        create_training_parser,
        parse_training_args,
    )

    torch.set_float32_matmul_precision("high")

    # ===================================================================
    # 1. Setup: Detect Checkpoint Mode
    # ===================================================================
    parser = create_training_parser()
    args, _unknown = parser.parse_known_args(list(argv))

    save_dir = Path("results") / args.experiment_name
    original_experiment_path = Path(args.experiment_path)

    # --create-config-only: Force fresh mode to just generate configs
    if args.create_config_only:
        log.info("CREATE CONFIG ONLY: Generating configs without training")
        mode, checkpoint_path, wandb_id = "fresh", None, None
    else:
        mode, checkpoint_path, wandb_id = detect_checkpoint_mode(args, save_dir)

    # ===================================================================
    # 2. Load or Build Configs Based on Mode
    # ===================================================================
    if mode == "resume":
        # Continuing the same run: load the frozen bundle, do not rebuild.
        bundle_path = save_dir / BUNDLE_FILENAME
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Resume requires {BUNDLE_FILENAME} but not found at {bundle_path}\n"
                "This may be an old checkpoint. Use --checkpoint flag for warm start instead."
            )
        log.info(f"Loading configs from {bundle_path}")
        configs = load_experiment_configs(bundle_path)
        configs.motion_lib.validate()

        args.checkpoint = checkpoint_path
        experiment_module = None  # Intentionally skip loading - frozen config

        if args.overrides:
            log.warning(
                "CLI overrides provided during RESUME will be IGNORED.\n"
                "Resume uses exact configs from the saved bundle.\n"
                "For a new run with modified configs, use --checkpoint for warm start instead."
            )

        # Fail now if an observation, reward, termination or evaluator metric
        # reads a context name this configuration will never produce.
        run_config_binding_check(
            configs.env, configs.robot, configs.terrain, configs.agent, log
        )
        if configs.agent is not None:
            from protomotions.agents.common.config import validate_module_order

            validate_module_order(configs.agent)
        log.info(
            "RESUME: Using exact configs from first run (no config building, no CLI overrides)"
        )
    else:
        # Warm start / fresh: build configs from the experiment file. build_configs
        # runs the binding check after the overrides are applied.
        log.info(f"{mode.upper()}: Building configs from experiment file")
        experiment_module = load_experiment_module(original_experiment_path)
        args = parse_training_args(argv, experiment_module)
        # Keep this a string: it is written verbatim into config.yaml by
        # save_configs (json.dump(vars(args))), and a Path is not JSON
        # serializable. agent.load() accepts the string path either way.
        args.checkpoint = str(checkpoint_path) if mode == "warm_start" else None
        overrides = parse_cli_overrides(args.overrides)
        configs = build_configs(experiment_module, args, overrides)

    # ===================================================================
    # 2b. Create Config Only Mode: Save configs and exit early
    # ===================================================================
    if args.create_config_only:
        _handle_create_config_only(args, save_dir, original_experiment_path, configs)
        return

    if configs.agent is None:
        raise ValueError(
            f"experiment {original_experiment_path} defines no agent_config(); "
            "training needs one"
        )

    robot_config = configs.robot
    simulator_config = configs.simulator
    motion_lib_config = configs.motion_lib
    env_config = configs.env
    agent_config = configs.agent

    # ===================================================================
    # 3. Fabric Configuration: Loggers, Callbacks, Distributed Setup
    # ===================================================================
    loggers = [
        {"_target_": "lightning.fabric.loggers.TensorBoardLogger", "root_dir": save_dir}
    ]
    if args.use_wandb:
        loggers.append(build_wandb_logger_config(args, save_dir, wandb_id))

    callbacks = []
    if args.use_slurm:
        callbacks.append(
            {
                "_target_": "protomotions.agents.callbacks.slurm_autoresume_srun.AutoResumeCallbackSrun",
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
    print(fabric_config.as_loggable_dict())
    fabric: Fabric = Fabric(**fabric_config.as_kwargs())
    fabric.launch()

    # ===================================================================
    # 4. Environment Setup: IsaacLab, Seeding
    # ===================================================================
    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(fabric.device)}
        if not args.headless:
            app_launcher_flags["visualizer"] = ["kit"]
        if fabric.world_size > 1:
            # This is needed when running with SLURM.
            # When launching multi-GPU/node jobs without SLURM, or differently, maybe this needs to be adapted accordingly.
            app_launcher_flags["distributed"] = True
            os.environ["LOCAL_RANK"] = str(fabric.local_rank)
            os.environ["RANK"] = str(fabric.global_rank)
        # MUST go through kit_args, NOT sys.argv.append(). AppLauncher appends its
        # OWN threadCount (os.cpu_count() // WORLD_SIZE) inside __init__, but only on
        # the distributed branch (app_launcher.py:1034-1064) -- so a single-GPU run
        # never sees it and a multi-GPU run silently loses our value. Kit takes the
        # LAST occurrence of a duplicated setting (verified: appending 1 then 16
        # aborts 2/2, appending 16 then 1 is clean), and _config_resolution calls
        # _resolve_device_settings BEFORE _resolve_kit_args, whose `sys.argv +=
        # self._kit_args` (app_launcher.py:1205) therefore lands last and wins.
        # getattr, not attribute access: `args` does not always come from this
        # parser (tests and other callers pass a plain namespace without the flag).
        kit_thread_count = getattr(args, "kit_thread_count", None)
        if kit_thread_count is not None:
            kit_arg = f"--/plugins/carb.tasking.plugin/threadCount={kit_thread_count}"
            existing = app_launcher_flags.get("kit_args", "")
            app_launcher_flags["kit_args"] = f"{existing} {kit_arg}".strip()

        launcher = app_launcher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = launcher.app

        # Suppress verbose PhysX/IsaacLab warnings that flood stdout.
        # These warnings (e.g., "Stiffness unsupported for articulation joints",
        # "Could not perform modify_collision_properties") are harmless but produce
        # 10K+ log lines across multi-GPU runs, causing Lustre I/O backpressure
        # and NCCL timeouts during initialization.
        # References:
        #   https://forums.developer.nvidia.com/t/isaac-sim-log-level-to-many-log-messages/308545/3
        #   https://github.com/isaac-sim/IsaacLab/issues/1691
        import omni.log

        _omni_log = omni.log.get_log()
        for _channel in [
            "omni.physx.plugin",
            "omni.physx.tensors.plugin",
            "isaaclab.sim.utils",
        ]:
            _omni_log.set_channel_enabled(
                _channel, False, omni.log.SettingBehavior.OVERRIDE
            )

    if args.seed is not None:
        from protomotions.utils.torch_utils import seeding

        rank = fabric.global_rank if fabric.global_rank is not None else 0
        fabric.seed_everything(args.seed + rank)
        seeding(args.seed + rank, torch_deterministic=args.torch_deterministic)

    motion_shard_cycle = 0
    if mode == "resume":
        motion_shard_cycle = resolve_resume_motion_shard_cycle(
            fabric, checkpoint_path, motion_lib_config
        )

    # ===================================================================
    # 5. Create Environment and Agent
    # ===================================================================
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

    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.utils.component_builder import build_env_from_configs

    # The friction conversion is written back into `configs`, so the bundle
    # saved below describes what the env was built from.
    env: BaseEnv = build_env_from_configs(
        configs,
        fabric.device,
        motion_shard_cycle=motion_shard_cycle,
        **simulator_extra_params,  # simulation_app for IsaacLab
    )
    fabric.call("on_env_init_end")

    from protomotions.agents.base_agent.agent import BaseAgent
    from protomotions.utils.hydra_replacement import get_class

    AgentClass = get_class(agent_config._target_)
    agent: BaseAgent = AgentClass(config=agent_config, env=env, fabric=fabric)

    agent.setup()
    agent.fabric.strategy.barrier()
    agent.load(args.checkpoint, load_training_state=(mode == "resume"))

    # ===================================================================
    # 6. Save Configs (First Run Only - Warm Start or Fresh)
    # ===================================================================
    is_first_run = mode in ["warm_start", "fresh"]
    if fabric.global_rank == 0 and is_first_run:
        if args.use_wandb:
            try_log_hyperparams_to_wandb(fabric, configs, fabric_config)
        save_training_configs(
            save_dir,
            args,
            configs,
            experiment_source_path=original_experiment_path,
        )

    agent.fabric.strategy.barrier()

    # Skip first policy update after resume to avoid training spike from full reset
    if mode == "resume":
        agent._skip_next_policy_update = True

    # ===================================================================
    # 7. Train
    # ===================================================================
    agent.fit()
