# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The training command line, declared once.

The train shim, :func:`protomotions.utils.config_builder.build_experiment` and
both cluster launchers parse against the flags added here, so a training flag
exists in exactly one place.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from protomotions.utils.cli_utils import parse_bool


def positive_int(value):
    """Parse a strictly positive integer CLI value."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """Add every training flag to ``parser``."""
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
        help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton')",
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
        "--wandb-project",
        type=str,
        default="physical_animation",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--kit-thread-count",
        type=int,
        default=None,
        help=(
            "Pin Kit's task scheduler to this many threads (IsaacLab only). Default "
            "None passes nothing: Kit keeps its own thread count on a single GPU, and "
            "AppLauncher sets os.cpu_count() // WORLD_SIZE on multi-GPU runs. Set to 1 "
            "to work around the IsaacLab 3 heap corruption during the first "
            "sim.reset(), which scales with the number of distinct spawn prototypes: "
            "measured 7/10 glibc aborts at the default vs 0/10 at 1, on 512 "
            "prototypes / 8000 envs. The cost is small: ~68 s/epoch with the pin vs "
            "~70 s/epoch without on 4xA100 with 4096 envs and 4500 furniture scenes. "
            "On resume, a value given on the command line overrides the saved one."
        ),
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
        "--headless",
        nargs="?",
        const=True,
        default=True,
        type=parse_bool,
        help="Run simulation in headless mode",
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
    training_limit_group = parser.add_mutually_exclusive_group()
    training_limit_group.add_argument(
        "--training-max-steps",
        type=int,
        default=10000000000000,
        help="Maximum number of training steps. Default to 'loads of steps'.",
    )
    training_limit_group.add_argument(
        "--training-max-iterations",
        type=positive_int,
        default=None,
        help=(
            "Maximum number of complete training iterations. Each iteration "
            "collects one rollout and performs its optimization updates."
        ),
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


def create_training_parser() -> argparse.ArgumentParser:
    """Build the training parser with every flag declared."""
    parser = argparse.ArgumentParser(
        description="Train reinforcement learning agent with configurable parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_training_arguments(parser)
    return parser


def experiment_path_from_argv(argv: Sequence[str]) -> Path:
    """Pre-parse ``--experiment-path`` to find the experiment module to load."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment-path", required=True)
    args, _ = parser.parse_known_args(list(argv))
    return Path(args.experiment_path)


def parse_training_args(argv: Sequence[str], experiment) -> argparse.Namespace:
    """Parse the full training command line, including experiment-added flags."""
    parser = create_training_parser()
    additional = getattr(experiment, "additional_experiment_arguments", None)
    if additional is not None:
        additional(parser)
    return parser.parse_args(list(argv))


def read_launch_facts(argv: Sequence[str]) -> argparse.Namespace:
    """Read the few training values a cluster launcher needs to submit a job.

    A launcher forwards the whole training command line to ``train_agent.py``
    verbatim, so it declares no training flag of its own. It still needs a
    handful of values to shape the submission -- ``simulator`` (container image
    and python), ``experiment_name`` (job name and checkpoint path), ``ngpu`` /
    ``nodes`` (the node shape), and ``use_wandb`` (whether to mount credentials)
    -- which this reads off the forwarded tail against the one training parser.
    ``parse_known_args`` ignores any experiment-contributed flag, so the
    experiment module need not load on the submit host; the standard training
    flags the parser requires must be present, which fails a malformed command
    on the submit host rather than inside the job.
    """
    parser = create_training_parser()
    facts, _ = parser.parse_known_args(list(argv))
    return facts


def apply_launch_facts(args, argv: Sequence[str], extra: Sequence[str] = ()) -> None:
    """Reflect the submission facts read from the forwarded tail onto ``args``.

    A cluster launcher parses only its own flags and forwards the training tail;
    this reads the handful of values it shapes the submission by -- simulator,
    experiment name, ngpu, nodes, use_wandb, plus any launcher-specific ``extra``
    (Lepton also reads checkpoint, for a relative-path warning) -- off that tail
    with :func:`read_launch_facts` and sets them on ``args``. The field list lives
    here, once, so no training flag is declared in a launcher.
    """
    facts = read_launch_facts(argv)
    for name in ("simulator", "experiment_name", "ngpu", "nodes", "use_wandb", *extra):
        setattr(args, name, getattr(facts, name))
