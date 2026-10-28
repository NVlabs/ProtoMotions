# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed configuration bundle a build produces and every consumer reads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional, Tuple


if TYPE_CHECKING:
    from protomotions.agents.base_agent.config import BaseAgentConfig
    from protomotions.components.motion_lib import MotionLibConfig
    from protomotions.components.scene_lib import SceneLibConfig
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.robot_configs.base import RobotConfig
    from protomotions.simulator.base_simulator.config import SimulatorConfig


# Every consumer uses the constant so the name lives in one place. The bundle
# is a JSON tree rebuilt through the real constructors (B4); inference derives
# its view from it at load (B2), so there is no separate inference bundle.
BUNDLE_FILENAME = "resolved_configs.json"

# Two concepts share a core because they mean different things. TRACKING_METADATA_KEYS
# (via is_tracking_metadata_key) is the release-hygiene denylist: tracking metadata a
# shipped bundle must never carry ANYWHERE -- in config.yaml, resolved_configs.json or
# provenance.json -- because it is never part of what the bundle describes.
# LAUNCHER_RUN_KEYS are one run's CLI arguments (cluster sizing and training limits);
# the portable writer drops them from config.yaml, but training_max_* are legitimate
# inside resolved_configs.json's agent root (the agent's own training-limit config), so
# they are not hygiene violations and only config.yaml, where they are run arguments,
# strips them.
TRACKING_METADATA_KEYS = frozenset(
    {"use_wandb", "use_slurm", "run_id", "entity", "web_demo_pr"}
)
LAUNCHER_RUN_KEYS = frozenset(
    {
        "kit_thread_count",
        "ngpu",
        "nodes",
        "training_max_steps",
        "training_max_iterations",
    }
)


def is_tracking_metadata_key(key: str) -> bool:
    """Whether a config key is tracking metadata a shipped bundle must never carry
    (any ``wandb``-prefixed key, or one named in :data:`TRACKING_METADATA_KEYS`)."""
    return key.startswith("wandb") or key in TRACKING_METADATA_KEYS


@dataclass
class ExperimentConfigs:
    """The seven configs a build produces, read by attribute everywhere.

    One typed object replaces the seven loose positional configs that travelled
    in four different orders and the string-keyed dict that indexed them.
    Training builds it, saves it, and resume loads it back; inference, export
    and the parity harness read the same attributes.
    """

    robot: "RobotConfig"
    simulator: "SimulatorConfig"
    terrain: "TerrainConfig"
    scene_lib: "SceneLibConfig"
    motion_lib: "MotionLibConfig"
    env: "EnvConfig"
    agent: "Optional[BaseAgentConfig]"

    ROOTS: ClassVar[Tuple[str, ...]] = (
        "robot",
        "simulator",
        "terrain",
        "scene_lib",
        "motion_lib",
        "env",
        "agent",
    )

    def copy(self) -> "ExperimentConfigs":
        """A deep copy, so an inference view never mutates the training bundle."""
        return deepcopy(self)


def save_experiment_configs(
    configs: ExperimentConfigs, directory, filename: str = BUNDLE_FILENAME
) -> Path:
    """Write the bundle as a tagged JSON tree and return its path.

    Every value is resolved and tagged (``config_schema.to_tree``) with a
    ``{format_version, protomotions, configs}`` header; load rebuilds it through
    the real constructors. No pickle, no tensor-valued field travelling opaquely.
    """
    import json

    from protomotions import __version__
    from protomotions.utils.config_schema import FORMAT_VERSION, to_tree

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    tree = {
        "format_version": FORMAT_VERSION,
        "protomotions": __version__,
        "configs": {
            root: to_tree(getattr(configs, root), where=root)
            for root in ExperimentConfigs.ROOTS
        },
    }
    path.write_text(json.dumps(tree, indent=2, allow_nan=False) + "\n")
    return path


def load_experiment_configs(path) -> ExperimentConfigs:
    """Rebuild the bundle through the real constructors; refuse a stale version."""
    import json

    from protomotions.utils.config_schema import _REGEN, FORMAT_VERSION, from_tree

    path = Path(path)
    data = json.loads(path.read_text())
    version = data.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"bundle {path} has format_version {version}; this code reads "
            f"{FORMAT_VERSION}; {_REGEN}"
        )
    trees = data.get("configs")
    missing = (
        []
        if not isinstance(trees, dict)
        else [root for root in ExperimentConfigs.ROOTS if root not in trees]
    )
    if not isinstance(trees, dict) or missing:
        raise ValueError(
            f"bundle {path} is not an ExperimentConfigs bundle (missing configs "
            f"{missing or 'section'}); {_REGEN}"
        )
    # The robot is rebuilt first: a baked-FK kernel object rebuilds its tensors
    # from the robot's kinematic_info, so env and agent need the built robot.
    # ``where`` is the location only; the refusal messages prepend "bundle ", so
    # passing it here too would read "bundle bundle <path>".
    robot = from_tree(trees["robot"], None, where=f"{path}:robot")
    roots = {"robot": robot}
    for root in ExperimentConfigs.ROOTS:
        if root != "robot":
            roots[root] = from_tree(
                trees[root], None, where=f"{path}:{root}", robot=robot
            )
    return ExperimentConfigs(**roots)


def load_experiment_configs_from_checkpoint(checkpoint_path) -> ExperimentConfigs:
    """Load the one (training) bundle saved beside a checkpoint, or raise naming it.

    Inference, export and ``--full-eval`` all derive their view from this bundle
    at load, so there is no second file to prefer.
    """
    bundle_path = Path(checkpoint_path).parent / BUNDLE_FILENAME
    if not bundle_path.exists():
        raise FileNotFoundError(f"resolved configs not found: {bundle_path}")
    return load_experiment_configs(bundle_path)
