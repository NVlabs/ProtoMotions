# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deriving the inference view from the training bundle.

Inference, export and ``--full-eval`` all read the one saved (training) bundle
and derive their view at load through :func:`derive_inference_configs`, in one
order: framework defaults (``apply_inference_defaults`` on each config class),
then overrides, then the recipe's and ``--inference-config``'s hooks. Nothing is
baked at train time; the save path dry-runs this derivation so a broken hook
fails during training, not at inference.
"""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Callable, List, Mapping, Optional, Sequence

from protomotions.utils.config_builder import load_experiment_module
from protomotions.utils.config_utils import apply_overrides, read_override
from protomotions.utils.experiment_configs import ExperimentConfigs

log = logging.getLogger(__name__)

# A recipe (or --inference-config) hook mutates the inference view in place.
Hook = Callable[[ExperimentConfigs, argparse.Namespace], None]


def inference_defaults(configs: ExperimentConfigs) -> None:
    """The framework's inference view, before any override or recipe hook.

    A fixed sequence, not a getattr scan: each config class says what inference
    changes about it, and the sensors switch to their clean (noise-free) read.
    """
    configs.env.apply_inference_defaults()
    if configs.env.sensors is not None:
        configs.env.sensors = configs.env.sensors.for_inference()
    configs.simulator.apply_inference_defaults()
    configs.motion_lib.apply_inference_defaults()
    if configs.agent is not None:
        configs.agent.apply_inference_defaults(configs.env)


def derive_inference_configs(
    training: ExperimentConfigs,
    args: argparse.Namespace,
    overrides: Mapping[str, object],
    hooks: Sequence[Hook],
) -> ExperimentConfigs:
    """The inference view of a training bundle: defaults, overrides, then hooks.

    Works on a deep copy, so the training bundle is never mutated. A hook that
    silently undoes an override is refused by name. The derived view is what
    inference runs, what export reads and what ``--full-eval`` evaluates.
    """
    from protomotions.envs.context_check import run_config_binding_check

    view = training.copy()
    inference_defaults(view)
    applied = apply_overrides(view, overrides)
    for hook in hooks:
        hook(view, args)
        for path, value in applied.items():
            current = read_override(view, path)
            if current != value:
                raise ValueError(
                    f"inference hook {hook.__module__}.{hook.__qualname__} undid "
                    f"override {path}={value!r}: the value is now {current!r}"
                )
    view.motion_lib.validate()
    run_config_binding_check(view.env, view.robot, view.terrain, view.agent, log)
    return view


def _check_hook_arity(hook: Hook) -> None:
    """A hook takes exactly ``(configs, args)``; the eight-parameter form is gone."""
    if len(inspect.signature(hook).parameters) != 2:
        raise TypeError(
            f"{hook.__module__}.apply_inference_overrides must take (configs, args); "
            "the eight-parameter form was retired"
        )


def recipe_hooks(
    checkpoint_dir, inference_config: Optional[Path]
) -> List[Hook]:
    """The inference hooks to run, in order: the recipe's, then ``--inference-config``.

    The recipe's hook is loaded from the ``experiment_config.py`` saved beside
    the checkpoint (so editing a recipe changes existing checkpoints' inference
    view, and a run directory without it is incomplete).
    """
    checkpoint_dir = Path(checkpoint_dir)
    experiment_path = checkpoint_dir / "experiment_config.py"
    if not experiment_path.exists():
        raise FileNotFoundError(
            f"no experiment_config.py beside the checkpoint in {checkpoint_dir}; "
            "the run directory is incomplete, regenerate it with "
            "train_agent.py --create-config-only"
        )

    hooks: List[Hook] = []
    recipe_hook = getattr(
        load_experiment_module(experiment_path), "apply_inference_overrides", None
    )
    if recipe_hook is not None:
        _check_hook_arity(recipe_hook)
        hooks.append(recipe_hook)

    if inference_config is not None:
        inference_path = Path(inference_config)
        inference_hook = getattr(
            load_experiment_module(inference_path), "apply_inference_overrides", None
        )
        if inference_hook is None:
            raise ValueError(
                f"{inference_path} must define apply_inference_overrides(configs, args)"
            )
        _check_hook_arity(inference_hook)
        hooks.append(inference_hook)

    return hooks
