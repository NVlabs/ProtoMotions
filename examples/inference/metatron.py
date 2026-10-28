# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metatron v1 empty-scene playback and interactive goals, cycled with X."""

from argparse import Namespace

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext

from protomotions.envs.control.masked_mimic_tasks.controller import (
    GetupRecoveryConfig,
    configure_masked_mimic_task,
)
from protomotions.utils.experiment_configs import ExperimentConfigs


def empty_scene_corners(dof_pos: Tensor = EnvContext.current.dof_pos) -> Tensor:
    """Two absent box tokens, each with eight XYZ corners in metres."""
    return dof_pos.new_zeros(dof_pos.shape[0], 2, 24)


def empty_scene_mask(dof_pos: Tensor = EnvContext.current.dof_pos) -> Tensor:
    """Neither box is visible; the transformer excludes both tokens."""
    return dof_pos.new_zeros((dof_pos.shape[0], 2), dtype=torch.bool)


def apply_inference_overrides(configs: ExperimentConfigs, args: Namespace) -> None:
    """Install interactive goals without changing the trained model or sensors."""
    if args.full_eval:
        raise ValueError("Interactive goals do not define a motion-tracking evaluation")
    configure_masked_mimic_task(
        configs.env,
        configs.robot,
        "idle",
        getup=GetupRecoveryConfig(),
    )
    task = configs.env.control_components["masked_mimic"]
    task.command_source = "script" if args.headless else "keyboard"
    task.reach_body_name = "right_rubber_hand"
    task.speed = 0.0
