# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metatron v1 empty-scene playback and interactive goals, cycled with X.

The saved configuration publishes its scene through ``SceneBoxesControl`` (empty
in the simulator, the host's at deployment), so nothing here touches the scene.
"""

from argparse import Namespace

from protomotions.envs.control.masked_mimic_tasks.controller import (
    GetupRecoveryConfig,
    configure_masked_mimic_task,
)
from protomotions.utils.experiment_configs import ExperimentConfigs


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
