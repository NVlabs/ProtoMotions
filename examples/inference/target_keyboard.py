# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the single target control component from the keyboard (W/A/S/D)."""

from protomotions.envs.control.command_source import (
    KeyboardTargetCommandSourceConfig,
)
from protomotions.envs.control.target_control import (
    TargetControlConfig,
)
from protomotions.utils.experiment_configs import ExperimentConfigs


def apply_inference_overrides(configs: ExperimentConfigs, args) -> None:
    targets = [
        config
        for config in configs.env.control_components.values()
        if isinstance(config, TargetControlConfig)
    ]
    if len(targets) != 1:
        raise ValueError("Keyboard targets require exactly one TargetControlConfig")
    targets[0].command_source = KeyboardTargetCommandSourceConfig()
