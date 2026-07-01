# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-specific utility functions.

This module contains utilities specific to inference mode.

For general config override utilities, see protomotions.utils.config_utils
"""

import logging
from protomotions.envs.base_env.config import EnvConfig
from protomotions.simulator.base_simulator.config import SimulatorConfig

log = logging.getLogger(__name__)


def apply_all_inference_overrides(
    robot_config,
    simulator_config: SimulatorConfig,
    env_config: EnvConfig,
    agent_config,
    terrain_config,
    motion_lib_config,
    scene_lib_config,
    experiment_module=None,
    args=None,
) -> None:
    """
    Apply all inference overrides (standard + experiment + CLI).

    This is the main entry point for inference configuration adjustments.
    Inference uses frozen configs from resolved_configs_inference.pt plus inference-specific overrides.

    Args:
        robot_config: Robot configuration to modify
        simulator_config: Simulator configuration to modify
        env_config: Environment configuration to modify
        agent_config: Agent configuration to modify (can be None for inference)
        experiment_module: Optional experiment module for apply_inference_overrides()
        args: Optional command line arguments
    """

    # Apply experiment-specific inference overrides if available
    if experiment_module is not None and args is not None:
        apply_inference_overrides_fn = getattr(
            experiment_module, "apply_inference_overrides", None
        )
        if apply_inference_overrides_fn is not None:
            try:
                log.info(
                    "Applying experiment inference overrides from apply_inference_overrides()"
                )
                apply_inference_overrides_fn(
                    robot_config, simulator_config, env_config, agent_config, terrain_config, motion_lib_config, scene_lib_config, args
                )
            except Exception as e:
                log.warning(f"Failed to apply experiment inference overrides: {e}")
