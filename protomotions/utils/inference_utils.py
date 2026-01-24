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
"""Inference-specific utility functions.

This module contains utilities specific to inference mode.

For general config override utilities, see protomotions.utils.config_utils
"""

import logging
from typing import Any
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

    # Apply backward compatibility fixes for old checkpoints
    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

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


def apply_backward_compatibility_fixes(
    robot_config: Any,
    simulator_config: SimulatorConfig = None,
    env_config: EnvConfig = None,
) -> None:
    """
    Apply backward compatibility fixes to loaded configs from older checkpoints.

    This function patches configs that are missing fields added in newer versions.
    Each fix should be independent and check if the field is missing before applying.

    For large refactors, it's recommended to use ``train_agent.py --create-config-only``
    to re-create configs and then use the old weights checkpoint with ``--checkpoint``.
    But this function can be handy to fix small changes in config system.

    Args:
        robot_config: Robot configuration that may be missing newer fields
        simulator_config: Optional simulator configuration for migration fixes
        env_config: Optional environment configuration for migration fixes
    """

    # Future fixes can be added here as independent blocks:
    # Fix 1: Example for future field
    # if hasattr(robot_config, 'some_config') and not hasattr(robot_config.some_config, 'new_field'):
    #     log.warning("Detected old checkpoint missing new_field...")
    #     # Apply fix
    #     pass
