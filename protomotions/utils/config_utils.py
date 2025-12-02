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
# =============================================================================
# General Config Override Utilities
# =============================================================================

import logging
from typing import Dict, Any, Callable

log = logging.getLogger(__name__)


def import_experiment_relative_eval_overrides(
    relative_experiment_path: str,
) -> Callable:
    """
    Dynamically import and return the apply_inference_overrides function from an experiment module.

    This utility uses Python's inspect module to determine the caller's directory and loads
    an experiment module relative to that location. This allows evaluation scripts to import
    their corresponding training experiment's eval override function without hardcoding paths.

    Args:
        relative_experiment_path: Path to the experiment module relative to the caller's directory.
                                 E.g., "mlp.py" if in the same directory, or "../other/experiment.py"

    Returns:
        The apply_inference_overrides callable from the loaded experiment module.

    Raises:
        AttributeError: If the loaded module doesn't have an apply_inference_overrides function.
        FileNotFoundError: If the experiment module file doesn't exist.
        ImportError: If the module cannot be loaded or executed.

    Example:
        # In examples/experiments/mimic/mlp_deploy.py
        apply_inference_overrides = import_experiment_relative_eval_overrides("mlp.py")
        # This loads apply_inference_overrides from examples/experiments/mimic/mlp.py
    """
    import os
    import importlib.util
    import inspect

    # Get the path of the file that called this function
    # This will be the frame outside this function's definition
    frame = inspect.stack()[1]
    caller_file_path = os.path.abspath(frame.filename)
    caller_dir = os.path.dirname(caller_file_path)

    # Construct the path to the experiment module in the same directory as the caller
    _experiment_path = os.path.join(caller_dir, relative_experiment_path)

    # Check if the file exists before attempting to load it
    if not os.path.exists(_experiment_path):
        raise FileNotFoundError(
            f"Experiment module not found: {_experiment_path}\n"
            f"Caller: {caller_file_path}\n"
            f"Relative path: {relative_experiment_path}\n"
            f"Make sure the experiment file exists and the relative path is correct."
        )

    # Load the experiment module
    spec = importlib.util.spec_from_file_location("experiment_module", _experiment_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from: {_experiment_path}")

    experiment_module = importlib.util.module_from_spec(spec)

    # Execute the module and catch any errors during execution
    try:
        spec.loader.exec_module(experiment_module)
    except Exception as e:
        raise ImportError(
            f"Failed to execute experiment module: {_experiment_path}\n"
            f"Error: {type(e).__name__}: {e}\n"
            f"Make sure the module has no syntax errors and all imports are available."
        ) from e

    # Check if the module has the apply_inference_overrides function
    if not hasattr(experiment_module, "apply_inference_overrides"):
        raise AttributeError(
            f"Module does not have 'apply_inference_overrides' function: {_experiment_path}\n"
            f"Available attributes: {[attr for attr in dir(experiment_module) if not attr.startswith('_')]}\n"
            f"Make sure the experiment module defines an apply_inference_overrides function."
        )

    return experiment_module.apply_inference_overrides


def apply_config_overrides(
    overrides: Dict[str, Any],
    env_config,
    simulator_config,
    robot_config,
    agent_config=None,
    terrain_config=None,
    motion_lib_config=None,
    scene_lib_config=None,
) -> None:
    """
    Apply configuration overrides to config objects.

    This is a general-purpose utility that works for both training and evaluation.
    Overrides are specified in dot notation: "env.field.subfield": value.
    Supports both object attribute access and dictionary key access for nested paths.
    Raises ValueError if any override fails (field not found or invalid).

    Args:
        overrides: Dictionary of overrides to apply. Format is
            {"config_type.field.subfield": value, ...}
        env_config: Environment configuration to modify in-place
        simulator_config: Simulator configuration to modify in-place
        robot_config: Robot configuration to modify in-place
        agent_config: Optional agent configuration to modify in-place
        terrain_config: Optional terrain configuration to modify in-place
        motion_lib_config: Optional motion library configuration to modify in-place
        scene_lib_config: Optional scene library configuration to modify in-place

    Supported config types:
        - 'env': Environment config
        - 'simulator': Simulator config
        - 'robot': Robot config
        - 'agent': Agent config (training only)
        - 'terrain': Terrain config
        - 'motion_lib': Motion library config
        - 'scene_lib': Scene library config

    Raises:
        ValueError: If override key is invalid or field not found (prevents typos)

    Example::

        apply_config_overrides(
            {
                "env.max_episode_length": 1000,
                "simulator.num_envs": 4096,
                "env.reward_config.pow_rew.weight": 2e-6,  # dict key access
                "terrain.horizontal_scale": 0.1,
            },
            env_config, simulator_config, robot_config,
            terrain_config=terrain_config
        )
    """
    if not overrides:
        return

    log.info(f"Applying {len(overrides)} config override(s)...")

    for key, value in overrides.items():
        # Parse the key to determine config and field path
        parts = key.split(".")

        if len(parts) < 2:
            raise ValueError(
                f"Invalid override key format: '{key}'. Expected 'config.field' or 'config.field.subfield'"
            )

        # Determine which config object to use
        config_type = parts[0]
        field_path = parts[1:]

        if config_type == "env":
            config_obj = env_config
        elif config_type == "simulator":
            config_obj = simulator_config
        elif config_type == "robot":
            config_obj = robot_config
        elif config_type == "agent":
            if agent_config is None:
                raise ValueError(
                    f"Cannot override '{key}': agent_config not provided to apply_config_overrides()\n"
                    f"Agent config overrides are only supported in training, not evaluation."
                )
            config_obj = agent_config
        elif config_type == "terrain":
            if terrain_config is None:
                raise ValueError(
                    f"Cannot override '{key}': terrain_config not provided to apply_config_overrides()"
                )
            config_obj = terrain_config
        elif config_type == "motion_lib":
            if motion_lib_config is None:
                raise ValueError(
                    f"Cannot override '{key}': motion_lib_config not provided to apply_config_overrides()"
                )
            config_obj = motion_lib_config
        elif config_type == "scene_lib":
            if scene_lib_config is None:
                raise ValueError(
                    f"Cannot override '{key}': scene_lib_config not provided to apply_config_overrides()"
                )
            config_obj = scene_lib_config
        else:
            raise ValueError(
                f"Unknown config type '{config_type}' in override key: '{key}'\n"
                f"Valid types: 'env', 'simulator', 'robot', 'agent', 'terrain', 'motion_lib', 'scene_lib'"
            )

        # Navigate to the target field (supports both object attributes and dict keys)
        target = config_obj
        for i, field in enumerate(field_path[:-1]):
            if isinstance(target, dict):
                if field not in target:
                    path_so_far = ".".join(parts[: i + 2])
                    raise ValueError(
                        f"Key '{field}' not found in dict at config path: '{key}'\n"
                        f"Failed at: '{path_so_far}'\n"
                        f"Available keys: {list(target.keys())}"
                    )
                target = target[field]
            else:
                if not hasattr(target, field):
                    path_so_far = ".".join(parts[: i + 2])
                    raise ValueError(
                        f"Field '{field}' not found in config path: '{key}'\n"
                        f"Failed at: '{path_so_far}'\n"
                        f"Check the config dataclass structure for valid field names."
                    )
                target = getattr(target, field)

        # Set the final field (supports both object attributes and dict keys)
        final_field = field_path[-1]
        allowed_field_types = [int, float, bool, str, type(None)]

        if isinstance(target, dict):
            if final_field not in target:
                raise ValueError(
                    f"Key '{final_field}' not found in dict at config path: '{key}'\n"
                    f"Available keys: {list(target.keys())}\n"
                    f"Check for typos in the override key."
                )
            old_value = target[final_field]
            field_type = type(old_value)
            if field_type not in allowed_field_types:
                raise ValueError(
                    f"Dict value '{final_field}' is of type '{field_type}' which is not allowed. "
                    f"Allowed types are: {allowed_field_types}"
                )
            target[final_field] = value
        else:
            if not hasattr(target, final_field):
                raise ValueError(
                    f"Field '{final_field}' not found in config path: '{key}'\n"
                    f"Available fields: {list(target.__dataclass_fields__.keys()) if hasattr(target, '__dataclass_fields__') else dir(target)}\n"
                    f"Check for typos in the override key."
                )
            old_value = getattr(target, final_field)
            field_type = type(old_value)
            if field_type not in allowed_field_types:
                raise ValueError(
                    f"Field '{final_field}' is of type '{field_type}' which is not allowed. "
                    f"Allowed types are: {allowed_field_types}"
                )
            setattr(target, final_field, value)

        log.info(f"  {key}: {old_value} -> {value}")


def parse_cli_overrides(override_strings: list) -> Dict[str, Any]:
    """
    Parse command-line override strings into a dictionary.

    Supports the format: "key=value" where value can be:
    - Numbers: "env.max_episode_length=1000"
    - Floats: "agent.learning_rate=1e-5"
    - Booleans: "env.enable_terrain=True"
    - Strings: "env.terrain.type=flat"
    - None: "env.early_termination=None"

    Args:
        override_strings: List of "key=value" strings

    Returns:
        Dictionary of parsed overrides

    Example:
        parse_cli_overrides(["env.max_episode_length=1000", "simulator.num_envs=4096"])
        # Returns: {"env.max_episode_length": 1000, "simulator.num_envs": 4096}
    """
    overrides = {}

    for override_str in override_strings:
        if "=" not in override_str:
            log.warning(f"Invalid override format (missing '='): {override_str}")
            continue

        key, value_str = override_str.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()

        # Parse the value
        try:
            # Try to evaluate as Python literal (handles int, float, bool, None, lists, etc.)
            import ast

            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            # If it fails, treat as string
            value = value_str

        overrides[key] = value

    return overrides
