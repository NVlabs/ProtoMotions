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
from dataclasses import dataclass
from typing import Dict, Any, Type, TypeVar, Union, get_origin, get_args, get_type_hints
from enum import Enum
import torch
from omegaconf import DictConfig

T = TypeVar("T")


@dataclass
class ConfigBuilder:
    """Mixin class providing dictionary conversion functionality."""

    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create an instance from a dictionary, converting lists to tensors where appropriate.

        Args:
            config_dict: Dictionary containing configuration values.

        Returns:
            Instance of the class with values from the dictionary.
        """

        field_types = get_type_hints(cls)
        processed_dict = {}

        # Helper function for type conversion
        def convert_value(
            val_to_convert: Any, target_type: Type, current_key: str
        ) -> Any:
            if val_to_convert is None:
                return None

            origin = get_origin(target_type)
            args = get_args(target_type)

            # 1. Nested dataclass (must have from_dict)
            if hasattr(target_type, "from_dict") and isinstance(
                val_to_convert, (dict, DictConfig)
            ):
                return target_type.from_dict(val_to_convert)

            # 2. Enum
            if isinstance(target_type, type) and issubclass(target_type, Enum):
                return target_type.from_str(val_to_convert)

            # 3. torch.Tensor from list
            if target_type is torch.Tensor and isinstance(val_to_convert, list):
                return torch.tensor(val_to_convert)

            # 4. Dictionary with torch.Tensor values (e.g., Dict[Any, torch.Tensor])
            if origin is dict and args and len(args) == 2 and args[1] is torch.Tensor:
                converted_dict = {}
                for k_dict, v_dict in val_to_convert.items():
                    converted_dict[k_dict] = torch.tensor(v_dict)
                return converted_dict

            # 5. List of torch.Tensors (e.g. List[torch.Tensor])
            if origin is list and args and len(args) == 1 and args[0] is torch.Tensor:
                converted_list = []
                for item_idx, item in enumerate(val_to_convert):
                    converted_list.append(torch.tensor(item))
                return converted_list

            # Default: return value as is
            # Dict and List of primitive values (without Enum or Tensor) are returned as is
            return val_to_convert

        for key, value in config_dict.items():
            if key not in field_types:
                print(
                    f"Note: '{key}' in config_dict is not a field in {cls.__name__}, it will be ignored."
                )
                continue

            field_type = field_types[key]

            if value is None:
                processed_dict[key] = None
                continue

            origin_ft = get_origin(field_type)
            args_ft = get_args(field_type)

            if (
                origin_ft is Union and len(args_ft) == 2 and args_ft[1] is type(None)
            ):  # Optional[T]
                inner_actual_type = args_ft[0]
                processed_dict[key] = convert_value(value, inner_actual_type, key)
            else:
                processed_dict[key] = convert_value(value, field_type, key)

        try:
            return cls(**processed_dict)
        except TypeError as e:
            print(
                f"Error instantiating {cls.__name__} with processed_dict. Keys in dict: {list(processed_dict.keys())}"
            )
            print(f"Original error: {str(e)}")

            # Show expected fields vs provided fields
            import inspect

            sig = inspect.signature(cls.__init__)
            expected_params = [
                param.name for param in sig.parameters.values() if param.name != "self"
            ]
            missing_params = [
                param for param in expected_params if param not in processed_dict.keys()
            ]
            if missing_params:
                print(f"Missing required parameters: {missing_params}")

            # Show field types for debugging
            for k_detail, v_detail in processed_dict.items():
                expected_type_detail = field_types.get(k_detail)
                print(
                    f"  Field '{k_detail}': Expected {expected_type_detail}, Got {type(v_detail)} = {v_detail}"
                )
            raise

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary, handling nested dataclasses.

        Returns:
            Dictionary representation of the config.
        """
        result = {}
        for field_name in self.__dataclass_fields__:  # Changed 'field' to 'field_name' to avoid conflict with 'field' from dataclasses
            value = getattr(self, field_name)

            if value is None:
                result[field_name] = None

            elif hasattr(value, "to_dict"):  # Handle nested dataclasses
                result[field_name] = value.to_dict()

            elif isinstance(value, Enum):  # Handle enums
                result[field_name] = value.value

            elif isinstance(value, (list, tuple)):  # Handle lists/tuples of dataclasses
                if value and hasattr(value[0], "to_dict"):
                    result[field_name] = [item.to_dict() for item in value]
                else:
                    result[field_name] = value

            elif isinstance(
                value, dict
            ):  # Handle dicts of dataclasses or other complex types
                processed_dict_val = {}
                for k, v in value.items():
                    if hasattr(v, "to_dict"):
                        processed_dict_val[k] = v.to_dict()
                    elif isinstance(v, torch.Tensor):
                        processed_dict_val[k] = v.tolist()
                    elif isinstance(v, Enum):
                        processed_dict_val[k] = v.value
                    else:
                        processed_dict_val[k] = v
                result[field_name] = processed_dict_val

            elif isinstance(value, torch.Tensor):
                result[field_name] = value.tolist()

            else:
                result[field_name] = value
        return result

    def __getitem__(self, key: str) -> Any:
        """Make configs behave like dicts for compatibility with external libraries."""
        return self.to_dict()[key]

    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for compatibility with external libraries."""
        return key in self.to_dict()

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method for compatibility with external libraries."""
        return self.to_dict().get(key, default)


def build_standard_configs(
    args,
    terrain_config_fn,
    scene_lib_config_fn,
    motion_lib_config_fn,
    env_config_fn,
    configure_robot_and_simulator_fn=None,
    agent_config_fn=None,
):
    """Build standard robot, simulator, terrain, scene_lib, motion_lib, env, and optionally agent configs.

    This is a helper function to reduce boilerplate in experiment files.
    All configs are built with training defaults - eval overrides applied separately via apply_inference_overrides().

    Parameter order matches execution order: robot → sim → terrain → scene_lib → motion_lib → env → agent

    Args:
        args: Command line arguments containing robot_name, simulator, etc.
        terrain_config_fn: REQUIRED function that takes (args) and returns TerrainConfig (or None for no terrain)
        scene_lib_config_fn: REQUIRED function that takes (args) and returns SceneLibConfig (scene_file can be None for empty)
        motion_lib_config_fn: REQUIRED function that takes (args) and returns MotionLibConfig (motion_file can be None for empty)
        env_config_fn: REQUIRED function that takes (robot_config, args) and returns env config
        configure_robot_and_simulator_fn: Optional function that takes (robot_config, simulator_config, args)
        agent_config_fn: Optional function that takes (robot_config, env_config, args) and returns agent config

    Returns:
        Dict with keys: robot, simulator, terrain, scene_lib, motion_lib, env, agent (optional)
    """
    from protomotions.robot_configs.factory import robot_config
    from protomotions.simulator.factory import simulator_config as simulator_config_func

    # Build robot config from factory
    robot_cfg = robot_config(args.robot_name)

    # Build simulator config from factory
    simulator_cfg = simulator_config_func(
        args.simulator, robot_cfg, args.headless, args.num_envs, args.experiment_name
    )

    # Configure robot and simulator for this experiment (if function provided)
    if configure_robot_and_simulator_fn is not None:
        configure_robot_and_simulator_fn(robot_cfg, simulator_cfg, args)

    # Build component configs (independent of robot_config)
    # These functions must always be provided
    terrain_cfg = terrain_config_fn(args)  # Can return None for no terrain (exception)
    scene_lib_cfg = scene_lib_config_fn(
        args
    )  # Must return SceneLibConfig (scene_file can be None)
    motion_lib_cfg = motion_lib_config_fn(
        args
    )  # Must return MotionLibConfig (motion_file can be None)

    # Build env config (depends on robot_config)
    env_cfg = env_config_fn(robot_cfg, args)

    # Build agent config if function provided (depends on robot_config and env_config)
    agent_cfg = (
        agent_config_fn(robot_cfg, env_cfg, args)
        if agent_config_fn is not None
        else None
    )

    return {
        "robot": robot_cfg,
        "simulator": simulator_cfg,
        "terrain": terrain_cfg,
        "scene_lib": scene_lib_cfg,
        "motion_lib": motion_lib_cfg,
        "env": env_cfg,
        "agent": agent_cfg,
    }
