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
"""Base class for observation, reward, and termination managers."""

from abc import ABC
import sys
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from protomotions.envs.managers.utils import resolve_body_indices
from protomotions.robot_configs.base import RobotConfig

# torch.compile unavailable on Python 3.8 (IsaacGym)
TORCH_COMPILE_AVAILABLE = hasattr(torch, 'compile') and sys.version_info >= (3, 9)


class BaseComponentManager(ABC):
    """Base class for environment component managers.
    
    Handles tensor device management, function compilation, variable evaluation,
    and body indices caching.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        robot_config: RobotConfig,
        device: torch.device,
        num_envs: int,
    ):
        self.config = config if config is not None else {}
        self.robot_config = robot_config
        self.device = device
        self.num_envs = num_envs
        self._compiled_functions: Dict[str, Callable] = {}
        self._indices_cache: Dict[str, Optional[Tensor]] = {}
        
        # Move static tensors to device (required for CUDA graphs)
        for component in self.config.values():
            for arg_name, var_value in list(component.variables.items()):
                if not isinstance(var_value, str):
                    component.variables[arg_name] = self._tensor_to_device(var_value)
    
    def _tensor_to_device(self, value: Any) -> Any:
        """Recursively move tensors to self.device."""
        if isinstance(value, torch.Tensor):
            return value.to(self.device) if value.device != self.device else value
        elif isinstance(value, dict):
            return {k: self._tensor_to_device(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return type(value)(self._tensor_to_device(v) for v in value)
        return value
    
    def _evaluate_variables(
        self, component: Any, context: Dict[str, Any], component_name: str
    ) -> Dict[str, Any]:
        """Evaluate component variables. Strings are eval'd against context."""
        func_kwargs = {}
        for arg_name, var_value in component.variables.items():
            if isinstance(var_value, str):
                func_kwargs[arg_name] = eval(var_value, {"__builtins__": {}}, context)
            else:
                func_kwargs[arg_name] = var_value
        return func_kwargs
    
    def _get_compiled_function(self, name: str, func: Callable) -> Callable:
        """Get torch.compiled function, caching result."""
        if name not in self._compiled_functions:
            compiled = None
            if TORCH_COMPILE_AVAILABLE:
                try:
                    compiled = torch.compile(func, mode="reduce-overhead")
                except Exception:
                    pass
            self._compiled_functions[name] = compiled or func
        return self._compiled_functions[name]
    
    def _get_cached_indices(self, name: str, component: Any) -> Optional[Tensor]:
        """Get cached body indices, resolving on first access."""
        if name in self._indices_cache:
            return self._indices_cache[name]
        
        if component.indices_subset is not None:
            resolved = resolve_body_indices(
                component.indices_subset,
                self.robot_config.kinematic_info.body_names,
                self.robot_config.common_naming_to_robot_body_names,
                self.device,
            )
            self._indices_cache[name] = resolved
            return resolved
        return None
