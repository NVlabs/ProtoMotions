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
"""Observation manager for dynamic observation component evaluation."""

import inspect
from typing import Dict, Any, Optional

import torch
from torch import Tensor

from protomotions.envs.managers.base_manager import BaseComponentManager
from protomotions.robot_configs.base import RobotConfig


class ObservationManager(BaseComponentManager):
    """Evaluates dynamic observation components and stores results in buffer."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        robot_config: RobotConfig,
        device: torch.device,
        num_envs: int,
        dt: float,
    ):
        super().__init__(config, robot_config, device, num_envs)
        self.dt = dt
        self.observation_buffer: Dict[str, Tensor] = {}
    
    def initialize(self, context: Dict[str, Any]):
        """Compute initial observations for all envs."""
        if self.config:
            all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            self.compute_observations(context, all_env_ids)
    
    def compute_observations(self, context: Dict[str, Any], env_ids: Optional[Tensor] = None):
        """Compute observations from config, updating buffer for specified envs."""
        if not self.config:
            return
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        
        for obs_name, component in self.config.items():
            if component.function is None:
                continue

            func_kwargs = self._evaluate_variables(component, context, f"obs '{obs_name}'")
            
            indices = self._get_cached_indices(obs_name, component)
            if indices is not None:
                func_kwargs["indices"] = indices
            
            sig = inspect.signature(component.function)
            func_accepts_env_ids = "env_ids" in sig.parameters
            if func_accepts_env_ids:
                func_kwargs["env_ids"] = env_ids

            compiled_fn = self._get_compiled_function(obs_name, component.function)
            obs_value = compiled_fn(**func_kwargs)
            
            if obs_name not in self.observation_buffer:
                self.observation_buffer[obs_name] = torch.zeros(
                    self.num_envs, obs_value.shape[-1], 
                    dtype=obs_value.dtype, device=self.device
                )
            
            if func_accepts_env_ids:
                self.observation_buffer[obs_name][env_ids] = obs_value
            else:
                self.observation_buffer[obs_name][env_ids] = obs_value[env_ids]
    
    def get_observations(self) -> Dict[str, Tensor]:
        """Return cloned observation tensors."""
        return {name: tensor.clone() for name, tensor in self.observation_buffer.items()}
