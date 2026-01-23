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
"""Termination manager for dynamic termination component evaluation."""

from typing import Dict, Any, Tuple

import torch
from torch import Tensor

from protomotions.envs.managers.base_manager import BaseComponentManager


class TerminationManager(BaseComponentManager):
    """Evaluates dynamic termination components. BaseEnv handles height/max_length checks."""

    def check_terminations(self, context: Dict[str, Any]) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
        """Check terminations, returning (reset_buf, terminate_buf, logging_dict)."""
        if not self.config:
            return (
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                {},
            )
        
        reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminate_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        logging_dict = {}
        
        for name, component in self.config.items():
            if component.function is None:
                continue

            func_kwargs = self._evaluate_variables(component, context, f"term '{name}'")
            
            indices = self._get_cached_indices(name, component)
            if indices is not None:
                func_kwargs["indices"] = indices

            compiled_fn = self._get_compiled_function(name, component.function)
            should_term = compiled_fn(**func_kwargs)
            
            if not component.terminate_on_true:
                should_term = ~should_term
            
            reset_buf = reset_buf | should_term
            terminate_buf = terminate_buf | should_term
            # Clone for logging (CUDA graphs reuse memory buffers)
            logging_dict[f"termination/{name}"] = should_term.float().clone()
        
        return reset_buf, terminate_buf, logging_dict
