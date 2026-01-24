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
"""Reward manager for dynamic reward component evaluation."""

from functools import cached_property
from typing import Dict, Any, Optional, Tuple

import torch
from torch import Tensor

from protomotions.components.pose_lib import compute_body_density_weights
from protomotions.envs.managers.base_manager import BaseComponentManager


class RewardManager(BaseComponentManager):
    """Evaluates dynamic reward components with grace periods and multiplicative/additive logic."""

    @cached_property
    def _density_weights(self) -> Tensor:
        """Per-body density weights based on kinematic chain distances."""
        return compute_body_density_weights(self.robot_config.kinematic_info).to(self.device)

    def compute_rewards(
        self, context: Dict[str, Any], grace_mask: Optional[Tensor]
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute combined reward and logging dict from config."""
        mult_reward = torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        add_reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        any_mult = False
        logging_dict = {}

        for name, component in self.config.items():
            if component.function is None:
                continue

            func_kwargs = self._evaluate_variables(component, context, f"reward '{name}'")
            
            indices = self._get_cached_indices(name, component)
            if indices is not None:
                func_kwargs["indices"] = indices
            if component.use_density_weights:
                func_kwargs["weights"] = self._density_weights

            compiled_fn = self._get_compiled_function(name, component.function)
            reward = compiled_fn(**func_kwargs)

            if component.zero_during_grace_period and grace_mask is not None:
                reward = reward.clone()
                reward[grace_mask] = 0.0

            assert torch.all(torch.isfinite(reward)), f"Reward '{name}' not finite"
            # Clone for logging (CUDA graphs reuse memory buffers)
            logging_dict[f"raw_r/{name}"] = reward.clone()

            if component.multiplicative:
                mult_reward *= reward
                any_mult = True
            elif component.weight != 0:
                scaled = reward * component.weight
                if component.min_value is not None:
                    scaled = torch.clamp(scaled, min=component.min_value)
                if component.max_value is not None:
                    scaled = torch.clamp(scaled, max=component.max_value)
                logging_dict[f"scaled_r/{name}"] = scaled.clone()
                add_reward += scaled

        if any_mult:
            logging_dict["multiplicative_reward"] = mult_reward.clone()
            logging_dict["additive_reward"] = add_reward.clone()
            return add_reward + mult_reward, logging_dict
        return add_reward, logging_dict
