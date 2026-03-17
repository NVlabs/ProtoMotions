# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
"""BaseEnv utility functions.

Contains RL combining logic for rewards/terminations.
Look here when debugging reward or termination behavior.

This module handles:
- Reward combining (multiplicative, additive, grace periods, clamping)
- Termination combining (OR logic, terminate_on_true inversion)
- Body indices resolution from names to indices
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch import Tensor

from protomotions.envs.mdp_component import is_mdp_component




# =============================================================================
# Reward Combining
# =============================================================================


def combine_rewards(
    raw_rewards: Dict[str, Tensor],
    configs: Dict[str, Any],
    grace_mask: Optional[Tensor] = None,
    region_weights: Optional[Tensor] = None,
    num_envs: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Combine raw rewards into final reward.
    
    RL Semantics:
    - multiplicative=True: rewards multiplied together (e.g., alive bonus)
    - Otherwise: weighted sum with optional clamping
    - grace_mask: zero rewards during grace period after reset
    - region_weights: per-body weights based on anatomical regions
    
    Args:
        raw_rewards: Dict of {name: reward_tensor} from component execution.
        configs: Dict of {name: MdpComponent} where params contain metadata.
        grace_mask: Boolean mask [num_envs] - True where in grace period.
        region_weights: Optional tensor of per-body weights.
        num_envs: Number of environments.
        device: Device for tensors.
    
    Returns:
        Tuple of (combined_reward, logging_dict) where logging_dict contains
        raw and scaled rewards for debugging.
    """
    mult_reward = torch.ones(num_envs, device=device, dtype=torch.float)
    add_reward = torch.zeros(num_envs, device=device, dtype=torch.float)
    any_mult = False
    logging_dict: Dict[str, Tensor] = {}
    
    for name, reward in raw_rewards.items():
        router = configs[name]
        # Extract params from MdpComponent
        cfg = router.get_params() if is_mdp_component(router) else router
        
        # Apply grace period zeroing
        if cfg.get("zero_during_grace_period", False) and grace_mask is not None:
            reward = reward.clone()
            reward[grace_mask] = 0.0
        
        # Sanity check
        assert torch.all(torch.isfinite(reward)), f"Reward '{name}' not finite"
        logging_dict[f"raw_r/{name}"] = reward.clone()
        
        # Apply multiplicative or additive combining
        if cfg.get("multiplicative", False):
            mult_reward *= reward
            any_mult = True
        else:
            weight = cfg.get("weight", 0.0)
            if weight != 0:
                scaled = reward * weight
                
                # Apply clamping
                min_val = cfg.get("min_value")
                max_val = cfg.get("max_value")
                if min_val is not None:
                    scaled = torch.clamp(scaled, min=min_val)
                if max_val is not None:
                    scaled = torch.clamp(scaled, max=max_val)
                
                logging_dict[f"scaled_r/{name}"] = scaled.clone()
                add_reward += scaled
    
    # Combine multiplicative and additive
    if any_mult:
        logging_dict["multiplicative_reward"] = mult_reward.clone()
        logging_dict["additive_reward"] = add_reward.clone()
        return add_reward + mult_reward, logging_dict
    
    return add_reward, logging_dict


# =============================================================================
# Termination Combining
# =============================================================================


def combine_terminations(
    raw_terms: Dict[str, Tensor],
    configs: Dict[str, Any],
    num_envs: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor]]:
    """Combine termination conditions into reset/terminate buffers.
    
    RL Semantics:
    - terminate_on_true=False: inverts condition (terminate when function returns False)
    - All conditions OR'd together (any termination triggers reset)
    
    Args:
        raw_terms: Dict of {name: bool_tensor} from component execution.
        configs: Dict of {name: MdpComponent} where params contain metadata.
        num_envs: Number of environments.
        device: Device for tensors.
    
    Returns:
        Tuple of (reset_buf, terminate_buf, logging_dict) where:
        - reset_buf: Boolean [num_envs] - environments to reset
        - terminate_buf: Boolean [num_envs] - environments terminated (for bootstrapping)
        - logging_dict: Per-termination condition flags for debugging.
    """
    reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
    terminate_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
    logging_dict: Dict[str, Tensor] = {}
    
    for name, should_term in raw_terms.items():
        router = configs[name]
        # Extract params from MdpComponent
        cfg = router.get_params() if is_mdp_component(router) else router
        
        # Invert if terminate_on_true is False
        if not cfg.get("terminate_on_true", True):
            should_term = ~should_term
        
        # OR all conditions together
        reset_buf = reset_buf | should_term
        terminate_buf = terminate_buf | should_term
        logging_dict[f"termination/{name}"] = should_term.float().clone()
    
    return reset_buf, terminate_buf, logging_dict


# =============================================================================
# Evaluation Combining
# =============================================================================


def combine_evaluation(
    raw_values: Dict[str, Tensor],
    configs: Dict[str, Any],
    num_envs: int,
    device: torch.device,
) -> Tuple[Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
    """Combine evaluation component results into failure flags.

    Evaluation components return numeric values [num_envs]. If a component has
    a ``threshold`` in its static_params, the value is compared against it to
    determine failure. ``fail_above`` (default True) controls the comparison
    direction.

    Args:
        raw_values: Dict of {name: value_tensor} from ComponentManager.execute_all().
        configs: Dict of {name: MdpComponent} where static_params may contain
                 ``threshold`` and ``fail_above`` metadata.
        num_envs: Number of environments.
        device: Device for tensors.

    Returns:
        Tuple of (failed_buf, component_values, component_failures) where:
        - failed_buf: Boolean [num_envs] - environments that failed any component
        - component_values: Dict[str, Tensor] - raw numeric values per component
        - component_failures: Dict[str, Tensor] - boolean failure per component
    """
    failed_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
    component_values: Dict[str, Tensor] = {}
    component_failures: Dict[str, Tensor] = {}

    for name, value in raw_values.items():
        router = configs[name]
        cfg = router.get_params() if is_mdp_component(router) else router

        component_values[name] = value.clone()

        threshold = cfg.get("threshold", None)
        if threshold is not None:
            fail_above = cfg.get("fail_above", True)
            if fail_above:
                failed = value > threshold
            else:
                failed = value < threshold
            component_failures[name] = failed
            failed_buf = failed_buf | failed

    return failed_buf, component_values, component_failures
