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
"""Utilities for MaskedMimic agent."""

import logging
from pathlib import Path
from typing import Dict, Any
from copy import deepcopy

import torch

log = logging.getLogger(__name__)


def load_expert_configs(expert_model_path: str) -> Dict[str, Any]:
    """Load expert's resolved_configs.pt from the checkpoint directory."""
    checkpoint_path = Path(expert_model_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expert checkpoint not found at {checkpoint_path}")
    
    resolved_configs_path = checkpoint_path.parent / "resolved_configs.pt"
    
    if not resolved_configs_path.exists():
        raise FileNotFoundError(f"Could not find resolved_configs.pt at {resolved_configs_path}")
    
    log.info(f"Loading expert configs from {resolved_configs_path}")
    return torch.load(resolved_configs_path, map_location="cpu", weights_only=False)


def get_expert_actor_in_keys(expert_agent_config: Any) -> list:
    """Extract the input keys required by the expert's actor."""
    if hasattr(expert_agent_config, 'model'):
        model_config = expert_agent_config.model
        
        if hasattr(model_config, 'actor') and hasattr(model_config.actor, 'in_keys'):
            return list(model_config.actor.in_keys)
        
        if hasattr(model_config, 'in_keys'):
            return list(model_config.in_keys)
    
    log.warning("Could not determine expert actor in_keys")
    return []


def get_expert_observation_keys(
    expert_env_config: Any,
    expert_agent_config: Any,
    prefix: str = "expert_",
) -> list:
    """Get the prefixed observation keys that would be added for the expert."""
    if not hasattr(expert_env_config, 'observation_components'):
        return []
    
    if expert_env_config.observation_components is None:
        return []
    
    actor_in_keys = get_expert_actor_in_keys(expert_agent_config)
    if not actor_in_keys:
        actor_in_keys = list(expert_env_config.observation_components.keys())
    
    return [
        f"{prefix}{obs_key}" 
        for obs_key in expert_env_config.observation_components.keys()
        if obs_key in actor_in_keys
    ]


def get_expert_observation_components(
    expert_env_config: Any,
    expert_agent_config: Any,
    existing_obs_keys: list = None,
    prefix: str = "expert_",
) -> Dict[str, Any]:
    """Extract observation components needed by expert's actor with prefixed keys."""
    expert_obs_components = {}
    
    if not hasattr(expert_env_config, 'observation_components'):
        return expert_obs_components
    
    if expert_env_config.observation_components is None:
        return expert_obs_components
    
    actor_in_keys = get_expert_actor_in_keys(expert_agent_config)
    if not actor_in_keys:
        actor_in_keys = list(expert_env_config.observation_components.keys())
    
    existing_keys_set = set(existing_obs_keys) if existing_obs_keys else set()
    
    for obs_key, obs_config in expert_env_config.observation_components.items():
        if obs_key not in actor_in_keys:
            continue
            
        prefixed_key = f"{prefix}{obs_key}"
        
        if prefixed_key in existing_keys_set:
            raise ValueError(f"Expert observation key '{prefixed_key}' conflicts with existing observation.")
        
        expert_obs_components[prefixed_key] = deepcopy(obs_config)
    
    log.info(f"Loaded {len(expert_obs_components)} expert observation components with prefix '{prefix}'")
    
    return expert_obs_components
