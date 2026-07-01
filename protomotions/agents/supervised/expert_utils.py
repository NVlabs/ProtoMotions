# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expert observation utilities for supervised agents."""

import logging
from typing import Dict, Any
from copy import deepcopy

log = logging.getLogger(__name__)


def _require_expert_actor_in_keys(expert_agent_config: Any) -> list:
    actor_in_keys = get_expert_actor_in_keys(expert_agent_config)
    if not actor_in_keys:
        raise ValueError(
            "Expert agent config must define model.actor.in_keys or model.in_keys "
            "so supervised expert observations can be wired explicitly."
        )
    return actor_in_keys


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
    
    actor_in_keys = _require_expert_actor_in_keys(expert_agent_config)
    
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
    
    actor_in_keys = _require_expert_actor_in_keys(expert_agent_config)
    
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
