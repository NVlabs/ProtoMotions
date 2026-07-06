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


def _action_config_vector_length(value: Any) -> int | None:
    try:
        return len(value)
    except TypeError:
        return None


def _action_fn_name(action_config: Dict[str, Any]) -> str:
    fn = action_config["fn"]
    return getattr(fn, "__name__", str(fn))


def _required_action_config_fields(action_config: Dict[str, Any]) -> tuple[str, ...]:
    fn_name = _action_fn_name(action_config)
    if fn_name == "bm_pd_action":
        return ("pd_action_offset", "action_scale", "stiffness", "damping")
    if fn_name == "normalized_pd_fixed_gains_action":
        return ("pd_action_offset", "pd_action_scale", "stiffness", "damping")
    if fn_name == "passthrough_pd_action":
        return ("stiffness", "damping")
    return ()


def _dof_names(robot_config: Any) -> list | None:
    kinematic_info = getattr(robot_config, "kinematic_info", None)
    dof_names = getattr(kinematic_info, "dof_names", None)
    if dof_names is None:
        return None
    return list(dof_names)


def validate_expert_action_config(
    action_config: Any,
    robot_config: Any,
    expert_robot_config: Any = None,
) -> None:
    """Fail fast if an expert action config cannot drive this robot."""
    if action_config is None:
        return

    if not isinstance(action_config, dict) or "fn" not in action_config:
        raise ValueError("Expert env action_config must be None or a dict with an 'fn' entry")

    expected = robot_config.number_of_actions
    for field_name in _required_action_config_fields(action_config):
        if field_name not in action_config:
            fn_name = _action_fn_name(action_config)
            raise ValueError(
                f"Expert action_config for {fn_name} is missing required "
                f"field '{field_name}'"
            )

    for field_name in (
        "pd_action_offset",
        "pd_action_scale",
        "action_scale",
        "stiffness",
        "damping",
    ):
        value = action_config.get(field_name)
        if value is None:
            continue
        length = _action_config_vector_length(value)
        if length is not None and length != expected:
            robot_name = getattr(robot_config, "robot_name", type(robot_config).__name__)
            raise ValueError(
                f"Expert action_config.{field_name} has length {length}; "
                f"expected {expected} actions for {robot_name}"
            )

    if expert_robot_config is None:
        return

    expert_dof_names = _dof_names(expert_robot_config)
    student_dof_names = _dof_names(robot_config)
    if expert_dof_names is None or student_dof_names is None:
        return
    if expert_dof_names != student_dof_names:
        raise ValueError(
            "Expert action_config DOF order does not match student robot: "
            f"expert={expert_dof_names}, student={student_dof_names}"
        )


def get_expert_action_config(
    expert_env_config: Any,
    robot_config: Any,
    expert_robot_config: Any = None,
) -> Any:
    """Copy and validate the expert env action interface for student rollouts."""
    if not hasattr(expert_env_config, "action_config"):
        raise ValueError("Expert env config must define action_config for distillation")

    action_config = deepcopy(expert_env_config.action_config)
    validate_expert_action_config(action_config, robot_config, expert_robot_config)
    return action_config
