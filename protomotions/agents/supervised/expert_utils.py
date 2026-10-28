# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expert observation utilities for supervised agents."""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional, TYPE_CHECKING
import torch
from copy import deepcopy

from protomotions.envs.context_sources import SENSOR_SOURCES, source_of_path

if TYPE_CHECKING:
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.envs.sensors.config import SensorConfig

log = logging.getLogger(__name__)

# The prefix every expert-labelled observation key carries in the student env,
# so the student's own observations and the expert's never collide. One spelling.
EXPERT_OBS_PREFIX = "expert_"


def validate_expert_action_contract(
    expert_robot, expert_env, student_robot, student_env
):
    """Expert labels and executed actions share one joint order and raw-action meaning."""
    if expert_robot.kinematic_info.dof_names != student_robot.kinematic_info.dof_names:
        raise ValueError("Expert and student must use the same ordered joints")
    if expert_robot.control != student_robot.control:
        raise ValueError("Expert and student must use the same controller parameters")
    source, target = expert_env.action_config, student_env.action_config
    if (
        source is None
        or target is None
        or source.fn is not target.fn
        or source.params.keys() != target.params.keys()
    ):
        raise ValueError("Expert and student must use the same action config")
    for key, value in source.params.items():
        other = target.params[key]
        if isinstance(value, torch.Tensor):
            matches = isinstance(other, torch.Tensor) and torch.equal(
                value.detach().cpu(), other.detach().cpu()
            )
        else:
            matches = value == other
        if not matches:
            raise ValueError(f"Expert action config mismatch: {key}")


def _sensor_contract_detail(
    expert: Optional[SensorConfig], target: Optional[SensorConfig], *, noisy: bool
) -> Optional[str]:
    """Why two sensor configs are an incompatible contract for a source, or None.

    A clean source needs the same channels and, for localization, the same body
    and offset; noise is not read. A noisy source needs the whole channel equal,
    noise and drift included. Random seeds are never compared.
    """
    if expert is None or target is None:
        return "one environment declares sensors and the other does not"
    expert_channels, target_channels = expert.channels(), target.channels()
    if expert_channels.keys() != target_channels.keys():
        return f"channels {sorted(expert_channels)} != {sorted(target_channels)}"
    for channel_name, expert_channel in expert_channels.items():
        target_channel = target_channels[channel_name]
        if noisy:
            if expert_channel != target_channel:
                return f"channel {channel_name!r} differs"
        elif channel_name == "localization":
            expert_placement = (expert_channel.body_name, tuple(expert_channel.offset))
            target_placement = (target_channel.body_name, tuple(target_channel.offset))
            if expert_placement != target_placement:
                return f"channel {channel_name!r} placement differs"
    return None


def validate_expert_sensor_contract(
    expert_env_config: EnvConfig,
    expert_agent_config: Any,
    target_sensors: Optional[SensorConfig],
) -> None:
    """Keep a frozen expert's sensor inputs unchanged in the student environment.

    Privileged bindings are independent of sensors. A clean estimate requires
    the same channels and their placement; a noisy estimate also requires the
    same noise and drift. Random seeds need not match: the contract describes
    measurements, not their draws.
    """
    actor_keys = get_expert_actor_in_keys(expert_agent_config)
    expert_sensors = expert_env_config.sensors
    for name, component in expert_env_config.observation_components.items():
        if name not in actor_keys:
            continue
        for path in component.dynamic_vars.values():
            source = source_of_path(path.path)
            if source not in SENSOR_SOURCES:
                continue
            detail = _sensor_contract_detail(
                expert_sensors, target_sensors, noisy=source == "noisy"
            )
            if detail is not None:
                raise ValueError(
                    f"Expert observation {name!r} binds {path.path!r} with a "
                    f"different sensor contract: {detail}"
                )


def get_expert_actor_in_keys(expert_agent_config: Any) -> list:
    """Input keys of the expert's actor, from model.actor.in_keys or model.in_keys."""
    model_config = getattr(expert_agent_config, "model", None)
    actor_config = getattr(model_config, "actor", model_config)
    actor_in_keys = list(getattr(actor_config, "in_keys", None) or [])
    if not actor_in_keys:
        raise ValueError(
            "Expert agent config must define model.actor.in_keys or model.in_keys "
            "so supervised expert observations can be wired explicitly."
        )
    return actor_in_keys


def get_expert_observation_keys(
    expert_env_config: EnvConfig,
    expert_agent_config: Any,
    prefix: str = EXPERT_OBS_PREFIX,
) -> list:
    """Get the prefixed observation keys that would be added for the expert."""
    actor_in_keys = get_expert_actor_in_keys(expert_agent_config)

    return [
        f"{prefix}{obs_key}"
        for obs_key in expert_env_config.observation_components.keys()
        if obs_key in actor_in_keys
    ]


def get_expert_observation_components(
    expert_env_config: EnvConfig,
    expert_agent_config: Any,
    existing_obs_keys: list = None,
    prefix: str = EXPERT_OBS_PREFIX,
) -> Dict[str, Any]:
    """Extract observation components needed by expert's actor with prefixed keys."""
    expert_obs_components = {}

    actor_in_keys = get_expert_actor_in_keys(expert_agent_config)

    existing_keys_set = set(existing_obs_keys) if existing_obs_keys else set()

    for obs_key, obs_config in expert_env_config.observation_components.items():
        if obs_key not in actor_in_keys:
            continue

        prefixed_key = f"{prefix}{obs_key}"

        if prefixed_key in existing_keys_set:
            raise ValueError(
                f"Expert observation key '{prefixed_key}' conflicts with existing observation."
            )

        expert_obs_components[prefixed_key] = deepcopy(obs_config)

    log.info(
        f"Loaded {len(expert_obs_components)} expert observation components with prefix '{prefix}'"
    )

    return expert_obs_components
