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
"""Configuration classes for the base environment.

This module defines the configuration dataclasses for environment settings,
rewards, terminations, and observation components.
"""

from typing import Optional, List, Union, Dict, Callable, Any
from dataclasses import dataclass, field

from protomotions.envs.obs.scene_obs import SceneObsConfig
from protomotions.envs.obs.observation_component import ObservationComponentConfig
from protomotions.envs.motion_manager.config import MotionManagerConfig
from protomotions.envs.control.base import ControlComponentConfig


@dataclass
class RewardComponentConfig:
    """Configuration for a single dynamic reward component.

    Defines a reward function with variable bindings, optional subsetting,
    and scaling parameters.
    """

    function: Callable[..., Any] = field(
        default=None,
        metadata={"help": "Callable reward function. Receives resolved variables and indices."}
    )
    variables: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Maps function argument names to context keys or constants."}
    )
    indices_subset: Optional[Union[List[int], List[str]]] = field(
        default=None,
        metadata={"help": "Body names or indices to subset tensors. Passed as 'indices' argument."}
    )
    weight: float = field(
        default=0.0,
        metadata={"help": "Scaling weight for the reward (applied after function call)."}
    )
    multiplicative: bool = field(
        default=False,
        metadata={"help": "If True, reward is multiplied into combined reward instead of added."}
    )
    min_value: Optional[float] = field(
        default=None,
        metadata={"help": "Lower bound cap for the reward (applied after weight scaling)."}
    )
    max_value: Optional[float] = field(
        default=None,
        metadata={"help": "Upper bound cap for the reward (applied after weight scaling)."}
    )
    zero_during_grace_period: bool = field(
        default=False,
        metadata={"help": "Zero reward during grace period after reset. For unreliable early rewards."}
    )
    use_density_weights: bool = field(
        default=False,
        metadata={"help": "Use automatic per-body density weights based on kinematic chain distances."}
    )


@dataclass
class TerminationComponentConfig:
    """Configuration for a single dynamic termination component.

    Similar to RewardComponentConfig but for termination conditions.
    """

    function: Optional[Callable[..., Any]] = field(
        default=None,
        metadata={"help": "Termination function returning boolean tensor [num_envs]."}
    )
    variables: Dict[str, str] = field(
        default_factory=dict,
        metadata={"help": "Maps function argument names to eval strings."}
    )
    indices_subset: Optional[Union[List[int], List[str]]] = field(
        default=None,
        metadata={"help": "Body names or indices to subset tensors."}
    )
    terminate_on_true: bool = field(
        default=True,
        metadata={"help": "Terminate when function returns True. If False, terminate on False."}
    )


@dataclass
class EnvConfig:
    """Main environment configuration."""

    max_episode_length: int = field(
        default=300,
        metadata={"help": "Maximum steps per episode before automatic reset.", "min": 1}
    )
    reset_grace_period: int = field(
        default=5,
        metadata={"help": "Steps after reset where grace period applies (for zeroing unreliable rewards).", "min": 0}
    )
    num_state_history_steps: int = field(
        default=0,
        metadata={"help": "Number of historical state steps to store. 0 = no history.", "min": 0}
    )

    _target_: str = "protomotions.envs.base_env.env.BaseEnv"

    scene_obs: SceneObsConfig = field(
        default_factory=SceneObsConfig,
        metadata={"help": "Scene observation configuration."}
    )

    motion_manager: MotionManagerConfig = field(
        default_factory=MotionManagerConfig,
        metadata={"help": "Motion manager for reference motion handling."}
    )

    ref_respawn_offset: float = field(
        default=0.05,
        metadata={"help": "Height offset for respawning relative to reference.", "min": 0.0}
    )
    ref_object_respawn_offset: float = field(
        default=0.0,
        metadata={"help": "Height offset for object respawning."}
    )
    ref_contact_smooth_window: int = field(
        default=0,
        metadata={"help": "Window length for smoothing contact labels. 0 = no smoothing.", "min": 0}
    )
    skip_correct_terrain_height_on_flat: bool = field(
        default=True,
        metadata={"help": "Skip terrain height correction when terrain is flat (optimization)."}
    )

    show_terrain_markers: bool = field(
        default=False,
        metadata={"help": "Show terrain markers during evaluation. Uses significant memory in IsaacGym."}
    )
    save_dir: str = field(
        default="",
        metadata={"help": "Directory for saving evaluation outputs."}
    )

    reward_components: Dict[str, RewardComponentConfig] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of named reward components."}
    )
    
    control_components: Dict[str, ControlComponentConfig] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of stateful task/control managers."}
    )
    
    termination_components: Dict[str, TerminationComponentConfig] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of stateless termination functions."}
    )
    
    observation_components: Dict[str, ObservationComponentConfig] = field(
        default_factory=dict,
        metadata={"help": "Dictionary of stateless observation functions."}
    )
