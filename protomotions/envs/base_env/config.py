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
from dataclasses import dataclass, field
from typing import Optional, List, Union, Dict, Callable, Any
from protomotions.utils.config_builder import ConfigBuilder

# Import obs configs from their proper location
from protomotions.envs.obs.config import (
    HumanoidObsConfig,
    SceneObsConfig,
)

# Import motion manager config from its proper location
from protomotions.envs.motion_manager.config import MotionManagerConfig


@dataclass
class RewardComponentConfig(ConfigBuilder):
    """Configuration for a single dynamic reward component.

    Attributes:
        function: Callable reward function to invoke. Receives resolved variables and indices.
        variables: Dict mapping function argument names to eval strings.
                   Strings are evaluated against a context dict containing current_state, ref_state, etc.
                   Example: {"x": "current_state.rigid_body_pos", "ref_x": "ref_state.rigid_body_pos"}
                   For exp functions, include coefficient: {"coefficient": "-100.0"}
        indices_subset: Optional body names (List[str]) or indices (List[int]) to subset tensors.
                        When provided, passed to function under the "indices" key.
        weight: Scaling weight for the reward (applied after function call).
        multiplicative: If True, reward is multiplied into combined reward instead of added.
        min_value: Optional lower bound cap for the reward (applied after weight scaling).
        max_value: Optional upper bound cap for the reward (applied after weight scaling).
        zero_during_grace_period: If True, reward is zeroed during the grace period after reset.
                                  Used for rewards that are unreliable immediately after reset (e.g., power, contact changes).
    """

    function: Callable[..., Any]  # Reward function to call
    variables: Dict[str, str] = field(
        default_factory=dict
    )  # {"arg_name": "eval_string"}
    indices_subset: Optional[Union[List[int], List[str]]] = (
        None  # Body names or indices
    )
    weight: float = 0.0
    multiplicative: bool = False
    min_value: Optional[float] = (
        None  # Lower bound cap for the reward (applied after weight scaling)
    )
    max_value: Optional[float] = (
        None  # Upper bound cap for the reward (applied after weight scaling)
    )
    zero_during_grace_period: bool = (
        False  # Zero this reward during grace period after reset
    )


@dataclass
class EnvConfig(ConfigBuilder):
    """Main environment configuration."""

    max_episode_length: int = 300

    # Target for the environment class
    _target_: str = "protomotions.envs.base_env.env.BaseEnv"

    # Observations
    humanoid_obs: HumanoidObsConfig = field(default_factory=HumanoidObsConfig)
    scene_obs: SceneObsConfig = field(default_factory=SceneObsConfig)

    # Termination
    termination_height: float = 0.15
    enable_height_termination: bool = False
    motion_manager: MotionManagerConfig = field(default_factory=MotionManagerConfig)

    # Respawn related params
    ref_respawn_offset: float = 0.05
    ref_object_respawn_offset: float = 0.0
    ref_contact_smooth_window: int = (
        0  # Window length for smoothing contact labels. 0 means no smoothing.
    )
    skip_correct_terrain_height_on_flat: bool = True  # Skip terrain height correction when terrain is completely flat (optimization)

    # Evaluation params
    show_terrain_markers: bool = False # took a tons of memory in IsaacGym
    save_dir: str = ""

    # Reward configuration - single unified dict for all reward components
    reward_config: Dict[str, RewardComponentConfig] = field(default_factory=dict)
