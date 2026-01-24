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
"""Configuration for dynamic observation components.

Observation components are stateless functions that compute observations from
the global context. Similar to the reward component system, they use eval strings
to extract variables from context and pass them to pure functions.

Examples:
    - Target poses: compute_target_poses(current_state, ref_state, future_steps)
    - Steering: compute_steering_obs(root_rot, tar_dir, tar_speed)
"""

from typing import Callable, Dict, Optional, Union, List, Any
from dataclasses import dataclass, field


@dataclass
class ObservationComponentConfig:
    """Configuration for a single dynamic observation component.
    
    Similar to RewardComponentConfig, this allows defining observations through
    a function and variable mappings that are evaluated from the global context.
    Components are enabled by being present in the observation_components dict.
    
    Attributes:
        function: Callable observation function to invoke. Receives resolved variables.
        variables: Dict mapping function argument names to context keys or constants.
                   String keys are looked up in the global context dict.
                   Non-string values are passed directly as constants.
                   Example: {"motion_ids": "motion_ids", "num_steps": 3}
        indices_subset: Optional body names (List[str]) or indices (List[int]) to subset tensors.
                        When provided, passed to function under the "indices" key.
        
    Example:
        >>> from protomotions.envs.obs import max_coords_obs_factory
        >>> obs_config = max_coords_obs_factory(local_obs=True)
    """
    
    function: Optional[Callable[..., Any]] = None  # Observation function to call
    variables: Dict[str, str] = field(
        default_factory=dict
    )  # {"arg_name": "context_key_or_value"}
    indices_subset: Optional[Union[List[int], List[str]]] = field(default=None)  # Body names or indices
