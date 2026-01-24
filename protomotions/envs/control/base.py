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
"""Base class for control components.

Control components are stateful task managers that define the objectives and
behaviors of the environment. They manage task-specific state, provide context
for observations and rewards, and can define custom termination conditions.

Examples:
    - MimicControlComponent: Manages motion tracking tasks
    - SteeringControlComponent: Manages heading and speed targets
    - PathFollowingControlComponent: Manages path generation and following
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from dataclasses import dataclass, field

if TYPE_CHECKING:
    from protomotions.simulator.base_simulator.config import VisualizationMarkerConfig, MarkerState
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class ControlComponentConfig:
    """Base configuration for control components.
    
    Note: Components are enabled by being present in the control_components dict.
    To disable a component, remove it from the dict.
    """
    pass


class ControlComponent(ABC):
    """Base class for control components.
    
    Control components are stateful modules that define task behavior. They:
    - Maintain task-specific state across timesteps
    - Provide context variables for observations, rewards, and terminations
    - Can define custom reset and termination logic
    - Can create visualization markers
    
    Attributes:
        config: Component configuration.
        env: Parent environment instance.
    """
    
    def __init__(self, config: ControlComponentConfig, env: "BaseEnv"):
        """Initialize the control component.
        
        Args:
            config: Component configuration.
            env: Parent environment instance.
        """
        self.config = config
        self.env = env
    
    def reset(self, env_ids: Tensor):
        """Reset component state for the given environments.
        
        Called when environments are reset. Should reinitialize any stateful
        buffers and sample new task parameters.
        
        Args:
            env_ids: Indices of environments to reset [num_reset_envs].
        """
        pass
    
    @abstractmethod
    def step(self):
        """Update component state after each physics step.
        
        Called during post_physics_step(). Should update any time-dependent
        state or check for task updates.
        """
        pass
    
    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """Check for component-specific reset and termination conditions.
        
        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors [num_envs].
            Default implementation returns all False.
        """
        device = self.env.device
        num_envs = self.env.num_envs
        return (
            torch.zeros(num_envs, dtype=torch.bool, device=device),
            torch.zeros(num_envs, dtype=torch.bool, device=device),
        )
    
    @abstractmethod
    def get_context(self) -> Dict[str, any]:
        """Get context variables to add to the global context.
        
        The returned dictionary will be merged into the global context dict
        used for computing observations, rewards, and terminations.
        
        Returns:
            Dictionary mapping variable names to tensors or other values.
            
        Example:
            >>> return {
            ...     "ref_state": self.get_reference_state(),
            ...     "motion_times": self.motion_manager.motion_times,
            ... }
        """
        pass
    
    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, "VisualizationMarkerConfig"]:
        """Create visualization marker configurations.
        
        Called during environment initialization. Should return marker configs
        for visualizing the task (e.g., target poses, waypoints).
        
        Args:
            headless: If True, should return empty dict (no visualization).
            
        Returns:
            Dictionary mapping marker names to VisualizationMarkerConfig.
            Default implementation returns empty dict.
        """
        return {}
    
    def get_markers_state(self) -> Dict[str, "MarkerState"]:
        """Compute current marker positions and orientations.
        
        Called each frame to update visualization markers. Should return
        marker states corresponding to the configs from create_visualization_markers().
        
        Returns:
            Dictionary mapping marker names to MarkerState with positions/orientations.
            Default implementation returns empty dict.
        """
        return {}

