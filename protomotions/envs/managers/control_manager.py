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
"""Control manager for BaseEnv.

Manages control_components lifecycle and orchestrates their interactions.
Control components define task behaviors and provide context for observations/rewards.
"""

from typing import Dict, Any, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.utils.hydra_replacement import get_class

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv
    from protomotions.simulator.base_simulator.config import VisualizationMarkerConfig, MarkerState
    from protomotions.envs.control.base import ControlComponent


class ControlManager:
    """Manages control_components lifecycle and orchestration.
    
    Control components are stateful task managers that:
    - Define task objectives and behaviors
    - Provide context variables for observations/rewards
    - Can define custom reset and termination logic
    - Can create visualization markers
    
    Note: Unlike other managers, control components DO receive env reference
    because they need deep integration with environment state (motion_manager,
    simulator, etc.). This manager orchestrates their lifecycle.
    
    Attributes:
        components: Dict mapping component names to ControlComponent instances
        device: Device for tensor operations
        num_envs: Number of parallel environments
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        env: "BaseEnv",
    ):
        """Initialize control manager.
        
        Args:
            config: Dictionary mapping component names to component configurations
            env: Parent environment instance (control components need env access)
        """
        self.device = env.device
        self.num_envs = env.num_envs
        
        # Initialize control components
        # Components are enabled by being present in the dict
        self.components: Dict[str, "ControlComponent"] = {}
        for name, comp_config in config.items():
            comp_class = get_class(comp_config._target_)
            self.components[name] = comp_class(comp_config, env)
    
    def step(self):
        """Update all control components after each physics step.
        
        Called during post_physics_step(). Components update their
        time-dependent state or check for task updates.
        """
        for component in self.components.values():
            component.step()
    
    def reset(self, env_ids: Tensor):
        """Reset all control components for specified environments.
        
        Called when environments are reset. Components reinitialize
        stateful buffers and sample new task parameters.
        
        Args:
            env_ids: Indices of environments to reset [num_reset_envs]
        """
        for component in self.components.values():
            component.reset(env_ids)
    
    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """Check control component-specific reset and termination conditions.
        
        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors [num_envs]
        """
        reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminate_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        for component in self.components.values():
            comp_reset, comp_terminate = component.check_resets_and_terminations()
            reset_buf = reset_buf | comp_reset
            terminate_buf = terminate_buf | comp_terminate
        
        return reset_buf, terminate_buf
    
    def get_context(self) -> Dict[str, Any]:
        """Get context variables from all control components.
        
        Merges context dicts from all components. These variables are
        added to the global context for observation/reward/termination
        function evaluation.
        
        Returns:
            Dictionary mapping variable names to tensors or other values
        """
        context = {}
        for component in self.components.values():
            comp_context = component.get_context()
            # Merge component context (no prefix for now, may add if conflicts arise)
            context.update(comp_context)
        return context
    
    def create_visualization_markers(
        self, headless: bool
    ) -> Dict[str, "VisualizationMarkerConfig"]:
        """Create visualization marker configurations from all components.
        
        Called during environment initialization. Collects marker configs
        from all components for visualizing task state (e.g., target poses,
        waypoints).
        
        Args:
            headless: If True, should return empty dict (no visualization)
        
        Returns:
            Dictionary mapping marker names to VisualizationMarkerConfig
        """
        if headless:
            return {}
        
        markers = {}
        for component in self.components.values():
            comp_markers = component.create_visualization_markers(headless)
            markers.update(comp_markers)
        return markers
    
    def get_markers_state(self) -> Dict[str, "MarkerState"]:
        """Compute current marker positions and orientations from all components.
        
        Called each frame to update visualization markers. Collects marker
        states from all components.
        
        Returns:
            Dictionary mapping marker names to MarkerState with positions/orientations
        """
        markers_state = {}
        for component in self.components.values():
            comp_markers_state = component.get_markers_state()
            markers_state.update(comp_markers_state)
        return markers_state

