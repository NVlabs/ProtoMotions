# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External kinematic control component.

Simple control component that applies externally-provided poses kinematically.
Used when pose computation happens outside the environment (e.g., prior model inference).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.simulator.base_simulator.simulator_state import ResetState

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class ExternalKinematicControlConfig(ControlComponentConfig):
    """Configuration for external kinematic control component."""
    _target_: str = "protomotions.envs.control.external_kinematic_control.ExternalKinematicControl"


class ExternalKinematicControl(ControlComponent):
    """Control component that applies externally-provided poses.
    
    This component receives poses from external code (e.g., a prior model)
    and applies them kinematically during step().
    
    Usage:
        hook = env.control_manager.components["external_kinematic"]
        hook.set_next_pose(reset_state)
        env.step(actions)  # Will apply the pose during step
    """
    
    config: ExternalKinematicControlConfig
    
    def __init__(self, config: ExternalKinematicControlConfig, env: "BaseEnv"):
        super().__init__(config, env)
        self._next_pose: Optional[ResetState] = None
        self._initialized = False
    
    def reset(self, env_ids: Tensor):
        """Reset component state."""
        self._next_pose = None
        self._initialized = True
    
    def set_next_pose(self, reset_state: ResetState):
        """Set the pose to apply on the next step.
        
        Args:
            reset_state: The robot state to apply kinematically.
        """
        self._next_pose = reset_state
    
    def step(self):
        """Apply the stored pose kinematically."""
        if not self._initialized or self._next_pose is None:
            return
        
        env_ids = torch.arange(self.env.num_envs, dtype=torch.long, device=self.env.device)
        
        # Apply the stored pose (no object support)
        self.env.simulator.reset_envs(self._next_pose, None, env_ids)
        
        # Prevent environment reset logic
        self.env.progress_buf[:] = 0
        self.env.reset_buf[:] = 0
        self.env.terminate_buf[:] = 0
        
        # Clear the pose after applying
        self._next_pose = None
    
    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """No resets from external kinematic control."""
        return (
            torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device),
            torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device),
        )
    
    def populate_context(self, ctx: EnvContext) -> None:
        """External kinematic control doesn't add any context variables."""
        pass

