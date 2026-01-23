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
"""Mimic control component for motion tracking tasks.

This component manages reference motion tracking, including:
- Motion manager for motion library sampling and playback
- Reference state computation and terrain correction
- Masked mimic conditioning state
- Visualization markers for target poses
"""

from dataclasses import dataclass
from typing import Dict, Tuple, TYPE_CHECKING

import torch
from torch import Tensor

from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.obs.humanoid import dof_to_local
from protomotions.simulator.base_simulator.config import (
    MarkerConfig,
    VisualizationMarkerConfig,
    MarkerState,
)

if TYPE_CHECKING:
    from protomotions.envs.base_env.env import BaseEnv


@dataclass
class MimicControlConfig(ControlComponentConfig):
    """Configuration for mimic control component.
    
    Attributes:
        bootstrap_on_episode_end: If True, don't terminate when motion clip ends.
        num_future_steps: Number of future reference poses to provide in context.
            Used by observation functions that need multi-step goal trajectories.
    """
    _target_: str = "protomotions.envs.control.mimic_control.MimicControl"
    
    bootstrap_on_episode_end: bool = True
    num_future_steps: int = 1


class MimicControl(ControlComponent):
    """Control component for motion tracking tasks.
    
    Provides context for mimic observations and rewards. Accesses the env's
    motion_manager rather than creating its own.
    """
    
    config: MimicControlConfig
    
    def __init__(self, config: MimicControlConfig, env: "BaseEnv"):
        """Initialize mimic control component.
        
        Args:
            config: Component configuration.
            env: Parent environment instance.
        """
        super().__init__(config, env)
    
    def step(self):
        """Control component step - motion manager is handled by env."""
        # Motion manager is updated by env.post_physics_step if needed
        pass
    
    def check_resets_and_terminations(self) -> Tuple[Tensor, Tensor]:
        """Check if motion clips have finished.
        
        Returns:
            Tuple of (reset_buf, terminate_buf) boolean tensors.
        """
        device = self.env.device
        num_envs = self.env.num_envs
        
        # Check if motion clip has finished (access via env)
        done_clip = self.env.motion_manager.get_done_tracks()
        reset_buf = done_clip
        
        # Only terminate if not bootstrapping
        if self.config.bootstrap_on_episode_end:
            terminate_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
        else:
            terminate_buf = done_clip
        
        return reset_buf, terminate_buf
    
    def get_context(self) -> Dict[str, any]:
        """Get mimic-specific context for observations and rewards.
        
        Returns:
            Dictionary with reference state and tracking variables.
            
            Key context variables:
            - ref_state: Single-step reference state at current time [batch, ...] (for rewards)
            - mimic_ref_pos: Future body positions [envs, future_steps, bodies, 3]
            - mimic_ref_rot: Future body rotations [envs, future_steps, bodies, 4]
            - mimic_ref_anchor_rot: Future anchor rotations [envs, future_steps, 4]
            - mimic_ref_vel: Future body velocities [envs, future_steps, bodies, 3]
            - mimic_ref_anchor_vel: Future anchor velocities [envs, future_steps, 3]
            - mimic_ref_ang_vel: Future body angular velocities [envs, future_steps, bodies, 3]
            - mimic_ref_anchor_ang_vel: Future anchor angular velocities [envs, future_steps, 3]
            - mimic_ref_dof_pos: Future DOF positions [envs, future_steps, dofs]
            - mimic_ref_dof_vel: Future DOF velocities [envs, future_steps, dofs]
        """
        num_envs = self.env.num_envs
        device = self.env.device
        motion_ids = self.env.motion_manager.motion_ids
        motion_times = self.env.motion_manager.motion_times
        
        # Get single-step reference state at current time (for rewards)
        ref_state = self.env.motion_lib.get_motion_state(motion_ids, motion_times)
        
        # Apply terrain height correction to reference state
        ref_gt = ref_state.rigid_body_pos.clone()
        ref_gt += self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            ref_gt
        )
        ref_state.rigid_body_pos = ref_gt
        
        # Build multi-step reference for observations
        # Times: t+dt, t+2*dt, ..., t+N*dt
        num_future_steps = self.config.num_future_steps
        dt = self.env.dt
        
        time_offsets = dt * torch.arange(
            1, num_future_steps + 1, device=device
        )
        future_times = motion_times.unsqueeze(-1) + time_offsets  # [envs, N]
        
        # Clamp to motion length
        motion_lengths = self.env.motion_lib.get_motion_length(motion_ids)
        future_times = torch.minimum(future_times, motion_lengths.unsqueeze(-1))
        
        # Flatten for motion_lib query: [envs*N]
        flat_motion_ids = motion_ids.unsqueeze(-1).expand(
            num_envs, num_future_steps
        ).reshape(-1)
        flat_future_times = future_times.reshape(-1)
        
        # Query motion lib for all future steps
        future_state = self.env.motion_lib.get_motion_state(
            flat_motion_ids, flat_future_times
        )
        
        # Reshape to [envs, future_steps, ...] and apply terrain correction
        num_bodies = future_state.rigid_body_pos.shape[1]
        num_dofs = future_state.dof_pos.shape[1]
        
        # Body positions with terrain correction
        mimic_ref_pos = future_state.rigid_body_pos.view(
            num_envs, num_future_steps, num_bodies, 3
        ).clone()
        offset = self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
            mimic_ref_pos[:, 0, :, :]  # Use first step for offset
        )
        mimic_ref_pos += offset.unsqueeze(1)
        
        # Body rotations
        mimic_ref_rot = future_state.rigid_body_rot.view(
            num_envs, num_future_steps, num_bodies, 4
        )
        
        # Body velocities
        mimic_ref_vel = future_state.rigid_body_vel.view(
            num_envs, num_future_steps, num_bodies, 3
        )
        
        # Body angular velocities
        mimic_ref_ang_vel = future_state.rigid_body_ang_vel.view(
            num_envs, num_future_steps, num_bodies, 3
        )
        
        # DOF positions and velocities
        mimic_ref_dof_pos = future_state.dof_pos.view(
            num_envs, num_future_steps, num_dofs
        )
        mimic_ref_dof_vel = future_state.dof_vel.view(
            num_envs, num_future_steps, num_dofs
        )
        
        hinge_axes_map = self.env.robot_config.kinematic_info.hinge_axes_map
        ref_lr = dof_to_local(ref_state.dof_pos, hinge_axes_map, True)
        
        return {
            "ref_state_rigid_body_pos": ref_state.rigid_body_pos,
            "ref_state_rigid_body_rot": ref_state.rigid_body_rot,
            "ref_state_rigid_body_vel": ref_state.rigid_body_vel,
            "ref_state_rigid_body_ang_vel": ref_state.rigid_body_ang_vel,
            "ref_state_rigid_body_contacts": ref_state.rigid_body_contacts,
            "ref_state_dof_pos": ref_state.dof_pos,
            "ref_state_dof_vel": ref_state.dof_vel,
            "ref_state_root_height": ref_state.rigid_body_pos[:, 0, 2],
            
            # Root values
            "ref_state_root_pos": ref_state.rigid_body_pos[:, 0, :],
            "ref_state_root_rot": ref_state.rigid_body_rot[:, 0, :],
            "ref_state_root_vel": ref_state.rigid_body_vel[:, 0, :],
            "ref_state_root_ang_vel": ref_state.rigid_body_ang_vel[:, 0, :],
            
            # Anchor values
            "ref_state_anchor_pos": ref_state.rigid_body_pos[:, self.env.robot_config.anchor_body_index, :],
            "ref_state_anchor_rot": ref_state.rigid_body_rot[:, self.env.robot_config.anchor_body_index, :],
            "ref_state_anchor_vel": ref_state.rigid_body_vel[:, self.env.robot_config.anchor_body_index, :],
            "ref_state_anchor_ang_vel": ref_state.rigid_body_ang_vel[:, self.env.robot_config.anchor_body_index, :],

            "ref_lr": ref_lr,

            "mimic_ref_pos": mimic_ref_pos,
            "mimic_ref_rot": mimic_ref_rot,
            "mimic_ref_vel": mimic_ref_vel,
            "mimic_ref_ang_vel": mimic_ref_ang_vel,
            "mimic_ref_dof_pos": mimic_ref_dof_pos,
            "mimic_ref_dof_vel": mimic_ref_dof_vel,

            # Root values
            "mimic_ref_root_pos": mimic_ref_pos[:, :, 0, :],
            "mimic_ref_root_rot": mimic_ref_rot[:, :, 0, :],
            "mimic_ref_root_vel": mimic_ref_vel[:, :, 0, :],
            "mimic_ref_root_ang_vel": mimic_ref_ang_vel[:, :, 0, :],

            # Anchor values
            "mimic_ref_anchor_pos": mimic_ref_pos[:, :, self.env.robot_config.anchor_body_index, :],
            "mimic_ref_anchor_rot": mimic_ref_rot[:, :, self.env.robot_config.anchor_body_index, :],
            "mimic_ref_anchor_vel": mimic_ref_vel[:, :, self.env.robot_config.anchor_body_index, :],
            "mimic_ref_anchor_ang_vel": mimic_ref_ang_vel[:, :, self.env.robot_config.anchor_body_index, :],
        }
    
    def create_visualization_markers(self, headless: bool) -> Dict[str, VisualizationMarkerConfig]:
        """Create visualization markers for reference poses.
        
        Args:
            headless: If True, returns empty dict.
            
        Returns:
            Dictionary of marker configurations.
        """
        if headless:
            return {}
        
        visualization_markers = {}
        
        # Standard mimic: visualize all bodies
        body_names = self.env.robot_config.kinematic_info.body_names
        
        body_markers = []
        for body_name in body_names:
            if (
                self.env.robot_config.mimic_small_marker_bodies is not None
                and body_name in self.env.robot_config.mimic_small_marker_bodies
            ):
                body_markers.append(MarkerConfig(size="small"))
            else:
                body_markers.append(MarkerConfig(size="regular"))
        
        # Red markers for target poses
        body_markers_red_cfg = VisualizationMarkerConfig(
            type="sphere", color=(1.0, 0.0, 0.0), markers=body_markers
        )
        visualization_markers["body_markers_red"] = body_markers_red_cfg
        
        return visualization_markers
    
    def get_markers_state(self) -> Dict[str, MarkerState]:
        """Compute marker positions for reference poses.
        
        Returns:
            Dictionary mapping marker names to MarkerState.
        """
        if self.env.simulator.headless:
            return {}
        
        markers_state = {}
        
        # Get reference state at current time (access motion_manager via env)
        ref_state = self.env.motion_lib.get_motion_state(
            self.env.motion_manager.motion_ids, self.env.motion_manager.motion_times
        )
        
        target_pos = ref_state.rigid_body_pos.clone()
        target_pos += (
            self.env.get_spawn_to_ref_pose_offset_with_terrain_height_correction(
                target_pos
            )
        )
        
        # Standard mimic: show all body markers in red
        target_pos = target_pos.view(self.env.num_envs, -1, 3)
        markers_state["body_markers_red"] = MarkerState(
            translation=target_pos,
            orientation=torch.zeros(
                self.env.num_envs, target_pos.shape[1], 4, device=self.env.device
            ),
        )
        
        return markers_state
