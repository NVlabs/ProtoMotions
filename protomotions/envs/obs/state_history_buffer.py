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
"""State history buffer for storing historical robot state tensors.

This module provides the StateHistoryBuffer class that stores raw 4D tensors
of historical robot state (not RobotState objects) to enable ONNX-compilable
historical observation functions.

The buffer can store both clean (privileged) and noisy observations to support
asymmetric actor-critic training with observation noise.
"""

from typing import Dict, Optional

import torch
from torch import Tensor


class StateHistoryBuffer:
    """Buffer for storing current + historical robot state as raw 4D tensors.
    
    Stores robot state tensors with shape [envs, num_history_steps + 1, ...].
    The buffer holds both current state (index 0) and historical states (index 1+).
    Uses raw tensors instead of RobotState objects to avoid 3D indexing assumptions
    and enable ONNX compilation of historical observation functions.
    
    Buffer layout after rotate_and_update():
        - Index 0: current state (t)
        - Index 1: previous state (t - dt)
        - Index 2: t - 2*dt
        - ... up to index num_history_steps
    
    The historical_* properties return [:, 1:] which gives exactly num_history_steps
    elements (excluding current).
    
    When store_noisy=True, additional noisy_* buffers are allocated to store
    observations with noise applied, enabling asymmetric actor-critic training.
    The main buffers store clean (privileged) data, noisy buffers store noisy data.
    
    Args:
        num_envs: Number of parallel environments.
        num_history_steps: Number of historical timesteps to store (not including current).
        num_bodies: Number of rigid bodies in the robot.
        num_dofs: Number of degrees of freedom.
        action_dim: Dimension of action space.
        device: Device for tensor storage.
        store_noisy: If True, allocate additional buffers for noisy observations.
    
    Attributes:
        rigid_body_pos: Buffer [envs, num_history_steps + 1, bodies, 3].
        rigid_body_rot: Buffer [envs, num_history_steps + 1, bodies, 4].
        rigid_body_vel: Buffer [envs, num_history_steps + 1, bodies, 3].
        rigid_body_ang_vel: Buffer [envs, num_history_steps + 1, bodies, 3].
        dof_pos: Buffer [envs, num_history_steps + 1, num_dofs].
        dof_vel: Buffer [envs, num_history_steps + 1, num_dofs].
        actions: Buffer [envs, num_history_steps + 1, action_dim].
        historical_*: Properties returning [:, 1:] with exactly num_history_steps elements.
        noisy_*: Optional noisy versions of buffers (only if store_noisy=True).
    """
    
    def __init__(
        self,
        num_envs: int,
        num_history_steps: int,
        num_bodies: int,
        num_dofs: int,
        action_dim: int,
        num_contact_bodies: int,
        anchor_body_index: int,
        device: torch.device,
        store_noisy: bool = False,
    ):
        self.num_envs = num_envs
        self.num_history_steps = num_history_steps
        self.num_bodies = num_bodies
        self.num_dofs = num_dofs
        self.action_dim = action_dim
        self.num_contact_bodies = num_contact_bodies
        self.anchor_body_index = anchor_body_index
        self.device = device
        self.store_noisy = store_noisy
        
        buffer_size = num_history_steps + 1
        
        # Clean (privileged) buffers - always allocated
        self.rigid_body_pos = torch.zeros(
            num_envs, buffer_size, num_bodies, 3,
            dtype=torch.float, device=device
        )
        self.rigid_body_rot = torch.zeros(
            num_envs, buffer_size, num_bodies, 4,
            dtype=torch.float, device=device
        )
        self.rigid_body_vel = torch.zeros(
            num_envs, buffer_size, num_bodies, 3,
            dtype=torch.float, device=device
        )
        self.rigid_body_ang_vel = torch.zeros(
            num_envs, buffer_size, num_bodies, 3,
            dtype=torch.float, device=device
        )
        self.dof_pos = torch.zeros(
            num_envs, buffer_size, num_dofs,
            dtype=torch.float, device=device
        )
        self.dof_vel = torch.zeros(
            num_envs, buffer_size, num_dofs,
            dtype=torch.float, device=device
        )
        self.actions = torch.zeros(
            num_envs, buffer_size, action_dim,
            dtype=torch.float, device=device
        )
        self.ground_heights = torch.zeros(
            num_envs, buffer_size,
            dtype=torch.float, device=device
        )
        self.body_contacts = torch.zeros(
            num_envs, buffer_size, num_contact_bodies,
            dtype=torch.bool, device=device
        )
        
        # Noisy buffers - only allocated if store_noisy=True
        if store_noisy:
            self.noisy_rigid_body_pos = torch.zeros(
                num_envs, buffer_size, num_bodies, 3,
                dtype=torch.float, device=device
            )
            self.noisy_rigid_body_rot = torch.zeros(
                num_envs, buffer_size, num_bodies, 4,
                dtype=torch.float, device=device
            )
            self.noisy_rigid_body_vel = torch.zeros(
                num_envs, buffer_size, num_bodies, 3,
                dtype=torch.float, device=device
            )
            self.noisy_rigid_body_ang_vel = torch.zeros(
                num_envs, buffer_size, num_bodies, 3,
                dtype=torch.float, device=device
            )
            self.noisy_dof_pos = torch.zeros(
                num_envs, buffer_size, num_dofs,
                dtype=torch.float, device=device
            )
            self.noisy_dof_vel = torch.zeros(
                num_envs, buffer_size, num_dofs,
                dtype=torch.float, device=device
            )
            self.noisy_ground_heights = torch.zeros(
                num_envs, buffer_size,
                dtype=torch.float, device=device
            )
        else:
            # When not storing noisy, point to clean buffers for memory efficiency
            self.noisy_rigid_body_pos = self.rigid_body_pos
            self.noisy_rigid_body_rot = self.rigid_body_rot
            self.noisy_rigid_body_vel = self.rigid_body_vel
            self.noisy_rigid_body_ang_vel = self.rigid_body_ang_vel
            self.noisy_dof_pos = self.dof_pos
            self.noisy_dof_vel = self.dof_vel
            self.noisy_ground_heights = self.ground_heights
    
    @property
    def historical_rigid_body_pos(self) -> Tensor:
        """Historical body positions [envs, history_steps-1, bodies, 3]."""
        return self.rigid_body_pos[:, 1:]
    
    @property
    def historical_rigid_body_rot(self) -> Tensor:
        """Historical body rotations [envs, history_steps-1, bodies, 4]."""
        return self.rigid_body_rot[:, 1:]
    
    @property
    def historical_rigid_body_vel(self) -> Tensor:
        """Historical body velocities [envs, history_steps-1, bodies, 3]."""
        return self.rigid_body_vel[:, 1:]
    
    @property
    def historical_rigid_body_ang_vel(self) -> Tensor:
        """Historical body angular velocities [envs, history_steps-1, bodies, 3]."""
        return self.rigid_body_ang_vel[:, 1:]
    
    @property
    def historical_dof_pos(self) -> Tensor:
        """Historical DOF positions [envs, history_steps-1, num_dofs]."""
        return self.dof_pos[:, 1:]
    
    @property
    def historical_dof_vel(self) -> Tensor:
        """Historical DOF velocities [envs, history_steps-1, num_dofs]."""
        return self.dof_vel[:, 1:]
    
    @property
    def historical_actions(self) -> Tensor:
        """Historical actions [envs, history_steps-1, action_dim]."""
        return self.actions[:, 1:]
    
    @property
    def historical_ground_heights(self) -> Tensor:
        """Historical ground heights beneath root [envs, history_steps-1]."""
        return self.ground_heights[:, 1:]
    
    @property
    def historical_body_contacts(self) -> Tensor:
        """Historical body contacts [envs, history_steps-1, num_contact_bodies]."""
        return self.body_contacts[:, 1:]
    
    @property
    def historical_root_pos(self) -> Tensor:
        """Historical root positions [envs, history_steps-1, 3]."""
        return self.rigid_body_pos[:, 1:, 0, :]
    
    @property
    def historical_root_rot(self) -> Tensor:
        """Historical root rotations [envs, history_steps-1, 4]."""
        return self.rigid_body_rot[:, 1:, 0, :]
    
    @property
    def historical_root_ang_vel(self) -> Tensor:
        """Historical root angular velocities [envs, history_steps-1, 3]."""
        return self.rigid_body_ang_vel[:, 1:, 0, :]
    
    @property
    def historical_anchor_pos(self) -> Tensor:
        """Historical anchor body positions [envs, history_steps-1, 3]."""
        return self.rigid_body_pos[:, 1:, self.anchor_body_index, :]
    
    @property
    def historical_anchor_rot(self) -> Tensor:
        """Historical anchor body rotations [envs, history_steps-1, 4]."""
        return self.rigid_body_rot[:, 1:, self.anchor_body_index, :]
    
    # =========================================================================
    # Noisy historical properties (for actor with observation noise)
    # =========================================================================
    
    @property
    def noisy_historical_rigid_body_pos(self) -> Tensor:
        """Noisy historical body positions [envs, history_steps-1, bodies, 3]."""
        return self.noisy_rigid_body_pos[:, 1:]
    
    @property
    def noisy_historical_rigid_body_rot(self) -> Tensor:
        """Noisy historical body rotations [envs, history_steps-1, bodies, 4]."""
        return self.noisy_rigid_body_rot[:, 1:]
    
    @property
    def noisy_historical_rigid_body_vel(self) -> Tensor:
        """Noisy historical body velocities [envs, history_steps-1, bodies, 3]."""
        return self.noisy_rigid_body_vel[:, 1:]
    
    @property
    def noisy_historical_rigid_body_ang_vel(self) -> Tensor:
        """Noisy historical body angular velocities [envs, history_steps-1, bodies, 3]."""
        return self.noisy_rigid_body_ang_vel[:, 1:]
    
    @property
    def noisy_historical_dof_pos(self) -> Tensor:
        """Noisy historical DOF positions [envs, history_steps-1, num_dofs]."""
        return self.noisy_dof_pos[:, 1:]
    
    @property
    def noisy_historical_dof_vel(self) -> Tensor:
        """Noisy historical DOF velocities [envs, history_steps-1, num_dofs]."""
        return self.noisy_dof_vel[:, 1:]
    
    @property
    def noisy_historical_root_pos(self) -> Tensor:
        """Noisy historical root positions [envs, history_steps-1, 3]."""
        return self.noisy_rigid_body_pos[:, 1:, 0, :]
    
    @property
    def noisy_historical_root_rot(self) -> Tensor:
        """Noisy historical root rotations [envs, history_steps-1, 4]."""
        return self.noisy_rigid_body_rot[:, 1:, 0, :]
    
    @property
    def noisy_historical_root_ang_vel(self) -> Tensor:
        """Noisy historical root angular velocities [envs, history_steps-1, 3]."""
        return self.noisy_rigid_body_ang_vel[:, 1:, 0, :]
    
    @property
    def noisy_historical_anchor_pos(self) -> Tensor:
        """Noisy historical anchor body positions [envs, history_steps-1, 3]."""
        return self.noisy_rigid_body_pos[:, 1:, self.anchor_body_index, :]
    
    @property
    def noisy_historical_anchor_rot(self) -> Tensor:
        """Noisy historical anchor body rotations [envs, history_steps-1, 4]."""
        return self.noisy_rigid_body_rot[:, 1:, self.anchor_body_index, :]
    
    @property
    def noisy_historical_ground_heights(self) -> Tensor:
        """Noisy historical ground heights [envs, history_steps-1]."""
        return self.noisy_ground_heights[:, 1:]
    
    @property
    def historical_anchor_vel(self) -> Tensor:
        """Historical anchor body velocities [envs, history_steps-1, 3]."""
        return self.rigid_body_vel[:, 1:, self.anchor_body_index, :]
    
    @property
    def historical_anchor_ang_vel(self) -> Tensor:
        """Historical anchor body angular velocities [envs, history_steps-1, 3]."""
        return self.rigid_body_ang_vel[:, 1:, self.anchor_body_index, :]
    
    @torch.no_grad()
    def rotate_and_update(
        self,
        rigid_body_pos: Tensor,
        rigid_body_rot: Tensor,
        rigid_body_vel: Tensor,
        rigid_body_ang_vel: Tensor,
        dof_pos: Tensor,
        dof_vel: Tensor,
        actions: Tensor,
        ground_heights: Tensor,
        body_contacts: Tensor,
        noisy_rigid_body_pos: Optional[Tensor] = None,
        noisy_rigid_body_rot: Optional[Tensor] = None,
        noisy_rigid_body_vel: Optional[Tensor] = None,
        noisy_rigid_body_ang_vel: Optional[Tensor] = None,
        noisy_dof_pos: Optional[Tensor] = None,
        noisy_dof_vel: Optional[Tensor] = None,
        noisy_ground_heights: Optional[Tensor] = None,
    ):
        """Rotate history buffer and insert current state at the front.
        
        Shifts all history one step back (discarding oldest) and inserts
        the current state at index 0.
        
        Args:
            rigid_body_pos: Current body positions [envs, bodies, 3] (clean/privileged).
            rigid_body_rot: Current body rotations [envs, bodies, 4] (clean/privileged).
            rigid_body_vel: Current body velocities [envs, bodies, 3] (clean/privileged).
            rigid_body_ang_vel: Current body angular velocities [envs, bodies, 3] (clean/privileged).
            dof_pos: Current DOF positions [envs, num_dofs] (clean/privileged).
            dof_vel: Current DOF velocities [envs, num_dofs] (clean/privileged).
            actions: Current actions [envs, action_dim].
            ground_heights: Ground heights beneath root [envs] (clean/privileged).
            body_contacts: Body contact flags [envs, num_contact_bodies].
            noisy_rigid_body_pos: Optional noisy body positions [envs, bodies, 3].
            noisy_rigid_body_rot: Optional noisy body rotations [envs, bodies, 4].
            noisy_rigid_body_vel: Optional noisy body velocities [envs, bodies, 3].
            noisy_rigid_body_ang_vel: Optional noisy body angular velocities [envs, bodies, 3].
            noisy_dof_pos: Optional noisy DOF positions [envs, num_dofs].
            noisy_dof_vel: Optional noisy DOF velocities [envs, num_dofs].
            noisy_ground_heights: Optional noisy ground heights [envs].
        """
        # Roll all clean tensors: shift history back by 1 (index 0 becomes 1, etc.)
        self.rigid_body_pos = self.rigid_body_pos.roll(shifts=1, dims=1)
        self.rigid_body_rot = self.rigid_body_rot.roll(shifts=1, dims=1)
        self.rigid_body_vel = self.rigid_body_vel.roll(shifts=1, dims=1)
        self.rigid_body_ang_vel = self.rigid_body_ang_vel.roll(shifts=1, dims=1)
        self.dof_pos = self.dof_pos.roll(shifts=1, dims=1)
        self.dof_vel = self.dof_vel.roll(shifts=1, dims=1)
        self.actions = self.actions.roll(shifts=1, dims=1)
        self.ground_heights = self.ground_heights.roll(shifts=1, dims=1)
        self.body_contacts = self.body_contacts.roll(shifts=1, dims=1)
        
        # Insert clean data at front
        self.rigid_body_pos[:, 0] = rigid_body_pos
        self.rigid_body_rot[:, 0] = rigid_body_rot
        self.rigid_body_vel[:, 0] = rigid_body_vel
        self.rigid_body_ang_vel[:, 0] = rigid_body_ang_vel
        self.dof_pos[:, 0] = dof_pos
        self.dof_vel[:, 0] = dof_vel
        self.actions[:, 0] = actions
        self.ground_heights[:, 0] = ground_heights
        self.body_contacts[:, 0] = body_contacts
        
        # Handle noisy buffers if store_noisy is enabled
        if self.store_noisy:
            self.noisy_rigid_body_pos = self.noisy_rigid_body_pos.roll(shifts=1, dims=1)
            self.noisy_rigid_body_rot = self.noisy_rigid_body_rot.roll(shifts=1, dims=1)
            self.noisy_rigid_body_vel = self.noisy_rigid_body_vel.roll(shifts=1, dims=1)
            self.noisy_rigid_body_ang_vel = self.noisy_rigid_body_ang_vel.roll(shifts=1, dims=1)
            self.noisy_dof_pos = self.noisy_dof_pos.roll(shifts=1, dims=1)
            self.noisy_dof_vel = self.noisy_dof_vel.roll(shifts=1, dims=1)
            self.noisy_ground_heights = self.noisy_ground_heights.roll(shifts=1, dims=1)
            
            # Insert noisy data at front (use clean if noisy not provided)
            self.noisy_rigid_body_pos[:, 0] = noisy_rigid_body_pos if noisy_rigid_body_pos is not None else rigid_body_pos
            self.noisy_rigid_body_rot[:, 0] = noisy_rigid_body_rot if noisy_rigid_body_rot is not None else rigid_body_rot
            self.noisy_rigid_body_vel[:, 0] = noisy_rigid_body_vel if noisy_rigid_body_vel is not None else rigid_body_vel
            self.noisy_rigid_body_ang_vel[:, 0] = noisy_rigid_body_ang_vel if noisy_rigid_body_ang_vel is not None else rigid_body_ang_vel
            self.noisy_dof_pos[:, 0] = noisy_dof_pos if noisy_dof_pos is not None else dof_pos
            self.noisy_dof_vel[:, 0] = noisy_dof_vel if noisy_dof_vel is not None else dof_vel
            self.noisy_ground_heights[:, 0] = noisy_ground_heights if noisy_ground_heights is not None else ground_heights
    
    @torch.no_grad()
    def reset_from_states(
        self,
        env_ids: Tensor,
        rigid_body_pos: Tensor,
        rigid_body_rot: Tensor,
        rigid_body_vel: Tensor,
        rigid_body_ang_vel: Tensor,
        dof_pos: Tensor,
        dof_vel: Tensor,
        ground_heights: Tensor,
        body_contacts: Tensor,
        actions: Optional[Tensor] = None,
    ):
        """Reset history for specified environments from historical state tensors.
        
        Used for reset-from-reference where we have historical states from motion_lib.
        Noisy buffers are also reset to clean data (no noise on reset).
        
        Args:
            env_ids: Environment indices to reset.
            rigid_body_pos: Historical body positions [len(env_ids), steps, bodies, 3].
            rigid_body_rot: Historical body rotations [len(env_ids), steps, bodies, 4].
            rigid_body_vel: Historical body velocities [len(env_ids), steps, bodies, 3].
            rigid_body_ang_vel: Historical body angular velocities [len(env_ids), steps, bodies, 3].
            dof_pos: Historical DOF positions [len(env_ids), steps, num_dofs].
            dof_vel: Historical DOF velocities [len(env_ids), steps, num_dofs].
            ground_heights: Historical ground heights [len(env_ids), steps].
            body_contacts: Historical body contacts [len(env_ids), steps, num_contact_bodies].
            actions: Historical actions [len(env_ids), steps, action_dim] or None to zero.
        """
        self.rigid_body_pos[env_ids] = rigid_body_pos
        self.rigid_body_rot[env_ids] = rigid_body_rot
        self.rigid_body_vel[env_ids] = rigid_body_vel
        self.rigid_body_ang_vel[env_ids] = rigid_body_ang_vel
        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel
        self.ground_heights[env_ids] = ground_heights
        self.body_contacts[env_ids] = body_contacts
        
        if actions is not None:
            self.actions[env_ids] = actions
        else:
            self.actions[env_ids] = 0.0
        
        # Reset noisy buffers to clean data (no noise on reset)
        if self.store_noisy:
            self.noisy_rigid_body_pos[env_ids] = rigid_body_pos
            self.noisy_rigid_body_rot[env_ids] = rigid_body_rot
            self.noisy_rigid_body_vel[env_ids] = rigid_body_vel
            self.noisy_rigid_body_ang_vel[env_ids] = rigid_body_ang_vel
            self.noisy_dof_pos[env_ids] = dof_pos
            self.noisy_dof_vel[env_ids] = dof_vel
            self.noisy_ground_heights[env_ids] = ground_heights
    
    @torch.no_grad()
    def reset_from_single_state(
        self,
        env_ids: Tensor,
        rigid_body_pos: Tensor,
        rigid_body_rot: Tensor,
        rigid_body_vel: Tensor,
        rigid_body_ang_vel: Tensor,
        dof_pos: Tensor,
        dof_vel: Tensor,
        ground_heights: Tensor,
        body_contacts: Tensor,
    ):
        """Reset history for specified environments by repeating a single state.
        
        Used for default reset where we just copy the current state to all buffer slots.
        Buffer has num_history_steps + 1 slots (current + history).
        Noisy buffers are also reset to clean data (no noise on reset).
        
        Args:
            env_ids: Environment indices to reset.
            rigid_body_pos: Current body positions [len(env_ids), bodies, 3].
            rigid_body_rot: Current body rotations [len(env_ids), bodies, 4].
            rigid_body_vel: Current body velocities [len(env_ids), bodies, 3].
            rigid_body_ang_vel: Current body angular velocities [len(env_ids), bodies, 3].
            dof_pos: Current DOF positions [len(env_ids), num_dofs].
            dof_vel: Current DOF velocities [len(env_ids), num_dofs].
            ground_heights: Current ground heights [len(env_ids)].
            body_contacts: Current body contacts [len(env_ids), num_contact_bodies].
        """
        buffer_size = self.rigid_body_pos.shape[1]
        
        # Expand tensors for clean buffers
        expanded_rigid_body_pos = rigid_body_pos.unsqueeze(1).expand(-1, buffer_size, -1, -1)
        expanded_rigid_body_rot = rigid_body_rot.unsqueeze(1).expand(-1, buffer_size, -1, -1)
        expanded_rigid_body_vel = rigid_body_vel.unsqueeze(1).expand(-1, buffer_size, -1, -1)
        expanded_rigid_body_ang_vel = rigid_body_ang_vel.unsqueeze(1).expand(-1, buffer_size, -1, -1)
        expanded_dof_pos = dof_pos.unsqueeze(1).expand(-1, buffer_size, -1)
        expanded_dof_vel = dof_vel.unsqueeze(1).expand(-1, buffer_size, -1)
        
        self.rigid_body_pos[env_ids] = expanded_rigid_body_pos
        self.rigid_body_rot[env_ids] = expanded_rigid_body_rot
        self.rigid_body_vel[env_ids] = expanded_rigid_body_vel
        self.rigid_body_ang_vel[env_ids] = expanded_rigid_body_ang_vel
        self.dof_pos[env_ids] = expanded_dof_pos
        self.dof_vel[env_ids] = expanded_dof_vel
        self.ground_heights[env_ids] = ground_heights.unsqueeze(1).expand(-1, buffer_size)
        self.body_contacts[env_ids] = body_contacts.unsqueeze(1).expand(-1, buffer_size, -1)
        self.actions[env_ids] = 0.0
        
        # Reset noisy buffers to clean data (no noise on reset)
        if self.store_noisy:
            self.noisy_rigid_body_pos[env_ids] = expanded_rigid_body_pos
            self.noisy_rigid_body_rot[env_ids] = expanded_rigid_body_rot
            self.noisy_rigid_body_vel[env_ids] = expanded_rigid_body_vel
            self.noisy_rigid_body_ang_vel[env_ids] = expanded_rigid_body_ang_vel
            self.noisy_dof_pos[env_ids] = expanded_dof_pos
            self.noisy_dof_vel[env_ids] = expanded_dof_vel
            self.noisy_ground_heights[env_ids] = ground_heights.unsqueeze(1).expand(-1, buffer_size)
    
    def save_state(self) -> Dict[str, Tensor]:
        """Save the entire buffer state for later restoration.
        
        Returns a dictionary containing cloned copies of all buffer tensors.
        Use with load_state() to cache and restore buffer state.
        
        Returns:
            Dictionary mapping tensor names to cloned tensors.
        """
        state = {
            'rigid_body_pos': self.rigid_body_pos.clone(),
            'rigid_body_rot': self.rigid_body_rot.clone(),
            'rigid_body_vel': self.rigid_body_vel.clone(),
            'rigid_body_ang_vel': self.rigid_body_ang_vel.clone(),
            'dof_pos': self.dof_pos.clone(),
            'dof_vel': self.dof_vel.clone(),
            'actions': self.actions.clone(),
            'ground_heights': self.ground_heights.clone(),
            'body_contacts': self.body_contacts.clone(),
            'store_noisy': self.store_noisy,
        }
        if self.store_noisy:
            state['noisy_rigid_body_pos'] = self.noisy_rigid_body_pos.clone()
            state['noisy_rigid_body_rot'] = self.noisy_rigid_body_rot.clone()
            state['noisy_rigid_body_vel'] = self.noisy_rigid_body_vel.clone()
            state['noisy_rigid_body_ang_vel'] = self.noisy_rigid_body_ang_vel.clone()
            state['noisy_dof_pos'] = self.noisy_dof_pos.clone()
            state['noisy_dof_vel'] = self.noisy_dof_vel.clone()
            state['noisy_ground_heights'] = self.noisy_ground_heights.clone()
        return state
    
    def load_state(self, state: Dict[str, Tensor]) -> None:
        """Restore buffer state from a previously saved state.
        
        Args:
            state: Dictionary from save_state() containing buffer tensors.
        """
        self.rigid_body_pos.copy_(state['rigid_body_pos'])
        self.rigid_body_rot.copy_(state['rigid_body_rot'])
        self.rigid_body_vel.copy_(state['rigid_body_vel'])
        self.rigid_body_ang_vel.copy_(state['rigid_body_ang_vel'])
        self.dof_pos.copy_(state['dof_pos'])
        self.dof_vel.copy_(state['dof_vel'])
        self.actions.copy_(state['actions'])
        self.ground_heights.copy_(state['ground_heights'])
        self.body_contacts.copy_(state['body_contacts'])
        if self.store_noisy and state.get('store_noisy', False):
            self.noisy_rigid_body_pos.copy_(state['noisy_rigid_body_pos'])
            self.noisy_rigid_body_rot.copy_(state['noisy_rigid_body_rot'])
            self.noisy_rigid_body_vel.copy_(state['noisy_rigid_body_vel'])
            self.noisy_rigid_body_ang_vel.copy_(state['noisy_rigid_body_ang_vel'])
            self.noisy_dof_pos.copy_(state['noisy_dof_pos'])
            self.noisy_dof_vel.copy_(state['noisy_dof_vel'])
            self.noisy_ground_heights.copy_(state['noisy_ground_heights'])

