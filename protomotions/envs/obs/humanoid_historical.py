# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
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
"""Compute helper functions for humanoid observations.

These are tensor-param helper functions that perform the actual observation
computations. They are separated from the ctx-taking wrappers in
humanoid_obs_functions.py to keep the code organization clean.
"""

from typing import List, Union

import torch
from torch import Tensor

from protomotions.envs.obs.humanoid import (
    compute_humanoid_max_coords_observations,
    compute_humanoid_reduced_coords_observations,
)
from protomotions.envs.obs.utils import select_step_indices


def compute_historical_time_offsets(
    num_state_history_steps: int,
    dt: float,
    num_envs: int,
    device: str,
    history_steps: Union[int, List[int]] = None,
) -> Tensor:
    """Compute time offsets for historical observations.
    
    Returns time offsets (negative, going back in time) for each selected history step.
    Combine with historical pose observations via concat for temporal conditioning.
    
    Args:
        num_state_history_steps: Total steps stored in history buffer.
        dt: Environment timestep.
        num_envs: Number of environments.
        device: Torch device.
        history_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
    
    Returns:
        Time offsets [num_envs, num_steps]. Uses 0-indexed buffer positions * dt,
        so step 1 -> 0*dt, step 9 -> 8*dt, etc. (matches original subsampling behavior).
    """
    # Create tensor of 1-indexed step numbers
    if history_steps is None:
        step_indices = torch.arange(1, num_state_history_steps + 1, device=device, dtype=torch.float32)
    elif isinstance(history_steps, int):
        step_indices = torch.arange(1, history_steps + 1, device=device, dtype=torch.float32)
    else:
        step_indices = torch.tensor(history_steps, device=device, dtype=torch.float32)
    
    # Convert to 0-indexed buffer positions, then multiply by dt
    time_offsets = (step_indices - 1) * dt
    return time_offsets.unsqueeze(0).expand(num_envs, -1)


def compute_historical_reduced_coords_from_state(
    historical_dof_pos: Tensor,
    historical_dof_vel: Tensor,
    historical_root_rot: Tensor,
    historical_root_local_ang_vel: Tensor,
    historical_anchor_rot: Tensor,
    w_last: bool = True,
    history_steps: Union[int, List[int]] = None,
) -> Tensor:
    """Compute historical reduced_coords observations from state history tensors.
    
    Takes 4D historical state tensors, flattens them for the existing observation
    function, then reshapes back to [envs, history_steps * obs_dim].
    
    Args:
        historical_dof_pos: Historical DOF positions [envs, history_steps, num_dofs].
        historical_dof_vel: Historical DOF velocities [envs, history_steps, num_dofs].
        historical_root_rot: Historical root rotation [envs, history_steps, 4].
        historical_root_local_ang_vel: Historical root angular velocity in local frame [envs, history_steps, 3].
        historical_anchor_rot: Historical anchor rotation [envs, history_steps, 4].
        w_last: Whether quaternions use w-last format.
        history_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
    
    Returns:
        Flattened historical observations [envs, history_steps * obs_dim].
    
    Raises:
        ValueError: If historical tensors are None (state history not enabled).
    """
    if historical_dof_pos is None:
        raise ValueError(
            "Historical state tensors are None. "
            "Set env_config.num_state_history_steps > 0 to enable state history buffer."
        )
    
    num_envs = historical_dof_pos.shape[0]
    
    if history_steps is None:
        dof_pos = historical_dof_pos
        dof_vel = historical_dof_vel
        root_rot = historical_root_rot
        root_local_ang_vel = historical_root_local_ang_vel
        anchor_rot = historical_anchor_rot
    else:
        dof_pos = select_step_indices(historical_dof_pos, history_steps)
        dof_vel = select_step_indices(historical_dof_vel, history_steps)
        root_rot = select_step_indices(historical_root_rot, history_steps)
        root_local_ang_vel = select_step_indices(historical_root_local_ang_vel, history_steps)
        anchor_rot = select_step_indices(historical_anchor_rot, history_steps)
    
    actual_steps = dof_pos.shape[1]
    
    flat_dof_pos = dof_pos.reshape(-1, dof_pos.shape[-1])
    flat_dof_vel = dof_vel.reshape(-1, dof_vel.shape[-1])
    flat_root_rot = root_rot.reshape(-1, 4)
    flat_root_local_ang_vel = root_local_ang_vel.reshape(-1, 3)
    flat_anchor_rot = anchor_rot.reshape(-1, 4)
    
    flat_obs = compute_humanoid_reduced_coords_observations(
        dof_pos=flat_dof_pos,
        dof_vel=flat_dof_vel,
        root_rot=flat_root_rot,
        root_local_ang_vel=flat_root_local_ang_vel,
        anchor_rot=flat_anchor_rot,
        w_last=w_last,
    )
    
    obs_dim = flat_obs.shape[-1]
    return flat_obs.view(num_envs, actual_steps * obs_dim)

# Context mapping for ONNX export


def compute_historical_max_coords_from_state(
    historical_rigid_body_pos: Tensor,
    historical_rigid_body_rot: Tensor,
    historical_rigid_body_vel: Tensor,
    historical_rigid_body_ang_vel: Tensor,
    historical_ground_heights: Tensor,
    historical_body_contacts: Tensor,
    local_obs: bool = True,
    root_height_obs: bool = True,
    observe_contacts: bool = False,
    w_last: bool = True,
    history_steps: Union[int, List[int]] = None,
) -> Tensor:
    """Compute historical max_coords observations from state history tensors.
    
    Args:
        historical_rigid_body_pos: Historical body positions [envs, history_steps, bodies, 3].
        historical_rigid_body_rot: Historical body rotations [envs, history_steps, bodies, 4].
        historical_rigid_body_vel: Historical body velocities [envs, history_steps, bodies, 3].
        historical_rigid_body_ang_vel: Historical body angular velocities [envs, history_steps, bodies, 3].
        historical_ground_heights: Historical ground heights [envs, history_steps].
        historical_body_contacts: Historical body contacts [envs, history_steps, num_contact_bodies].
        local_obs: Whether to use local coordinate frame.
        root_height_obs: Whether to include root height.
        observe_contacts: Whether to include contact observations.
        w_last: Whether quaternions use w-last format.
        history_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
    
    Returns:
        Flattened historical observations [envs, history_steps * obs_dim].
    
    Raises:
        ValueError: If historical tensors are None (state history not enabled).
    """
    if historical_rigid_body_pos is None:
        raise ValueError(
            "Historical state tensors are None. "
            "Set env_config.num_state_history_steps > 0 to enable state history buffer."
        )
    
    num_envs = historical_rigid_body_pos.shape[0]
    num_bodies = historical_rigid_body_pos.shape[2]
    num_contact_bodies = historical_body_contacts.shape[2]
    
    if history_steps is None:
        body_pos = historical_rigid_body_pos
        body_rot = historical_rigid_body_rot
        body_vel = historical_rigid_body_vel
        body_ang_vel = historical_rigid_body_ang_vel
        ground_heights = historical_ground_heights
        body_contacts = historical_body_contacts
    else:
        body_pos = select_step_indices(historical_rigid_body_pos, history_steps)
        body_rot = select_step_indices(historical_rigid_body_rot, history_steps)
        body_vel = select_step_indices(historical_rigid_body_vel, history_steps)
        body_ang_vel = select_step_indices(historical_rigid_body_ang_vel, history_steps)
        ground_heights = select_step_indices(historical_ground_heights, history_steps)
        body_contacts = select_step_indices(historical_body_contacts, history_steps)
    
    actual_steps = body_pos.shape[1]
    batch_size = num_envs * actual_steps
    flat_body_pos = body_pos.reshape(batch_size, num_bodies, 3)
    flat_body_rot = body_rot.reshape(batch_size, num_bodies, 4)
    flat_body_vel = body_vel.reshape(batch_size, num_bodies, 3)
    flat_body_ang_vel = body_ang_vel.reshape(batch_size, num_bodies, 3)
    flat_ground_height = ground_heights.reshape(batch_size, 1)
    flat_contacts = body_contacts.reshape(batch_size, num_contact_bodies)
    
    flat_obs = compute_humanoid_max_coords_observations(
        body_pos=flat_body_pos,
        body_rot=flat_body_rot,
        body_vel=flat_body_vel,
        body_ang_vel=flat_body_ang_vel,
        ground_height=flat_ground_height,
        body_contacts=flat_contacts,
        local_obs=local_obs,
        root_height_obs=root_height_obs,
        observe_contacts=observe_contacts,
        w_last=w_last,
    )
    
    obs_dim = flat_obs.shape[-1]
    return flat_obs.view(num_envs, actual_steps * obs_dim)

# Context mapping for ONNX export


def compute_historical_actions_from_state(
    historical_actions: Tensor,
    history_steps: Union[int, List[int]] = None,
) -> Tensor:
    """Compute flattened historical actions from state history.
    
    Args:
        historical_actions: Historical actions [envs, history_steps, action_dim].
        history_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 3, 5]). None = use all.
    
    Returns:
        Flattened historical actions [envs, history_steps * action_dim].
    
    Raises:
        ValueError: If historical tensors are None (state history not enabled).
    """
    if historical_actions is None:
        raise ValueError(
            "Historical actions tensor is None. "
            "Set env_config.num_state_history_steps > 0 to enable state history buffer."
        )
    
    num_envs = historical_actions.shape[0]
    
    if history_steps is None:
        actions = historical_actions
    else:
        actions = select_step_indices(historical_actions, history_steps)
    
    actual_steps = actions.shape[1]
    action_dim = actions.shape[2]
    
    return actions.reshape(num_envs, actual_steps * action_dim)


def compute_historical_poses_with_time(
    historical_rigid_body_pos: Tensor,
    historical_rigid_body_rot: Tensor,
    historical_rigid_body_vel: Tensor,
    historical_rigid_body_ang_vel: Tensor,
    historical_ground_heights: Tensor,
    historical_body_contacts: Tensor,
    history_steps: Union[int, List[int]],
    local_obs: bool,
    root_height_obs: bool,
    w_last: bool,
    dt: float,
) -> Tensor:
    """Compute historical pose observations with time offsets from raw state.
    
    Wraps compute_historical_max_coords_from_state and compute_historical_time_offsets,
    concatenating poses with time per step for backward compatibility.
    
    Args:
        historical_rigid_body_pos: Historical body positions [envs, history_steps, bodies, 3].
        historical_rigid_body_rot: Historical body rotations [envs, history_steps, bodies, 4].
        historical_rigid_body_vel: Historical body velocities [envs, history_steps, bodies, 3].
        historical_rigid_body_ang_vel: Historical body angular velocities [envs, history_steps, bodies, 3].
        historical_ground_heights: Historical ground heights [envs, history_steps].
        historical_body_contacts: Historical body contacts [envs, history_steps, num_contact_bodies].
        history_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 9, 17, ...]).
        local_obs: Whether to use local coordinate frame.
        root_height_obs: Whether to include root height.
        w_last: Whether quaternions use w-last format.
        dt: Environment timestep.
    
    Returns:
        Historical poses with time [num_envs, num_steps * (obs_dim + 1)].
    """
    num_envs = historical_rigid_body_pos.shape[0]
    num_state_history_steps = historical_rigid_body_pos.shape[1]
    device = historical_rigid_body_pos.device
    
    # Get historical poses using the modular function
    poses_flat = compute_historical_max_coords_from_state(
        historical_rigid_body_pos=historical_rigid_body_pos,
        historical_rigid_body_rot=historical_rigid_body_rot,
        historical_rigid_body_vel=historical_rigid_body_vel,
        historical_rigid_body_ang_vel=historical_rigid_body_ang_vel,
        historical_ground_heights=historical_ground_heights,
        historical_body_contacts=historical_body_contacts,
        local_obs=local_obs,
        root_height_obs=root_height_obs,
        observe_contacts=False,
        w_last=w_last,
        history_steps=history_steps,
    )
    
    # Get time offsets using the modular function
    time_offsets = compute_historical_time_offsets(
        num_state_history_steps=num_state_history_steps,
        dt=dt,
        num_envs=num_envs,
        device=device,
        history_steps=history_steps,
    )
    
    # Compute num_steps from history_steps
    num_steps = history_steps if isinstance(history_steps, int) else len(history_steps)
    obs_dim = poses_flat.shape[-1] // num_steps
    poses = poses_flat.view(num_envs, num_steps, obs_dim)
    
    # Reshape times to [envs, num_steps, 1]
    times = time_offsets.unsqueeze(-1)
    
    # Concatenate and flatten: [envs, num_steps, obs_dim + 1] -> [envs, num_steps * (obs_dim + 1)]
    combined = torch.cat([poses, times], dim=-1)
    return combined.view(num_envs, -1)


def compute_historical_poses_with_time_reduced_coords(
    historical_dof_pos: Tensor,
    historical_dof_vel: Tensor,
    historical_root_local_ang_vel: Tensor,
    historical_root_rot: Tensor,
    historical_anchor_rot: Tensor,
    history_steps: Union[int, List[int]],
    w_last: bool,
    dt: float,
) -> Tensor:
    """Compute historical reduced coords observations with time offsets.
    
    Wraps compute_historical_reduced_coords_from_state and compute_historical_time_offsets,
    concatenating poses with time per step for backward compatibility.
    
    Args:
        historical_dof_pos: Historical DOF positions [envs, history_steps, num_dofs].
        historical_dof_vel: Historical DOF velocities [envs, history_steps, num_dofs].
        historical_root_local_ang_vel: Historical root angular velocity in local frame [envs, history_steps, 3].
        historical_root_rot: Historical root rotation [envs, history_steps, 4].
        historical_anchor_rot: Historical anchor rotation [envs, history_steps, 4].
        history_steps: Steps to select. Int N for first N consecutive steps,
            list for specific step indices (e.g., [1, 9, 17, ...]).
        w_last: Whether quaternions use w-last format.
        dt: Environment timestep.
    
    Returns:
        Historical poses with time [num_envs, num_steps * (obs_dim + 1)].
    """
    num_envs = historical_dof_pos.shape[0]
    num_state_history_steps = historical_dof_pos.shape[1]
    device = historical_dof_pos.device
    
    # Get historical poses using the modular function
    poses_flat = compute_historical_reduced_coords_from_state(
        historical_dof_pos=historical_dof_pos,
        historical_dof_vel=historical_dof_vel,
        historical_root_rot=historical_root_rot,
        historical_root_local_ang_vel=historical_root_local_ang_vel,
        historical_anchor_rot=historical_anchor_rot,
        w_last=w_last,
        history_steps=history_steps,
    )
    
    # Get time offsets using the modular function
    time_offsets = compute_historical_time_offsets(
        num_state_history_steps=num_state_history_steps,
        dt=dt,
        num_envs=num_envs,
        device=device,
        history_steps=history_steps,
    )
    
    # Compute num_steps from history_steps
    num_steps = history_steps if isinstance(history_steps, int) else len(history_steps)
    obs_dim = poses_flat.shape[-1] // num_steps
    poses = poses_flat.view(num_envs, num_steps, obs_dim)
    
    # Reshape times to [envs, num_steps, 1]
    times = time_offsets.unsqueeze(-1)
    
    # Concatenate and flatten: [envs, num_steps, obs_dim + 1] -> [envs, num_steps * (obs_dim + 1)]
    combined = torch.cat([poses, times], dim=-1)
    return combined.view(num_envs, -1)


def compute_historical_max_coords_from_motion_lib(
    motion_lib,
    motion_ids: Tensor,
    motion_times: Tensor,
    num_state_history_steps: int,
    dt: float,
    local_obs: bool = True,
    root_height_obs: bool = True,
    history_steps: Union[int, List[int]] = None,
) -> Tensor:
    """Compute historical max_coords observations from motion library reference data.
    
    Used by AMP discriminator to generate expert reference observations from
    sampled motions in the motion library.
    
    Args:
        motion_lib: Motion library instance with get_motion_state method.
        motion_ids: Motion indices to sample [num_samples].
        motion_times: Current times in each motion [num_samples].
        num_state_history_steps: Total history steps stored (used when history_steps=None).
        dt: Time step between historical observations.
        local_obs: Whether to use local (heading-aligned) coordinate frame.
        root_height_obs: Whether to include root height observation.
        history_steps: Steps to select. Int N for first N consecutive steps (1 to N),
            list for specific step indices (e.g., [1, 4, 8, 12, 16]).
            None = use all num_state_history_steps.
    
    Returns:
        Flattened historical observations [num_samples, num_steps * obs_dim].
    """
    num_samples = motion_ids.shape[0]
    device = motion_ids.device
    
    # Convert to list of step indices (1-indexed, going back in time)
    if history_steps is None:
        step_indices = list(range(1, num_state_history_steps + 1))
    elif isinstance(history_steps, int):
        step_indices = list(range(1, history_steps + 1))
    else:
        step_indices = history_steps
    
    # Collect historical states from motion library
    all_obs = []
    for step_idx in step_indices:
        # Time offset going backwards from current time (step 1 = 1*dt back, etc.)
        time_offset = step_idx * dt
        sample_times = (motion_times - time_offset).clamp(min=0.0)
        
        # Get motion state at this time
        ref_state = motion_lib.get_motion_state(motion_ids, sample_times)
        
        # Compute observation for this frame
        # Use zero ground height for reference (flat ground assumption)
        ground_height = torch.zeros(num_samples, 1, device=device)
        # Use empty contacts (reference doesn't have contact info)
        num_bodies = ref_state.rigid_body_pos.shape[1]
        body_contacts = torch.zeros(num_samples, num_bodies, dtype=torch.bool, device=device)
        
        frame_obs = compute_humanoid_max_coords_observations(
            body_pos=ref_state.rigid_body_pos,
            body_rot=ref_state.rigid_body_rot,
            body_vel=ref_state.rigid_body_vel,
            body_ang_vel=ref_state.rigid_body_ang_vel,
            ground_height=ground_height,
            body_contacts=body_contacts,
            local_obs=local_obs,
            root_height_obs=root_height_obs,
            observe_contacts=False,
            w_last=True,
        )
        all_obs.append(frame_obs)
    
    num_steps = len(step_indices)
    # Stack and flatten: [num_samples, num_steps, obs_dim] -> [num_samples, num_steps * obs_dim]
    historical_obs = torch.stack(all_obs, dim=1)
    obs_dim = historical_obs.shape[-1]
    
    return historical_obs.view(num_samples, num_steps * obs_dim)
