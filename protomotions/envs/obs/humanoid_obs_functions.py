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
"""Stateless observation functions for humanoid robots.

These functions compute observations from robot state and can be used
with the ObservationComponentConfig system.
"""

import torch
from torch import Tensor

from protomotions.envs.obs.humanoid import (
    compute_humanoid_max_coords_observations,
    compute_humanoid_reduced_coords_observations,
)
from protomotions.envs.obs.observation_component import ObservationComponentConfig


def max_coords_obs_factory(
    local_obs: bool = True,
    root_height_obs: bool = True,
    observe_contacts: bool = False,
    observation_noise: bool = False,
) -> ObservationComponentConfig:
    """Factory for max_coords_obs with sensible defaults.
    
    Args:
        local_obs: Whether to use local (heading-aligned) coordinate frame.
        root_height_obs: Whether to include root height observation.
        observe_contacts: Whether to include contact observations.
        observation_noise: If True, use noisy state variables (noisy_* prefix).
                          If False (default), use clean state variables.
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    # Select variable prefix based on noise flag
    prefix = "noisy_" if observation_noise else ""
    
    return ObservationComponentConfig(
        function=compute_humanoid_max_coords_observations,
        variables={
            "body_pos": f"{prefix}current_state_rigid_body_pos",
            "body_rot": f"{prefix}current_state_rigid_body_rot",
            "body_vel": f"{prefix}current_state_rigid_body_vel",
            "body_ang_vel": f"{prefix}current_state_rigid_body_ang_vel",
            "ground_height": f"{prefix}ground_heights_beneath_root",
            "body_contacts": "body_contacts",
            "local_obs": local_obs,
            "root_height_obs": root_height_obs,
            "observe_contacts": observe_contacts,
            "w_last": True,
        },
    )


def reduced_coords_obs_factory(
    root_height_obs: bool = False,
    root_vel_obs: bool = False,
    observation_noise: bool = False,
) -> ObservationComponentConfig:
    """Factory for reduced_coords_obs.
    
    Output format (in order):
        - dof_pos [num_dofs]
        - dof_vel [num_dofs]
        - root_ang_vel [3]: root angular velocity in local frame
        - proj_gravity [3]: projected gravity vector
        - root_vel [3]: root linear velocity in local frame (if root_vel_obs=True)
        - root_height [1]: root height above ground (if root_height_obs=True)
    
    Args:
        root_height_obs: Whether to include root height observation.
        root_vel_obs: Whether to include root linear velocity observation.
        observation_noise: If True, use noisy state variables (noisy_* prefix).
                          If False (default), use clean state variables.
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    # Select variable prefix based on noise flag
    prefix = "noisy_" if observation_noise else ""
    
    variables = {
        "dof_pos": f"{prefix}current_state_dof_pos",
        "dof_vel": f"{prefix}current_state_dof_vel",
        "root_local_ang_vel": f"{prefix}current_state_root_local_ang_vel",
        "anchor_rot": f"{prefix}current_state_anchor_rot",
        "w_last": True,
        "root_height_obs": root_height_obs,
        "root_vel_obs": root_vel_obs,
    }
    
    if root_height_obs:
        variables["root_pos"] = f"{prefix}current_state_root_pos"
        variables["ground_height"] = f"{prefix}ground_heights_beneath_root"
    
    if root_vel_obs:
        variables["root_rot"] = f"{prefix}current_state_root_rot"
        variables["root_vel"] = f"{prefix}current_state_root_vel"
    
    return ObservationComponentConfig(
        function=compute_humanoid_reduced_coords_observations,
        variables=variables,
    )


def historical_reduced_coords_obs_factory(
    num_steps: int = None,
    observation_noise: bool = False,
    root_height_obs: bool = False,
    root_vel_obs: bool = False,
) -> ObservationComponentConfig:
    """Factory for historical reduced_coords_obs from state history buffer.
    
    Args:
        num_steps: Number of historical steps to use (None = use all available).
        observation_noise: If True, use noisy historical state (noisy_* prefix).
                          If False (default), use clean historical state.
        root_height_obs: Whether to include root height observation.
        root_vel_obs: Whether to include root linear velocity observation.
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    # Select variable prefix based on noise flag
    prefix = "noisy_" if observation_noise else ""
    
    variables= {
        "historical_dof_pos": f"{prefix}historical_dof_pos",
        "historical_dof_vel": f"{prefix}historical_dof_vel",
        "historical_root_rot": f"{prefix}historical_root_rot",
        "historical_root_local_ang_vel": f"{prefix}historical_root_local_ang_vel",
        "historical_anchor_rot": f"{prefix}historical_anchor_rot",
        "num_steps": num_steps,
        "w_last": True,
    }
    
    if root_height_obs:
        variables["root_pos"] = f"{prefix}historical_root_pos"
        variables["ground_height"] = f"{prefix}historical_ground_heights"
    
    if root_vel_obs:
        variables["root_vel"] = f"{prefix}historical_root_vel"
    
    return ObservationComponentConfig(
        function=compute_historical_reduced_coords_from_state,
        variables=variables,
    )


def historical_max_coords_obs_factory(
    local_obs: bool = True,
    root_height_obs: bool = True,
    observe_contacts: bool = False,
    num_steps: int = None,
    observation_noise: bool = False,
) -> ObservationComponentConfig:
    """Factory for historical max_coords_obs from state history buffer.
    
    Args:
        local_obs: Whether to use local (heading-aligned) coordinate frame.
        root_height_obs: Whether to include root height observation.
        observe_contacts: Whether to include contact observations.
        num_steps: Number of historical steps to use (None = use all available).
        observation_noise: If True, use noisy historical state (noisy_* prefix).
                          If False (default), use clean historical state.
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    prefix = "noisy_" if observation_noise else ""
    
    return ObservationComponentConfig(
        function=compute_historical_max_coords_from_state,
        variables={
            "historical_rigid_body_pos": f"{prefix}historical_rigid_body_pos",
            "historical_rigid_body_rot": f"{prefix}historical_rigid_body_rot",
            "historical_rigid_body_vel": f"{prefix}historical_rigid_body_vel",
            "historical_rigid_body_ang_vel": f"{prefix}historical_rigid_body_ang_vel",
            "historical_ground_heights": f"{prefix}historical_ground_heights",
            "historical_body_contacts": "historical_body_contacts",  # contacts not noisy
            "num_steps": num_steps,
            "local_obs": local_obs,
            "root_height_obs": root_height_obs,
            "observe_contacts": observe_contacts,
            "w_last": True,
        },
    )


def historical_actions_factory(
    num_steps: int = None,
) -> ObservationComponentConfig:
    """Factory for historical actions from state history buffer.
    
    Args:
        num_steps: Number of historical steps to use (None = use all available).
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    return ObservationComponentConfig(
        function=compute_historical_actions_from_state,
        variables={
            "historical_actions": "historical_actions",
            "num_steps": num_steps,
        },
    )


def previous_actions_factory() -> ObservationComponentConfig:
    """Factory for previous actions observation (single timestep t-1).
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    from protomotions.envs.obs.general import passthrough
    
    return ObservationComponentConfig(
        function=passthrough,
        variables={"tensor": "previous_actions"},
    )


def compute_historical_reduced_coords_from_state(
    historical_dof_pos: Tensor,
    historical_dof_vel: Tensor,
    historical_root_rot: Tensor,
    historical_root_local_ang_vel: Tensor,
    historical_anchor_rot: Tensor,
    historical_root_pos: Tensor = None,
    historical_ground_heights: Tensor = None,
    w_last: bool = True,
    num_steps: int = None,
    root_height_obs: bool = False,
    root_vel_obs: bool = False,
) -> Tensor:
    """Compute historical reduced_coords observations from state history tensors.
    
    Takes 4D historical state tensors, flattens them for the existing observation
    function, then reshapes back to [envs, num_steps * obs_dim].
    
    Args:
        historical_dof_pos: Historical DOF positions [envs, history_steps, num_dofs].
        historical_dof_vel: Historical DOF velocities [envs, history_steps, num_dofs].
        historical_root_rot: Historical root rotation [envs, history_steps, 4].
        historical_root_local_ang_vel: Historical root angular velocity in local frame [envs, history_steps, 3].
        historical_anchor_rot: Historical anchor rotation [envs, history_steps, 4].
        historical_root_pos: Historical root position [envs, history_steps, 3].
        historical_ground_heights: Historical ground heights [envs, history_steps].
        w_last: Whether quaternions use w-last format.
        num_steps: Number of steps to use (None = use all).
    
    Returns:
        Flattened historical observations [envs, num_steps * obs_dim].
    
    Raises:
        ValueError: If historical tensors are None (state history not enabled).
    """
    if historical_dof_pos is None:
        raise ValueError(
            "Historical state tensors are None. "
            "Set env_config.num_state_history_steps > 0 to enable state history buffer."
        )
    
    num_envs = historical_dof_pos.shape[0]
    total_steps = historical_dof_pos.shape[1]
    
    if num_steps is None:
        num_steps = total_steps
    else:
        assert num_steps <= total_steps, "num_steps must be less than or equal to total_steps"
    
    dof_pos = historical_dof_pos[:, :num_steps]
    dof_vel = historical_dof_vel[:, :num_steps]
    root_rot = historical_root_rot[:, :num_steps]
    root_local_ang_vel = historical_root_local_ang_vel[:, :num_steps]
    anchor_rot = historical_anchor_rot[:, :num_steps]
    
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
        root_height_obs=root_height_obs,
        root_vel_obs=root_vel_obs,
    )
    
    obs_dim = flat_obs.shape[-1]
    return flat_obs.view(num_envs, num_steps * obs_dim)


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
    num_steps: int = None,
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
        num_steps: Number of steps to use (None = use all).
    
    Returns:
        Flattened historical observations [envs, num_steps * obs_dim].
    
    Raises:
        ValueError: If historical tensors are None (state history not enabled).
    """
    if historical_rigid_body_pos is None:
        raise ValueError(
            "Historical state tensors are None. "
            "Set env_config.num_state_history_steps > 0 to enable state history buffer."
        )
    
    num_envs = historical_rigid_body_pos.shape[0]
    total_steps = historical_rigid_body_pos.shape[1]
    num_bodies = historical_rigid_body_pos.shape[2]
    num_contact_bodies = historical_body_contacts.shape[2]
    
    if num_steps is None:
        num_steps = total_steps
    else:
        assert num_steps <= total_steps, "num_steps must be less than or equal to total_steps"
    
    body_pos = historical_rigid_body_pos[:, :num_steps]
    body_rot = historical_rigid_body_rot[:, :num_steps]
    body_vel = historical_rigid_body_vel[:, :num_steps]
    body_ang_vel = historical_rigid_body_ang_vel[:, :num_steps]
    ground_heights = historical_ground_heights[:, :num_steps]
    body_contacts = historical_body_contacts[:, :num_steps]
    
    batch_size = num_envs * num_steps
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
        observe_contacts=False,
        w_last=w_last,
    )
    
    obs_dim = flat_obs.shape[-1]
    return flat_obs.view(num_envs, num_steps * obs_dim)


def compute_historical_actions_from_state(
    historical_actions: Tensor,
    num_steps: int = None,
) -> Tensor:
    """Compute flattened historical actions from state history.
    
    Args:
        historical_actions: Historical actions [envs, history_steps, action_dim].
        num_steps: Number of steps to use (None = use all).
    
    Returns:
        Flattened historical actions [envs, num_steps * action_dim].
    
    Raises:
        ValueError: If historical tensors are None (state history not enabled).
    """
    if historical_actions is None:
        raise ValueError(
            "Historical actions tensor is None. "
            "Set env_config.num_state_history_steps > 0 to enable state history buffer."
        )
    
    num_envs = historical_actions.shape[0]
    total_steps = historical_actions.shape[1]
    action_dim = historical_actions.shape[2]
    
    if num_steps is None:
        num_steps = total_steps
    else:
        assert num_steps <= total_steps, "num_steps must be less than or equal to total_steps"
    
    return historical_actions[:, :num_steps].reshape(num_envs, num_steps * action_dim)


def historical_poses_with_time_factory(
    num_historical_conditioned_steps: int = 15,
    total_stored_historical_steps: int = 120,
    local_obs: bool = True,
    root_height_obs: bool = True,
    w_last: bool = True,
) -> ObservationComponentConfig:
    """Factory for historical poses with time from raw state tensors.
    
    Computes max_coords observations from historical state, subsamples them,
    and appends time offset information for each step. Useful for temporal
    conditioning in transformer-based policies.
    
    Args:
        num_historical_conditioned_steps: Number of steps to condition on (after subsampling).
        total_stored_historical_steps: Total steps stored in buffer.
        local_obs: Whether to use local (heading-aligned) coordinate frame.
        root_height_obs: Whether to include root height observation.
        w_last: Whether quaternions use w-last format.
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    return ObservationComponentConfig(
        function=compute_historical_poses_with_time,
        variables={
            "historical_rigid_body_pos": "historical_rigid_body_pos",
            "historical_rigid_body_rot": "historical_rigid_body_rot",
            "historical_rigid_body_vel": "historical_rigid_body_vel",
            "historical_rigid_body_ang_vel": "historical_rigid_body_ang_vel",
            "historical_ground_heights": "historical_ground_heights",
            "num_historical_conditioned_steps": num_historical_conditioned_steps,
            "total_stored_historical_steps": total_stored_historical_steps,
            "local_obs": local_obs,
            "root_height_obs": root_height_obs,
            "w_last": w_last,
            "dt": "dt",
        },
    )
    

def historical_poses_with_time_reduced_coords_factory(
    num_historical_conditioned_steps: int,
    total_stored_historical_steps: int,
    w_last: bool = True,
) -> ObservationComponentConfig:
    """Factory for historical_pose_obs using reduced coords with time offsets.
    
    Uses DOF-based reduced coords observations instead of body-based max coords.
    Subsamples historical steps and appends time offset information.
    
    Args:
        num_historical_conditioned_steps: Number of steps to condition on (after subsampling).
        total_stored_historical_steps: Total steps stored in buffer.
        w_last: Whether quaternions use w-last format.
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    return ObservationComponentConfig(
        function=compute_historical_poses_with_time_reduced_coords,
        variables={
            "historical_dof_pos": "historical_dof_pos",
            "historical_dof_vel": "historical_dof_vel",
            "historical_root_local_ang_vel": "historical_root_local_ang_vel",
            "historical_root_rot": "historical_root_rot",
            "historical_anchor_rot": "historical_anchor_rot",
            "num_historical_conditioned_steps": num_historical_conditioned_steps,
            "total_stored_historical_steps": total_stored_historical_steps,
            "w_last": w_last,
            "dt": "dt",
        },
    )


def compute_historical_poses_with_time_reduced_coords(
    historical_dof_pos: Tensor,
    historical_dof_vel: Tensor,
    historical_root_local_ang_vel: Tensor,
    historical_root_rot: Tensor,
    historical_anchor_rot: Tensor,
    num_historical_conditioned_steps: int,
    total_stored_historical_steps: int,
    w_last: bool,
    dt: float,
) -> Tensor:
    """Compute historical reduced coords observations with time offsets.
    
    Args:
        historical_dof_pos: Historical DOF positions [envs, history_steps, num_dofs].
        historical_dof_vel: Historical DOF velocities [envs, history_steps, num_dofs].
        historical_root_local_ang_vel: Historical root angular velocity in local frame [envs, history_steps, 3].
        historical_root_rot: Historical root rotation [envs, history_steps, 4].
        historical_anchor_rot: Historical anchor rotation [envs, history_steps, 4].
        num_historical_conditioned_steps: Number of steps to condition on.
        total_stored_historical_steps: Total steps stored in buffer.
        w_last: Whether quaternions use w-last format.
        dt: Environment timestep.
    
    Returns:
        Historical poses with time [num_envs, num_historical_conditioned_steps * (obs_dim + 1)].
    """
    if historical_dof_pos is None:
        raise ValueError(
            "Historical state tensors are None. "
            "Set env_config.num_state_history_steps > 0 to enable state history buffer."
        )
    
    num_envs = historical_dof_pos.shape[0]
    actual_history_steps = historical_dof_pos.shape[1]
    device = historical_dof_pos.device
    
    if actual_history_steps < total_stored_historical_steps:
        sub_sampling_factor = max(1, actual_history_steps // num_historical_conditioned_steps)
    else:
        sub_sampling_factor = total_stored_historical_steps // num_historical_conditioned_steps
    
    subsample_indices = torch.arange(0, actual_history_steps, sub_sampling_factor, device=device)
    subsample_indices = subsample_indices[:num_historical_conditioned_steps]
    actual_conditioned_steps = len(subsample_indices)
    
    dof_pos = historical_dof_pos[:, subsample_indices]
    dof_vel = historical_dof_vel[:, subsample_indices]
    root_local_ang_vel = historical_root_local_ang_vel[:, subsample_indices]
    root_rot = historical_root_rot[:, subsample_indices]
    anchor_rot = historical_anchor_rot[:, subsample_indices]
    
    batch_size = num_envs * actual_conditioned_steps
    flat_dof_pos = dof_pos.reshape(batch_size, -1)
    flat_dof_vel = dof_vel.reshape(batch_size, -1)
    flat_root_local_ang_vel = root_local_ang_vel.reshape(batch_size, 3)
    flat_root_rot = root_rot.reshape(batch_size, 4)
    flat_anchor_rot = anchor_rot.reshape(batch_size, 4)
    
    flat_obs = compute_humanoid_reduced_coords_observations(
        dof_pos=flat_dof_pos,
        dof_vel=flat_dof_vel,
        root_local_ang_vel=flat_root_local_ang_vel,
        root_rot=flat_root_rot,
        anchor_rot=flat_anchor_rot,
        w_last=w_last,
    )
    
    obs_dim = flat_obs.shape[-1]
    historical_poses = flat_obs.view(num_envs, actual_conditioned_steps, obs_dim)
    
    time_indices = subsample_indices.float() * dt
    time_offsets = time_indices.view(1, actual_conditioned_steps, 1).expand(num_envs, -1, -1)
    
    historical_poses_with_time = torch.cat(
        [historical_poses, time_offsets], dim=-1
    ).view(num_envs, -1)
    
    return historical_poses_with_time


def compute_historical_poses_with_time(
    historical_rigid_body_pos: Tensor,
    historical_rigid_body_rot: Tensor,
    historical_rigid_body_vel: Tensor,
    historical_rigid_body_ang_vel: Tensor,
    historical_ground_heights: Tensor,
    num_historical_conditioned_steps: int,
    total_stored_historical_steps: int,
    local_obs: bool,
    root_height_obs: bool,
    w_last: bool,
    dt: float,
) -> Tensor:
    """Compute historical pose observations with time offsets from raw state.
    
    Computes max_coords observations from historical state tensors, subsamples
    to the desired number of conditioned steps, and appends time offset 
    information for each step.
    
    Args:
        historical_rigid_body_pos: Historical body positions [envs, history_steps, bodies, 3].
        historical_rigid_body_rot: Historical body rotations [envs, history_steps, bodies, 4].
        historical_rigid_body_vel: Historical body velocities [envs, history_steps, bodies, 3].
        historical_rigid_body_ang_vel: Historical body angular velocities [envs, history_steps, bodies, 3].
        historical_ground_heights: Historical ground heights [envs, history_steps].
        num_historical_conditioned_steps: Number of steps to condition on.
        total_stored_historical_steps: Total steps stored in buffer.
        local_obs: Whether to use local coordinate frame.
        root_height_obs: Whether to include root height.
        w_last: Whether quaternions use w-last format.
        dt: Environment timestep.
    
    Returns:
        Historical poses with time [num_envs, num_historical_conditioned_steps * (obs_dim + 1)].
    """
    if historical_rigid_body_pos is None:
        raise ValueError(
            "Historical state tensors are None. "
            "Set env_config.num_state_history_steps > 0 to enable state history buffer."
        )
    
    num_envs = historical_rigid_body_pos.shape[0]
    actual_history_steps = historical_rigid_body_pos.shape[1]
    num_bodies = historical_rigid_body_pos.shape[2]
    device = historical_rigid_body_pos.device
    
    if actual_history_steps < total_stored_historical_steps:
        sub_sampling_factor = max(1, actual_history_steps // num_historical_conditioned_steps)
    else:
        sub_sampling_factor = total_stored_historical_steps // num_historical_conditioned_steps
    
    subsample_indices = torch.arange(0, actual_history_steps, sub_sampling_factor, device=device)
    subsample_indices = subsample_indices[:num_historical_conditioned_steps]
    actual_conditioned_steps = len(subsample_indices)
    
    body_pos = historical_rigid_body_pos[:, subsample_indices]
    body_rot = historical_rigid_body_rot[:, subsample_indices]
    body_vel = historical_rigid_body_vel[:, subsample_indices]
    body_ang_vel = historical_rigid_body_ang_vel[:, subsample_indices]
    ground_heights = historical_ground_heights[:, subsample_indices]
    
    batch_size = num_envs * actual_conditioned_steps
    flat_body_pos = body_pos.reshape(batch_size, num_bodies, 3)
    flat_body_rot = body_rot.reshape(batch_size, num_bodies, 4)
    flat_body_vel = body_vel.reshape(batch_size, num_bodies, 3)
    flat_body_ang_vel = body_ang_vel.reshape(batch_size, num_bodies, 3)
    flat_ground_height = ground_heights.reshape(batch_size, 1)
    flat_contacts = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
    
    flat_obs = compute_humanoid_max_coords_observations(
        body_pos=flat_body_pos,
        body_rot=flat_body_rot,
        body_vel=flat_body_vel,
        body_ang_vel=flat_body_ang_vel,
        ground_height=flat_ground_height,
        body_contacts=flat_contacts,
        local_obs=local_obs,
        root_height_obs=root_height_obs,
        observe_contacts=False,
        w_last=w_last,
    )
    
    obs_dim = flat_obs.shape[-1]
    historical_poses = flat_obs.view(num_envs, actual_conditioned_steps, obs_dim)
    
    time_indices = subsample_indices.float() * dt
    time_offsets = time_indices.view(1, actual_conditioned_steps, 1).expand(num_envs, -1, -1)
    
    historical_poses_with_time = torch.cat(
        [historical_poses, time_offsets], dim=-1
    ).view(num_envs, -1)
    
    return historical_poses_with_time


# =============================================================================
# AMP Reference Observation Functions (for computing expert obs from motion lib)
# =============================================================================


def historical_max_coords_ref_obs_factory(
    local_obs: bool = True,
    root_height_obs: bool = True,
) -> ObservationComponentConfig:
    """Factory for computing historical max_coords observations from motion library.
    
    Used by AMP discriminator to generate expert reference observations from
    sampled motions in the motion library.
    
    Args:
        local_obs: Whether to use local (heading-aligned) coordinate frame.
        root_height_obs: Whether to include root height observation.
    
    Returns:
        ObservationComponentConfig for reference observation computation.
    """
    return ObservationComponentConfig(
        function=compute_historical_max_coords_from_motion_lib,
        variables={
            "motion_lib": "motion_lib",
            "motion_ids": "motion_ids",
            "motion_times": "motion_times",
            "num_historical_steps": "num_historical_steps",
            "dt": "dt",
            "local_obs": local_obs,
            "root_height_obs": root_height_obs,
        },
    )


def compute_historical_max_coords_from_motion_lib(
    motion_lib,
    motion_ids: Tensor,
    motion_times: Tensor,
    num_historical_steps: int,
    dt: float,
    local_obs: bool = True,
    root_height_obs: bool = True,
) -> Tensor:
    """Compute historical max_coords observations from motion library reference data.
    
    Samples historical states from the motion library and computes max_coords
    observations, matching the format of compute_historical_max_coords_from_state.
    
    Args:
        motion_lib: Motion library instance with get_motion_state method.
        motion_ids: Motion indices to sample [num_samples].
        motion_times: Current times in each motion [num_samples].
        num_historical_steps: Number of historical steps to compute.
        dt: Time step between historical observations.
        local_obs: Whether to use local (heading-aligned) coordinate frame.
        root_height_obs: Whether to include root height observation.
    
    Returns:
        Flattened historical observations [num_samples, num_historical_steps * obs_dim].
    """
    num_samples = motion_ids.shape[0]
    device = motion_ids.device
    
    # Collect historical states from motion library
    all_obs = []
    for step in range(num_historical_steps):
        # Time offset going backwards from current time
        time_offset = (num_historical_steps - 1 - step) * dt
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
    
    # Stack and flatten: [num_samples, num_historical_steps, obs_dim] -> [num_samples, num_historical_steps * obs_dim]
    historical_obs = torch.stack(all_obs, dim=1)  # [num_samples, num_historical_steps, obs_dim]
    obs_dim = historical_obs.shape[-1]
    
    return historical_obs.view(num_samples, num_historical_steps * obs_dim)
