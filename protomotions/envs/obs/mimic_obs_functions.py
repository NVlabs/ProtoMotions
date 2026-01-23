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
"""Observation functions for mimic tasks.

These functions compute target pose observations for motion imitation.
"""

import torch
from torch import Tensor

from typing import Dict

from protomotions.envs.obs.target_poses import (
    build_max_coords_target_poses,
    build_max_coords_target_poses_future_rel,
    build_reduced_coords_target_poses,
)
from protomotions.envs.obs.observation_component import ObservationComponentConfig
from protomotions.envs.obs.humanoid import dof_to_obs


def compute_phase_obs(motion_ids: Tensor, motion_times: Tensor, motion_lib) -> Tensor:
    """Compute phase observations as (sin, cos) of normalized motion time.
    
    Args:
        motion_ids: Motion indices [batch].
        motion_times: Current times in motion [batch].
        motion_lib: Motion library instance.
    
    Returns:
        Phase observations [batch, 2] with (sin(phase), cos(phase)).
    """
    phase = motion_times / motion_lib.get_motion_length(motion_ids)
    sin_phase = phase.sin().unsqueeze(-1)
    cos_phase = phase.cos().unsqueeze(-1)
    
    phase_obs = torch.cat([sin_phase, cos_phase], dim=-1)
    return phase_obs


def compute_time_left_obs(motion_ids: Tensor, motion_times: Tensor, motion_lib) -> Tensor:
    """Compute time remaining in motion clip.
    
    Args:
        motion_ids: Motion indices [batch].
        motion_times: Current times in motion [batch].
        motion_lib: Motion library instance.
    
    Returns:
        Time left observations [batch, 1].
    """
    time_left = (
        motion_lib.get_motion_length(motion_ids) - motion_times
    ).unsqueeze(-1)
    return time_left


def mimic_target_poses_max_coords_factory(
    with_velocities: bool = True,
    with_relative: bool = True,
    num_future_steps: int = None,
    observation_noise: bool = False,
) -> ObservationComponentConfig:
    """Factory for max_coords target poses (full body).
    
    Uses mimic_ref_* tensors from context [envs, future_steps, bodies, dim].
    If num_future_steps is provided, uses only the first N steps from context.
    
    Args:
        with_velocities: Whether to include velocity information.
        with_relative: Whether to include relative pose observations (pos_rel, rot_rel).
        num_future_steps: Number of future steps to use (None = use all from context).
        observation_noise: If True, use noisy current state (noisy_* prefix).
                          If False (default), use clean current state.
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    # Select variable prefix based on noise flag
    prefix = "noisy_" if observation_noise else ""
    
    return ObservationComponentConfig(
        function=build_max_coords_target_poses,
        variables={
            "current_state_body_pos": f"{prefix}current_state_rigid_body_pos",
            "current_state_body_rot": f"{prefix}current_state_rigid_body_rot",
            "current_state_body_vel": f"{prefix}current_state_rigid_body_vel",
            "current_state_body_ang_vel": f"{prefix}current_state_rigid_body_ang_vel",
            "mimic_ref_pos": "mimic_ref_pos",
            "mimic_ref_rot": "mimic_ref_rot",
            "mimic_ref_vel": "mimic_ref_vel",
            "mimic_ref_ang_vel": "mimic_ref_ang_vel",
            "num_future_steps": num_future_steps,
            "with_velocities": with_velocities,
            "with_relative": with_relative,
            "w_last": True,
        },
    )


def mimic_target_poses_max_coords_future_rel_factory(
    num_future_steps: int = None,
    observation_noise: bool = False,
) -> ObservationComponentConfig:
    """Factory for max_coords_future_rel target poses (full body).
    
    Uses mimic_ref_* tensors from context [envs, future_steps, bodies, dim].
    If num_future_steps is provided, uses only the first N steps from context.
    
    Args:
        num_future_steps: Number of future steps to use (None = use all from context).
        observation_noise: If True, use noisy current state (noisy_* prefix).
                          If False (default), use clean current state.
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    # Select variable prefix based on noise flag
    prefix = "noisy_" if observation_noise else ""
    
    return ObservationComponentConfig(
        function=build_max_coords_target_poses_future_rel,
        variables={
            "current_state_body_pos": f"{prefix}current_state_rigid_body_pos",
            "current_state_body_rot": f"{prefix}current_state_rigid_body_rot",
            "mimic_ref_pos": "mimic_ref_pos",
            "mimic_ref_rot": "mimic_ref_rot",
            "num_future_steps": num_future_steps,
            "w_last": True,
        },
    )


def mimic_target_poses_simple_factory(
    num_future_steps: int = None,
    include_dof_vel: bool = True,
    include_height: bool = False,
    include_anchor_vel: bool = False,
    include_anchor_ang_vel: bool = False,
    observation_noise: bool = False,
) -> ObservationComponentConfig:
    """Factory for reduced_coords target poses with XY offset and optional extras.
    
    Output format (in order):
        - target_root_rot [6]: relative root rotation (6D tan-norm)
        - dof_vel [num_dofs]: DOF velocities (if include_dof_vel=True)
        - dof_pos [num_dofs]: DOF positions
        - xy_offset [2]: XY offset in heading frame
        - height [1]: absolute height (if include_height=True)
        - anchor_vel [3]: anchor linear velocity in local frame (if include_anchor_vel=True)
        - anchor_ang_vel [3]: anchor angular velocity in local frame (if include_anchor_ang_vel=True)
    
    Args:
        num_future_steps: Number of future steps (None = use all).
        include_dof_vel: If True, includes DOF velocities [num_dofs].
        include_height: If True, includes absolute height [1].
        include_anchor_vel: If True, includes anchor linear velocity [3].
        include_anchor_ang_vel: If True, includes anchor angular velocity [3].
        observation_noise: If True, use noisy current state (noisy_* prefix).
                          If False (default), use clean current state.
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    # Select variable prefix based on noise flag
    prefix = "noisy_" if observation_noise else ""
    
    variables = {
        "current_state_anchor_pos": f"{prefix}current_state_anchor_pos",
        "current_state_anchor_rot": f"{prefix}current_state_anchor_rot",
        "mimic_ref_anchor_rot": "mimic_ref_anchor_rot",
        "mimic_ref_anchor_pos": "mimic_ref_anchor_pos",
        "mimic_ref_dof_vel": "mimic_ref_dof_vel",
        "mimic_ref_dof_pos": "mimic_ref_dof_pos",
        "num_future_steps": num_future_steps,
        "w_last": True,
        "include_xy_offset": True,
        "include_height": include_height,
        "include_dof_vel": include_dof_vel,
        "include_anchor_vel": include_anchor_vel,
        "include_anchor_ang_vel": include_anchor_ang_vel,
    }
    
    if include_anchor_vel:
        variables["mimic_ref_anchor_vel"] = "mimic_ref_anchor_vel"
    
    if include_anchor_ang_vel:
        variables["mimic_ref_anchor_ang_vel"] = "mimic_ref_anchor_ang_vel"
    
    return ObservationComponentConfig(
        function=build_reduced_coords_target_poses,
        variables=variables,
    )


def mimic_target_poses_reduced_coords_factory(
    num_future_steps: int = None,
    observation_noise: bool = False,
) -> ObservationComponentConfig:
    """Factory for reduced_coords target poses (root + DOFs).
    
    Uses mimic_ref_* tensors from context [envs, future_steps, bodies/dofs, dim].
    If num_future_steps is provided, uses only the first N steps from context.
    
    Args:
        num_future_steps: Number of future steps to use (None = use all from context).
        observation_noise: If True, use noisy current state (noisy_* prefix).
                          If False (default), use clean current state.
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    # Select variable prefix based on noise flag
    prefix = "noisy_" if observation_noise else ""
    
    return ObservationComponentConfig(
        function=build_reduced_coords_target_poses,
        variables={
            "current_state_anchor_rot": f"{prefix}current_state_anchor_rot",
            "mimic_ref_anchor_rot": "mimic_ref_anchor_rot",
            "mimic_ref_dof_vel": "mimic_ref_dof_vel",
            "mimic_ref_dof_pos": "mimic_ref_dof_pos",
            "num_future_steps": num_future_steps,
            "w_last": True,
            "include_xy_offset": False,
        },
    )


# =============================================================================
# Individual Target Component Factories
# =============================================================================
# These factories provide modular observation components for tokenizers.
# Combine multiple components in observation_components dict as needed.
# All factories support num_future_steps for multi-frame targets.


def target_dof_pos_factory(
    num_future_steps: int = 1,
    use_6d_rotation: bool = False,
    hinge_axes_map: Dict[int, torch.Tensor] = None,
) -> ObservationComponentConfig:
    """Factory for target DOF positions.
    
    Args:
        num_future_steps: Number of future frames to include.
        use_6d_rotation: If True, convert to 6D tan-norm representation.
        hinge_axes_map: Joint axes map from robot_config.kinematic_info.hinge_axes_map.
                        Required when use_6d_rotation=True.
    
    Output: [num_future_steps * dim] where dim is num_dofs or num_joints*6 (if use_6d_rotation).
    """
    if use_6d_rotation:
        assert hinge_axes_map is not None, (
            "hinge_axes_map is required when use_6d_rotation=True. "
            "Pass robot_config.kinematic_info.hinge_axes_map from your experiment file."
        )
    
    variables = {
        "mimic_ref_dof_pos": "mimic_ref_dof_pos",
        "num_future_steps": num_future_steps,
        "use_6d_rotation": use_6d_rotation,
    }
    if use_6d_rotation:
        variables["hinge_axes_map"] = hinge_axes_map
    
    return ObservationComponentConfig(
        function=_build_target_dof_pos,
        variables=variables,
    )


def _build_target_dof_pos(
    mimic_ref_dof_pos: torch.Tensor,
    num_future_steps: int = 1,
    use_6d_rotation: bool = False,
    hinge_axes_map: Dict[int, torch.Tensor] = None,
) -> torch.Tensor:
    """Extract target DOF positions from reference."""
    num_envs = mimic_ref_dof_pos.shape[0]
    dof_pos = mimic_ref_dof_pos[:, :num_future_steps]
    num_dofs = dof_pos.shape[-1]
    
    if use_6d_rotation:
        flat_dof_pos = dof_pos.reshape(num_envs * num_future_steps, num_dofs)
        flat_6d = dof_to_obs(flat_dof_pos, hinge_axes_map, w_last=True)
        return flat_6d.reshape(num_envs, num_future_steps * flat_6d.shape[-1])
    
    return dof_pos.reshape(num_envs, -1)


def target_dof_vel_factory(num_future_steps: int = 1) -> ObservationComponentConfig:
    """Factory for target DOF velocities.
    
    Args:
        num_future_steps: Number of future frames to include.
    
    Output: dof_vel [num_future_steps * num_dofs]
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    return ObservationComponentConfig(
        function=_build_target_dof_vel,
        variables={
            "mimic_ref_dof_vel": "mimic_ref_dof_vel",
            "num_future_steps": num_future_steps,
        },
    )


def _build_target_dof_vel(
    mimic_ref_dof_vel: torch.Tensor,
    num_future_steps: int = 1,
) -> torch.Tensor:
    """Extract target DOF velocities from reference.
    
    Args:
        mimic_ref_dof_vel: [envs, future_steps, num_dofs]
        num_future_steps: Number of steps to use
    
    Returns:
        [envs, num_future_steps * num_dofs]
    """
    num_envs = mimic_ref_dof_vel.shape[0]
    return mimic_ref_dof_vel[:, :num_future_steps].reshape(num_envs, -1)


def target_root_rot_factory(num_future_steps: int = 1) -> ObservationComponentConfig:
    """Factory for target root rotation (6D representation).
    
    Args:
        num_future_steps: Number of future frames to include.
    
    Output: root_rot [num_future_steps * 6] - relative rotation in 6D tan-norm format
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    from protomotions.envs.obs.target_poses import build_target_root_rot
    return ObservationComponentConfig(
        function=build_target_root_rot,
        variables={
            "current_state_root_rot": "current_state_root_rot",
            "mimic_ref_anchor_rot": "mimic_ref_anchor_rot",
            "num_future_steps": num_future_steps,
            "w_last": True,
        },
    )


def target_xy_offset_factory(num_future_steps: int = 1) -> ObservationComponentConfig:
    """Factory for target XY offset in heading frame.
    
    Args:
        num_future_steps: Number of future frames to include.
    
    Output: xy_offset [num_future_steps * 2]
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    from protomotions.envs.obs.target_poses import build_target_xy_offset
    return ObservationComponentConfig(
        function=build_target_xy_offset,
        variables={
            "current_state_anchor_pos": "current_state_anchor_pos",
            "current_state_anchor_rot": "current_state_anchor_rot",
            "mimic_ref_anchor_pos": "mimic_ref_anchor_pos",
            "num_future_steps": num_future_steps,
            "w_last": True,
        },
    )


def target_height_factory(num_future_steps: int = 1) -> ObservationComponentConfig:
    """Factory for target absolute height.
    
    Args:
        num_future_steps: Number of future frames to include.
    
    Output: height [num_future_steps * 1]
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    from protomotions.envs.obs.target_poses import build_target_height
    return ObservationComponentConfig(
        function=build_target_height,
        variables={
            "mimic_ref_anchor_pos": "mimic_ref_anchor_pos",
            "num_future_steps": num_future_steps,
        },
    )


def target_root_vel_factory(num_future_steps: int = 1) -> ObservationComponentConfig:
    """Factory for target root linear velocity in local frame.
    
    Args:
        num_future_steps: Number of future frames to include.
    
    Output: root_vel [num_future_steps * 3]
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    from protomotions.envs.obs.target_poses import build_target_root_vel
    return ObservationComponentConfig(
        function=build_target_root_vel,
        variables={
            "current_state_anchor_rot": "current_state_anchor_rot",
            "mimic_ref_root_vel": "mimic_ref_root_vel",
            "num_future_steps": num_future_steps,
            "w_last": True,
        },
    )


def target_root_ang_vel_factory(num_future_steps: int = 1) -> ObservationComponentConfig:
    """Factory for target root angular velocity in local frame.
    
    Args:
        num_future_steps: Number of future frames to include.
    
    Output: root_ang_vel [num_future_steps * 3]
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    from protomotions.envs.obs.target_poses import build_target_root_ang_vel
    return ObservationComponentConfig(
        function=build_target_root_ang_vel,
        variables={
            "current_state_anchor_rot": "current_state_anchor_rot",
            "mimic_ref_root_ang_vel": "mimic_ref_root_ang_vel",
            "num_future_steps": num_future_steps,
            "w_last": True,
        },
    )
