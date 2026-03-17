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
"""Typed context views for observations, rewards, and terminations.

This module provides lightweight view classes that wrap existing data structures
(RobotState, StateHistoryBuffer) without copying. Context paths use FieldPath
and NestedField descriptors for dual access:

- Class access: Returns FieldPath for configuration (e.g., EnvContext.current.rigid_body_pos)
- Instance access: Returns actual tensor values (e.g., ctx.current.rigid_body_pos)

Example usage in experiment configs:
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    observation_components = {
        "max_coords_obs": MdpComponent(
            compute_func=compute_humanoid_max_coords_observations,
            dynamic_vars={
                "body_pos": EnvContext.current.rigid_body_pos,  # FieldPath object
                "body_rot": EnvContext.current.rigid_body_rot,
            },
            static_params={"local_obs": True},
        ),
    }
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from torch import Tensor

from protomotions.envs.context_paths import FieldPath, NestedField
from protomotions.envs.obs.humanoid import compute_local_ang_vel

if TYPE_CHECKING:
    from protomotions.simulator.base_simulator.simulator_state import RobotState
    from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer


# =============================================================================
# Robot State View (for nested FieldPath access to RobotState fields)
# =============================================================================


class RobotStateView:
    """Lightweight path-proxy for RobotState fields.

    Used as the nested_class in NestedField so that class-level access like
    ``EnvContext.mimic.ref_state.rigid_body_pos`` produces the correct FieldPath.

    At instance level the actual RobotState object is stored directly, so
    attribute access falls through to the real dataclass fields.
    """

    rigid_body_pos: Tensor = FieldPath()
    rigid_body_rot: Tensor = FieldPath()
    rigid_body_vel: Tensor = FieldPath()
    rigid_body_ang_vel: Tensor = FieldPath()
    rigid_body_contacts: Tensor = FieldPath()
    rigid_body_contact_forces: Tensor = FieldPath()
    dof_pos: Tensor = FieldPath()
    dof_vel: Tensor = FieldPath()
    dof_forces: Tensor = FieldPath()
    local_rigid_body_rot: Tensor = FieldPath()


# =============================================================================
# Current State Views
# =============================================================================


class CurrentStateView:
    """View into current robot state with anchor and local angular velocity properties.

    Wraps a RobotState and adds:
    - Anchor body properties (anchor_pos, anchor_rot, anchor_vel, etc.)
    - Local angular velocities (root_local_ang_vel, anchor_local_ang_vel)

    This view does not copy data - it provides typed access to the underlying
    RobotState with additional precomputed properties.

    All fields are FieldPath descriptors for dual class/instance access.
    """

    # Direct state attributes (stored)
    rigid_body_pos: Tensor = FieldPath()
    rigid_body_rot: Tensor = FieldPath()
    rigid_body_vel: Tensor = FieldPath()
    rigid_body_ang_vel: Tensor = FieldPath()
    rigid_body_contacts: Tensor = FieldPath()
    dof_pos: Tensor = FieldPath()
    dof_vel: Tensor = FieldPath()
    dof_forces: Tensor = FieldPath()
    anchor_idx: int = FieldPath()

    # Root properties (precomputed from state)
    root_pos: Tensor = FieldPath()
    root_rot: Tensor = FieldPath()
    root_vel: Tensor = FieldPath()
    root_ang_vel: Tensor = FieldPath()
    root_height: Tensor = FieldPath()
    root_local_ang_vel: Tensor = FieldPath()

    # Anchor properties (precomputed)
    anchor_pos: Tensor = FieldPath()
    anchor_rot: Tensor = FieldPath()
    anchor_vel: Tensor = FieldPath()
    anchor_ang_vel: Tensor = FieldPath()
    anchor_local_ang_vel: Tensor = FieldPath()

    def __init__(self, state: "RobotState", anchor_idx: int):
        """Initialize CurrentStateView with precomputed derived values.

        Args:
            state: The underlying RobotState containing all body poses and velocities.
            anchor_idx: Index of anchor body (typically pelvis) for anchor properties.
        """
        # Store direct values from state
        # Note: state may be RobotState (full) or NoisyObservations (subset),
        # so optional fields use getattr with None default.
        self.rigid_body_pos = state.rigid_body_pos
        self.rigid_body_rot = state.rigid_body_rot
        self.rigid_body_vel = state.rigid_body_vel
        self.rigid_body_ang_vel = state.rigid_body_ang_vel
        self.rigid_body_contacts = getattr(state, "rigid_body_contacts", None)
        self.dof_pos = state.dof_pos
        self.dof_vel = state.dof_vel
        self.dof_forces = getattr(state, "dof_forces", None)
        self.anchor_idx = anchor_idx

        # Precompute root properties
        self.root_pos = state.root_pos
        self.root_rot = state.root_rot
        self.root_vel = state.root_vel
        self.root_ang_vel = state.root_ang_vel
        self.root_height = state.rigid_body_pos[:, 0, 2]
        self.root_local_ang_vel = compute_local_ang_vel(
            state.root_rot, state.root_ang_vel
        )

        # Precompute anchor properties
        self.anchor_pos = state.rigid_body_pos[:, anchor_idx, :]
        self.anchor_rot = state.rigid_body_rot[:, anchor_idx, :]
        self.anchor_vel = state.rigid_body_vel[:, anchor_idx, :]
        self.anchor_ang_vel = state.rigid_body_ang_vel[:, anchor_idx, :]
        self.anchor_local_ang_vel = compute_local_ang_vel(
            self.anchor_rot, self.anchor_ang_vel
        )


# =============================================================================
# Historical State Views
# =============================================================================


class HistoricalView:
    """View into historical state from StateHistoryBuffer.

    Provides access to historical state tensors with shape [num_envs, history_steps, ...].
    Wraps the buffer without copying data. Can be used for either clean (ground-truth)
    or noisy observations based on the use_noisy parameter.

    All fields are FieldPath descriptors for dual class/instance access.
    """

    # Historical state attributes (stored from buffer)
    rigid_body_pos: Tensor = FieldPath()
    rigid_body_rot: Tensor = FieldPath()
    rigid_body_vel: Tensor = FieldPath()
    rigid_body_ang_vel: Tensor = FieldPath()
    dof_pos: Tensor = FieldPath()
    dof_vel: Tensor = FieldPath()
    actions: Tensor = FieldPath()
    processed_actions: Tensor = FieldPath()
    ground_heights: Tensor = FieldPath()
    body_contacts: Tensor = FieldPath()

    # Root properties (precomputed from buffer)
    root_pos: Tensor = FieldPath()
    root_rot: Tensor = FieldPath()
    root_ang_vel: Tensor = FieldPath()
    root_local_ang_vel: Tensor = FieldPath()

    # Anchor properties (precomputed from buffer)
    anchor_pos: Tensor = FieldPath()
    anchor_rot: Tensor = FieldPath()
    anchor_vel: Tensor = FieldPath()
    anchor_ang_vel: Tensor = FieldPath()

    def __init__(self, buffer: "StateHistoryBuffer", use_noisy: bool = False):
        """Initialize HistoricalView with precomputed derived values.

        Args:
            buffer: The underlying StateHistoryBuffer containing historical state.
            use_noisy: If True, use noisy historical data. If False, use clean data.
        """
        # Determine prefix based on use_noisy
        prefix = "noisy_historical_" if use_noisy else "historical_"

        # Store direct values from buffer
        self.rigid_body_pos = getattr(buffer, f"{prefix}rigid_body_pos")
        self.rigid_body_rot = getattr(buffer, f"{prefix}rigid_body_rot")
        self.rigid_body_vel = getattr(buffer, f"{prefix}rigid_body_vel")
        self.rigid_body_ang_vel = getattr(buffer, f"{prefix}rigid_body_ang_vel")
        self.dof_pos = getattr(buffer, f"{prefix}dof_pos")
        self.dof_vel = getattr(buffer, f"{prefix}dof_vel")
        self.ground_heights = getattr(buffer, f"{prefix}ground_heights")

        # Clean-only fields (not available in noisy version)
        if not use_noisy:
            self.actions = buffer.historical_actions
            self.processed_actions = buffer.historical_processed_actions
            self.body_contacts = buffer.historical_body_contacts
        else:
            self.actions = None
            self.processed_actions = None
            self.body_contacts = None

        # Precompute root properties
        self.root_pos = getattr(buffer, f"{prefix}root_pos")
        self.root_rot = getattr(buffer, f"{prefix}root_rot")
        self.root_ang_vel = getattr(buffer, f"{prefix}root_ang_vel")
        self.root_local_ang_vel = compute_local_ang_vel(
            self.root_rot, self.root_ang_vel
        )

        # Precompute anchor properties
        self.anchor_pos = getattr(buffer, f"{prefix}anchor_pos")
        self.anchor_rot = getattr(buffer, f"{prefix}anchor_rot")

        # Clean-only anchor properties
        if not use_noisy:
            self.anchor_vel = buffer.historical_anchor_vel
            self.anchor_ang_vel = buffer.historical_anchor_ang_vel
        else:
            self.anchor_vel = None
            self.anchor_ang_vel = None


# =============================================================================
# Control-Specific Contexts
# =============================================================================


class MimicContext:
    """View for mimic control context.

    Contains reference state for *reward* computation and multi-step future poses
    for *observation* computation. The reference state is at the current timestep,
    while future poses are at t+dt, t+2*dt, etc.

    All fields are FieldPath descriptors for dual class/instance access.
    """

    # Reference state (nested – needs NestedField so class-level path proxy works)
    ref_state: "RobotState" = NestedField(RobotStateView)

    # Future state attributes (stored)
    future_pos: Tensor = FieldPath()
    future_rot: Tensor = FieldPath()
    future_vel: Tensor = FieldPath()
    future_ang_vel: Tensor = FieldPath()
    future_dof_pos: Tensor = FieldPath()
    future_dof_vel: Tensor = FieldPath()
    anchor_idx: int = FieldPath()
    ref_lr: Tensor = FieldPath()

    # Future root properties (precomputed)
    future_root_pos: Tensor = FieldPath()
    future_root_rot: Tensor = FieldPath()
    future_root_vel: Tensor = FieldPath()
    future_root_ang_vel: Tensor = FieldPath()

    # Current-frame reference anchor position (precomputed)
    ref_anchor_pos: Tensor = FieldPath()

    # Future anchor properties (precomputed)
    future_anchor_pos: Tensor = FieldPath()
    future_anchor_rot: Tensor = FieldPath()
    future_anchor_vel: Tensor = FieldPath()
    future_anchor_ang_vel: Tensor = FieldPath()

    def __init__(
        self,
        ref_state: "RobotState",
        future_pos: Tensor,
        future_rot: Tensor,
        future_vel: Tensor,
        future_ang_vel: Tensor,
        future_dof_pos: Tensor,
        future_dof_vel: Tensor,
        anchor_idx: int,
        ref_lr: Tensor,
    ):
        """Initialize MimicContext with precomputed derived values.

        Args:
            ref_state: Single-step reference state at current time for reward computation.
            future_pos: Future body positions [num_envs, future_steps, num_bodies, 3].
            future_rot: Future body rotations [num_envs, future_steps, num_bodies, 4].
            future_vel: Future body velocities [num_envs, future_steps, num_bodies, 3].
            future_ang_vel: Future angular velocities [num_envs, future_steps, num_bodies, 3].
            future_dof_pos: Future DOF positions [num_envs, future_steps, num_dofs].
            future_dof_vel: Future DOF velocities [num_envs, future_steps, num_dofs].
            anchor_idx: Index of anchor body for computing anchor-relative values.
            ref_lr: Reference DOF in local rotation format for DOF tracking rewards.
        """
        # Store direct values
        self.ref_state = ref_state
        self.future_pos = future_pos
        self.future_rot = future_rot
        self.future_vel = future_vel
        self.future_ang_vel = future_ang_vel
        self.future_dof_pos = future_dof_pos
        self.future_dof_vel = future_dof_vel
        self.anchor_idx = anchor_idx
        self.ref_lr = ref_lr

        # Precompute future root properties
        self.future_root_pos = future_pos[:, :, 0, :]
        self.future_root_rot = future_rot[:, :, 0, :]
        self.future_root_vel = future_vel[:, :, 0, :]
        self.future_root_ang_vel = future_ang_vel[:, :, 0, :]

        # Precompute current-frame reference anchor position
        self.ref_anchor_pos = ref_state.rigid_body_pos[:, anchor_idx, :]

        # Precompute future anchor properties
        self.future_anchor_pos = future_pos[:, :, anchor_idx, :]
        self.future_anchor_rot = future_rot[:, :, anchor_idx, :]
        self.future_anchor_vel = future_vel[:, :, anchor_idx, :]
        self.future_anchor_ang_vel = future_ang_vel[:, :, anchor_idx, :]


class MaskedMimicContext:
    """View for masked mimic control context.

    Extends MimicContext with sparse target poses and visibility masks for
    conditional motion generation. Only a subset of bodies and timesteps
    are revealed to the policy.

    All fields are FieldPath descriptors for dual class/instance access.
    """

    # Base mimic context (nested)
    mimic: MimicContext = NestedField(MimicContext)

    # Sparse target attributes (stored)
    ref_pos: Tensor = FieldPath()
    ref_rot: Tensor = FieldPath()
    target_times: Tensor = FieldPath()
    time_offsets: Tensor = FieldPath()
    target_poses_masks: Tensor = FieldPath()
    target_bodies_masks: Tensor = FieldPath()

    def __init__(
        self,
        mimic: MimicContext,
        ref_pos: Tensor,
        ref_rot: Tensor,
        target_times: Tensor,
        time_offsets: Tensor,
        target_poses_masks: Tensor,
        target_bodies_masks: Tensor,
    ):
        """Initialize MaskedMimicContext.

        Args:
            mimic: Base mimic context with full reference state and future poses.
            ref_pos: Target body positions at sparse times [num_envs, future_steps, num_bodies, 3].
            ref_rot: Target body rotations at sparse times [num_envs, future_steps, num_bodies, 4].
            target_times: Absolute target times [num_envs, future_steps].
            time_offsets: Time offsets from current time [num_envs, future_steps].
            target_poses_masks: Which timesteps are visible [num_envs, future_steps].
            target_bodies_masks: Which bodies are visible [num_envs, future_steps * num_bodies * 2].
        """
        self.mimic = mimic
        self.ref_pos = ref_pos
        self.ref_rot = ref_rot
        self.target_times = target_times
        self.time_offsets = time_offsets
        self.target_poses_masks = target_poses_masks
        self.target_bodies_masks = target_bodies_masks


class SteeringContext:
    """View for steering control context.

    Contains target direction, speed, and facing direction for locomotion
    steering tasks. Also stores previous root position for velocity computation.

    All fields are FieldPath descriptors for dual class/instance access.
    """

    tar_dir: Tensor = FieldPath()
    tar_dir_theta: Tensor = FieldPath()
    tar_speed: Tensor = FieldPath()
    tar_face_dir: Tensor = FieldPath()
    prev_root_pos: Tensor = FieldPath()

    def __init__(
        self,
        tar_dir: Tensor,
        tar_dir_theta: Tensor,
        tar_speed: Tensor,
        tar_face_dir: Tensor,
        prev_root_pos: Tensor,
    ):
        """Initialize SteeringContext.

        Args:
            tar_dir: Target movement direction in world frame [num_envs, 2] (xy only).
            tar_dir_theta: Target direction as angle from forward [num_envs].
            tar_speed: Target movement speed [num_envs].
            tar_face_dir: Target facing direction in world frame [num_envs, 2] (xy only).
            prev_root_pos: Previous root position for velocity computation [num_envs, 3].
        """
        self.tar_dir = tar_dir
        self.tar_dir_theta = tar_dir_theta
        self.tar_speed = tar_speed
        self.tar_face_dir = tar_face_dir
        self.prev_root_pos = prev_root_pos


class PathContext:
    """View for path follower control context.

    Contains target position, head position, and trajectory samples for
    path following tasks.

    All fields are FieldPath descriptors for dual class/instance access.
    """

    tar_pos: Tensor = FieldPath()
    head_pos: Tensor = FieldPath()
    traj_samples: Tensor = FieldPath()
    height_conditioned: bool = FieldPath()
    head_body_id: int = FieldPath()

    def __init__(
        self,
        tar_pos: Tensor,
        head_pos: Tensor,
        traj_samples: Tensor,
        height_conditioned: bool,
        head_body_id: int,
    ):
        """Initialize PathContext.

        Args:
            tar_pos: Current target position on path [num_envs, 3].
            head_pos: Current head body position [num_envs, 3].
            traj_samples: Future waypoint positions [num_envs, num_samples, 3].
            height_conditioned: Whether observations include height information.
            head_body_id: Index of head body in kinematic chain.
        """
        self.tar_pos = tar_pos
        self.head_pos = head_pos
        self.traj_samples = traj_samples
        self.height_conditioned = height_conditioned
        self.head_body_id = head_body_id


# =============================================================================
# Main Context Class
# =============================================================================


class EnvContext:
    """Complete typed context for observation, reward, and termination functions.

    This class provides typed access to all environment state needed for
    computing observations, rewards, and terminations. MdpComponent functions
    receive this context and bind their kernels to context paths with full IDE
    autocomplete:

        from protomotions.envs.context_views import EnvContext
        from protomotions.envs.mdp_component import MdpComponent

        observation_components = {
            "max_coords_obs": MdpComponent(
                compute_func=compute_humanoid_max_coords_observations,
                dynamic_vars={
                    "body_pos": EnvContext.current.rigid_body_pos,  # FieldPath
                },
            ),
        }

    Core state views (current, noisy) are always present. Historical and control-
    specific views are optional and populated based on environment configuration.

    All fields are FieldPath or NestedField descriptors for dual class/instance access.
    """

    # Core state (always present)
    current: CurrentStateView = NestedField(CurrentStateView)
    noisy: CurrentStateView = NestedField(CurrentStateView)

    # Historical state (optional - depends on state history config)
    historical: Optional[HistoricalView] = NestedField(HistoricalView)
    noisy_historical: Optional[HistoricalView] = NestedField(HistoricalView)

    # Actions (historical)
    previous_action: Optional[Tensor] = FieldPath()
    current_processed_action: Optional[Tensor] = FieldPath()
    previous_processed_action: Optional[Tensor] = FieldPath()

    # Environment state
    ground_heights: Optional[Tensor] = FieldPath()
    noisy_ground_heights: Optional[Tensor] = FieldPath()
    body_contacts: Optional[Tensor] = FieldPath()
    current_contact_force_magnitudes: Optional[Tensor] = FieldPath()
    prev_contact_force_magnitudes: Optional[Tensor] = FieldPath()
    dt: float = FieldPath()

    # Contact tracking
    contact_body_ids: Optional[Tensor] = FieldPath()

    # Control-specific contexts (populated by controllers via populate_context)
    mimic: Optional[MimicContext] = NestedField(MimicContext)
    masked_mimic: Optional[MaskedMimicContext] = NestedField(MaskedMimicContext)
    steering: Optional[SteeringContext] = NestedField(SteeringContext)
    path: Optional[PathContext] = NestedField(PathContext)

    def __init__(
        self,
        current: CurrentStateView,
        noisy: CurrentStateView,
        dt: float,
        historical: Optional[HistoricalView] = None,
        noisy_historical: Optional[HistoricalView] = None,
        previous_action: Optional[Tensor] = None,
        current_processed_action: Optional[Tensor] = None,
        previous_processed_action: Optional[Tensor] = None,
        ground_heights: Optional[Tensor] = None,
        noisy_ground_heights: Optional[Tensor] = None,
        body_contacts: Optional[Tensor] = None,
        current_contact_force_magnitudes: Optional[Tensor] = None,
        prev_contact_force_magnitudes: Optional[Tensor] = None,
        contact_body_ids: Optional[Tensor] = None,
        mimic: Optional[MimicContext] = None,
        masked_mimic: Optional[MaskedMimicContext] = None,
        steering: Optional[SteeringContext] = None,
        path: Optional[PathContext] = None,
    ):
        """Initialize EnvContext with all state views.

        Args:
            current: Clean (ground-truth) current robot state for rewards/critic.
            noisy: Noisy current robot state for actor observations.
            dt: Simulation timestep in seconds.
            historical: Clean historical state view (optional).
            noisy_historical: Noisy historical state view (optional).
            previous_action: Previous raw action [num_envs, action_dim] (optional).
            current_processed_action: Current processed action (optional).
            previous_processed_action: Previous processed action (optional).
            ground_heights: Ground height beneath root position [num_envs] (optional).
            noisy_ground_heights: Noisy ground height for actor (optional).
            body_contacts: Boolean contact flags for tracked bodies (optional).
            current_contact_force_magnitudes: Current contact force magnitudes (optional).
            prev_contact_force_magnitudes: Previous contact forces (optional).
            contact_body_ids: Indices of bodies to track contacts for (optional).
            mimic: Mimic control context (optional).
            masked_mimic: Masked mimic context (optional).
            steering: Steering control context (optional).
            path: Path following context (optional).
        """
        # Core state
        self.current = current
        self.noisy = noisy
        self.dt = dt

        # Historical state
        self.historical = historical
        self.noisy_historical = noisy_historical

        # Actions
        self.previous_action = previous_action
        self.current_processed_action = current_processed_action
        self.previous_processed_action = previous_processed_action

        # Environment state
        self.ground_heights = ground_heights
        self.noisy_ground_heights = noisy_ground_heights
        self.body_contacts = body_contacts
        self.current_contact_force_magnitudes = current_contact_force_magnitudes
        self.prev_contact_force_magnitudes = prev_contact_force_magnitudes

        # Contact tracking
        self.contact_body_ids = contact_body_ids

        # Control-specific views
        self.mimic = mimic
        self.masked_mimic = masked_mimic
        self.steering = steering
        self.path = path


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CurrentStateView",
    "HistoricalView",
    "MimicContext",
    "MaskedMimicContext",
    "SteeringContext",
    "PathContext",
    "EnvContext",
]
