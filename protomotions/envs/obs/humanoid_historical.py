# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Historical observation kernels wired to state-history tensors by MdpComponent.

Every kernel here reads a window published as ``[envs, steps, ...]`` and names
the 1-based slots it takes with ``history_steps: StepSelection`` (no default:
the recipe writes ``history_steps=8`` or ``history_steps=[1, 2, 4]``). The age
of a selected slot comes from ``step_ages`` with a named origin: ``"current"``
makes slot ``k`` ``k * dt`` old (the proprioception tokens), ``"first_slot"``
makes it ``(k - 1) * dt`` old (the pose-token convention, the first stored
slot at age zero). Both conventions are kept for the policies trained on them.
"""

from typing import Literal, Optional

import torch
from torch import Tensor

from protomotions.envs.context_views import EnvContext
from protomotions.envs.obs.humanoid import (
    compute_humanoid_max_coords_observations,
    compute_humanoid_reduced_coords_observations,
)
from protomotions.envs.steps import StepSelection, select_steps, step_ages

TimeOrigin = Literal["current", "first_slot"]

_NO_HISTORY = (
    "Historical state tensors are None: bind a component that reads the history "
    "(history_steps=[...])"
)


def compute_historical_proprioception(
    historical_dof_pos: Tensor = EnvContext.historical.dof_pos,
    historical_dof_vel: Tensor = EnvContext.historical.dof_vel,
    historical_root_local_ang_vel: Tensor = EnvContext.historical.root_local_ang_vel,
    historical_anchor_rot: Tensor = EnvContext.historical.anchor_rot,
    historical_actions: Tensor = EnvContext.historical.raw_actions,
    dt: float = EnvContext.dt,
    time_origin: TimeOrigin = "current",
    *,
    fk,
    history_steps: StepSelection,
) -> Tensor:
    """Previous-only FK/sensor/action tokens, with positive age in seconds.

    Inputs already exclude the current slot. At lag k the incoming raw action
    belongs to state s(t-k), i.e. it is a(t-k-1), not the outgoing command.
    """

    selected = [
        select_steps(value, history_steps).flatten(0, 1)
        for value in (
            historical_dof_pos, historical_dof_vel, historical_root_local_ang_vel,
            historical_anchor_rot, historical_actions,
        )
    ]
    dof_pos, dof_vel, gyro, anchor_rot, actions = selected
    batch = historical_dof_pos.shape[0]
    reduced = compute_humanoid_reduced_coords_observations(
        dof_pos, dof_vel, anchor_rot, gyro, w_last=True,
    )
    ages = step_ages(history_steps, dt, time_origin, dof_pos.device).to(dof_pos.dtype)
    ages = ages[None, :, None].expand(batch, -1, -1).reshape(-1, 1)
    tokens = torch.cat((fk(dof_pos, anchor_rot), reduced, actions, ages), dim=-1)
    return tokens.reshape(batch, -1)


def compute_historical_reduced_coords_from_state(
    historical_dof_pos: Tensor = EnvContext.historical.dof_pos,
    historical_dof_vel: Tensor = EnvContext.historical.dof_vel,
    historical_root_rot: Tensor = EnvContext.historical.root_rot,
    historical_root_local_ang_vel: Tensor = EnvContext.historical.root_local_ang_vel,
    historical_anchor_rot: Tensor = EnvContext.historical.anchor_rot,
    w_last: bool = True,
    *,
    history_steps: StepSelection,
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
        history_steps: The 1-based slots to read, in output order.
        w_last: Whether quaternions use w-last format.

    Returns:
        Flattened historical observations [envs, len(history_steps) * obs_dim].

    Raises:
        ValueError: If historical tensors are None (no history is published).
    """
    if historical_dof_pos is None:
        raise ValueError(_NO_HISTORY)

    num_envs = historical_dof_pos.shape[0]

    dof_pos = select_steps(historical_dof_pos, history_steps)
    dof_vel = select_steps(historical_dof_vel, history_steps)
    root_rot = select_steps(historical_root_rot, history_steps)
    root_local_ang_vel = select_steps(historical_root_local_ang_vel, history_steps)
    anchor_rot = select_steps(historical_anchor_rot, history_steps)

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


def compute_historical_max_coords_from_state(
    historical_rigid_body_pos: Tensor = EnvContext.historical.rigid_body_pos,
    historical_rigid_body_rot: Tensor = EnvContext.historical.rigid_body_rot,
    historical_rigid_body_vel: Tensor = EnvContext.historical.rigid_body_vel,
    historical_rigid_body_ang_vel: Tensor = EnvContext.historical.rigid_body_ang_vel,
    historical_ground_heights: Optional[Tensor] = EnvContext.historical.ground_heights,
    historical_body_contacts: Optional[Tensor] = None,
    local_obs: bool = True,
    root_height_obs: bool = True,
    observe_contacts: bool = False,
    w_last: bool = True,
    anchor_body_index: int = EnvContext.historical.anchor_idx,
    *,
    history_steps: StepSelection,
) -> Tensor:
    """Compute historical max_coords observations from state history tensors.

    Args:
        historical_rigid_body_pos: Historical body positions [envs, history_steps, bodies, 3].
        historical_rigid_body_rot: Historical body rotations [envs, history_steps, bodies, 4].
        historical_rigid_body_vel: Historical body velocities [envs, history_steps, bodies, 3].
        historical_rigid_body_ang_vel: Historical body angular velocities [envs, history_steps, bodies, 3].
        history_steps: The 1-based slots to read, in output order.
        historical_ground_heights: Historical ground heights [envs, history_steps].
        historical_body_contacts: Historical body contacts [envs, history_steps, num_contact_bodies].
        local_obs: Whether to use local coordinate frame.
        root_height_obs: Whether to include root height.
        observe_contacts: Whether to include contact observations.
        w_last: Whether quaternions use w-last format.
        anchor_body_index: The anchor body; the published anchor index by default.

    Returns:
        Flattened historical observations [envs, len(history_steps) * obs_dim].

    Raises:
        ValueError: If historical tensors are None (no history is published).
    """
    if historical_rigid_body_pos is None:
        raise ValueError(_NO_HISTORY)

    num_envs = historical_rigid_body_pos.shape[0]
    num_bodies = historical_rigid_body_pos.shape[2]

    body_pos = select_steps(historical_rigid_body_pos, history_steps)
    body_rot = select_steps(historical_rigid_body_rot, history_steps)
    body_vel = select_steps(historical_rigid_body_vel, history_steps)
    body_ang_vel = select_steps(historical_rigid_body_ang_vel, history_steps)
    ground_heights = (
        None
        if historical_ground_heights is None
        else select_steps(historical_ground_heights, history_steps)
    )
    body_contacts = (
        None
        if historical_body_contacts is None
        else select_steps(historical_body_contacts, history_steps)
    )

    actual_steps = body_pos.shape[1]
    batch_size = num_envs * actual_steps
    flat_body_pos = body_pos.reshape(batch_size, num_bodies, 3)
    flat_body_rot = body_rot.reshape(batch_size, num_bodies, 4)
    flat_body_vel = body_vel.reshape(batch_size, num_bodies, 3)
    flat_body_ang_vel = body_ang_vel.reshape(batch_size, num_bodies, 3)
    flat_ground_height = (
        None if ground_heights is None else ground_heights.reshape(batch_size, 1)
    )
    flat_contacts = (
        None
        if body_contacts is None
        else body_contacts.reshape(batch_size, body_contacts.shape[2])
    )

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
        anchor_body_index=anchor_body_index,
    )

    obs_dim = flat_obs.shape[-1]
    return flat_obs.view(num_envs, actual_steps * obs_dim)


def compute_historical_actions_from_state(
    historical_actions: Tensor = EnvContext.raw_action_history,
    *,
    history_steps: StepSelection,
) -> Tensor:
    """Compute flattened historical actions from state history.

    Args:
        historical_actions: Historical actions [envs, history_steps, action_dim].
        history_steps: The 1-based slots to read, in output order.

    Returns:
        Flattened historical actions [envs, len(history_steps) * action_dim].

    Raises:
        ValueError: If historical tensors are None (no history is published).
    """
    if historical_actions is None:
        raise ValueError(_NO_HISTORY)

    num_envs = historical_actions.shape[0]

    actions = select_steps(historical_actions, history_steps)

    actual_steps = actions.shape[1]
    action_dim = actions.shape[2]

    return actions.reshape(num_envs, actual_steps * action_dim)


def _append_historical_time(
    poses_flat: Tensor,
    steps: StepSelection,
    dt: float,
    time_origin: TimeOrigin,
) -> Tensor:
    """Append each selected slot's age to its flattened pose token."""
    num_envs = poses_flat.shape[0]
    num_steps = len(steps)
    ages = step_ages(steps, dt, time_origin, poses_flat.device)
    poses = poses_flat.view(num_envs, num_steps, poses_flat.shape[-1] // num_steps)
    times = ages.view(1, num_steps, 1).expand(num_envs, -1, -1)
    return torch.cat((poses, times), dim=-1).view(num_envs, -1)


def compute_historical_poses_with_time(
    historical_rigid_body_pos: Tensor,
    historical_rigid_body_rot: Tensor,
    historical_rigid_body_vel: Tensor,
    historical_rigid_body_ang_vel: Tensor,
    historical_ground_heights: Tensor,
    history_steps: StepSelection,
    local_obs: bool,
    root_height_obs: bool,
    w_last: bool,
    dt: float,
    anchor_body_index: int = EnvContext.current.anchor_idx,
    time_origin: TimeOrigin = "first_slot",
) -> Tensor:
    """Compute historical pose observations with time offsets from raw state.

    Wraps compute_historical_max_coords_from_state and appends each selected
    slot's age (``step_ages`` with ``time_origin``) to its pose token.

    Args:
        historical_rigid_body_pos: Historical body positions [envs, history_steps, bodies, 3].
        historical_rigid_body_rot: Historical body rotations [envs, history_steps, bodies, 4].
        historical_rigid_body_vel: Historical body velocities [envs, history_steps, bodies, 3].
        historical_rigid_body_ang_vel: Historical body angular velocities [envs, history_steps, bodies, 3].
        historical_ground_heights: Historical ground heights [envs, history_steps].
        history_steps: The 1-based slots to read, in output order.
        local_obs: Whether to use local coordinate frame.
        root_height_obs: Whether to include root height.
        w_last: Whether quaternions use w-last format.
        dt: Environment timestep.
        anchor_body_index: The anchor body; the published anchor index by default.
        time_origin: ``"first_slot"`` ages slot k as (k - 1) * dt, ``"current"``
            as k * dt.

    Returns:
        Historical poses with time [num_envs, num_steps * (obs_dim + 1)].
    """
    poses_flat = compute_historical_max_coords_from_state(
        historical_rigid_body_pos=historical_rigid_body_pos,
        historical_rigid_body_rot=historical_rigid_body_rot,
        historical_rigid_body_vel=historical_rigid_body_vel,
        historical_rigid_body_ang_vel=historical_rigid_body_ang_vel,
        history_steps=history_steps,
        historical_ground_heights=historical_ground_heights,
        local_obs=local_obs,
        root_height_obs=root_height_obs,
        observe_contacts=False,
        w_last=w_last,
        anchor_body_index=anchor_body_index,
    )

    return _append_historical_time(poses_flat, history_steps, dt, time_origin)


def compute_historical_poses_with_time_reduced_coords(
    historical_dof_pos: Tensor,
    historical_dof_vel: Tensor,
    historical_root_local_ang_vel: Tensor,
    historical_root_rot: Tensor,
    historical_anchor_rot: Tensor,
    history_steps: StepSelection,
    w_last: bool,
    dt: float,
    time_origin: TimeOrigin = "first_slot",
) -> Tensor:
    """Compute historical reduced coords observations with time offsets.

    Wraps compute_historical_reduced_coords_from_state and appends each
    selected slot's age (``step_ages`` with ``time_origin``) to its pose token.

    Args:
        historical_dof_pos: Historical DOF positions [envs, history_steps, num_dofs].
        historical_dof_vel: Historical DOF velocities [envs, history_steps, num_dofs].
        historical_root_local_ang_vel: Historical root angular velocity in local frame [envs, history_steps, 3].
        historical_root_rot: Historical root rotation [envs, history_steps, 4].
        historical_anchor_rot: Historical anchor rotation [envs, history_steps, 4].
        history_steps: The 1-based slots to read, in output order.
        w_last: Whether quaternions use w-last format.
        dt: Environment timestep.
        time_origin: ``"first_slot"`` ages slot k as (k - 1) * dt, ``"current"``
            as k * dt.

    Returns:
        Historical poses with time [num_envs, num_steps * (obs_dim + 1)].
    """
    poses_flat = compute_historical_reduced_coords_from_state(
        historical_dof_pos=historical_dof_pos,
        historical_dof_vel=historical_dof_vel,
        historical_root_rot=historical_root_rot,
        historical_root_local_ang_vel=historical_root_local_ang_vel,
        historical_anchor_rot=historical_anchor_rot,
        history_steps=history_steps,
        w_last=w_last,
    )

    return _append_historical_time(poses_flat, history_steps, dt, time_origin)


def compute_historical_max_coords_from_motion_lib(
    motion_lib,
    motion_ids: Tensor,
    motion_times: Tensor,
    dt: float,
    history_steps: StepSelection,
    local_obs: bool = True,
    root_height_obs: bool = True,
    observe_contacts: bool = False,
    contact_body_ids: Tensor = None,
    anchor_body_index: int = EnvContext.current.anchor_idx,
) -> Tensor:
    """Compute historical max_coords observations from motion library reference data.

    Used by AMP discriminator to generate expert reference observations from
    sampled motions in the motion library.

    Args:
        motion_lib: Motion library instance with get_motion_state method.
        motion_ids: Motion indices to sample [num_samples].
        motion_times: Current times in each motion [num_samples].
        dt: Time step between historical observations.
        history_steps: The 1-based steps back in time to read, in output order
            (step k samples the motion k * dt before ``motion_times``).
        local_obs: Whether to use local (heading-aligned) coordinate frame.
        root_height_obs: Whether to include root height observation.
        observe_contacts: Whether to include contact observations.
        contact_body_ids: Optional simulator contact body subset. When provided,
            reference contacts are sliced the same way as BaseEnv reset history.
        anchor_body_index: The anchor body; the published anchor index by default.

    Returns:
        Flattened historical observations [num_samples, len(history_steps) * obs_dim].
    """
    num_samples = motion_ids.shape[0]
    device = motion_ids.device

    # Collect historical states from motion library
    all_obs = []
    for step_idx in history_steps:
        # Time offset going backwards from current time (step 1 = 1*dt back, etc.)
        time_offset = step_idx * dt
        sample_times = (motion_times - time_offset).clamp(min=0.0)
        motion_lengths = getattr(motion_lib, "motion_lengths", None)
        if motion_lengths is not None:
            max_times = motion_lengths[motion_ids].to(
                device=sample_times.device,
                dtype=sample_times.dtype,
            )
            sample_times = torch.min(sample_times, max_times)

        # Get motion state at this time
        ref_state = motion_lib.get_motion_state(motion_ids, sample_times)

        # Compute observation for this frame
        # Use zero ground height for reference (flat ground assumption)
        ground_height = torch.zeros(num_samples, 1, device=device)
        num_bodies = ref_state.rigid_body_pos.shape[1]
        rigid_body_contacts = getattr(ref_state, "rigid_body_contacts", None)
        if observe_contacts and rigid_body_contacts is not None:
            if contact_body_ids is None:
                body_contacts = rigid_body_contacts.bool()
            else:
                contact_body_ids = contact_body_ids.to(
                    device=rigid_body_contacts.device,
                    dtype=torch.long,
                )
                body_contacts = rigid_body_contacts[:, contact_body_ids].bool()
            body_contacts = body_contacts.to(device=device)
        else:
            num_contact_bodies = (
                num_bodies if contact_body_ids is None else contact_body_ids.numel()
            )
            body_contacts = torch.zeros(
                num_samples,
                num_contact_bodies,
                dtype=torch.bool,
                device=device,
            )

        frame_obs = compute_humanoid_max_coords_observations(
            body_pos=ref_state.rigid_body_pos,
            body_rot=ref_state.rigid_body_rot,
            body_vel=ref_state.rigid_body_vel,
            body_ang_vel=ref_state.rigid_body_ang_vel,
            ground_height=ground_height,
            body_contacts=body_contacts,
            local_obs=local_obs,
            root_height_obs=root_height_obs,
            observe_contacts=observe_contacts,
            w_last=True,
            anchor_body_index=anchor_body_index,
        )
        all_obs.append(frame_obs)

    num_steps = len(history_steps)
    # Stack and flatten: [num_samples, num_steps, obs_dim] -> [num_samples, num_steps * obs_dim]
    historical_obs = torch.stack(all_obs, dim=1)
    obs_dim = historical_obs.shape[-1]

    return historical_obs.view(num_samples, num_steps * obs_dim)
