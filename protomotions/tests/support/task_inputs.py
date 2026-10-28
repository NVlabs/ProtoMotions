# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one fake every control test builds its control from.

A control reads its :class:`~protomotions.envs.control.task_inputs.TaskInputs`
and nothing else, so a test needs no env: :func:`make_task_inputs` assembles
a ``TaskInputs`` over a small robot config (:func:`make_robot`), a real
:class:`EpisodeClock` over real :class:`EpisodeBuffers` (write
``inputs.clock.progress_buf[:] = ...`` to set the step count), a flat or
custom ground query, a :class:`UserInterface` with the env's reset key
registered, a :class:`Teleports` log for the animation players' pose writes
and the episode buffers' own ``restart``. The reference query, when a test
needs one, comes from :func:`~protomotions.tests.support.fake_reference.fake_reference`
over motion fakes; :func:`make_robot_state` builds the ``RobotState`` the
lifecycle calls take.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from protomotions.envs.base_env.owners import EpisodeBuffers, EpisodeClock
from protomotions.envs.control.task_inputs import TaskInputs
from protomotions.simulator.base_simulator.simulator_state import (
    ObjectState,
    ResetState,
    RobotState,
    StateConversion,
)
from protomotions.simulator.base_simulator.user_interface import UserInterface


def identity_quat(*shape: int) -> Tensor:
    quat = torch.zeros(*shape, 4)
    quat[..., 3] = 1.0
    return quat


def make_robot(
    body_names: Sequence[str] = ("root",),
    *,
    anchor_body_index: int = 0,
    trackable_bodies_subset: Optional[Sequence[str]] = None,
    head_body_name: Optional[str] = None,
    non_termination_contact_bodies: Union[str, Sequence[str]] = "all",
    mimic_small_marker_bodies: Optional[Sequence[str]] = None,
    semantic_forward_axis_xy: Tuple[float, float] = (1.0, 0.0),
    default_root_height: float = 0.8,
    hinge_axes_map=None,
    default_dof_pos=None,
) -> SimpleNamespace:
    """A robot config with the fields the controls read."""
    body_names = list(body_names)
    naming = {}
    if head_body_name is not None:
        naming["head_body_name"] = [head_body_name]
    return SimpleNamespace(
        anchor_body_index=anchor_body_index,
        trackable_bodies_subset=(
            body_names
            if trackable_bodies_subset is None
            else list(trackable_bodies_subset)
        ),
        kinematic_info=SimpleNamespace(
            body_names=body_names,
            num_bodies=len(body_names),
            hinge_axes_map=hinge_axes_map,
        ),
        common_naming_to_robot_body_names=naming,
        non_termination_contact_bodies=non_termination_contact_bodies,
        mimic_small_marker_bodies=mimic_small_marker_bodies,
        semantic_forward_axis_xy=semantic_forward_axis_xy,
        default_root_height=default_root_height,
        default_dof_pos=default_dof_pos,
    )


def make_robot_state(
    positions,
    *,
    rotations=None,
    velocities=None,
    ang_velocities=None,
    dof_pos=None,
    dof_vel=None,
    contacts=None,
    num_bodies: Optional[int] = None,
) -> RobotState:
    """A ``RobotState`` from body positions ``[E, B, 3]`` (or roots ``[E, 3]``,
    repeated over ``num_bodies`` bodies); everything else zero or identity."""
    positions = torch.as_tensor(positions, dtype=torch.float32)
    if positions.ndim == 2:
        positions = positions[:, None, :].repeat(1, num_bodies or 1, 1)
    num_envs, bodies = positions.shape[:2]

    def tensor_or(value, default):
        return default if value is None else torch.as_tensor(value, dtype=torch.float32)

    dof_dim = 1 if dof_pos is None else torch.as_tensor(dof_pos).shape[-1]
    return RobotState(
        state_conversion=StateConversion.COMMON,
        rigid_body_pos=positions,
        rigid_body_rot=tensor_or(rotations, identity_quat(num_envs, bodies)),
        rigid_body_vel=tensor_or(velocities, torch.zeros(num_envs, bodies, 3)),
        rigid_body_ang_vel=tensor_or(ang_velocities, torch.zeros(num_envs, bodies, 3)),
        rigid_body_contacts=(
            torch.zeros(num_envs, bodies, dtype=torch.bool)
            if contacts is None
            else contacts
        ),
        dof_pos=tensor_or(dof_pos, torch.zeros(num_envs, dof_dim)),
        dof_vel=tensor_or(dof_vel, torch.zeros(num_envs, dof_dim)),
    )


def make_reset_state(num_envs: int = 2, dof_dim: int = 1) -> ResetState:
    return ResetState(
        state_conversion=StateConversion.COMMON,
        root_pos=torch.zeros(num_envs, 3),
        root_rot=identity_quat(num_envs),
        root_vel=torch.zeros(num_envs, 3),
        root_ang_vel=torch.zeros(num_envs, 3),
        dof_pos=torch.zeros(num_envs, dof_dim),
        dof_vel=torch.zeros(num_envs, dof_dim),
    )


def make_object_state(root_pos, root_rot=None, root_vel=None) -> ObjectState:
    root_pos = torch.as_tensor(root_pos, dtype=torch.float32)
    num_envs, num_objects = root_pos.shape[:2]
    return ObjectState(
        state_conversion=StateConversion.COMMON,
        root_pos=root_pos,
        root_rot=(
            identity_quat(num_envs, num_objects)
            if root_rot is None
            else torch.as_tensor(root_rot, dtype=torch.float32)
        ),
        root_vel=(
            torch.zeros(num_envs, num_objects, 3)
            if root_vel is None
            else torch.as_tensor(root_vel, dtype=torch.float32)
        ),
        root_ang_vel=torch.zeros(num_envs, num_objects, 3),
    )


class Teleports:
    """The animation players' pose writes, logged instead of applied."""

    def __init__(self):
        self.requests: List[Tuple[Tensor, ResetState, Optional[ObjectState]]] = []

    def __call__(self, env_ids: Tensor, robot: ResetState, objects) -> None:
        self.requests.append((env_ids.clone(), robot, objects))


def flat_ground(height: float = 0.0) -> Callable[[Tensor], Tensor]:
    """A ground query answering ``height`` everywhere (``[n]``)."""

    def ground_height(points: Tensor) -> Tensor:
        return torch.full((points.shape[0],), height, device=points.device)

    return ground_height


def make_task_inputs(
    num_envs: int,
    *,
    robot: Optional[SimpleNamespace] = None,
    reference=None,
    dt: float = 0.02,
    device: Union[str, torch.device] = "cpu",
    max_episode_length: int = 300,
    ground_height: Union[float, Callable[[Tensor], Tensor]] = 0.0,
    headless: bool = True,
    ui: Optional[UserInterface] = None,
    generator: Optional[torch.Generator] = None,
    teleports: Optional[Teleports] = None,
) -> TaskInputs:
    """A ``TaskInputs`` over fakes; see the module docstring."""
    device = torch.device(device)
    episode = EpisodeBuffers(num_envs, device, max_episode_length)
    ui = UserInterface() if ui is None else ui
    scope = ui.scope("env")
    scope.register("R", "reset", "Reset all environments")
    if not callable(ground_height):
        ground_height = flat_ground(ground_height)
    return TaskInputs(
        num_envs=num_envs,
        device=device,
        dt=dt,
        robot=make_robot() if robot is None else robot,
        clock=EpisodeClock(episode, dt),
        reference=reference,
        ground_height=ground_height,
        generator=generator,
        headless=headless,
        ui=ui,
        viewer_reset=scope.reset,
        restart_episode=episode.restart,
        teleport=Teleports() if teleports is None else teleports,
    )


def motion_fakes(inputs: TaskInputs):
    """The motion fakes a ``fake_reference`` was built over (the test's
    ``SimpleNamespace`` with ``motion_manager`` and ``motion_lib``), for a
    test that moves the clip time or swaps a fake's method after the control
    was built."""
    return inputs.reference._motion


__all__ = [
    "Teleports",
    "motion_fakes",
    "flat_ground",
    "identity_quat",
    "make_object_state",
    "make_reset_state",
    "make_robot",
    "make_robot_state",
    "make_task_inputs",
]
