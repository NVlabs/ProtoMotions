# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor values shared by physical state capture, estimation, and history."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Callable, Iterator, Literal, Optional

from torch import Tensor

StateSource = Literal["privileged", "clean", "noisy"]


@dataclass
class BodyState:
    """Body and joint state with optional, explicitly unavailable quantities.

    Leading dimensions are environments, optionally followed by history. Body
    positions and velocities refer to link origins in world coordinates (m,
    m/s); rotations are xyzw; angular velocities are world-frame rad/s. Scalar
    joint positions and velocities are rad and rad/s; joint forces are Nm.
    ``rigid_body_contacts`` stores flags for the selected contact rows, reduced
    across selected partners in the privileged matrix, not independent sensor
    measurements. Its body axis follows ``ContactLayout.common_body_ids``. The
    force matrix itself is current-only and is never stored in this history.

    ``ground_heights`` is world z: terrain under the observation anchor for
    privileged state, or estimated root z minus measured root clearance for a
    sensor estimate. Odometry displacement is anchor XY in its episode-start
    heading frame. The start pose belongs to that same source.

    None means unavailable, not a zero reading. History does not allocate these
    fields and context views do not expose them to component bindings.
    """

    rigid_body_rot: Tensor
    rigid_body_ang_vel: Tensor
    dof_pos: Tensor
    dof_vel: Tensor
    rigid_body_pos: Optional[Tensor] = None
    rigid_body_vel: Optional[Tensor] = None
    ground_heights: Optional[Tensor] = None
    rigid_body_contacts: Optional[Tensor] = None
    dof_forces: Optional[Tensor] = None
    odom_start_xy: Optional[Tensor] = None
    odom_start_heading_inv: Optional[Tensor] = None
    odom_disp_start: Optional[Tensor] = None

    def tensors(self) -> Iterator[tuple[str, Tensor]]:
        """Iterate only available tensor fields, in declaration order."""
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                yield field.name, value

    def map(self, transform: Callable[[Tensor], Tensor]) -> BodyState:
        """Apply a tensor operation without filling unavailable fields."""
        return BodyState(**{name: transform(value) for name, value in self.tensors()})
