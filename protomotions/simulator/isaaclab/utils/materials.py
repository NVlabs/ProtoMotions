# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IsaacLab runtime physics-material helpers."""

from typing import Optional, Union

import torch


def set_material_friction(
    materials: torch.Tensor,
    static_friction: Union[float, torch.Tensor],
    dynamic_friction: Optional[Union[float, torch.Tensor]] = None,
) -> None:
    """Update friction columns in-place without changing restitution.

    This operation is shared by the robot-wide baseline and body-level domain
    randomization so both paths operate on the same material tensor.
    """
    if dynamic_friction is None:
        dynamic_friction = static_friction
    materials[..., 0] = static_friction
    materials[..., 1] = dynamic_friction
