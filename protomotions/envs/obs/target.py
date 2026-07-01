# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observation compute kernel for target-reaching tasks."""

from torch import Tensor

from protomotions.envs.obs.utils import heading_local_xy_delta


def compute_target_obs(root_pos: Tensor, root_rot: Tensor, tar_pos: Tensor) -> Tensor:
    """Return target XY offset in the humanoid heading frame."""
    return heading_local_xy_delta(root_pos, root_rot, tar_pos, w_last=True)


__all__ = ["compute_target_obs"]
