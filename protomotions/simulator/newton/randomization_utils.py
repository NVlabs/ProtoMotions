# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import torch

from protomotions.simulator.base_simulator.utils import (
    get_friction_bucket_count,
    get_friction_table,
)


def move_friction_tables_to_device(
    friction_dr: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """Return friction bucket tables on the target device without mutating input."""
    moved = dict(friction_dr)
    for key in ("static_friction", "dynamic_friction", "restitution"):
        table = moved.get(key)
        if table is not None:
            moved[key] = table.to(device)
    return moved

__all__ = [
    "get_friction_bucket_count",
    "get_friction_table",
    "move_friction_tables_to_device",
]
