# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import torch


def move_friction_tables_to_device(
    friction_dr: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """Return the friction domain-randomization dict with bucket tables on ``device``.

    The base simulator samples the friction bucket tables on CPU because
    IsaacGym and IsaacLab consume them there. Newton indexes the tables with
    bucket ids on the simulation device, so on GPU the tables must be moved
    first or advanced indexing raises a device-mismatch RuntimeError.
    """
    moved = dict(friction_dr)
    for key in ("static_friction", "dynamic_friction", "restitution"):
        table = moved.get(key)
        if table is not None:
            moved[key] = table.to(device)
    return moved
