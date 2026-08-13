# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small fail-fast helpers for distributed agent coordination."""

from typing import Optional

import torch
import torch.distributed as dist


def raise_if_any_rank_failed(
    error: Optional[Exception], operation: str, device: torch.device
) -> None:
    """Raise on every rank if ``operation`` failed on any rank."""
    if dist.is_initialized():
        failed = torch.tensor(error is not None, device=device, dtype=torch.long)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX)
        any_failed = bool(failed.item())
    else:
        any_failed = error is not None

    if not any_failed:
        return
    if error is not None:
        raise RuntimeError(f"{operation} failed") from error
    raise RuntimeError(f"{operation} failed on another rank")
