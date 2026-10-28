# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared float-cast kernel, marked to run eagerly."""

from torch import Tensor

from protomotions.envs.mdp_component import eager_kernel


@eager_kernel
def to_float(x: Tensor) -> Tensor:
    """Convert to float; run eagerly.

    This shared cast receives different tensor ranks and dtypes, which can
    exhaust Dynamo's specialization cache. ``@eager_kernel`` keeps it out of
    ``torch.compile`` so it recompiles nothing; ONNX export can still trace the
    underlying cast.
    """
    return x.float()


__all__ = ["to_float"]
