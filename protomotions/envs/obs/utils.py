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
"""Utility functions for observation processing."""

from typing import Union, List

import torch
from torch import Tensor


def select_step_indices(
    tensor: Tensor,
    steps: Union[int, List[int]],
    dim: int = 1
) -> Tensor:
    """Select steps from tensor by index.

    Supports both consecutive steps (int) and arbitrary step indices (list).
    Uses 1-indexed step numbers that are converted to 0-indexed tensor positions.

    Args:
        tensor: Input tensor with steps along dim.
        steps: If int N, selects first N steps (like arange(1, N+1)).
               If list, selects specific 1-indexed steps (e.g., [1, 3, 5] -> indices [0, 2, 4]).
        dim: Dimension containing steps.

    Returns:
        Tensor with selected steps.
    """
    if isinstance(steps, int):
        return tensor.narrow(dim, 0, steps)
    else:
        indices = torch.tensor([s - 1 for s in steps], device=tensor.device, dtype=torch.long)
        return tensor.index_select(dim, indices)
