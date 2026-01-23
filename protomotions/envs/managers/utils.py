# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
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
"""Utility functions for dynamic component managers."""

from typing import List, Optional, Union, Dict

import torch
from torch import Tensor


def resolve_body_indices(
    names_or_indices: Optional[Union[List[int], List[str]]],
    body_names: List[str],
    common_naming_to_robot_body_names: Dict[str, List[str]],
    device: torch.device,
) -> Optional[Tensor]:
    """Convert body names or indices to a tensor of body indices.

    Args:
        names_or_indices: Either:
            - List of body names (strings) - can include common names like "all_left_foot_bodies"
            - List of indices (ints), do nothing just convert to tensor
            - None, return None
        body_names: List of all body names in the robot
        common_naming_to_robot_body_names: Dict mapping common names to lists of actual body names
        device: PyTorch device for the output tensor

    Returns:
        Tensor of body indices on device, or None if input is None/empty.
    """
    if not names_or_indices:  # Handles None and empty list
        return None

    # If already indices, just convert to tensor
    if isinstance(names_or_indices[0], int):
        return torch.tensor(names_or_indices, dtype=torch.long, device=device)

    # Convert names to indices
    indices = []
    for name in names_or_indices:
        # Expand common names to actual body names, or use name directly
        actual_names = common_naming_to_robot_body_names.get(name, [name])

        for actual_name in actual_names:
            if actual_name not in body_names:
                source = (
                    f" (from common name '{name}')" if name in common_naming_to_robot_body_names else ""
                )
                raise ValueError(
                    f"Body name '{actual_name}'{source} not found in: {body_names}"
                )
            indices.append(body_names.index(actual_name))

    return torch.tensor(indices, dtype=torch.long, device=device)

