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
"""General utility observation functions."""

from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from protomotions.envs.obs.observation_component import ObservationComponentConfig


def passthrough(tensor: Tensor) -> Tensor:
    """Passthrough function that returns the tensor as-is.
    
    Args:
        tensor: Input tensor.
    
    Returns:
        Same tensor unchanged.
    """
    return tensor


def passthrough_float(tensor: Tensor) -> Tensor:
    """Passthrough function that converts a tensor to float for observations.
    
    Used for masks that need to be included in observations.
    
    Args:
        tensor: Input tensor (can be bool or any type).
    
    Returns:
        Float tensor for observation buffer.
    """
    return tensor.float()


def passthrough_factory(variable: str) -> "ObservationComponentConfig":
    """Factory for passthrough observation that passes a context variable as-is.
    
    Use this to expose any context variable directly as an observation.
    
    Args:
        variable: Name of the context variable to pass through.
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    from protomotions.envs.obs.observation_component import ObservationComponentConfig
    
    return ObservationComponentConfig(
        function=passthrough,
        variables={"tensor": variable},
    )


def passthrough_float_factory(variable: str) -> "ObservationComponentConfig":
    """Factory for passthrough observation that converts a context variable to float.
    
    Use this to expose masks or boolean tensors as float observations.
    
    Args:
        variable: Name of the context variable to pass through as float.
    
    Returns:
        Pre-configured ObservationComponentConfig.
    """
    from protomotions.envs.obs.observation_component import ObservationComponentConfig
    
    return ObservationComponentConfig(
        function=passthrough_float,
        variables={"tensor": variable},
    )

