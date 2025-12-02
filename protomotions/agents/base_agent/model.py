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
"""Base model interface for agent neural networks.

This module defines the abstract base class that all agent models must implement.
It provides a TensorDictModule interface for clean, compilable models.

Key Classes:
    - BaseModel: Abstract base class for all agent models (TensorDictModule)
"""

from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase
from protomotions.agents.base_agent.config import BaseModelConfig
from abc import abstractmethod


class BaseModel(TensorDictModuleBase):
    """Base class for all agent models.

    All models are TensorDictModules with a single forward method that
    processes observations and returns all model outputs in a TensorDict.

    Args:
        config: Model configuration with architecture parameters.

    Attributes:
        config: Stored configuration for the model.
        in_keys: Input keys for TensorDict (set by subclasses).
        out_keys: Output keys for TensorDict (default: ["action"]).
    """

    def __init__(self, config: BaseModelConfig):
        super().__init__()
        self.config = config

        # Default output keys (subclasses can override)
        self.out_keys = ["action"]
        # in_keys will be set by subclasses based on their architecture
        self.in_keys = []

    @abstractmethod
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """Forward pass through the model.

        Args:
            tensordict: TensorDict containing observations.

        Returns:
            TensorDict with model outputs added.
        """
        pass
