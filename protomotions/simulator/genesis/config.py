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
from dataclasses import dataclass, field
from protomotions.simulator.base_simulator.config import SimParams, SimulatorConfig


@dataclass
class GenesisSimParams(SimParams):
    """Genesis-specific simulation parameters."""

    substeps: int


@dataclass
class GenesisSimulatorConfig(SimulatorConfig):
    """Configuration specific to Genesis simulator."""

    _target_: str = "protomotions.simulator.genesis.simulator.GenesisSimulator"
    w_last: bool = False
    sim: GenesisSimParams = field(default_factory=GenesisSimParams)
