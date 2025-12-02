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
from dataclasses import dataclass
from typing import Dict, Any, Union, Optional, List
from omegaconf import DictConfig
from protomotions.utils.config_builder import ConfigBuilder
from lightning import fabric

from protomotions.utils.hydra_replacement import instantiate


@dataclass
class FabricConfig(ConfigBuilder):
    """Configuration for Lightning Fabric"""

    accelerator: str = "gpu"
    devices: Union[int, str] = 1
    num_nodes: Union[int, str] = 1
    strategy: Union[Dict, fabric.strategies.Strategy] = fabric.strategies.DDPStrategy()
    precision: Union[str, int] = "32-true"
    loggers: Optional[List[Union[Dict, fabric.loggers.Logger]]] = None
    callbacks: Optional[List[Union[Dict, Any]]] = None

    def __post_init__(self):
        if self.strategy is not None and (
            isinstance(self.strategy, dict) or isinstance(self.strategy, DictConfig)
        ):
            self.strategy = instantiate(self.strategy)
        if self.loggers is not None:
            loggers = []
            for logger in self.loggers:
                if isinstance(logger, dict) or isinstance(logger, DictConfig):
                    loggers.append(instantiate(logger))
                else:
                    loggers.append(logger)
            self.loggers = loggers
        if self.callbacks is not None:
            callbacks = []
            for callback in self.callbacks:
                if isinstance(callback, dict) or isinstance(callback, DictConfig):
                    callbacks.append(instantiate(callback))
                else:
                    callbacks.append(callback)
            self.callbacks = callbacks
