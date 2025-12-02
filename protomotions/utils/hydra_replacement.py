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
"""
Simple replacement for hydra.utils functions to avoid the heavy hydra-core dependency.
Provides get_class and instantiate functions with compatible APIs.
"""

import importlib
from typing import Any


def get_class(path: str) -> type:
    """
    Import and return a class from a string path.

    Args:
        path: Fully qualified class path, e.g., "torch.optim.Adam"

    Returns:
        The class object

    Example:
        >>> Adam = get_class("torch.optim.Adam")
        >>> optimizer = Adam(params, lr=0.001)
    """
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def instantiate(config, **kwargs) -> Any:
    """
    Instantiate a class from a config object.

    Args:
        config: Config object with _target_ attribute specifying the class path,
                or a dict with '_target_' key
        **kwargs: Additional keyword arguments to pass to the constructor,
                  overriding config values

    Returns:
        Instance of the specified class

    Example:
        >>> class Config:
        ...     _target_ = "torch.optim.Adam"
        ...     lr = 0.001
        >>> optimizer = instantiate(config, params=model.parameters())
    """
    # Handle both dict and object configs
    if isinstance(config, dict):
        target = config.get("_target_")
        config_dict = {k: v for k, v in config.items() if k != "_target_"}
    else:
        target = getattr(config, "_target_", None)
        config_dict = {
            k: v
            for k, v in vars(config).items()
            if k != "_target_" and not k.startswith("_")
        }

    if target is None:
        raise ValueError(
            "Config must have a '_target_' attribute or key specifying the class path"
        )

    # Get the class
    cls = get_class(target)

    # Merge config and kwargs (kwargs take precedence)
    merged_kwargs = {**config_dict, **kwargs}

    # Instantiate and return
    return cls(**merged_kwargs)
