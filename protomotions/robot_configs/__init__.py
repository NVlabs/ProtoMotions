# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in robot configurations."""

import os

__path__.extend(
    p for p in os.environ.get("PROTOMOTIONS_ROBOT_CONFIG_PATH", "").split(os.pathsep) if p
)
