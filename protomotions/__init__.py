# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ProtoMotions public package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("protomotions")
except PackageNotFoundError:
    __version__ = "source"

__all__ = ["__version__"]
