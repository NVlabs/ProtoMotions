# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_namespace_packages, setup

setup(
    name="protomotions",
    version="3.1",
    packages=find_namespace_packages(include=["protomotions", "protomotions.*"]),
    description="Physics-based Character Animation with Reinforcement Learning",
    author="Chen Tessler, Yifeng Jiang",
    python_requires=">=3.8",
)
