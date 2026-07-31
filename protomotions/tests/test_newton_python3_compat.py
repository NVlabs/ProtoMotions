# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path


NEWTON_SIMULATOR = (
    Path(__file__).resolve().parents[1] / "simulator" / "newton" / "simulator.py"
)


def test_newton_articulation_view_receives_joint_name_sequence():
    tree = ast.parse(NEWTON_SIMULATOR.read_text(), filename=str(NEWTON_SIMULATOR))
    include_joints_values = [
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ArticulationView"
        for keyword in node.keywords
        if keyword.arg == "include_joints"
    ]

    assert len(include_joints_values) == 1
    value = include_joints_values[0]
    assert isinstance(value, ast.Call)
    assert isinstance(value.func, ast.Name)
    assert value.func.id == "list"
