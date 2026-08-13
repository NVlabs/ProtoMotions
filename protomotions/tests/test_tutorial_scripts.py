# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulator-free checks for tutorial script source contracts."""
from __future__ import annotations

import ast
from pathlib import Path


TUTORIAL_DIR = Path(__file__).resolve().parents[2] / "examples" / "tutorial"


def test_direct_tutorial_robot_configs_declare_semantic_forward_axis():
    """Fresh base RobotConfig examples must declare anatomical forward."""
    missing: list[str] = []
    for path in sorted(TUTORIAL_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "RobotConfig":
                continue
            keyword_names = {keyword.arg for keyword in node.keywords}
            if "semantic_forward_axis_xy" not in keyword_names:
                missing.append(f"{path.name}:{node.lineno}")

    assert missing == []
