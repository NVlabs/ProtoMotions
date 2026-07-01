# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulator-free checks for tutorial script source contracts."""
from __future__ import annotations

import ast
from pathlib import Path


TUTORIAL_DIR = Path(__file__).resolve().parents[2] / "examples" / "tutorial"


def test_direct_tutorial_robot_configs_use_current_fields():
    """Tutorials should not pass stale RobotConfig constructor fields."""
    stale: list[str] = []
    stale_fields = {"semantic_forward_axis_xy"}
    for path in sorted(TUTORIAL_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "RobotConfig":
                continue
            keyword_names = {keyword.arg for keyword in node.keywords}
            for keyword in sorted(keyword_names & stale_fields):
                stale.append(f"{path.name}:{node.lineno}:{keyword}")

    assert stale == []
