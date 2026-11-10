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


def test_scene_creation_uses_the_scene_library_enum_values():
    """The scene tutorial constructs the enum values consumed by SceneLib."""
    from protomotions.components.scene_lib import ReplicationMethod, SubsetMethod

    path = TUTORIAL_DIR / "3_scene_creation.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    statements = [
        node for node in main.body
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "protomotions.components.scene_lib"
        ) or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "scene_lib_config"
                for target in node.targets
            )
        )
    ]
    namespace = {}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"),
         namespace)
    config = namespace["scene_lib_config"]
    assert config.replicate_method is ReplicationMethod.RANDOM
    assert config.subset_method is SubsetMethod.RANDOM
