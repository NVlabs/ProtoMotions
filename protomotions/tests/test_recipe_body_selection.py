# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example recipes name the bodies a kernel reads, never resolve them by hand.

A recipe spells a body selection as ``BodySelection.named(...)`` /
``.group(...)`` (resolved to indices once by ``build_env``, kept as names in the
saved configuration) or, when it needs the indices at config time, through
``robot_config.body_indices(...)``. Resolving a name with ``body_names.index``
in the recipe hard-codes robot-specific indices into the config and bypasses
both paths, so this test refuses it anywhere under ``examples/experiments``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RECIPES_DIR = Path(__file__).resolve().parents[2] / "examples" / "experiments"
RECIPES = sorted(RECIPES_DIR.rglob("*.py"))


def _body_names_index_lines(tree: ast.AST) -> list[int]:
    """Lines where the module resolves a body name with ``body_names.index``.

    Catches both ``<...>.body_names.index(name)`` and the aliased form
    ``names = <...>.body_names`` followed by ``names.index(name)``.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute):
            if node.value.attr == "body_names":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases.add(target.id)

    offenders: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "index":
            continue
        receiver = node.func.value
        attr_body_names = (
            isinstance(receiver, ast.Attribute) and receiver.attr == "body_names"
        )
        alias_body_names = isinstance(receiver, ast.Name) and receiver.id in aliases
        if attr_body_names or alias_body_names:
            offenders.append(node.lineno)
    return offenders


@pytest.mark.parametrize(
    "recipe", RECIPES, ids=[str(p.relative_to(RECIPES_DIR)) for p in RECIPES]
)
def test_recipe_does_not_resolve_bodies_with_body_names_index(recipe: Path) -> None:
    offenders = _body_names_index_lines(ast.parse(recipe.read_text()))
    assert not offenders, (
        f"{recipe.relative_to(RECIPES_DIR)} resolves body names with "
        f"body_names.index at line(s) {offenders}; use "
        "robot_config.body_indices(...) or BodySelection.named/.group instead."
    )
