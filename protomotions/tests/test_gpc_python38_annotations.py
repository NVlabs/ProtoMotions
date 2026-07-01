# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_PYTHON_PATHS = [
    REPO_ROOT / "deployment",
    REPO_ROOT / "examples",
    REPO_ROOT / "protomotions",
    REPO_ROOT / "scripts",
    REPO_ROOT / "usd_convert",
]
MODERN_BUILTIN_ANNOTATIONS = {"dict", "list", "set", "tuple", "type"}


class ModernAnnotationVisitor(ast.NodeVisitor):
    def __init__(self):
        self.requires_future_annotations = False

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.BitOr):
            self.requires_future_annotations = True
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in MODERN_BUILTIN_ANNOTATIONS
        ):
            self.requires_future_annotations = True
        self.generic_visit(node)


def _annotation_nodes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                if arg.annotation is not None:
                    yield arg.annotation
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                yield node.args.vararg.annotation
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                yield node.args.kwarg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign) and node.annotation is not None:
            yield node.annotation


def _has_future_annotations(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _requires_future_annotations(tree: ast.Module) -> bool:
    visitor = ModernAnnotationVisitor()
    for annotation in _annotation_nodes(tree):
        visitor.visit(annotation)
    return visitor.requires_future_annotations


def test_public_python_paths_postpone_modern_annotations():
    offenders = []
    for directory in PUBLIC_PYTHON_PATHS:
        for path in sorted(directory.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            if _requires_future_annotations(tree) and not _has_future_annotations(tree):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert offenders == []
