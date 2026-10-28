# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config dataclass annotations stay in syntax every supported Python evaluates.

The saved form (B4) and the override funnel (B3) evaluate a config field's
annotation at run time, ``eval``-ed against the field's module. Every config
module has ``from __future__ import annotations``, so a PEP 604 union
(``float | None``) or a builtin generic (``tuple[float, ...]``) imports fine on
every Python and fails only when evaluated, and only under 3.8 (3.9 for the
unions), where saving a configuration whose field holds ``None``, or overriding
such a field, raises. CI runs 3.10 and cannot see this, so the rule is static:
this test parses every module that defines a config dataclass the saved form can
reach and refuses those two spellings in dataclass field annotations.
``typing.Optional``, ``Union``, ``Tuple``, ``List``, ``Dict``, ``Set`` and ``Type``
are the forms to use.
"""

import ast
import inspect
import sys

from protomotions.envs.control.masked_mimic_prompt import GoalCanvas
from protomotions.envs.control.masked_mimic_tasks.controller import (
    MaskedMimicTaskControlConfig,
)
from protomotions.tests.test_config_annotations_resolve import (
    reachable_config_classes,
)
from protomotions.utils.config_schema import to_tree

# builtin generic -> the typing form that every supported Python evaluates
TYPING_FORMS = {
    "list": "List",
    "tuple": "Tuple",
    "dict": "Dict",
    "set": "Set",
    "frozenset": "FrozenSet",
    "type": "Type",
}


def _is_dataclass(node: ast.ClassDef) -> bool:
    """``@dataclass``, ``@dataclass(...)`` or ``@dataclasses.dataclass`` forms."""
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Attribute):
            name = target.attr
        else:
            name = getattr(target, "id", None)
        if name == "dataclass":
            return True
    return False


def _unsupported_spellings(annotation: ast.AST):
    reasons = set()
    for node in ast.walk(annotation):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            reasons.add("PEP 604 union, use typing.Optional / typing.Union")
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in TYPING_FORMS
        ):
            builtin = node.value.id
            reasons.add(
                f"builtin generic {builtin}[...], use typing.{TYPING_FORMS[builtin]}"
            )
    return sorted(reasons)


def dataclass_field_annotation_violations(source: str, filename: str = "<source>"):
    """Every dataclass field annotation in ``source`` that Python 3.8 cannot evaluate.

    One line per field: where it is, the annotation as written, and what to use.
    """
    violations = []
    for node in ast.walk(ast.parse(source, filename)):
        if not (isinstance(node, ast.ClassDef) and _is_dataclass(node)):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.AnnAssign):
                continue
            reasons = _unsupported_spellings(statement.annotation)
            if reasons:
                field_name = getattr(statement.target, "id", "?")
                text = ast.get_source_segment(source, statement.annotation)
                violations.append(
                    f"{filename}:{statement.lineno} {node.name}.{field_name}: "
                    f"{text}  -- {'; '.join(reasons)}"
                )
    return violations


def test_the_scanner_refuses_both_spellings_and_accepts_the_typing_forms():
    source = (
        "from __future__ import annotations\n"
        "import dataclasses\n"
        "from dataclasses import dataclass, field\n"
        "from typing import Dict, Optional, Tuple\n"
        "\n"
        "@dataclass\n"
        "class A:\n"
        "    a: float | None = None\n"
        "    b: tuple[float, ...] = (1.0,)\n"
        "    c: Optional[Tuple[float, ...]] = None\n"
        "    d: Optional[list[int]] = None\n"
        "    e: Dict[str, int] = field(default_factory=dict)\n"
        "\n"
        "@dataclasses.dataclass(frozen=True)\n"
        "class B:\n"
        "    f: dict[str, int] = field(default_factory=dict)\n"
        "\n"
        "class NotAConfig:\n"
        "    g: str | None = None\n"
    )
    found = dataclass_field_annotation_violations(source, "snippet.py")
    assert [line.split()[1] for line in found] == ["A.a:", "A.b:", "A.d:", "B.f:"]
    assert "PEP 604 union" in found[0] and "typing.Optional" in found[0]
    assert "tuple[...]" in found[1] and "typing.Tuple" in found[1]
    assert "list[...]" in found[2] and "typing.List" in found[2]
    assert "dict[...]" in found[3] and "typing.Dict" in found[3]


def test_every_module_defining_a_reachable_config_dataclass_evaluates_on_38():
    modules = sorted({cls.__module__ for cls in reachable_config_classes()})
    # A floor well below the true count (dozens of config modules across agents,
    # env, control, simulator, sensors, terrains and robots) still catches the
    # module set silently shrinking.
    assert len(modules) > 25, modules

    violations = []
    for name in modules:
        path = inspect.getsourcefile(sys.modules[name])
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        violations.extend(dataclass_field_annotation_violations(source, path))
    assert not violations, "\n".join(
        ["dataclass field annotations Python 3.8 cannot evaluate:"] + violations
    )


def test_the_task_controller_config_saves_with_its_none_defaults():
    """The live break: ``facing``, ``getup`` and ``reach_body_name`` default to None.

    Spelled ``float | None``, the annotation did not evaluate under 3.8, and under
    3.10 it evaluated to a ``types.UnionType`` the saver's Optional check does not
    recognise, so the save was refused on every Python. With ``Optional[float]``
    it saves everywhere, and the None defaults land in the tree as None.
    """
    canvas = GoalCanvas(
        future_steps=(1,),
        num_masked_future_steps=1,
        conditionable_bodies=("pelvis",),
        anchor_body="pelvis",
        anchor_pose_submasking=False,
        has_local_stream=False,
        style_frames=None,
        has_velocity_command=False,
    )
    tree = to_tree(MaskedMimicTaskControlConfig(canvas=canvas), where="task")
    assert tree["facing"] is None
    assert tree["getup"] is None
    assert tree["reach_body_name"] is None
    assert tree["reach_offset"] == [0.3, 0.0, 0.1]  # tuples save as lists
