# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# =============================================================================
# General Config Override Utilities
# =============================================================================

from __future__ import annotations

import logging
import sys
import types
import typing
from dataclasses import MISSING as _MISSING
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from typing import Any, Collection, Dict, Mapping, Optional, Union

# ``X | None`` (PEP 604, Python 3.10+) evaluates to ``types.UnionType``, a
# different origin from ``typing.Union``; the two spell the same thing. On
# 3.8 and 3.9 there is no ``types.UnionType`` and the tuple holds ``Union`` twice.
_UNION_ORIGINS = (Union, getattr(types, "UnionType", Union))


def is_union(ann) -> bool:
    """Whether a resolved annotation is a union in either spelling."""
    return typing.get_origin(ann) in _UNION_ORIGINS


def declaring_module(cls, name: str) -> str:
    """The module of the class in ``cls``'s MRO that declares field ``name``.

    A field's annotation string (``from __future__ import annotations``) names
    types imported where the field is DECLARED; a subclass in another module
    inherits the field without those imports, so the string is evaluated against
    the declaring class's module, as ``typing.get_type_hints`` does. Falls back to
    ``cls.__module__`` for a name no class in the MRO annotates.
    """
    for klass in cls.__mro__:
        if name in vars(klass).get("__annotations__", {}):
            return klass.__module__
    return cls.__module__


log = logging.getLogger(__name__)

ROOTS_SENTENCE = "robot, simulator, terrain, scene_lib, motion_lib, env, agent"


def resolve_field_annotation(cls, name: str):
    """The resolved annotation of ``cls.name``, from that field alone; raises TypeError
    naming the field when the annotation string cannot be evaluated.

    The one per-field resolver shared by the B4 saved form (``config_schema``) and the
    B3 override funnel. Resolving the whole class (``get_type_hints``) would fail
    whenever any OTHER field carries a name that does not resolve; resolving per field
    keeps one unrelated sibling from breaking every field being saved or overridden. The
    field's type must be importable at the level of the module that DECLARES the field
    (an inherited field resolves in its base class's module, as ``get_type_hints`` does;
    config.py imports MdpComponent and TerminationRule there) and spelled so this Python
    can evaluate it (PEP 604 unions and builtin generics need 3.10 / 3.9); an annotation
    that does not resolve is a real error, not a case to silence by returning ``Any``
    (which would drop the None-for-non-Optional refusal at save).
    """
    field = next((f for f in fields(cls) if f.name == name), None)
    if field is None:
        return Any
    ann = field.type
    if isinstance(ann, str):
        module = declaring_module(cls, name)
        try:
            ann = eval(  # noqa: S307 - evaluating a dataclass field annotation
                ann, vars(sys.modules[module])
            )
        except Exception as error:
            raise TypeError(
                f"annotation {ann!r} of {cls.__name__}.{name} does not resolve: its "
                f"type is not importable at {module} module level (only under "
                "TYPE_CHECKING), or the annotation uses syntax this Python cannot "
                "evaluate (PEP 604 unions and builtin generics need 3.10 / 3.9)"
            ) from error
    return ann


def _override_annotation(cls, name: str, path: str):
    """``resolve_field_annotation`` with the override funnel's error context."""
    try:
        return resolve_field_annotation(cls, name)
    except TypeError as error:
        raise TypeError(f"cannot type override {path!r}: {error}") from error


def _require_field(cls, name: str, path: str) -> None:
    """A dataclass step must name a declared ``init=True`` field, not a property."""
    by_name = {f.name: f for f in fields(cls)}
    field = by_name.get(name)
    if field is not None and field.init:
        return
    derived = (field is not None and not field.init) or isinstance(
        getattr(cls, name, None), property
    )
    suffix = (
        " (it is derived; override the field it is derived from)" if derived else ""
    )
    raise ValueError(
        f"{path!r}: {name!r} is not a declared field of {cls.__name__}{suffix}"
    )


def _coerce_enum(value, enum_cls, path: str):
    for member in enum_cls:
        if value == member.name or value == member.value:
            return member
    values = [m.value for m in enum_cls]
    raise ValueError(
        f"invalid value {value!r} for {enum_cls.__name__} at {path!r}; valid: {values}"
    )


def _mismatch(path, value, ann, cls, name) -> ValueError:
    return ValueError(
        f"override {path}={value!r}: expected {ann} "
        f"({cls.__name__}.{name}), got {type(value).__name__}"
    )


def coerce(value, ann, path, cls, name):
    """Coerce a parsed literal to a field's declared annotation (never its runtime value)."""
    if ann is Any or ann is None:
        return value
    if ann is type(None):
        if value is None:
            return None
        raise _mismatch(path, value, ann, cls, name)
    if ann is bool:
        if isinstance(value, bool):
            return value
        raise _mismatch(path, value, ann, cls, name)
    if ann is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _mismatch(path, value, ann, cls, name)
        return value
    if ann is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _mismatch(path, value, ann, cls, name)
        return float(value)
    if ann is str:
        if isinstance(value, str):
            return value
        raise _mismatch(path, value, ann, cls, name)
    if isinstance(ann, type) and issubclass(ann, Enum):
        return _coerce_enum(value, ann, path)

    origin = typing.get_origin(ann)
    args = typing.get_args(ann)

    if is_union(ann):
        if value is None and type(None) in args:
            return None
        members = [m for m in args if m is not type(None)]
        if isinstance(value, dict):
            for member in members:
                if is_dataclass(member):
                    return member(**value)  # constructor errors surface
        for member in members:
            try:
                return coerce(value, member, path, cls, name)
            except (ValueError, TypeError):
                continue
        raise ValueError(
            f"override {path}={value!r}: no member of {ann} "
            f"({cls.__name__}.{name}) accepts it; members are {list(members)}"
        )
    if origin in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise _mismatch(path, value, ann, cls, name)
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            args = (args[0],) * len(value)
        if args and origin is list:
            args = (args[0],) * len(value)
        items = [coerce(v, a, path, cls, name) for v, a in zip(value, args or value)]
        return items if origin is list else tuple(items)
    if origin is dict:
        return value
    if origin is typing.Literal:
        if value in args:
            return value
        raise ValueError(
            f"override {path}={value!r}: not one of {list(args)} ({cls.__name__}.{name})"
        )
    if is_dataclass(ann) and isinstance(value, dict):
        return ann(**value)
    if ann is dict or ann is list or ann is tuple:
        return value
    raise _mismatch(path, value, ann, cls, name)


def _descend(obj, name: str, path: str):
    """One step into ``obj`` by ``name``; raises if the step is not legal."""
    if is_dataclass(obj) and not isinstance(obj, type):
        _require_field(type(obj), name, path)
        return getattr(obj, name)
    if isinstance(obj, dict):
        if name not in obj:
            raise ValueError(
                f"{path!r}: {name!r} is not a key of the dict at this step"
            )
        return obj[name]
    if not hasattr(obj, name):
        raise ValueError(f"{path!r}: {type(obj).__name__} has no attribute {name!r}")
    return getattr(obj, name)


def _construct_optional(obj, name: str, path: str):
    """A ``None`` on the way is built from its annotation when it has no required fields."""
    if not (is_dataclass(obj) and not isinstance(obj, type)):
        return None
    ann = _override_annotation(type(obj), name, path)
    members = typing.get_args(ann) if is_union(ann) else (ann,)
    target = next((m for m in members if isinstance(m, type) and is_dataclass(m)), None)
    if target is None:
        return None
    required = [
        f.name
        for f in fields(target)
        if f.default is _MISSING and f.default_factory is _MISSING
    ]
    if required:
        raise ValueError(
            f"{path!r} crosses {name} which is None and {target.__name__} has "
            f"required fields; set {name} first"
        )
    return target()


def _assign(obj, name: str, value):
    """Write ``value`` at ``obj.name``, rebuilding a frozen dataclass in place."""
    if isinstance(obj, dict):
        obj[name] = value
        return obj
    if is_dataclass(obj) and obj.__dataclass_params__.frozen:
        return replace(obj, **{name: value})
    setattr(obj, name, value)
    return obj


def _set_path(obj, parts, path, value):
    """Set ``value`` at ``parts`` under ``obj``; returns (new ``obj``, coerced value)."""
    name = parts[0]
    if len(parts) == 1:
        if is_dataclass(obj) and not isinstance(obj, type):
            _require_field(type(obj), name, path)
            ann = _override_annotation(type(obj), name, path)
            coerced = coerce(value, ann, path, type(obj), name)
        elif isinstance(obj, dict):
            if name not in obj:
                raise ValueError(
                    f"{path!r}: {name!r} is not a key of the dict at this step"
                )
            coerced = value
        else:
            if not hasattr(obj, name):
                raise ValueError(
                    f"{path!r}: {type(obj).__name__} has no attribute {name!r}"
                )
            coerced = value
        return _assign(obj, name, coerced), coerced
    child = _descend(obj, name, path)
    if child is None:
        child = _construct_optional(obj, name, path)
    new_child, coerced = _set_path(child, parts[1:], path, value)
    return _assign(obj, name, new_child), coerced


def _resolve_root(configs, root: str, path: str):
    from protomotions.utils.experiment_configs import ExperimentConfigs

    if root not in ExperimentConfigs.ROOTS:
        raise ValueError(
            f"unknown override root {root!r} in {path!r}; roots are {ROOTS_SENTENCE}"
        )
    obj = getattr(configs, root)
    if root == "agent" and obj is None:
        raise ValueError(
            f"cannot override {path!r}: this experiment has no agent config"
        )
    return obj


def _apply_one(configs, path: str, value):
    root, *rest = path.split(".")
    obj = _resolve_root(configs, root, path)
    if not rest:
        raise ValueError(f"override {path!r} names only a root; expected root.field")
    if root == "robot":
        # Robot writes re-derive __post_init__ facts through update_fields, which
        # also refuses a derived field (D's anchor property). The walker refuses a
        # property first, so anchor_body_index=... fails with the generic message.
        first = rest[0]
        _require_field(type(obj), first, path)
        if len(rest) == 1:
            ann = _override_annotation(type(obj), first, path)
            coerced = coerce(value, ann, path, type(obj), first)
        else:
            child = getattr(obj, first)
            if child is None:
                child = _construct_optional(obj, first, path)
            child, coerced = _set_path(child, rest[1:], path, value)
        obj.update_fields(**{first: child if len(rest) > 1 else coerced})
        return coerced
    new_obj, coerced = _set_path(obj, rest, path, value)
    if new_obj is not obj:
        setattr(configs, root, new_obj)
    return coerced


def apply_overrides(
    configs,
    overrides: Mapping[str, Any],
    roots: Optional[Collection[str]] = None,
) -> Dict[str, Any]:
    """Apply ``{root.path: value}`` overrides to an ExperimentConfigs in place.

    The leaf is typed by its dataclass annotation, never by the runtime value, so
    a typo or a wrong type fails at parse time naming the field. ``roots`` limits
    which roots are applied (``None`` = all), which lets the build funnel apply
    ``robot`` and ``simulator`` early and the rest after the build. Returns the
    coerced ``{path: value}`` that landed, for the undo check.
    """
    from protomotions.utils.experiment_configs import ExperimentConfigs

    applied: Dict[str, Any] = {}
    for path, value in overrides.items():
        root = path.split(".", 1)[0]
        # Refuse an unknown root even when this pass filters it out, so a typo'd
        # root is not silently dropped by the early/late split in build_configs.
        if root not in ExperimentConfigs.ROOTS:
            raise ValueError(
                f"unknown override root {root!r} in {path!r}; roots are {ROOTS_SENTENCE}"
            )
        if roots is not None and root not in roots:
            continue
        applied[path] = _apply_one(configs, path, value)
    if applied:
        log.info(f"Applied {len(applied)} config override(s): {sorted(applied)}")
    return applied


def read_override(configs, path: str):
    """Read the current value at a ``root.path`` override path on an ExperimentConfigs.

    Walks the way the funnel does (a dict step by key, a dataclass or any other
    object by attribute). The inference derivation and the build-time undo check
    read back through this to see whether a builder step changed an early override.
    """
    root, *rest = path.split(".")
    obj = _resolve_root(configs, root, path)
    for part in rest:
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    return obj


def parse_cli_overrides(override_strings: list) -> Dict[str, Any]:
    """
    Parse command-line override strings into a dictionary.

    Supports the format: "key=value" where value can be:
    - Numbers: "env.max_episode_length=1000"
    - Floats: "agent.learning_rate=1e-5"
    - Booleans: "env.enable_terrain=True"
    - Strings: "env.terrain.type=flat"
    - None: "env.early_termination=None"

    Args:
        override_strings: List of "key=value" strings

    Returns:
        Dictionary of parsed overrides

    Raises:
        ValueError: If any override string is not in "key=value" format or key is empty.

    Example:
        parse_cli_overrides(["env.max_episode_length=1000", "simulator.num_envs=4096"])
        # Returns: {"env.max_episode_length": 1000, "simulator.num_envs": 4096}
    """
    overrides = {}

    for override_str in override_strings:
        if "=" not in override_str:
            raise ValueError(
                f"Invalid override format (required key=value): {override_str!r}. "
                "Use e.g. robot.asset.asset_file_name=mjcf/g1_holo_compat.xml"
            )

        key, value_str = override_str.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()

        if not key:
            raise ValueError(
                f"Invalid override: key is empty in {override_str!r}. "
                "Use format key=value (e.g. robot.asset.asset_file_name=path.xml)."
            )

        try:
            import ast

            value = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            value = value_str

        overrides[key] = value

    return overrides
