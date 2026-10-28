# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The text saved form of a config bundle: a tagged tree rebuilt through the real constructors.

``to_tree`` emits a JSON-able tagged tree for any config value; ``from_tree``
rebuilds it by calling every object's real constructor, so ``__post_init__`` runs
and validation happens by construction. An old or hand-edited bundle fails naming
the field, the class and the file with the regenerate command. No pickle, no
tensor-valued field and no kernel referenced by anything but its importable name
survive on the config path.
"""

from __future__ import annotations

import dataclasses
import enum
import inspect
import logging
import math
import typing
from typing import Any

import torch

from protomotions.utils.config_utils import is_union, resolve_field_annotation
from protomotions.utils.hydra_replacement import get_class

log = logging.getLogger(__name__)

FORMAT_VERSION = 1

_REGEN = "regenerate from the experiment_config.py and config.yaml saved beside it"

# Reserved tag keys. The leading "$" is what makes a tag unambiguous: a config
# dict can never carry a "$"-key (refused at save), so a dict holding one of these
# is always a tag, never user data. The companion keys (name, dtype, bindings,
# params, meta, required_keys, init) are plain -- they only appear inside a tag,
# so a dataclass may even declare a field named "type", "name" or "dtype".
_TYPE = "$type"
_ENUM = "$enum"
_TENSOR = "$tensor"
_FUNCTION = "$function"
_PATH = "$path"
_KERNEL = "$kernel"
_KERNEL_OBJECT = "$kernel_object"


# --------------------------------------------------------------------------- #
# Annotation resolution (per field, shared with the B3 override funnel:
# config_utils.resolve_field_annotation).
# --------------------------------------------------------------------------- #
def _admits_none(ann) -> bool:
    """Whether a resolved annotation permits ``None``: ``Any``, ``NoneType`` or an
    Optional in either spelling (``Optional[X]`` / ``Union[X, None]``, or ``X | None``,
    which on Python 3.10+ evaluates to a ``types.UnionType``, not a ``typing.Union``).
    """
    if ann is Any or ann is None or ann is type(None):
        return True
    if is_union(ann):
        return type(None) in typing.get_args(ann)
    return False


def _unwrap_optional(ann):
    """Strip a single Optional wrapper (``Optional[C]`` / ``Union[C, None]``) to the
    contained type, so a field typed ``Optional[List[X]]`` or ``Optional[Dict[K, V]]``
    still threads its container's element / value annotation on load. A union of two
    non-None types is left as is -- there is no single container to thread through.
    """
    if is_union(ann):
        non_none = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return ann


# --------------------------------------------------------------------------- #
# Emit.
# --------------------------------------------------------------------------- #
def _is_named_function(obj) -> bool:
    """A module-level function, importable by ``module.qualname``.

    A lambda or a function defined inside another (``<locals>`` in its qualname)
    is not importable, so it is not a named function and is refused at save
    rather than saving a name that fails to load.
    """
    import types

    return (
        isinstance(obj, types.FunctionType)
        and obj.__name__ != "<lambda>"
        and "<locals>" not in obj.__qualname__
    )


def _qualname(obj) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _component_tree(component, where: str) -> dict:
    """An MdpComponent as kernel (or kernel object) plus bindings, params and meta.

    The static params are canonicalized at save: the kernel signature's
    unoverridden plain defaults are filled in (and FieldPath defaults become
    bindings), exactly what ``from_kernel`` produces, so the saved form is
    complete and self-contained whether the component was built directly or by
    ``from_kernel``, and the two forms are tree-equal.
    """
    from protomotions.envs.context_paths import FieldPath

    kernel = component.get_compute_func()
    tree: dict
    if _is_named_function(kernel):
        tree = {_KERNEL: _qualname(kernel)}
    elif hasattr(kernel, "export_init") and hasattr(type(kernel), "from_init"):
        # A saveable kernel object (export_init/from_init) persists exactly as to_tree
        # writes it in any position ({$kernel_object, init}).
        tree = to_tree(kernel, where=where)
    else:
        # Neither a module-level function nor a saveable kernel object. Refuse at save
        # with a clear message, rather than delegating to to_tree: a callable dataclass
        # kernel would otherwise save as a $type config and reload as a dataclass used
        # as a kernel, and to_tree's generic refusal would mis-advise "regenerate".
        name = getattr(kernel, "__qualname__", None) or type(kernel).__name__
        raise ValueError(
            f"cannot save {where}: the component's kernel {name} is neither a "
            "module-level function nor a saveable kernel object (export_init/from_init); "
            "write a named function or a kernel object"
        )

    bindings = dict(component.get_bindings_dict())
    params = dict(component.get_params())
    overridden = set(bindings) | set(params)
    for name, param in inspect.signature(kernel).parameters.items():
        if name in overridden or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        default = param.default
        if default is inspect.Parameter.empty:
            continue
        if isinstance(default, FieldPath):
            bindings[name] = default.path
        else:
            params[name] = default

    tree["bindings"] = bindings
    tree["params"] = {
        name: to_tree(value, where=f"{where}.params.{name}")
        for name, value in params.items()
    }
    if component.meta is not None:
        tree["meta"] = to_tree(component.meta, where=f"{where}.meta")
    if component.required_keys:
        tree["required_keys"] = {k: list(v) for k, v in component.required_keys.items()}
    return tree


def to_tree(obj, *, where: str):
    """A JSON-able tagged tree for a config value; raises naming ``where`` on a bad value."""
    from protomotions.envs.context_paths import FieldPath
    from protomotions.envs.mdp_component import is_mdp_component

    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"cannot save {where}: non-finite float {obj!r}")
        return obj
    if isinstance(obj, enum.Enum):
        return {_ENUM: _qualname(type(obj)), "name": obj.name}
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        return {_TENSOR: t.tolist(), "dtype": str(t.dtype).replace("torch.", "")}
    if isinstance(obj, FieldPath):
        return {_PATH: obj.path}
    if is_mdp_component(obj):
        return _component_tree(obj, where)
    if hasattr(obj, "export_init") and hasattr(type(obj), "from_init"):
        # A kernel object (baked-FK and friends) in any position -- the kernel slot of
        # a component, or here a static param -- persists as its exported init and
        # rebuilds through from_init on load, the same form _component_tree writes in
        # the kernel slot. Checked BEFORE the dataclass branch: a kernel object that is
        # also a @dataclass must save as {$kernel_object, init}, not as a $type config
        # (which would reload as a dataclass used as a kernel and die at the binding
        # check with a misleading "has no field 'bindings'" message).
        return {
            _KERNEL_OBJECT: _qualname(type(obj)),
            "init": to_tree(obj.export_init(), where=f"{where}.init"),
        }
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        cls = type(obj)
        tree = {_TYPE: _qualname(cls)}
        for field in dataclasses.fields(cls):
            if not field.init:
                continue
            value = getattr(obj, field.name)
            # The emit refusal is for a None where the field's declared type is not
            # Optional -- a hook that left a component table ``= None`` dies here.
            # Other values emit their natural tagged form (a str for an enum field,
            # a tensor for a List[float] field) and are validated on load by the
            # real constructor.
            try:
                annotation = resolve_field_annotation(cls, field.name)
            except TypeError as error:
                raise ValueError(
                    f"cannot save {where}.{field.name}: {error}"
                ) from error
            if value is None and not _admits_none(annotation):
                raise ValueError(
                    f"cannot save {where}.{field.name}: value None does not match "
                    f"the declared type {annotation}"
                )
            tree[field.name] = to_tree(value, where=f"{where}.{field.name}")
        return tree
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"cannot save {where}: dict key {key!r} is not a string"
                )
            if key.startswith("$"):
                raise ValueError(
                    f"cannot save {where}: dict key {key!r} starts with '$', which "
                    "is reserved for the saved form's tags"
                )
            out[key] = to_tree(value, where=f"{where}.{key}")
        return out
    if isinstance(obj, (list, tuple)):
        return [to_tree(v, where=f"{where}[{i}]") for i, v in enumerate(obj)]
    if _is_named_function(obj):
        return {_FUNCTION: _qualname(obj)}
    if callable(obj):
        raise ValueError(
            f"cannot save {where}: {type(obj).__name__} is not a module-level "
            "function; write a named function"
        )
    raise ValueError(f"cannot save {where}: cannot serialize {type(obj).__name__}")


# --------------------------------------------------------------------------- #
# Load.
# --------------------------------------------------------------------------- #
_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "int32": torch.int32,
    "bool": torch.bool,
}


def _import(name: str, kind: str, where: str):
    try:
        return get_class(name)
    except Exception:
        raise ValueError(
            f"bundle {where}: {name!r} is not importable or not a {kind}; it was "
            f"renamed or removed; {_REGEN}"
        )


def _kernel_from_init(obj_cls_name: str, init_tree, robot, where: str):
    """Rebuild a kernel object from its saved ``{$kernel_object, init}``.

    Import the class, refuse by name if it is not a saveable kernel object (no
    ``from_init``), and call ``from_init(robot, **init)`` under the bundle-path
    prefix. Shared by the component kernel slot and a bare kernel object saved as
    a static param, so both positions behave identically. from_init's own message
    (e.g. ``baked_fk._check_init``'s saved-constants mismatch) carries the
    regenerate command, so it is added only when a rarer error leaves it absent.
    """
    obj_cls = _import(obj_cls_name, "kernel object", where)
    if not hasattr(obj_cls, "from_init"):
        raise ValueError(
            f"bundle {where}: kernel object {obj_cls_name!r} has no "
            f"from_init(robot, **init); it is not a saveable kernel object; {_REGEN}"
        )
    init = from_tree(init_tree, Any, where=f"{where}.init", robot=robot)
    try:
        return obj_cls.from_init(robot, **init)
    except (TypeError, ValueError) as error:
        text = str(error)
        suffix = "" if "regenerate" in text else f"; {_REGEN}"
        raise ValueError(f"bundle {where}: {text}{suffix}") from error


def _rebuild_component(tree: dict, robot, where: str):
    from protomotions.envs.context_paths import resolve_path
    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    # The writer always emits "bindings" and "params" (each an object, even
    # when empty), and "init" for a kernel object; a subtree that is missing one
    # or carries a non-object in its place is truncated or hand-edited. Refuse it
    # here, naming the key, rather than defaulting the absent key to {} -- which
    # would silently load an unbound component that fails much later at build
    # with a confusing missing-binding message.
    is_kernel_object = _KERNEL_OBJECT in tree
    required = ["bindings", "params"]
    if is_kernel_object:
        required.append("init")
    malformed = [key for key in required if not isinstance(tree.get(key), dict)]
    if malformed:
        raise ValueError(
            f"bundle {where}: component subtree is missing or malformed at "
            f"{malformed} (truncated or hand-edited); {_REGEN}"
        )

    # Refuse a key the writer never emits. A misspelled "meta" would otherwise
    # load as meta=None and fail later at build with a recipe-pointing message; a
    # misspelled "required_keys" would drop the check with no error at all. The
    # tag plus bindings/params/meta/required_keys (and init for a kernel object)
    # are the whole component grammar.
    allowed = {
        _KERNEL_OBJECT if is_kernel_object else _KERNEL,
        "bindings",
        "params",
        "meta",
        "required_keys",
    }
    if is_kernel_object:
        allowed.add("init")
    unknown = sorted(set(tree) - allowed)
    if unknown:
        raise ValueError(
            f"bundle {where}: component subtree has unknown key(s) {unknown} "
            f"(the grammar is {sorted(allowed)}); {_REGEN}"
        )

    # The optional keys, when present, must have the writer's shape: required_keys
    # an object of string lists (a stray string would be split into characters by
    # list(v) below, sidestepping MdpComponent's own guard), and meta a tagged
    # object (a scalar would load as a bogus meta).
    raw_required_keys = tree.get("required_keys", {})
    if not isinstance(raw_required_keys, dict) or not all(
        isinstance(v, list) and all(isinstance(e, str) for e in v)
        for v in raw_required_keys.values()
    ):
        raise ValueError(
            f"bundle {where}: component 'required_keys' must be an object of "
            f"string lists; {_REGEN}"
        )
    if "meta" in tree and not isinstance(tree["meta"], dict):
        raise ValueError(
            f"bundle {where}: component 'meta' must be an object; {_REGEN}"
        )

    if is_kernel_object:
        kernel = _kernel_from_init(tree[_KERNEL_OBJECT], tree["init"], robot, where)
    else:
        kernel = _import(tree[_KERNEL], "kernel", where)
    bindings = {
        name: resolve_path(EnvContext, path) for name, path in tree["bindings"].items()
    }
    params = {
        name: from_tree(value, Any, where=f"{where}.params.{name}", robot=robot)
        for name, value in tree["params"].items()
    }
    meta = (
        from_tree(tree["meta"], Any, where=f"{where}.meta", robot=robot)
        if "meta" in tree
        else None
    )
    required_keys = {k: list(v) for k, v in raw_required_keys.items()}
    # Direct reconstruction (not from_kernel): the writer canonicalizes the saved
    # bindings + params so they fully describe the component, whereas from_kernel
    # re-injects the kernel's signature defaults -- absent from a directly-built
    # component -- which would make the round-trip unequal. The gate is
    # tree-equality, not __eq__.
    return MdpComponent(
        kernel, bindings, params, meta=meta, required_keys=required_keys or None
    )


def _validate_target(target, where: str) -> None:
    """Import-check a config's ``_target_`` class, so a rename fails at load.

    A ``_target_`` names the class instantiated for this config at build. Resolving
    it here turns a renamed or moved class into a load-time refusal naming the
    bundle path, like a bad ``$type``, instead of a bare ``AttributeError`` much
    later at build. A missing THIRD-PARTY package (e.g. an uninstalled simulator a
    sim2sim bundle still names) is logged and skipped -- an install matter, not a
    bundle defect; a missing or renamed ``protomotions`` module, or a class gone
    from a module that does import, is refused with the regenerate command.

    Simulator targets (``protomotions.simulator.*``) are the one exception: they
    are NOT import-checked here, because isaacgym and isaaclab must be imported
    before torch (which this module imports at the top), so importing a
    stored-but-not-running backend's class at load would violate that order and
    hard-fail. The backend actually built is import-checked where the launcher has
    already imported it in the right order -- at the inference swap
    (``factory.update_simulator_config_for_test``) and again at build.
    """
    if not isinstance(target, str) or not target:
        return
    if "." not in target:
        raise ValueError(
            f"bundle {where}: _target_ {target!r} is not a module.Class path; {_REGEN}"
        )
    if target.startswith("protomotions.simulator."):
        return  # import-order-sensitive; checked at the backend swap and at build
    try:
        get_class(target)
    except ModuleNotFoundError as error:
        missing = error.name or ""
        if missing == "protomotions" or missing.startswith("protomotions."):
            raise ValueError(
                f"bundle {where}: _target_ {target!r} needs module {missing!r}, "
                f"gone (renamed or moved); {_REGEN}"
            ) from error
        log.info(
            "bundle %s: _target_ %r not checkable here -- %s is not installed",
            where,
            target,
            missing,
        )
    except ImportError as error:
        missing = getattr(error, "name", None) or ""
        if missing == "protomotions" or missing.startswith("protomotions."):
            raise ValueError(
                f"bundle {where}: _target_ {target!r} is not importable "
                f"({error}); {_REGEN}"
            ) from error
        log.info(
            "bundle %s: _target_ %r not checkable here -- %s", where, target, error
        )
    except AttributeError as error:
        raise ValueError(
            f"bundle {where}: _target_ {target!r} is importable but the class is "
            f"gone (renamed or removed); {_REGEN}"
        ) from error


def from_tree(tree, annotation, *, where: str, robot=None):
    """Rebuild a config value from its tagged tree through the real constructor."""
    if isinstance(tree, dict):
        # A tag is recognized by its reserved "$"-key, which a plain config dict
        # can never carry (refused at save), so a user dict keyed "path", "kernel"
        # or "type" is never mistaken for a tag.
        if _TYPE in tree:
            cls = _import(tree[_TYPE], "dataclass", where)
            if not dataclasses.is_dataclass(cls):
                raise ValueError(
                    f"bundle {where}: {tree[_TYPE]!r} is not importable or not a "
                    f"dataclass; it was renamed or removed; {_REGEN}"
                )
            declared = {f.name: f for f in dataclasses.fields(cls)}
            kwargs = {}
            for key, sub in tree.items():
                if key == _TYPE:
                    continue
                field = declared.get(key)
                if field is None or not field.init:
                    names = sorted(f.name for f in dataclasses.fields(cls) if f.init)
                    raise ValueError(
                        f"bundle {where}: {cls.__name__} has no field {key!r} "
                        f"(declared fields: {names}); {_REGEN}"
                    )
                try:
                    key_annotation = resolve_field_annotation(cls, key)
                except TypeError as error:
                    raise ValueError(
                        f"bundle {where}.{key}: {error}; {_REGEN}"
                    ) from error
                kwargs[key] = from_tree(
                    sub, key_annotation, where=f"{where}.{key}", robot=robot
                )
            try:
                obj = cls(**kwargs)
            except (TypeError, ValueError) as error:
                # TypeError: a wrong/missing constructor argument. ValueError: a
                # __post_init__ that refuses its inputs (e.g. a robot config
                # missing its heading axis). Both keep the "bundle <where>" prefix
                # so the failure names the field, not just the constructor.
                raise ValueError(f"bundle {where}: {cls.__name__}: {error}; {_REGEN}")
            # A config's _target_ names the class instantiated for it at build;
            # resolve it now so a renamed class fails at load, not bare at build.
            _validate_target(getattr(obj, "_target_", None), where)
            return obj
        if _ENUM in tree:
            return _import(tree[_ENUM], "enum", where)[tree["name"]]
        if _TENSOR in tree:
            dtype = _DTYPES.get(tree["dtype"])
            if dtype is None:
                raise ValueError(
                    f"bundle {where}: tensor dtype {tree['dtype']!r} is not one of "
                    f"{sorted(_DTYPES)}; {_REGEN}"
                )
            return torch.tensor(tree[_TENSOR], dtype=dtype)
        if _FUNCTION in tree:
            return _import(tree[_FUNCTION], "function", where)
        if _PATH in tree:
            from protomotions.envs.context_paths import resolve_path
            from protomotions.envs.context_views import EnvContext

            return resolve_path(EnvContext, tree[_PATH])
        if _KERNEL in tree or _KERNEL_OBJECT in tree:
            # A full component carries bindings + params; a kernel object saved as
            # a bare static param carries neither (just the tag + init). Route by
            # shape: a subtree with either key goes to the component path, whose
            # #247 guard refuses it if the other is missing. A subtree with neither
            # is taken as a bare kernel object. The writer always emits both keys for
            # a real component, so only a hand-edit that strips both reaches here as a
            # bare object; it then rebuilds INTO the component slot (EnvConfig does not
            # type its component maps as Dict[str, MdpComponent]) and dies later at the
            # binding check with a bare AttributeError and no bundle path -- the one
            # truncation this dispatch cannot name.
            if _KERNEL in tree or "bindings" in tree or "params" in tree:
                return _rebuild_component(tree, robot, where)
            # The bare form's whole grammar is the tag plus init; refuse an extra key
            # (dropped silently otherwise, the looseness #247 removed from the
            # component form) and a non-object init (which would otherwise surface as
            # "from_init() argument after ** must be a mapping").
            unknown = sorted(set(tree) - {_KERNEL_OBJECT, "init"})
            if unknown:
                raise ValueError(
                    f"bundle {where}: a bare kernel object has unknown key(s) "
                    f"{unknown} (the grammar is {sorted({_KERNEL_OBJECT, 'init'})}); "
                    f"{_REGEN}"
                )
            if "init" not in tree:
                raise ValueError(
                    f"bundle {where}: a bare kernel object needs init; {_REGEN}"
                )
            if not isinstance(tree["init"], dict):
                raise ValueError(
                    f"bundle {where}: a bare kernel object's init must be an object; "
                    f"{_REGEN}"
                )
            return _kernel_from_init(tree[_KERNEL_OBJECT], tree["init"], robot, where)
        # A Dict[K, V] field threads V to its values, so an annotated tuple value
        # rebuilds as a tuple; an untyped dict keeps Any. Unwrap an Optional[Dict[...]]
        # first so the value annotation still threads through.
        container = _unwrap_optional(annotation)
        value_annotation = Any
        if typing.get_origin(container) is dict and typing.get_args(container):
            value_annotation = typing.get_args(container)[-1]
        return {
            key: from_tree(sub, value_annotation, where=f"{where}.{key}", robot=robot)
            for key, sub in tree.items()
        }
    if isinstance(tree, list):
        # Unwrap an Optional[List[...]] / Optional[Tuple[...]] so the element annotation
        # (and the tuple-vs-list choice) comes from the container, not the Optional.
        container = _unwrap_optional(annotation)
        origin = typing.get_origin(container)
        args = typing.get_args(container)
        elem = args[0] if args else Any
        items = [
            from_tree(v, elem, where=f"{where}[{i}]", robot=robot)
            for i, v in enumerate(tree)
        ]
        return tuple(items) if origin is tuple or container is tuple else items
    return tree
