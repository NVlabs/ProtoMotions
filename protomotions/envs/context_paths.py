# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Context path descriptors: how a binding spells a name in the context map.

``FieldPath`` and ``NestedField`` are class-level declarations only. Reading
``EnvContext.current.rigid_body_pos`` gives a ``FieldPath`` whose ``.path`` is
the string ``"current.rigid_body_pos"``; the environment publishes a value
under that same string in its flat context map (``protomotions.envs.context``).
The vocabulary classes in ``context_views.py`` are never instantiated: both
descriptors refuse instance access so a stale ``view.field = value`` fails
loudly instead of storing a value nothing reads.

A declaration can say what a configuration must have for the name to exist:
``FieldPath(requires="localization")`` is a name only a sensor suite with a
localization body produces, ``NestedField(..., requires="sensors")`` a whole
namespace that needs a sensor suite. ``context_views.py`` lists the
requirement names and ``context_requirements.py`` says how a configuration
satisfies each; a name with no ``requires`` is published by every
configuration.

Both descriptors also carry a ``deploy`` slot (``None`` by default): one string,
the host's name for the quantity, written by the declarations in
``context_views.py`` and read only by the exporter, which derives everything
else (shape, dtype, the future and history axes, the column labels, the sensor
channel) from a simulator-free publish. Nothing in ``envs`` reads it; it is
copied with the path and shown in ``repr``.

Usage in experiment configs::

    from protomotions.envs.context_views import EnvContext
    from protomotions.envs.mdp_component import MdpComponent

    observation_components = {
        "max_coords_obs": MdpComponent(
            compute_func=compute_humanoid_max_coords_observations,
            dynamic_vars={"body_pos": EnvContext.current.rigid_body_pos},
        ),
    }

``resolve_path(ctx, path)`` is the one lookup every reader uses: a mapping
context is indexed by the path string; any other object (the ONNX exporters'
attribute stand-ins, test doubles) is walked attribute by attribute.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable, Optional, Tuple, Union

Requires = Union[str, Iterable[str]]


def _as_requirements(requires: Requires) -> Tuple[str, ...]:
    """One requirement name or several, as a tuple of names."""
    if isinstance(requires, str):
        return (requires,)
    names = tuple(requires)
    for name in names:
        if not isinstance(name, str):
            raise TypeError(
                f"requires= takes requirement names (strings), got {name!r}"
            )
    return names


def _requires_repr(requires: Tuple[str, ...]) -> str:
    if not requires:
        return ""
    if len(requires) == 1:
        return f", requires={requires[0]!r}"
    return f", requires={requires!r}"


def _deploy_repr(deploy: Optional[str]) -> str:
    if deploy is None:
        return ""
    return f", deploy={deploy!r}"


class FieldPath:
    """A declared context name; ``.path`` is the key in the context map.

    ``requires`` names what a configuration must have for this name to be
    published (``"localization"``, ``"contacts"``, a control component's
    config switch ...); empty means every configuration publishes it. A name
    with no requirement that a publisher forgets fails the environment's
    build check. ``deploy`` is the host-kind deployment slot (module docstring).
    """

    __slots__ = ("name", "requires", "deploy", "_parent_path")

    def __init__(
        self,
        parent_path: str = "",
        requires: Requires = (),
        deploy: Optional[str] = None,
    ):
        self.name: str = ""
        self.requires: Tuple[str, ...] = _as_requirements(requires)
        self.deploy: Optional[str] = deploy
        self._parent_path = parent_path

    @classmethod
    def from_path(
        cls, path: str, requires: Requires = (), deploy: Optional[str] = None
    ) -> "FieldPath":
        """Build a FieldPath whose ``.path`` is this full dot-separated string.

        ``sensed`` uses it to repoint a binding at another namespace
        (``current.rigid_body_pos`` -> ``noisy.rigid_body_pos``). A dotless
        ``path`` is a top-level name; an empty ``path`` is refused.
        """
        if not path:
            raise ValueError("FieldPath.from_path needs a non-empty path")
        parent, _, name = path.rpartition(".")
        field = cls(parent_path=parent, requires=requires, deploy=deploy)
        field.name = name
        return field

    def __set_name__(self, owner, name):
        self.name = name

    @property
    def path(self) -> str:
        """Full dot-separated path to this field."""
        if self._parent_path:
            return f"{self._parent_path}.{self.name}"
        return self.name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        raise AttributeError(
            f"'{self.path}' is a declared context name, not an attribute of "
            f"{type(obj).__name__}: read it from the context map"
        )

    def __set__(self, obj, value):
        raise AttributeError(
            f"'{self.path}' is a declared context name: publish it from an owner "
            "instead of assigning it on a view instance"
        )

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return (
            f"FieldPath('{self.path}'{_requires_repr(self.requires)}"
            f"{_deploy_repr(self.deploy)})"
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, FieldPath):
            return self.path == other.path
        if isinstance(other, str):
            return self.path == other
        return False

    def __hash__(self) -> int:
        return hash(self.path)


class NestedField:
    """A declared namespace: a class whose FieldPaths live under this name.

    Class access returns a proxy class whose descriptors carry the full path
    (``EnvContext.mimic.ref_state.rigid_body_pos.path`` is
    ``"mimic.ref_state.rigid_body_pos"``). ``requires`` names what a
    configuration must have for the namespace to be published at all (a
    sensor suite, history); the names inside keep their own ``requires``.
    Only a namespace declared on ``EnvContext`` may carry it: one nested
    inside another view (``mimic.ref_state``) is refused with ``TypeError``
    when the paths are built, because nothing would read the tag there.
    ``owner`` tags who publishes the namespace: ``"control"`` for a control
    component's namespace, which is how ``context_views.CONTROL_NAMESPACES``
    is derived. ``deploy`` is the host-kind deployment slot (module docstring).
    """

    __slots__ = (
        "nested_class",
        "name",
        "requires",
        "owner",
        "deploy",
        "_parent_path",
        "_path_proxy",
    )

    def __init__(
        self,
        nested_class: type,
        parent_path: str = "",
        requires: Requires = (),
        owner: Optional[str] = None,
        deploy: Optional[str] = None,
    ):
        self.nested_class = nested_class
        self.name: str = ""
        self.requires: Tuple[str, ...] = _as_requirements(requires)
        self.owner = owner
        self.deploy: Optional[str] = deploy
        self._parent_path = parent_path
        self._path_proxy: Optional[type] = None

    def __set_name__(self, owner, name):
        self.name = name

    @property
    def path(self) -> str:
        if self._parent_path:
            return f"{self._parent_path}.{self.name}"
        return self.name

    def _create_path_proxy(self) -> type:
        full_path = self.path

        class PathProxy:
            """Path access to the names declared under one namespace."""

        # The raw descriptors come from __dict__ (getattr would trigger them);
        # a class's own declaration wins over one inherited from a base.
        for cls in self.nested_class.__mro__:
            for attr_name, attr in cls.__dict__.items():
                if attr_name.startswith("_") or hasattr(PathProxy, attr_name):
                    continue
                if isinstance(attr, FieldPath):
                    field = FieldPath(
                        full_path, requires=attr.requires, deploy=attr.deploy
                    )
                    field.name = attr_name
                    setattr(PathProxy, attr_name, field)
                elif isinstance(attr, NestedField):
                    # Only a namespace hung on EnvContext is checked for its
                    # own requires=; a tag here would be read by nothing.
                    if attr.requires:
                        raise TypeError(
                            f"'{full_path}.{attr_name}': a nested namespace cannot "
                            "carry requires=; put the tag on its names"
                        )
                    nested = NestedField(
                        attr.nested_class,
                        full_path,
                        owner=attr.owner,
                        deploy=attr.deploy,
                    )
                    nested.name = attr_name
                    setattr(PathProxy, attr_name, nested)
        return PathProxy

    def __get__(self, obj, objtype=None):
        if obj is None:
            if self._path_proxy is None:
                self._path_proxy = self._create_path_proxy()
            return self._path_proxy
        raise AttributeError(
            f"'{self.path}' is a declared context namespace, not an attribute of "
            f"{type(obj).__name__}: read its names from the context map"
        )

    def __set__(self, obj, value):
        raise AttributeError(
            f"'{self.path}' is a declared context namespace: its owner publishes "
            "its names instead of assigning a view instance"
        )

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return (
            f"NestedField('{self.path}'{_requires_repr(self.requires)}"
            f"{_deploy_repr(self.deploy)})"
        )


def declared_fields(namespace_class: type) -> Dict[str, FieldPath]:
    """Every FieldPath declared by a class or path proxy, by full path, recursively.

    Each FieldPath carries its own ``requires``; a namespace's requirements are
    on its ``NestedField`` (``context_views.namespace_requires``).
    """
    fields: Dict[str, FieldPath] = {}
    for cls in namespace_class.__mro__:
        for attr_name, attr in cls.__dict__.items():
            if attr_name.startswith("_"):
                continue
            if isinstance(attr, FieldPath):
                fields.setdefault(attr.path, attr)
            elif isinstance(attr, NestedField):
                for path, field in declared_fields(attr.__get__(None)).items():
                    fields.setdefault(path, field)
    return fields


def resolve_path(ctx: Any, path: str) -> Any:
    """Look ``path`` up in a context: a mapping by key, anything else by attributes.

    Raises ``AttributeError`` for a missing name in both modes, so the build
    check, the exporters and the parity recorder handle one exception type.
    """
    if isinstance(ctx, Mapping):
        try:
            return ctx[path]
        except KeyError:
            raise AttributeError(f"context has no '{path}'") from None
    value = ctx
    try:
        for part in path.split("."):
            value = getattr(value, part)
    except AttributeError:
        raise AttributeError(f"context has no '{path}'") from None
    return value


__all__ = ["FieldPath", "NestedField", "declared_fields", "resolve_path"]
