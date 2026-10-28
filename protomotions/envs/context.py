# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flat context: one read-only map from path string to value.

``BaseEnv._publish(env_ids)`` asks every publish owner and the control
components for a dict of ``{path: value}`` and merges them here. The paths
are the vocabulary of ``context_views.py`` (``"current.root_pos"``,
``"mimic.ref_state.rigid_body_pos"``, ``"prev_contact_force_magnitudes"``);
the values are whatever the owner holds: tensors (views or slices of owned
state, cloned only where the owner says so), dicts (``contact_body_index``)
and scalars (``dt``, ``anchor_idx``). A name a configuration does not
produce is absent, never None: ``path in ctx`` is the test.

``env_ids`` and ``dt`` are the context's own: ``merge`` publishes them as keys
and keeps them as attributes, because the component manager's compile rule
reads ``getattr(ctx, "env_ids", None)``. ``producers`` maps every key to the
name of the producer that published it (``merge`` records it; the env's
wiring report groups the context by it instead of publishing again).

The checks below run when the environment is built, on its first context,
and compare three things: what was published against the vocabulary
(``published_vocabulary_errors`` for the names, ``requirement_errors`` for
the ``requires=`` tags), the namespaces each control component published
against the ones its class declares (``control_namespace_errors``), and the
paths the components bind against both (``binding_errors``). Whether a
configuration publishes a tagged name is one rule, ``context_requirements.
status``, which the config-only check (``context_check.py``) calls on the
same declarations before anything is built, so both checks word a missing
name identically.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from torch import Tensor

from protomotions.envs.context_requirements import UNKNOWN, Facts, status
from protomotions.envs.context_views import (
    CONTROL_NAMESPACES,
    NAMESPACES,
    declared_paths,
    namespace_requires,
)

# The producer label of the names the context publishes itself (dt, env_ids).
_OWN_PRODUCER = "FrozenContext"


def control_producer(component_name: str) -> str:
    """The producer name ``BaseEnv._publish`` merges one control component's
    dict under (``context.producers`` records it for every key)."""
    return f"control/{component_name}"


def merge_with_producers(
    parts: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Union of the named producers' dicts, and the producer of every key.

    Both keep the order the parts were published in; a key published twice
    names both producers.
    """
    merged: Dict[str, Any] = {}
    producers: Dict[str, str] = {}
    for producer, published in parts.items():
        for key, value in published.items():
            if key in merged:
                raise ValueError(
                    f"context key '{key}' is published twice: by "
                    f"{producers[key]} and by {producer}"
                )
            merged[key] = value
            producers[key] = producer
    return merged, producers


def merge_published(parts: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Union of the named producers' dicts; a key published twice names both."""
    return merge_with_producers(parts)[0]


class FrozenContext(Mapping):
    """What the owners published for one build; immutable after construction.

    ``producers`` is the read-only ``{key: producer name}`` map ``merge``
    records (empty for a context built directly from a dict).
    """

    __slots__ = ("_values", "producers", "env_ids", "dt")

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        env_ids: Optional[Tensor],
        dt: float,
        producers: Optional[Mapping[str, str]] = None,
    ):
        object.__setattr__(self, "_values", MappingProxyType(dict(values)))
        object.__setattr__(self, "producers", MappingProxyType(dict(producers or {})))
        object.__setattr__(self, "env_ids", env_ids)
        object.__setattr__(self, "dt", dt)

    @classmethod
    def merge(
        cls,
        parts: Mapping[str, Mapping[str, Any]],
        *,
        env_ids: Optional[Tensor],
        dt: float,
    ):
        """Merge the producers' dicts (``{producer name: published}``) with the
        context's own ``dt`` and ``env_ids``; a key published twice names both
        producers. The resulting context remembers who published what."""
        own = {"dt": dt, "env_ids": env_ids}
        values, producers = merge_with_producers({_OWN_PRODUCER: own, **parts})
        return cls(values, env_ids=env_ids, dt=dt, producers=producers)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: object) -> bool:
        return key in self._values

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"FrozenContext is read-only (tried to set '{name}')")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"FrozenContext is read-only (tried to delete '{name}')")

    def __repr__(self) -> str:
        rows = "every env" if self.env_ids is None else f"{len(self.env_ids)} envs"
        return f"FrozenContext({len(self)} names, {rows})"

    # Mapping equality would compare tensor values with ``==``; one build of the
    # context is only ever meaningfully compared by identity.
    __eq__ = object.__eq__
    __hash__ = object.__hash__


def _namespace_of(path: str) -> str:
    return path.split(".", 1)[0] if "." in path else ""


def _published_by(
    published: Mapping[str, Any], paths: Sequence[str], separator: str
) -> str:
    """``"<separator>published by <producers>"`` for ``paths``, when
    ``published`` is a context that remembers who published what
    (``FrozenContext.producers``); empty for a plain dict, which has no
    producers to name."""
    producers = getattr(published, "producers", {})
    names = sorted({producers[path] for path in paths if path in producers})
    return f"{separator}published by {', '.join(names)}" if names else ""


def _has_provider(namespace: str, facts: Facts) -> bool:
    """Whether a configured control component declares ``namespace``."""
    return any(
        namespace in component_class.namespaces
        for component_class, _ in facts.control.values()
    )


def published_vocabulary_errors(published: Mapping[str, Any]) -> List[str]:
    """Where the published names disagree with the declaration.

    For every namespace with at least one published key: nothing undeclared
    may be published, and every declared name without a ``requires=`` tag
    must be present (a tagged name may be absent; whether it should be is
    :func:`requirement_errors`'s question). An undeclared name is reported
    with its producer when ``published`` is a context that recorded one.
    """
    by_namespace: Dict[str, set] = {}
    for key in published:
        by_namespace.setdefault(_namespace_of(key), set()).add(key)
    errors = []
    for namespace in sorted(by_namespace):
        keys = by_namespace[namespace]
        label = f"namespace '{namespace}'" if namespace else "the top level"
        if namespace not in NAMESPACES:
            errors.append(
                f"{label} is not declared in context_views; published {sorted(keys)}"
            )
            continue
        declared = declared_paths(namespace)
        undeclared = sorted(keys - declared.keys())
        missing = sorted(
            path
            for path, field in declared.items()
            if not field.requires and path not in keys
        )
        if undeclared:
            errors.append(
                f"{label}: published but not declared: {undeclared}"
                + _published_by(published, undeclared, ", ")
            )
        if missing:
            errors.append(
                f"{label}: declared as always present but not published: {missing}"
            )
    return errors


def requirement_errors(published: Mapping[str, Any], facts: Facts) -> List[str]:
    """Tagged names that were published although the configuration does not
    satisfy their ``requires=``: a publisher bug the tag catches (a sensor
    source emitting body positions without localization, a contact name
    without contact bodies). The error lists every tag that applies to the
    name (its namespace's and its own), the fact that is missing and, when
    ``published`` is a context that recorded one, the producer. A tag only
    the data can decide is allowed either way. Undeclared names are
    :func:`published_vocabulary_errors`'s; a name under a control namespace
    that no configured component declares is
    :func:`control_namespace_errors`'s, which names the component that
    published it, so it is skipped here.
    """
    errors = []
    for path in published:
        namespace = _namespace_of(path)
        declared = declared_paths(namespace) if namespace in NAMESPACES else {}
        field = declared.get(path)
        if field is None:
            continue
        if namespace in CONTROL_NAMESPACES and not _has_provider(namespace, facts):
            continue
        verdict, why = status(path, facts)
        if verdict is False:
            tags = ", ".join(
                repr(tag) for tag in (*namespace_requires(namespace), *field.requires)
            )
            errors.append(
                f"'{path}' (requires {tags}) is published although this "
                f"configuration does not satisfy it: {why}"
                + _published_by(published, [path], "; ")
            )
    return errors


def binding_errors(
    components: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    group: str,
    facts: Facts,
) -> List[str]:
    """Bound paths that are undeclared, or declared and not published.

    A bound path that is not in the vocabulary is an error naming the file to
    look in. A declared path that is absent from ``context`` is worded by
    :func:`context_requirements.status`: when the configuration does not
    satisfy the name's ``requires=``, the error says which fact is missing;
    when it does, the declaration and the publisher disagree and the error
    says so. ``context`` is the built context, or the config-only check's
    map of the names the configs imply.
    """
    errors = []
    for name, component in components.items():
        for argument, field_path in component.dynamic_vars.items():
            path = field_path.path
            where = (
                f"{group} component '{name}', argument '{argument}': "
                f"context path '{path}'"
            )
            namespace = _namespace_of(path)
            declared = declared_paths(namespace) if namespace in NAMESPACES else {}
            if path not in declared:
                errors.append(
                    f"{where} is not a declared context name (see context_views.py)"
                )
                continue
            if path in context:
                continue
            verdict, why = status(path, facts)
            if verdict is True:
                errors.append(
                    f"{where} is declared available under this configuration but "
                    "was not published (the declaration in context_views.py and "
                    "the publisher disagree)"
                )
            else:
                # Unmet, or decided by the loaded data (which turned out not
                # to carry it): either way the name is absent for that reason.
                errors.append(f"{where} is absent: {why}")
    return errors


def control_namespace_errors(
    context: FrozenContext,
    declared: Mapping[str, Sequence[str]],
) -> List[str]:
    """Where each control component's published namespaces disagree with its
    class's ``namespaces`` declaration.

    ``declared`` maps each configured component's name to its class's
    ``namespaces``; what that component published is read off
    ``context.producers``, which records every key under the component's own
    producer name (:func:`control_producer`), so a component is checked
    against what IT published, not against the union of all components. A
    namespace a component publishes without declaring it, or declares without
    publishing anything under it, is an error naming that component: the
    config-only check trusts the declarations, so each must match what its
    component does in a built env.
    """
    errors = []
    for name, namespaces in declared.items():
        producer = control_producer(name)
        published = {
            _namespace_of(key)
            for key, who in context.producers.items()
            if who == producer
        }
        for namespace in sorted(published - set(namespaces)):
            errors.append(
                f"control component '{name}' publishes namespace '{namespace}' "
                f"but does not declare it in its 'namespaces'"
            )
        for namespace in namespaces:
            if namespace not in published:
                errors.append(
                    f"control component '{name}' declares namespace "
                    f"'{namespace}' but published nothing under it"
                )
    return errors


__all__ = [
    "FrozenContext",
    "UNKNOWN",
    "binding_errors",
    "control_namespace_errors",
    "control_producer",
    "merge_published",
    "merge_with_producers",
    "published_vocabulary_errors",
    "requirement_errors",
]
