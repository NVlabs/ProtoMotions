# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where a state namespace's values come from: simulator truth or a sensor estimate.

The state history keeps one ``BodyState`` per source. The privileged source is
published under ``current`` / ``historical``; each sensor source under its own
name (``noisy`` / ``noisy_historical``). Every place that needs this naming
rule (the context publisher, :func:`sensed`, the exporters, the
expert-contract check) reads it from here instead of spelling the lists out.

The sources themselves are declared once, as the ``StateSource`` type in
``protomotions/envs/state.py`` (what :func:`sensed` takes); ``SOURCES`` is that
declaration's runtime form. An observation kernel binds the privileged names by
default; a recipe that wants the sensor suite's estimates instead wraps the
bound components in ``sensed(components, "noisy")``, which repoints every
``current`` / ``historical`` binding at that source's namespaces.
"""

from typing import Dict, Mapping, Optional, Tuple, TYPE_CHECKING, get_args

from protomotions.envs.context_paths import FieldPath, declared_fields, resolve_path
from protomotions.envs.state import StateSource

if TYPE_CHECKING:
    from protomotions.envs.mdp_component import MdpComponent

PRIVILEGED = "privileged"
SOURCES: Tuple[str, ...] = get_args(StateSource)
SENSOR_SOURCES: Tuple[str, ...] = tuple(
    source for source in SOURCES if source != PRIVILEGED
)


def _check(source: str) -> None:
    if source not in SOURCES:
        raise ValueError(f"unknown state source {source!r}; expected one of {SOURCES}")


def current_namespace(source: str) -> str:
    """Namespace of a source's current state: ``current`` for simulator truth."""
    _check(source)
    return "current" if source == PRIVILEGED else source


def historical_namespace(source: str) -> str:
    """Namespace of a source's past states: ``historical`` for simulator truth."""
    _check(source)
    return "historical" if source == PRIVILEGED else f"{source}_historical"


# The two namespaces of simulator truth, as plain strings for the exporters
# and the deployment protocol (which prefix ONNX input names with them).
CURRENT_NAMESPACE: str = current_namespace(PRIVILEGED)
HISTORICAL_NAMESPACE: str = historical_namespace(PRIVILEGED)


# ("clean", "clean_historical", "noisy", "noisy_historical"): present only
# when the experiment declares a sensor suite.
SENSOR_NAMESPACES: Tuple[str, ...] = tuple(
    namespace
    for source in SENSOR_SOURCES
    for namespace in (current_namespace(source), historical_namespace(source))
)
HISTORICAL_NAMESPACES: Tuple[str, ...] = tuple(
    historical_namespace(source) for source in SOURCES
)


def source_of_path(path: str) -> Optional[str]:
    """The state source a context path reads, or None for a non-state path."""
    namespace = path.split(".", 1)[0]
    for source in SOURCES:
        if namespace in (current_namespace(source), historical_namespace(source)):
            return source
    return None


def _privileged_only_paths() -> frozenset:
    """Context names only the simulator produces: no sensor estimates them.

    Every name tagged ``requires="contacts"`` in ``context_views.py`` plus the
    contact-index names that carry no requirement. ``sensed`` refuses a
    component that binds one.
    """
    from protomotions.envs.context_views import EnvContext

    tagged = {
        path
        for path, field in declared_fields(EnvContext).items()
        if "contacts" in field.requires
    }
    return frozenset(
        tagged
        | {
            "current.rigid_body_contacts",
            "historical.body_contacts",
            "contact_body_ids",
            "non_termination_contact_body_ids",
        }
    )


PRIVILEGED_ONLY_PATHS: frozenset = _privileged_only_paths()


def sensed(
    components: Mapping[str, "MdpComponent"], source: StateSource
) -> Dict[str, "MdpComponent"]:
    """Repoint every privileged state binding in ``components`` at ``source``.

    For each component, a binding under the ``current`` or ``historical``
    namespace is rewritten to the source's namespace (``noisy`` /
    ``noisy_historical``); control namespaces, ``dt``, ``env_ids``,
    ``progress_buf`` and ``current_action`` pass unchanged. A binding to a name
    only the simulator produces (:data:`PRIVILEGED_ONLY_PATHS`: contacts, joint
    forces) is refused; no sensor estimates it. A namespace this configuration
    lacks is left to the config-only check, not a second capability list.

    ``sensed(actor_obs, "privileged")`` is the identity: nothing moves, so
    nothing is refused.

    A moved binding is the target name's own declaration (looked up by path
    on ``EnvContext``), so it carries that name's requirement and deploy slot:
    ``noisy.root_pos`` needs localization where ``current.root_pos`` needs
    nothing, and the config-only check refuses it on a suite without one.
    """
    from protomotions.envs.context_views import EnvContext

    _check(source)
    out: Dict[str, "MdpComponent"] = {}
    for name, component in components.items():
        rebinds: Dict[str, FieldPath] = {}
        for argument, field_path in component.dynamic_vars.items():
            path = field_path.path
            # A sensor-sourced component may not read a name only the simulator
            # produces, whether the name sits under current.* / historical.*
            # (it would have to move) or at the top level (contact_forces, the
            # contact indices: it would stay privileged beside sensor values).
            if source != PRIVILEGED and path in PRIVILEGED_ONLY_PATHS:
                raise ValueError(
                    f"sensed('{source}'): component '{name}' binds '{path}', "
                    "which only the simulator produces (no sensor estimates it); "
                    "unbind it or keep the component privileged"
                )
            namespace, _, rest = path.partition(".")
            if namespace == "current":
                new_namespace = current_namespace(source)
            elif namespace == "historical":
                new_namespace = historical_namespace(source)
            else:
                continue  # a control name, dt, env_ids, ...: never moves
            if new_namespace == namespace:
                continue  # privileged source: this binding stays put
            try:
                rebinds[argument] = resolve_path(EnvContext, f"{new_namespace}.{rest}")
            except AttributeError as error:
                # The same sentence as every other refusal here, as a ValueError.
                raise ValueError(str(error)) from None
        out[name] = component.rebind(**rebinds) if rebinds else component
    return out


__all__ = [
    "CURRENT_NAMESPACE",
    "HISTORICAL_NAMESPACE",
    "HISTORICAL_NAMESPACES",
    "PRIVILEGED",
    "PRIVILEGED_ONLY_PATHS",
    "SENSOR_NAMESPACES",
    "SENSOR_SOURCES",
    "SOURCES",
    "current_namespace",
    "historical_namespace",
    "sensed",
    "source_of_path",
]
