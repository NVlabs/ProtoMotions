# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a configuration satisfies the ``requires=`` tags of the context vocabulary.

Researchers do not need this file; open it only to add a new kind of
requirement, which is one row in :data:`REQUIREMENTS` (a predicate over
:class:`Facts` and the sentence a refusal shows) plus a line in the list in
``context_views.py``'s docstring.

``context_views.py`` tags a name that not every configuration publishes with
what it needs (``FieldPath(requires="localization")``). This module is the
infrastructure behind those tags: :class:`Facts` reduces a configuration (or
a built environment) to the handful of facts the tags talk about,
:data:`REQUIREMENTS` is the one table saying which fact satisfies each tag
and how to word it when it does not, and :func:`status` is the one rule that
decides whether a path is published by a configuration. The config-only check
(``context_check.py``) and the build-time check (``context.py``,
``base_env/build.py``) both call :func:`status`, so a missing name is worded
the same before and after the environment is built.

A tag that is not in the table is a switch of the control component that
publishes the namespace: ``requires="style"`` on a ``masked_mimic`` name asks
every configured component class declaring ``masked_mimic`` whether its
config turns ``style`` on (``ControlComponent.provides(config, "style")``).

This module never imports a control component: it receives the component
classes through :class:`Facts` (``Facts.from_configs`` resolves them with
``get_class`` on the config's ``_target_``, ``Facts.from_env`` reads them off
the built control manager). The import graph is
``context_paths -> context_views -> context_requirements -> context ->
context_check``; keep it that way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from protomotions.envs.context_views import (
    CONTROL_NAMESPACES,
    NAMESPACES,
    declared_paths,
    namespace_requires,
)
from protomotions.envs.obs.state_history_buffer import history_depth
from protomotions.envs.terminations.rule import components_of
from protomotions.utils.hydra_replacement import get_class

# The verdict of a requirement only the loaded data can decide (the motion
# file's contact labels): neither satisfied nor unmet from the configs.
UNKNOWN = None


def has_heightfield(terrain: Any) -> bool:
    """Whether a terrain has a height field to publish (``terrain.*``).

    The one predicate for it: ``TerrainPublisher.publish`` uses it to decide
    whether to publish, and ``Facts.from_env`` to decide what the vocabulary
    expects, so the two cannot disagree. Every real ``Terrain`` has both
    tensors; the flat test doubles have neither.
    """
    return hasattr(terrain, "height_points") and hasattr(terrain, "height_samples")


def control_terminations(
    controls: Mapping[str, Tuple[type, Any]], robot_config
) -> Dict[str, Any]:
    """Every control's checks under ``<control>/<check>``, from ``{name: (class,
    config)}`` (the shape ``Facts.control`` carries; ``ControlComponent.
    terminations`` is a classmethod, so the table exists before any component
    does). The one computation behind ``ControlManager.terminations`` and the
    build's termination table.
    """
    entries: Dict[str, Any] = {}
    for name, (component_class, config) in controls.items():
        for check, rule in component_class.terminations(config, robot_config).items():
            entries[f"{name}/{check}"] = rule
    return entries


def bound_components(env_config, robot_config) -> Dict[str, Any]:
    """Every component the environment binds, keyed as the build's binding
    check keys them (``observation/<name>``, ``reward/<name>``,
    ``termination/<name>``): the observation and reward tables as the recipe
    wrote them and the termination table as its rules plus the controls'
    checks. The history ring sizes itself from this table
    (``history_depth``), and the config-only check reads the same depth.
    """
    controls = {
        name: (get_class(config._target_), config)
        for name, config in dict(env_config.control_components or {}).items()
    }
    terminations = {
        **dict(env_config.termination_components or {}),
        **control_terminations(controls, robot_config),
    }
    table: Dict[str, Any] = {}
    for group, components in (
        ("observation", env_config.observation_components),
        ("reward", env_config.reward_components),
        ("termination", components_of(terminations)),
    ):
        for name, component in dict(components or {}).items():
            table[f"{group}/{name}"] = component
    return table


@dataclass(frozen=True)
class Facts:
    """What a configuration has, as far as the ``requires=`` tags care.

    ``history_steps`` is the history ring's depth, what the bound components
    read of the past (:func:`bound_components`); ``contacts`` is
    whether the simulator reports contacts, which is exactly whether the
    robot declares contact bodies; ``sensors`` is the SensorConfig or None;
    ``terrain_heightfield`` whether the terrain has height points;
    ``has_objects`` and ``has_pointclouds`` whether the scene library has
    scene objects and computed their pointclouds (None from the configs
    alone: only the built library can tell, so those names are checked at
    build); ``control`` maps each configured control component's name to its
    class and its config, which is what the component switches are read from.
    """

    history_steps: int
    contacts: bool
    sensors: Optional[Any]
    terrain_heightfield: bool
    control: Dict[str, Tuple[type, Any]]
    has_objects: Optional[bool] = None
    has_pointclouds: Optional[bool] = None

    @classmethod
    def from_configs(cls, env_config, robot_config, terrain_config) -> "Facts":
        """The facts from the configs alone, before anything is built.

        ``terrain_heightfield`` is ``terrain_config is not None``: a real
        terrain always has height points, so the configs and the built env
        agree on it; only a test double built without a terrain config can
        differ, and the tests pass None for it. The control component classes
        are imported (``get_class`` on ``_target_``), never instantiated.
        """
        control: Dict[str, Tuple[type, Any]] = {}
        for name, config in dict(env_config.control_components or {}).items():
            component_class = get_class(config._target_)
            if not (
                isinstance(component_class, type)
                and hasattr(component_class, "namespaces")
                and callable(getattr(component_class, "provides", None))
            ):
                raise ValueError(
                    f"control component '{name}' ({config._target_}) is not a "
                    "ControlComponent"
                )
            control[name] = (component_class, config)
        sensors = env_config.sensors
        return cls(
            history_steps=history_depth(bound_components(env_config, robot_config)),
            contacts=bool(robot_config.contact_bodies),
            sensors=sensors,
            terrain_heightfield=terrain_config is not None,
            control=control,
        )

    @classmethod
    def from_env(cls, env) -> "Facts":
        """The same facts read off a built environment."""
        sensors = env.config.sensors
        return cls(
            history_steps=env._state_history.num_history_steps,
            contacts=bool(env.simulator.contacts_enabled),
            sensors=sensors,
            terrain_heightfield=has_heightfield(env.terrain),
            control={
                name: (type(component), component.config)
                for name, component in env.control_manager.components.items()
            },
            has_objects=env.scene_lib.num_scenes() > 0,
            has_pointclouds=bool(env.scene_lib.has_pointclouds),
        )


def _localized(facts: Facts) -> bool:
    # An optional channel is a sensor object when declared and None when not.
    return facts.sensors is not None and facts.sensors.localization is not None


def _has_base_velocity(facts: Facts) -> bool:
    return facts.sensors is not None and facts.sensors.base_velocity is not None


def _has_height_sensor(facts: Facts) -> bool:
    return facts.sensors is not None and facts.sensors.height is not None


# The table: requirement name -> (is it satisfied?, why not). The sentence
# follows "is absent:" in a refusal and may use {name} for the last part of
# the path. A predicate returning UNKNOWN says the configs cannot tell.
REQUIREMENTS: Dict[str, Tuple[Callable[[Facts], Optional[bool]], str]] = {
    "history": (
        lambda facts: facts.history_steps > 0,
        "no observation, reward or termination component reads a past state, so "
        "the history ring has depth 0",
    ),
    "contacts": (
        lambda facts: facts.contacts,
        "the robot declares no contact bodies (robot.contact_bodies is empty, so "
        "the simulator reports no contacts)",
    ),
    "sensors": (
        lambda facts: facts.sensors is not None,
        "this config has no sensors (config.sensors is None)",
    ),
    "localization": (
        _localized,
        "the sensor config has no localization sensor "
        "(config.sensors.localization is None)",
    ),
    "base_velocity": (
        _has_base_velocity,
        "the sensor config has no base-velocity sensor "
        "(config.sensors.base_velocity is None)",
    ),
    "height_sensor": (
        _has_height_sensor,
        "the sensor config has no height sensor (config.sensors.height is None)",
    ),
    "terrain_heightfield": (
        lambda facts: facts.terrain_heightfield,
        "the terrain has no height points",
    ),
    "motion_file": (
        lambda facts: UNKNOWN,
        "the motion library carries '{name}' only when the motion file has it",
    ),
    "objects": (
        lambda facts: UNKNOWN if facts.has_objects is None else facts.has_objects,
        "the scene has no objects (scene_lib.num_scenes() == 0)",
    ),
    "pointclouds": (
        lambda facts: (
            UNKNOWN if facts.has_pointclouds is None else facts.has_pointclouds
        ),
        "the scene computed no pointclouds (SceneLib.has_pointclouds is False)",
    ),
}


def _namespace_of(path: str) -> str:
    return path.split(".", 1)[0] if "." in path else ""


def _switch_status(
    tag: str, path: str, namespace: str, facts: Facts
) -> Tuple[Optional[bool], str]:
    """A tag that is a switch of the control component publishing ``namespace``."""
    providers = [
        (component_class, config)
        for component_class, config in facts.control.values()
        if namespace in component_class.namespaces
    ]
    if namespace not in CONTROL_NAMESPACES or not providers:
        raise ValueError(
            f"requirement '{tag}' on context name '{path}' is neither in "
            "context_requirements.REQUIREMENTS nor a switch of a control "
            "component publishing its namespace; add it to the table in "
            "protomotions/envs/context_requirements.py"
        )
    if any(
        component_class.provides(config, tag) for component_class, config in providers
    ):
        return True, ""
    # Name the switch's value when some provider's config has a field of that
    # name; otherwise the tag matches no field, which is a misspelling or a
    # component whose switch is not a plain field.
    with_field = [config for _, config in providers if hasattr(config, tag)]
    if with_field:
        config = with_field[0]
        detail = f"{type(config).__name__}.{tag} is {getattr(config, tag)!r}"
    else:
        _, config = providers[0]
        detail = (
            f"{type(config).__name__} has no field named '{tag}': is the tag "
            "misspelled, or does the component need a `provides` override?"
        )
    return (
        False,
        f"the '{namespace}' control component's config does not turn on '{tag}' "
        f"({detail})",
    )


def _tag_status(
    tag: str, path: str, namespace: str, facts: Facts
) -> Tuple[Optional[bool], str]:
    if tag in REQUIREMENTS:
        predicate, sentence = REQUIREMENTS[tag]
        verdict = predicate(facts)
        if verdict is True:
            return True, ""
        return verdict, sentence.format(name=path.rsplit(".", 1)[-1])
    return _switch_status(tag, path, namespace, facts)


def status(path: str, facts: Facts) -> Tuple[Optional[bool], str]:
    """THE rule: does this configuration publish ``path``?

    ``(True, "")`` when it does; ``(False, why)`` when it does not, with the
    sentence a refusal shows after "is absent:"; ``(UNKNOWN, why)`` when only
    the loaded data can tell. A control namespace with no configured component
    declaring it is ``(False, "no control component publishes '<ns>' ...")``.
    The namespace's tags are checked first, then the name's own, and the
    first unmet one decides; an unknown tag only counts when none is unmet.
    ``path`` must be a declared name (``ValueError`` otherwise).
    """
    namespace = _namespace_of(path)
    declared = declared_paths(namespace) if namespace in NAMESPACES else {}
    if path not in declared:
        raise ValueError(f"'{path}' is not a declared context name")
    if namespace in CONTROL_NAMESPACES and not any(
        namespace in component_class.namespaces
        for component_class, _ in facts.control.values()
    ):
        return (
            False,
            f"no control component publishes '{namespace}' "
            f"(config.control_components: {sorted(facts.control)})",
        )
    unknown: Optional[str] = None
    for tag in (*namespace_requires(namespace), *declared[path].requires):
        verdict, why = _tag_status(tag, path, namespace, facts)
        if verdict is False:
            return False, why
        if verdict is UNKNOWN and unknown is None:
            unknown = why
    return (True, "") if unknown is None else (UNKNOWN, unknown)


def published_paths(facts: Facts) -> Tuple[Mapping[str, str], Mapping[str, str]]:
    """Every declared path by verdict: ``(expected, unknown)``.

    ``expected`` maps the paths this configuration publishes to ``""``;
    ``unknown`` maps the paths only the data can decide to the reason. The
    paths it does not publish are in neither.
    """
    expected: Dict[str, str] = {}
    unknown: Dict[str, str] = {}
    for namespace in NAMESPACES:
        for path in declared_paths(namespace):
            verdict, why = status(path, facts)
            if verdict is True:
                expected[path] = ""
            elif verdict is UNKNOWN:
                unknown[path] = why
    return expected, unknown


__all__ = [
    "Facts",
    "REQUIREMENTS",
    "UNKNOWN",
    "has_heightfield",
    "published_paths",
    "status",
]
