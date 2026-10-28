# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The config-only binding check: refuse a bad binding before anything is built.

Every observation, reward, termination and evaluator metric reads context
names it binds (``EnvContext.noisy.root_pos`` ...). The environment checks
those bindings against the first context it publishes, and the evaluator
checks its own against the same context at its first evaluation; both need a
built environment, so a binding to a name this configuration can never
produce (a sensor name without a sensor suite, a history name when no bound
component reads past slot 0, a task namespace with no control component
for it) would otherwise surface only after the simulator is built, and the
evaluator's only after the first training epochs.

:func:`check_bindings_against_config` reports those bindings from the configs
alone, in seconds, with the same wording the build would use. What it refuses
is decided by the ``requires=`` tags in ``context_views.py`` and the table in
``context_requirements.py``: a bound name whose requirement the configuration
does not satisfy is an error naming the missing fact. What it leaves to the
build: names that depend on the motion file (the reference motion's contact
labels and local rotations), which only the loaded motion library can
confirm; they are reported as "can only be checked once the motion data is
loaded" and the build checks them.

The AMP reference-observation table (``AMPAgentConfig.reference_obs_components``)
is checked for its static keys and metadata only: its kernels read the motion
library through the training component's own call, not the env's context, and
that call ignores a key the kernel no longer takes, so a stale key would
otherwise survive silently.

``train_agent.py`` and ``inference_agent.py`` call
:func:`run_config_binding_check` right after the configs are built. The check
never builds a simulator, a terrain or a motion library and draws no random
number: it imports the control component classes (``Facts.from_configs``,
``get_class`` on their ``_target_``) and reads config fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from protomotions.agents.evaluators.components import as_failure
from protomotions.envs.context import binding_errors
from protomotions.envs.context_requirements import (
    Facts,
    control_terminations,
    published_paths,
)
from protomotions.envs.control.base import bound_paths
from protomotions.envs.mdp_component import static_component_errors
from protomotions.envs.terminations.rule import components_of

# The value every name the derived vocabulary expects to be published maps to:
# the check needs presence, never a value.
EXPECTED = object()

# A name whose fact the configs cannot tell (the motion file's contents, the
# built scene library's) is deferred to the build check with this sentence; the
# reason after the colon names the fact.
UNKNOWN_WORDING = "can only be checked once the environment is built"


@dataclass(frozen=True)
class ConfigVocabulary:
    """What the configs say the first context will hold.

    ``expected`` maps every path the build will publish to :data:`EXPECTED`
    (a mapping, so ``binding_errors`` can read it like a context);
    ``unknown`` maps the paths only the loaded data can decide to the reason;
    ``facts`` is what the configs were reduced to (``Facts.from_configs``).
    """

    expected: Dict[str, object]
    unknown: Dict[str, str]
    facts: Facts


@dataclass(frozen=True)
class UnknownPath:
    """One bound context path only the build can confirm: the reason, and
    the components that bind it (``"reward component 'match'"`` ...), in
    binding order."""

    path: str
    reason: str
    bound_by: Tuple[str, ...]


@dataclass(frozen=True)
class ConfigBindingReport:
    """The outcome of :func:`check_bindings_against_config`.

    ``errors`` are the bindings the build would refuse. For observations,
    rewards and terminations each line is the one the env's build check
    (``build.validate_bindings``) would raise. The evaluator's metric and
    failure components are checked under the keys its manager gives them
    (``metric/<name>``, ``failure/<name>``) with the reason added; the
    evaluator's own check at the first evaluation words its refusal
    differently ("context path ... is unavailable"), so those lines are this
    check's. ``unknown`` lists, one line per binding, the bound paths only
    the build can confirm; ``unknown_paths`` groups the same bindings by path
    (the entry points log one line per path). Empty ``errors`` means the
    build's check of published names against declared ones will pass for
    everything the configs can decide; the motion-file names are checked at
    build.
    """

    errors: List[str]
    unknown: List[str]
    unknown_paths: Tuple[UnknownPath, ...] = ()

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError(
                "the configured components bind context names this "
                "configuration will not publish (checked from the configs, "
                "before building anything):\n  " + "\n  ".join(self.errors)
            )


def config_vocabulary(env_config, robot_config, terrain_config) -> ConfigVocabulary:
    """Derive the names the build will publish from the three configs.

    One call of ``context_requirements.status`` per declared path: published,
    not published, or decided by the loaded data.
    """
    facts = Facts.from_configs(env_config, robot_config, terrain_config)
    expected, unknown = published_paths(facts)
    return ConfigVocabulary(
        expected={path: EXPECTED for path in sorted(expected)},
        unknown=dict(unknown),
        facts=facts,
    )


def _termination_components(env_config, controls, robot_config) -> Dict[str, Any]:
    """The bound kernels of the termination table the build lays out
    (``build.termination_table``): the recipe's rules plus every configured
    control's checks under ``<control>/<check>`` (``ControlComponent.
    terminations`` is a classmethod, so the table exists before any component
    does), so a control's check is refused here with a recipe termination's
    wording. ``controls`` is ``Facts.control``'s ``{name: (class, config)}``.
    A name on both sides is the build's refusal, not a binding one."""
    recipe = dict(getattr(env_config, "termination_components", None) or {})
    return components_of({**recipe, **control_terminations(controls, robot_config)})


def _component_groups(
    env_config, agent_config, controls, robot_config
) -> List[Tuple[str, Dict[str, Any]]]:
    """The component dicts to check, by group, in the order the build checks them.

    A group left as None (an inference override clears the terminations that
    way) checks as empty. The termination group is the build's whole table,
    the controls' checks included (:func:`_termination_components`). The
    evaluator's components are keyed the way its manager keys them
    (``metric/<name>``, ``failure/<name>``); a component object shared between
    a metric and a failure rule is checked once, under the first key, as
    ``agents/evaluators/manager.py`` routes it.
    """
    groups = [
        (group, dict(getattr(env_config, f"{group}_components", None) or {}))
        for group in ("observation", "reward")
    ]
    groups.append(
        ("termination", _termination_components(env_config, controls, robot_config))
    )
    evaluator = getattr(agent_config, "evaluator", None)
    if evaluator is not None:
        evaluation: Dict[str, Any] = {}
        identity_keys: Dict[int, str] = {}
        failures = evaluator.failure_components or {}
        for kind, group in (
            ("metric", evaluator.metric_components or {}),
            (
                "failure",
                {
                    name: as_failure(name, entry).component
                    for name, entry in failures.items()
                },
            ),
        ):
            for name, component in group.items():
                key = identity_keys.setdefault(id(component), f"{kind}/{name}")
                evaluation[key] = component
        groups.append(("evaluation", evaluation))
    return groups


def _reference_table_errors(agent_config) -> List[str]:
    """The AMP reference-observation table's static keys and metadata.

    ``AMPAgentConfig.reference_obs_components`` holds kernels the training
    component calls on the motion library's state with the inputs it supplies
    at the call (``AMPTrainingComponent._call_ref_obs_fn``), never on the env's
    context: nothing to check against the vocabulary, but a static key the
    kernel no longer takes is ignored by that call, so it is refused here with
    the kernel named, as the build words a stale key on an env table.
    """
    table = dict(getattr(agent_config, "reference_obs_components", None) or {})
    return static_component_errors(table, group="reference", runtime_inputs=True)


def _control_binding_errors(
    groups: List[Tuple[str, Dict[str, Any]]], controls, robot_config
) -> List[str]:
    """Each configured control class's own refusal of the bound paths.

    The build's last binding check asks every control component's class
    (``ControlComponent.check_bindings``) whether it can serve the union of
    the three env tables' bound paths (the controls' own checks included); a
    class whose published layout depends on its config (the masked mimic goal
    canvas) refuses with its own sentence. Asked here from the configs once
    the name checks pass (the build's order), so the refusal is worded before
    the build. ``groups`` is :func:`_component_groups`'s list; the evaluator's
    group is not the env's and is left out, as the build leaves it out.
    """
    bound = bound_paths(
        *(
            components
            for group, components in groups
            if group in ("observation", "reward", "termination")
        )
    )
    errors: List[str] = []
    for component_class, config in controls.values():
        try:
            component_class.check_bindings(config, robot_config, bound)
        except ValueError as error:
            errors.append(str(error))
    return errors


def check_bindings_against_config(
    env_config,
    robot_config,
    terrain_config,
    agent_config: Optional[Any] = None,
) -> ConfigBindingReport:
    """Check every bound context path against the vocabulary the configs imply.

    ``errors`` holds, in ``binding_errors``'s wording, every observation,
    reward, termination (the configured controls' own checks included, keyed
    ``<control>/<check>``) and (with an agent config) evaluation binding to a
    name that is not declared or that this configuration will not publish,
    plus the reward-metadata placement errors ``static_component_errors``
    reports (a reward with no ``RewardMeta``, or a non-reward carrying one),
    the stale static keys of the AMP reference table (:func:`_reference_table_errors`)
    and every configured control class's own refusal of the bound paths
    (``check_bindings``), worded as the build words them; ``unknown`` the
    bindings to names only the build can confirm (see the module docstring).
    Nothing is built and no random number is drawn.
    """
    vocabulary = config_vocabulary(env_config, robot_config, terrain_config)
    # The unknown names count as published for the error pass, so that pass
    # reports only what the build would refuse; they are listed separately.
    context: Dict[str, object] = dict(vocabulary.expected)
    context.update({path: EXPECTED for path in vocabulary.unknown})

    errors: List[str] = []
    unknown: List[str] = []
    bound_by: Dict[str, List[str]] = {}  # unknown path -> who binds it
    controls = vocabulary.facts.control  # {name: (class, config)}, imported once
    groups = _component_groups(env_config, agent_config, controls, robot_config)
    for group, components in groups:
        errors.extend(static_component_errors(components, group=group))
        errors.extend(
            binding_errors(components, context, group=group, facts=vocabulary.facts)
        )
        for name, component in components.items():
            for argument, field_path in component.dynamic_vars.items():
                path = field_path.path
                if path in vocabulary.unknown:
                    unknown.append(
                        f"{group} component '{name}', argument '{argument}': "
                        f"context path '{path}' {UNKNOWN_WORDING}: "
                        f"{vocabulary.unknown[path]}"
                    )
                    who = f"{group} component '{name}'"
                    binders = bound_by.setdefault(path, [])
                    if who not in binders:
                        binders.append(who)
    errors.extend(_reference_table_errors(agent_config))
    if not errors:  # the build's last check: asked once the names check out
        errors.extend(_control_binding_errors(groups, controls, robot_config))
    unknown_paths = tuple(
        UnknownPath(path, vocabulary.unknown[path], tuple(binders))
        for path, binders in bound_by.items()
    )
    return ConfigBindingReport(
        errors=errors, unknown=unknown, unknown_paths=unknown_paths
    )


def run_config_binding_check(
    env_config, robot_config, terrain_config, agent_config, log
) -> ConfigBindingReport:
    """The entry points' call: refuse the errors, log one INFO line per
    unknown path naming the components that bind it, return the report."""
    report = check_bindings_against_config(
        env_config, robot_config, terrain_config, agent_config
    )
    report.raise_for_errors()
    for entry in report.unknown_paths:
        log.info(
            "config check: context path '%s' %s (%s); bound by %s",
            entry.path,
            UNKNOWN_WORDING,
            entry.reason,
            ", ".join(entry.bound_by),
        )
    return report


__all__ = [
    "EXPECTED",
    "UNKNOWN_WORDING",
    "ConfigBindingReport",
    "ConfigVocabulary",
    "UnknownPath",
    "check_bindings_against_config",
    "config_vocabulary",
    "run_config_binding_check",
]
