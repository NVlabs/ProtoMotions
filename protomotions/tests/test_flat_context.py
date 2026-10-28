# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The flat context: immutability, dual-mode lookup, and the build-time check
that the published names, the declared names and the bindings agree."""

import re
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest
import torch

from protomotions.envs.base_env.build import (
    bind_managers,
    declare_owners,
    validate_bindings,
)
from protomotions.envs.base_env.publishers import (
    StatePublisher,
    StaticPublisher,
    TerrainPublisher,
)
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.context import (
    FrozenContext,
    binding_errors,
    control_namespace_errors,
    control_producer,
    merge_published,
    merge_with_producers,
    published_vocabulary_errors,
    requirement_errors,
)
from protomotions.envs.context_paths import FieldPath, resolve_path
from protomotions.envs.context_requirements import (
    REQUIREMENTS,
    UNKNOWN,
    Facts,
    has_heightfield,
    status,
)
from protomotions.envs.context_sources import (
    CURRENT_NAMESPACE,
    HISTORICAL_NAMESPACE,
    HISTORICAL_NAMESPACES,
    PRIVILEGED,
    SENSOR_NAMESPACES,
    SENSOR_SOURCES,
    SOURCES,
    current_namespace,
    historical_namespace,
    sensed,
    source_of_path,
)
from protomotions.envs.context_views import (
    CONTROL_NAMESPACES,
    NAMESPACES,
    EnvContext,
    declared_paths,
)
from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig
from protomotions.envs.control.mimic_control import MimicControl, MimicControlConfig
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
from protomotions.envs.obs.humanoid import (
    compute_humanoid_max_coords_observations,
    compute_humanoid_reduced_coords_observations,
)
from protomotions.envs.obs.humanoid_historical import (
    compute_historical_reduced_coords_from_state,
)
from protomotions.envs.obs.state_history_buffer import StateHistoryBuffer
from protomotions.envs.rewards.meta import RewardMeta
from protomotions.envs.rewards.regularization import (
    compute_action_rate_l2,
    compute_contact_force_change_rew,
    compute_contact_match_rew,
)
from protomotions.envs.rewards.tracking import compute_gt_rew
from protomotions.envs.sensors.config import (
    BaseVelocitySensor,
    HeightSensor,
    IMUSensor,
    JointEncoderSensor,
    LocalizationSensor,
    NoNoise,
    SensorConfig,
)
from protomotions.envs.state import StateSource
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.tests.parity.chain_motions import build_chain_motion_lib
from protomotions.tests.support.sensor_simulator import make_sensor_test_env
from protomotions.tests.test_base_env_helpers import _make_env


def _reward(kernel):
    """A reward binding the kernel's own context defaults, at unit weight."""
    return MdpComponent.from_kernel(kernel, meta=RewardMeta(weight=1.0))


# =============================================================================
# FrozenContext
# =============================================================================


def test_frozen_context_is_read_only_and_keeps_env_ids_and_dt_as_attributes():
    env_ids = torch.tensor([2, 0])
    context = FrozenContext({"a": torch.ones(2), "b": None}, env_ids=env_ids, dt=0.02)

    assert context.env_ids is env_ids and context.dt == 0.02
    assert len(context) == 2 and list(context) == ["a", "b"]
    assert "b" in context and context["b"] is None and "c" not in context
    with pytest.raises(TypeError):
        context["c"] = 1
    with pytest.raises(AttributeError, match="read-only"):
        context.extra = 1
    with pytest.raises(AttributeError, match="read-only"):
        del context.dt
    assert "2 envs" in repr(context)


def test_merge_names_the_key_and_both_producers_of_a_duplicate():
    merged = merge_published({"state": {"x": 1}, "control": {"y": 2}})
    assert merged == {"x": 1, "y": 2}
    with pytest.raises(
        ValueError, match="'x' is published twice: by state and by control"
    ):
        merge_published({"state": {"x": 1}, "control": {"x": 2}})
    # The context publishes dt and env_ids itself, as keys and as attributes.
    env_ids = torch.tensor([1])
    context = FrozenContext.merge({"only": {"x": 1}}, env_ids=env_ids, dt=0.1)
    assert dict(context) == {"dt": 0.1, "env_ids": env_ids, "x": 1}
    assert context.env_ids is env_ids and context.dt == 0.1
    # ... and the merged context remembers who published what (read-only,
    # in publish order); a context built directly from a dict records nobody.
    assert dict(context.producers) == {
        "dt": "FrozenContext",
        "env_ids": "FrozenContext",
        "x": "only",
    }
    with pytest.raises(TypeError):
        context.producers["x"] = "other"
    assert dict(FrozenContext({"a": 1}, env_ids=None, dt=0.1).producers) == {}
    assert merge_with_producers({"s": {"x": 1}, "c": {"y": 2}}) == (
        {"x": 1, "y": 2},
        {"x": "s", "y": "c"},
    )
    assert dict(FrozenContext.merge({}, env_ids=None, dt=0.1)) == {
        "dt": 0.1,
        "env_ids": None,
    }
    # ... so no producer may publish them.
    with pytest.raises(
        ValueError, match="'dt' is published twice: by FrozenContext and by static"
    ):
        FrozenContext.merge({"static": {"dt": 0.1}}, env_ids=None, dt=0.1)


# =============================================================================
# resolve_path: one lookup for the map and for attribute stand-ins
# =============================================================================


def test_resolve_path_indexes_mappings_and_walks_attribute_stand_ins():
    value = torch.ones(2)
    mapping = {"current.dof_pos": value, "mimic.anchor_idx": 1}
    stand_in = SimpleNamespace(
        current=SimpleNamespace(dof_pos=value), mimic=SimpleNamespace(anchor_idx=1)
    )

    for context in (mapping, FrozenContext(mapping, env_ids=None, dt=0.1), stand_in):
        assert resolve_path(context, "current.dof_pos") is value
        assert resolve_path(context, "mimic.anchor_idx") == 1
        with pytest.raises(AttributeError, match="current.dof_vel"):
            resolve_path(context, "current.dof_vel")
    # A dotted key in a mapping is one key, not a walk.
    assert resolve_path({"a.b": 3}, "a.b") == 3


# =============================================================================
# The build-time check on the real environment: published names, declared
# names and bindings agree
# =============================================================================


def _publishing(owner_class, extra=None, drop=()):
    """``owner_class.publish`` with ``extra`` added and ``drop`` removed."""
    original = owner_class.publish

    def publish(self, env_ids):
        published = original(self, env_ids)
        published.update(extra or {})
        for key in drop:
            del published[key]
        return published

    return publish


def _mimic_env(**kwargs):
    """The sensor test env tracking one synthetic chain clip (no contact labels)."""
    return make_sensor_test_env(
        motion_lib=build_chain_motion_lib(1),
        motion_manager=MimicMotionManagerConfig(),
        control_components={"mimic": MimicControlConfig()},
        **kwargs,
    )


def test_clean_configuration_builds_and_publishes_only_declared_names():
    env = make_sensor_test_env()
    try:
        assert published_vocabulary_errors(env.context) == []
        assert isinstance(env.context, FrozenContext)
        # dt and env_ids come from the orchestrator, not from a publisher.
        assert env.context["dt"] == env.dt and env.context["env_ids"] is None
        rows = torch.tensor([1])
        assert env._publish(rows)["env_ids"] is rows
        assert published_vocabulary_errors(env._publish(rows)) == []
        # The static publisher holds the two id tensors, not the env.
        static = env._publish_owners[-1]
        assert isinstance(static, StaticPublisher) and not hasattr(static, "env")
        assert static.publish(None) == {
            "contact_body_ids": env.contact_body_ids,
            "non_termination_contact_body_ids": env.non_termination_contact_body_ids,
        }
        assert static.publish(None)["contact_body_ids"] is env.contact_body_ids
    finally:
        env.close()


def test_publish_is_keyed_by_owner_name_and_the_names_are_unique(monkeypatch):
    names = [owner.name for owner in make_sensor_test_env()._publish_owners]
    assert names == [
        "state",
        "terrain",
        "scene",
        "contacts",
        "episode",
        "contact_memory",
        "static",
    ]
    # A key two owners publish is reported with both names.
    monkeypatch.setattr(
        TerrainPublisher, "publish", lambda self, env_ids: {"current.dof_pos": 1}
    )
    with pytest.raises(
        ValueError,
        match="'current.dof_pos' is published twice: by state and by terrain",
    ):
        make_sensor_test_env()
    monkeypatch.undo()
    # Two owners with one name would collapse in the merge; the bind refuses.
    monkeypatch.setattr(TerrainPublisher, "name", "state")
    with pytest.raises(AssertionError, match=r"share a name: \['state'\]"):
        make_sensor_test_env()


def test_privileged_source_requires_what_a_sensor_source_may_lack():
    env = make_sensor_test_env()
    try:
        published = dict(env.context)
    finally:
        env.close()
    assert published_vocabulary_errors(published) == []

    # The same StatePublisher serves every source, so only the vocabulary can
    # tell a privileged omission from a sensor suite that measures less.
    without_privileged = dict(published)
    del without_privileged["current.root_pos"]
    assert published_vocabulary_errors(without_privileged) == [
        "namespace 'current': declared as always present but not published: "
        "['current.root_pos']"
    ]
    without_sensor = dict(published)
    del without_sensor["noisy.root_pos"]
    assert published_vocabulary_errors(without_sensor) == []

    for name in (
        "rigid_body_pos",
        "rigid_body_vel",
        "ground_heights",
        "root_pos",
        "root_vel",
        "root_height",
        "anchor_pos",
        "anchor_vel",
    ):
        assert declared_paths("current")[f"current.{name}"].requires == (), name
        assert declared_paths("historical")[f"historical.{name}"].requires == (), name
        for namespace in SENSOR_NAMESPACES:
            assert declared_paths(namespace)[f"{namespace}.{name}"].requires, name
    assert declared_paths("current")["current.rigid_body_contacts"].requires == (
        "contacts",
    )
    # Odometry and joint forces exist on one side only: the sensor sources
    # have an odometer, the simulator reports joint forces.
    assert "current.odom_disp_start" not in declared_paths("current")
    assert "noisy.dof_forces" not in declared_paths("noisy")
    assert "noisy.rigid_body_contacts" not in declared_paths("noisy")


def test_undeclared_published_name_fails_the_build(monkeypatch):
    monkeypatch.setattr(
        StatePublisher, "publish", _publishing(StatePublisher, {"current.bogus": 1})
    )
    with pytest.raises(
        ValueError,
        match=(
            r"'current': published but not declared: \['current.bogus'\], "
            "published by state"
        ),
    ):
        make_sensor_test_env()


def test_missing_required_name_fails_the_build(monkeypatch):
    monkeypatch.setattr(
        StatePublisher,
        "publish",
        _publishing(StatePublisher, drop=("noisy.anchor_rot",)),
    )
    with pytest.raises(
        ValueError,
        match=(
            r"'noisy': declared as always present but not published: "
            r"\['noisy.anchor_rot'\]"
        ),
    ):
        make_sensor_test_env()


def test_undeclared_namespace_fails_the_build(monkeypatch):
    monkeypatch.setattr(
        StatePublisher, "publish", _publishing(StatePublisher, {"bogus.name": 1})
    )
    with pytest.raises(ValueError, match="namespace 'bogus' is not declared"):
        make_sensor_test_env()


def test_bound_name_the_suite_does_not_measure_says_which_sensor_is_missing():
    with pytest.raises(ValueError) as error:
        make_sensor_test_env(
            observations=sensed(
                {
                    "bad": MdpComponent.from_kernel(
                        compute_humanoid_max_coords_observations
                    )
                },
                "noisy",
            )
        )
    message = str(error.value)
    assert "observation component 'bad', argument 'body_vel'" in message
    assert (
        "'noisy.rigid_body_vel' is absent: the sensor config has no base-velocity "
        "sensor (config.sensors.base_velocity is None)"
    ) in message
    # Every offender is listed, not only the first.
    assert (
        "'noisy.ground_heights' is absent: the sensor config has no height sensor "
        "(config.sensors.height is None)"
    ) in message


def test_control_namespace_errors_name_the_component_whose_declaration_is_off():
    """Each component is checked against what IT published (its own producer
    name in ``context.producers``), not against the union: a component that
    declares a namespace only another component publishes is named."""
    context = FrozenContext.merge(
        {
            control_producer("a"): {},
            control_producer("b"): {"steering.tar_dir": torch.zeros(1, 3)},
        },
        env_ids=None,
        dt=0.1,
    )
    assert control_namespace_errors(
        context, {"a": ("steering",), "b": ("steering",)}
    ) == [
        "control component 'a' declares namespace 'steering' but published "
        "nothing under it"
    ]
    assert control_namespace_errors(context, {"a": (), "b": ()}) == [
        "control component 'b' publishes namespace 'steering' but does not "
        "declare it in its 'namespaces'"
    ]
    assert control_namespace_errors(context, {"a": (), "b": ("steering",)}) == []


def test_bound_control_namespace_without_its_component_says_so():
    with pytest.raises(ValueError, match="no control component publishes 'mimic'"):
        make_sensor_test_env(rewards={"gt": _reward(compute_gt_rew)})


def test_bound_motion_file_name_the_data_lacks_says_so_at_build():
    # The mimic namespace IS published; only the contact labels are missing,
    # which the configs could not tell and the loaded clips decided.
    with pytest.raises(ValueError) as error:
        _mimic_env(rewards={"match": _reward(compute_contact_match_rew)})
    message = str(error.value)
    assert "no control component publishes" not in message
    assert (
        "'mimic.ref_state.rigid_body_contacts' is absent: the motion library "
        "carries 'rigid_body_contacts' only when the motion file has it"
    ) in message


def test_bound_control_switch_that_is_off_names_the_switch():
    with pytest.raises(ValueError) as error:
        _mimic_env(
            terminations={
                "low": TerminationRule(
                    MdpComponent(
                        lambda h: h,
                        {"h": EnvContext.mimic.smoothed_ref_anchor_height},
                    )
                )
            }
        )
    assert (
        "'mimic.smoothed_ref_anchor_height' is absent: the 'mimic' control "
        "component's config does not turn on 'reference_height_smoothing_seconds' "
        "(MimicControlConfig.reference_height_smoothing_seconds is 0.0)"
    ) in str(error.value)


def test_bound_history_name_sizes_the_ring():
    """A binding to a history name is what gives the ring its depth: with no
    history observation, the action-rate reward's ``previous_action`` alone
    makes the ring keep one past slot, and the name is published."""
    env = make_sensor_test_env(
        history_steps=0, rewards={"rate": _reward(compute_action_rate_l2)}
    )
    assert env._state_history.num_history_steps == 1
    assert "previous_action" in env._publish()


def test_bound_terrain_name_without_height_points_says_so():
    component = MdpComponent(
        lambda height_points: height_points,
        {"height_points": EnvContext.terrain.height_points},
    )
    with pytest.raises(
        ValueError,
        match="'terrain.height_points' is absent: the terrain has no height points",
    ):
        make_sensor_test_env(observations={"terrain": component})


def test_bound_sensor_name_without_a_suite_and_bound_contact_name_without_contacts():
    env = _make_env(history=False)
    env._component_manager = ComponentManager(torch.device("cpu"))
    # The check walks the managers' resolved tables, so a rebound config
    # table reaches it through ``bind_managers``.
    env.config.observation_components = sensed(
        {
            "noisy": MdpComponent.from_kernel(
                compute_humanoid_reduced_coords_observations
            )
        },
        "noisy",
    )
    bind_managers(env)
    with pytest.raises(
        ValueError,
        match="'noisy.dof_pos' is absent: this config has no sensors",
    ):
        validate_bindings(env)

    env.config.observation_components = {}
    env.config.reward_components = {"impact": _reward(compute_contact_force_change_rew)}
    bind_managers(env)
    env.simulator.contacts_enabled = False
    # The ring was seeded with contact flags; re-seed it as a robot without
    # contact bodies would, or the requires="contacts" tag catches the
    # publisher emitting flags the configuration cannot have.
    env._state_history = StateHistoryBuffer(
        0, env._history_sampler.sample(env.simulator.state, None)
    )
    declare_owners(env)
    env._current_context = None
    with pytest.raises(
        ValueError,
        match=(
            "'prev_contact_force_magnitudes' is absent: the robot declares no "
            "contact bodies"
        ),
    ):
        validate_bindings(env)


def _sensors(*, localization=False, base_velocity=False, height=False):
    """A SensorConfig declaring the requested optional channels; noise is unused here."""
    exact = NoNoise("flat context")
    return SensorConfig(
        joint_encoders=JointEncoderSensor(exact, exact),
        imu=IMUSensor(exact, exact),
        localization=LocalizationSensor("pelvis", exact) if localization else None,
        base_velocity=BaseVelocitySensor(exact) if base_velocity else None,
        height=HeightSensor(exact) if height else None,
    )


def _facts(**overrides):
    """Facts of a configuration with everything on, except ``overrides``."""
    values = dict(
        history_steps=3,
        contacts=True,
        sensors=_sensors(localization=True, base_velocity=True, height=True),
        terrain_heightfield=True,
        control={"mimic": (MimicControl, MimicControlConfig())},
    )
    values.update(has_objects=True, has_pointclouds=True)
    values.update(overrides)
    return Facts(**values)


def test_binding_errors_reject_undeclared_paths_and_report_absent_ones():
    bogus = FieldPath()
    bogus.name = "nothing"
    components = {
        "undeclared": MdpComponent(lambda nothing: nothing, {"nothing": bogus}),
        "present": MdpComponent(lambda p: p, {"p": EnvContext.masked_mimic.ref_pos}),
        "disabled": MdpComponent(
            lambda mask: mask, {"mask": EnvContext.masked_mimic.local_ref_pos}
        ),
    }
    context = {"masked_mimic.ref_pos": torch.ones(1)}
    facts = _facts(control={"mm": (MaskedMimicControlLike, MaskedMimicControlConfig())})
    errors = binding_errors(components, context, group="observation", facts=facts)
    assert errors == [
        "observation component 'undeclared', argument 'nothing': context path "
        "'nothing' is not a declared context name (see context_views.py)",
        "observation component 'disabled', argument 'mask': context path "
        "'masked_mimic.local_ref_pos' is absent: the 'masked_mimic' control "
        "component's config does not turn on 'local_stream' "
        "(MaskedMimicControlConfig.local_stream is False)",
    ]


class MaskedMimicControlLike(MimicControl):
    """A class declaring ``masked_mimic`` with the base ``provides`` rule, so
    the facts above need no built component."""

    namespaces = ("mimic", "masked_mimic")


def test_binding_errors_tell_a_forgotten_publisher_from_an_unmet_requirement():
    components = {
        "forgotten": MdpComponent(lambda d: d, {"d": EnvContext.current.dof_pos}),
        "steer": MdpComponent(lambda d: d, {"d": EnvContext.steering.tar_dir}),
    }
    errors = binding_errors(components, {}, group="reward", facts=_facts())
    assert errors == [
        "reward component 'forgotten', argument 'd': context path "
        "'current.dof_pos' is declared available under this configuration but "
        "was not published (the declaration in context_views.py and the "
        "publisher disagree)",
        "reward component 'steer', argument 'd': context path 'steering.tar_dir' "
        "is absent: no control component publishes 'steering' "
        "(config.control_components: ['mimic'])",
    ]


def test_status_words_every_unmet_requirement():
    """``status`` is the one rule both checks use: a function of the facts a
    configuration has. Every row of the table, every switch kind, and the
    order (namespace tags before the name's own, first unmet wins)."""
    everything = _facts()
    for path in (
        "noisy.root_pos",
        "noisy.ground_heights",
        "clean_historical.rigid_body_vel",
        "historical.body_contacts",
        "contact_forces",
        "previous_action",
        "terrain.height_points",
        "mimic.future_pos",
        "current.dof_pos",
        "dt",
    ):
        assert status(path, everything) == (True, ""), path
    # The scene names: known once the library is built, unknown from the
    # configs alone (the built env answers both facts).
    assert status("scene.object_pos", everything) == (True, "")
    assert status("scene.neutral_pointclouds", everything) == (True, "")
    assert status("scene.object_pos", _facts(has_objects=False)) == (
        False,
        "the scene has no objects (scene_lib.num_scenes() == 0)",
    )
    assert status("scene.neutral_pointclouds", _facts(has_pointclouds=False)) == (
        False,
        "the scene computed no pointclouds (SceneLib.has_pointclouds is False)",
    )
    assert status("scene.object_rot", _facts(has_objects=None))[0] is UNKNOWN
    assert (
        status("scene.neutral_pointclouds", _facts(has_pointclouds=None))[0] is UNKNOWN
    )
    assert status("mimic.ref_state.rigid_body_contacts", everything) == (
        UNKNOWN,
        "the motion library carries 'rigid_body_contacts' only when the motion "
        "file has it",
    )
    assert status("mimic.ref_state.local_rigid_body_rot", everything)[0] is UNKNOWN

    no_history = (
        "no observation, reward or termination component reads a past state, "
        "so the history ring has depth 0"
    )
    without_history = _facts(history_steps=0)
    for path in ("historical.dof_pos", "noisy_historical.dof_pos", "previous_action"):
        assert status(path, without_history) == (False, no_history), path

    without_contacts = _facts(contacts=False)
    why = (
        "the robot declares no contact bodies (robot.contact_bodies is empty, so "
        "the simulator reports no contacts)"
    )
    for path in (
        "contact_forces",
        "contact_body_index",
        "prev_contact_force_magnitudes",
        "current.rigid_body_contacts",
        "historical.body_contacts",
        "historical.rigid_body_contacts",
    ):
        assert status(path, without_contacts) == (False, why), path

    without_suite = _facts(sensors=None)
    why = "this config has no sensors (config.sensors is None)"
    for path in ("noisy.dof_pos", "clean.root_pos", "noisy_historical.dof_pos"):
        assert status(path, without_suite) == (False, why), path
    # The namespace's requirement is checked before the name's own.
    assert status("noisy.ground_heights", _facts(sensors=None, history_steps=0)) == (
        False,
        why,
    )

    imu_only = _facts(sensors=_sensors())
    assert status("noisy.dof_pos", imu_only) == (True, "")
    no_localization = (
        "the sensor config has no localization sensor "
        "(config.sensors.localization is None)"
    )
    for path in (
        "noisy.rigid_body_pos",
        "noisy.root_pos",
        "noisy.anchor_pos",
        "noisy.root_height",
        "noisy.odom_disp_start",
        "clean.odom_start_xy",
        "noisy.ground_heights",
    ):
        assert status(path, imu_only) == (False, no_localization), path
    for path in ("noisy.rigid_body_vel", "noisy.root_vel", "clean.anchor_vel"):
        assert status(path, imu_only) == (
            False,
            "the sensor config has no base-velocity sensor "
            "(config.sensors.base_velocity is None)",
        ), path
    localized = _facts(sensors=_sensors(localization=True))
    assert status("noisy.root_pos", localized) == (True, "")
    assert status("noisy.ground_heights", localized) == (
        False,
        "the sensor config has no height sensor (config.sensors.height is None)",
    )

    assert status("terrain.height_points", _facts(terrain_heightfield=False)) == (
        False,
        "the terrain has no height points",
    )

    assert status("steering.tar_dir", everything) == (
        False,
        "no control component publishes 'steering' (config.control_components: "
        "['mimic'])",
    )
    assert status("mimic.future_pos", _facts(control={})) == (
        False,
        "no control component publishes 'mimic' (config.control_components: [])",
    )
    assert status("mimic.smoothed_ref_anchor_height", everything) == (
        False,
        "the 'mimic' control component's config does not turn on "
        "'reference_height_smoothing_seconds' "
        "(MimicControlConfig.reference_height_smoothing_seconds is 0.0)",
    )
    smoothing = MimicControlConfig(reference_height_smoothing_seconds=0.5)
    assert status(
        "mimic.smoothed_ref_anchor_height",
        _facts(control={"mimic": (MimicControl, smoothing)}),
    ) == (True, "")

    with pytest.raises(ValueError, match="not a declared context name"):
        status("current.bogus", everything)
    import protomotions.envs.context_requirements as requirements

    original = requirements.declared_paths

    def with_stray(namespace, tag):
        """``declared_paths`` with one extra name, ``<namespace>.stray``,
        tagged ``requires=tag``."""
        stray = FieldPath(requires=tag)
        stray.name = "stray"
        extra = dict(original(namespace), **{f"{namespace}.stray": stray})
        return lambda ns: extra if ns == namespace else original(ns)

    # A tag with no rule is an infrastructure error, never a lookup failure.
    requirements.declared_paths = with_stray("current", "no_such_fact")
    try:
        with pytest.raises(ValueError, match="'no_such_fact'.*'current.stray'"):
            status("current.stray", everything)
    finally:
        requirements.declared_paths = original
    # A switch tag that matches no field of the publishing component's config
    # (``styles`` written for ``style``) says so, instead of claiming the
    # field is None.
    requirements.declared_paths = with_stray("mimic", "styles")
    try:
        assert status("mimic.stray", everything) == (
            False,
            "the 'mimic' control component's config does not turn on 'styles' "
            "(MimicControlConfig has no field named 'styles': is the tag "
            "misspelled, or does the component need a `provides` override?)",
        )
    finally:
        requirements.declared_paths = original


def test_requirement_errors_catch_a_publisher_emitting_what_the_config_lacks():
    published = {
        "noisy.root_pos": torch.zeros(1, 3),  # no localization in this suite
        "noisy.dof_pos": torch.zeros(1, 2),
        "contact_forces": torch.zeros(1, 1, 1, 3),  # no contact bodies
        "mimic.ref_state.rigid_body_contacts": torch.zeros(1, 2),  # data decides
        "current.bogus": 1,  # undeclared: the vocabulary check's business
    }
    facts = _facts(sensors=_sensors(), contacts=False)
    # Every tag that applies is listed: the namespace's, then the name's own.
    no_localization = (
        "'noisy.root_pos' (requires 'sensors', 'localization') is published "
        "although this configuration does not satisfy it: the sensor config has "
        "no localization sensor (config.sensors.localization is None)"
    )
    no_contacts = (
        "'contact_forces' (requires 'contacts') is published although this "
        "configuration does not satisfy it: the robot declares no contact bodies "
        "(robot.contact_bodies is empty, so the simulator reports no contacts)"
    )
    assert requirement_errors(published, facts) == [no_localization, no_contacts]
    assert requirement_errors(published, _facts()) == []
    # A context that remembers who published what names the producer too.
    context = FrozenContext.merge(
        {
            "state": {"noisy.root_pos": published["noisy.root_pos"]},
            "contacts": {"contact_forces": published["contact_forces"]},
        },
        env_ids=None,
        dt=0.1,
    )
    assert requirement_errors(context, facts) == [
        no_localization + "; published by state",
        no_contacts + "; published by contacts",
    ]
    # A name under a control namespace no configured component declares is
    # the namespace check's business (it names the component), not this one's.
    assert requirement_errors({"steering.tar_dir": torch.zeros(1, 3)}, facts) == []


def test_a_publisher_emitting_an_unsatisfied_name_fails_the_build(monkeypatch):
    """The tag catches the publisher, not only the binding: a sensor source
    that emits body positions from a suite without localization. The error
    lists the tags, the missing fact and the publisher."""
    monkeypatch.setattr(
        StatePublisher,
        "publish",
        _publishing(StatePublisher, {"noisy.odom_disp_start": torch.zeros(3, 2)}),
    )
    with pytest.raises(
        ValueError,
        match=(
            r"'noisy.odom_disp_start' \(requires 'sensors', 'localization'\) is "
            "published although this configuration does not satisfy it: the "
            "sensor config has no localization sensor "
            r"\(config.sensors.localization is None\); published by state"
        ),
    ):
        make_sensor_test_env(sensors=_sensors(), target_observations=False)


def test_a_control_component_publishing_an_undeclared_namespace_is_named(
    monkeypatch,
):
    """A component that publishes a namespace its class does not list is
    reported once, by name, before the vocabulary checks can complain about
    every name of that namespace."""
    monkeypatch.setattr(
        MimicControl,
        "publish",
        _publishing(MimicControl, {"steering.tar_dir": torch.zeros(3, 3)}),
    )
    with pytest.raises(ValueError) as error:
        _mimic_env()
    message = str(error.value)
    assert (
        "control component 'mimic' publishes namespace 'steering' but does not "
        "declare it in its 'namespaces'"
    ) in message
    assert "steering.tar_dir' (requires" not in message
    assert "declared as always present" not in message


def test_a_forgotten_tag_fails_the_build(monkeypatch):
    """A name declared without ``requires=`` is promised to every
    configuration, so a configuration that does not publish it fails."""
    monkeypatch.setattr(
        StatePublisher,
        "publish",
        _publishing(StatePublisher, drop=("current.dof_forces",)),
    )
    with pytest.raises(
        ValueError,
        match=(
            r"'current': declared as always present but not published: "
            r"\['current.dof_forces'\]"
        ),
    ):
        make_sensor_test_env()


def test_facts_read_the_same_predicate_as_the_terrain_publisher():
    env = make_sensor_test_env()
    try:
        facts = Facts.from_env(env)
        assert facts.terrain_heightfield is has_heightfield(env.terrain) is False
        assert "terrain.height_points" not in env.context
        assert facts.contacts is env.simulator.contacts_enabled is True
        assert facts.sensors is env.config.sensors
        assert facts.history_steps == env._state_history.num_history_steps
        assert facts.control == {}
    finally:
        env.close()
    assert has_heightfield(SimpleNamespace(height_points=1, height_samples=1))
    assert not has_heightfield(SimpleNamespace(height_points=1))
    assert set(REQUIREMENTS) == {
        "history",
        "objects",
        "pointclouds",
        "contacts",
        "sensors",
        "localization",
        "base_velocity",
        "height_sensor",
        "terrain_heightfield",
        "motion_file",
    }
    assert HISTORICAL_NAMESPACES == (
        "historical",
        "clean_historical",
        "noisy_historical",
    )


def test_install_motion_lib_reruns_the_binding_check(monkeypatch):
    env = _mimic_env(rewards={"gt": _reward(compute_gt_rew)})
    try:
        # A compatible switch passes, draws nothing and leaves the cache empty.
        rng = torch.get_rng_state()
        env._current_context = None
        env.install_motion_lib(env.reference_motion.motion_lib)
        assert env._current_context is None
        assert torch.equal(torch.get_rng_state(), rng)
        # A switch after which the reference lacks a bound name is refused.
        monkeypatch.setattr(
            MimicControl,
            "publish",
            _publishing(MimicControl, drop=("mimic.ref_state.rigid_body_pos",)),
        )
        with pytest.raises(
            ValueError,
            match=(
                r"'mimic': declared as always present but not published: "
                r"\['mimic.ref_state.rigid_body_pos'\]"
            ),
        ):
            env.install_motion_lib(env.reference_motion.motion_lib)
    finally:
        env.close()


# =============================================================================
# context_sources: one naming rule for factories, exporters and the expert check
# =============================================================================


def test_context_sources_agree_with_the_factories_and_the_exporters():
    assert SOURCES == (PRIVILEGED, *SENSOR_SOURCES) == ("privileged", "clean", "noisy")
    # One declaration: the StateSource type the factories take.
    assert SOURCES == get_args(StateSource)
    assert SENSOR_NAMESPACES == (
        "clean",
        "clean_historical",
        "noisy",
        "noisy_historical",
    )
    assert HISTORICAL_NAMESPACES == (
        "historical",
        "clean_historical",
        "noisy_historical",
    )
    reduced = MdpComponent.from_kernel(compute_humanoid_reduced_coords_observations)
    history = MdpComponent.from_kernel(
        compute_historical_reduced_coords_from_state, history_steps=1
    )
    for source in SOURCES:
        moved = sensed({"now": reduced, "past": history}, source)
        assert (
            moved["now"].dynamic_vars["dof_pos"].path
            == f"{current_namespace(source)}.dof_pos"
        )
        assert (
            moved["past"].dynamic_vars["historical_dof_pos"].path
            == f"{historical_namespace(source)}.dof_pos"
        )
        assert source_of_path(f"{current_namespace(source)}.dof_pos") == source
        assert source_of_path(f"{historical_namespace(source)}.dof_pos") == source
    assert source_of_path("dt") is None and source_of_path("mimic.future_pos") is None
    with pytest.raises(ValueError, match="unknown state source"):
        current_namespace("imagined")
    # The plain constants the exporters and the deployment protocol prefix
    # ONNX names with are the helpers' answers for simulator truth.
    assert (
        (CURRENT_NAMESPACE, HISTORICAL_NAMESPACE)
        == (
            current_namespace(PRIVILEGED),
            historical_namespace(PRIVILEGED),
        )
        == ("current", "historical")
    )
    from examples.experiments.metatron.g1_student import (
        TEACHER_CLEAN_TO_PRIVILEGED,
    )

    assert TEACHER_CLEAN_TO_PRIVILEGED == {
        current_namespace("clean"): current_namespace(PRIVILEGED),
        historical_namespace("clean"): historical_namespace(PRIVILEGED),
    }

    from deployment.tracker_inputs import tracker_input_keys

    assert tracker_input_keys(
        ["noisy.dof_pos", "clean.anchor_rot", "mimic.future_pos"]
    ) == [
        "current.dof_pos",
        "current.anchor_rot",
        "mimic.future_pos",
    ]


def test_control_namespaces_are_the_ones_tagged_owner_control():
    assert CONTROL_NAMESPACES == ("mimic", "masked_mimic", "steering", "path", "target")
    fields = vars(EnvContext)
    for namespace in CONTROL_NAMESPACES:
        # A control namespace carries no requires=: it exists exactly when a
        # configured component declares it.
        assert fields[namespace].owner == "control"
        assert fields[namespace].requires == ()
    for namespace in ("current", "historical", "terrain", "scene"):
        assert fields[namespace].owner is None
    # The tag travels onto the path proxies' nested fields too.
    assert vars(EnvContext.mimic)["ref_state"].owner is None


# =============================================================================
# The docs spell only declared paths
# =============================================================================

_DOCS = Path(__file__).resolve().parents[2] / "docs" / "source" / "concepts"
_DOC_PATHS = re.compile(r'EnvContext\.([A-Za-z_][\w.]*\w)|ctx\["([^"]+)"\]')
# The "Adding Custom Context Names" section shows a hypothetical namespace.
_ILLUSTRATIVE_NAMESPACES = {"my_custom"}


def _is_declared(path: str) -> bool:
    """A namespace, a declared name, or a prefix of one (``mimic.ref_state``)."""
    if path in NAMESPACES:
        return True
    namespace = path.split(".", 1)[0] if "." in path else ""
    if namespace not in NAMESPACES:
        return False
    declared = declared_paths(namespace)
    return path in declared or any(name.startswith(f"{path}.") for name in declared)


@pytest.mark.parametrize("doc", ["environment_context.rst", "contact_matrix.rst"])
def test_docs_spell_only_declared_context_paths(doc):
    text = (_DOCS / doc).read_text()
    spelled = {dotted or quoted for dotted, quoted in _DOC_PATHS.findall(text)}
    assert spelled, f"{doc} spells no context path the way this test looks for"
    undeclared = sorted(
        path
        for path in spelled
        if path.split(".", 1)[0] not in _ILLUSTRATIVE_NAMESPACES
        and not _is_declared(path)
    )
    assert undeclared == [], f"{doc} spells undeclared context paths: {undeclared}"
    assert _is_declared("mimic.ref_state") and not _is_declared("mimic.ref_root_height")


def test_terrain_publisher_publishes_the_horizontal_scale_with_the_height_field():
    """A terrain with a height field also publishes ``terrain.horizontal_scale``
    (metres per cell, the terrain config's value); a flat double without one
    publishes nothing."""
    terrain = SimpleNamespace(
        height_points=torch.zeros(2, 3, 2),
        height_samples=torch.zeros(4, 4),
        config=SimpleNamespace(horizontal_scale=0.25),
    )
    published = TerrainPublisher(terrain).publish(None)
    assert published["terrain.horizontal_scale"] == 0.25
    assert set(published) == {
        "terrain.height_points",
        "terrain.height_samples",
        "terrain.horizontal_scale",
    }
    assert TerrainPublisher(SimpleNamespace(config=terrain.config)).publish(None) == {}
