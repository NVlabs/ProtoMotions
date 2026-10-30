# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The config-only binding check (``protomotions/envs/context_check.py``).

Four kinds of test:

- the vocabulary it derives from the configs is exactly the context a built
  env publishes (on the fake sensor simulator: with and without history, with
  and without a mimic control, with and without contact bodies, and over the
  sensor suites' features), so the check refuses exactly what the build
  refuses;
- the refusals carry the build check's wording, and the names only the
  motion data can confirm are reported as such instead of refused;
- every ``requires=`` tag in the vocabulary has a rule: a row of
  ``context_requirements.REQUIREMENTS`` or a switch of an in-tree control
  component that publishes the namespace.

The check over every shipped recipe, and the stages a run goes through after
it, is ``test_recipe_stages.py``; the recipe tests here build the configs
offline the same way (``protomotions/tests/support/recipe_stages.py``).
"""

import importlib
import io
import logging
import os
from contextlib import redirect_stdout
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from protomotions.envs.context_check import (
    EXPECTED,
    check_bindings_against_config,
    config_vocabulary,
    run_config_binding_check,
)
from protomotions.envs.context_paths import resolve_path
from protomotions.envs.context_requirements import REQUIREMENTS
from protomotions.envs.context_sources import sensed
from protomotions.envs.context_views import (
    CONTROL_NAMESPACES,
    NAMESPACES,
    EnvContext,
    declared_paths,
    namespace_requires,
)
from protomotions.envs.control.base import ControlComponent, ControlComponentConfig
from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig
from protomotions.envs.control.mimic_control import MimicControlConfig
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig
from protomotions.envs.obs.cast import to_float
from protomotions.envs.obs.humanoid import compute_humanoid_max_coords_observations
from protomotions.envs.obs.humanoid_historical import (
    compute_historical_reduced_coords_from_state,
)
from protomotions.envs.obs.masked_mimic_velocity_command import (
    compute_masked_mimic_velocity_command_obs,
)
from protomotions.envs.rewards.meta import RewardMeta
from protomotions.envs.rewards.regularization import (
    compute_action_rate_l2,
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
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.envs.terminations.tracking import mean_body_pos_error
from protomotions.tests.support.recipe_stages import fake_args

# The scene names: only the built scene library decides them (requires="objects" /
# "pointclouds"), so the config-only check lists them as undecidable.
SCENE_NAMES = frozenset(declared_paths("scene"))


# =============================================================================
# The derived vocabulary against a built env
# =============================================================================


def _mimic_kwargs(clips=1):
    from protomotions.tests.parity.chain_motions import build_chain_motion_lib

    return dict(
        motion_lib=build_chain_motion_lib(clips),
        motion_manager=MimicMotionManagerConfig(),
        control_components={"mimic": MimicControlConfig()},
    )


# The sensor suites the derived vocabulary is checked against a built env for:
# the IMU alone, the height sensor without localization (which cannot place
# its reading, so no ground height is published), localization alone, each
# extra feature with localization, and all of them.
def _sensors(*, localization=False, base_velocity=False, height=False):
    """A SensorConfig declaring the requested optional channels; noise is unused here."""
    exact = NoNoise("context check")
    return SensorConfig(
        joint_encoders=JointEncoderSensor(exact, exact),
        imu=IMUSensor(exact, exact),
        localization=LocalizationSensor("pelvis", exact) if localization else None,
        base_velocity=BaseVelocitySensor(exact) if base_velocity else None,
        height=HeightSensor(exact) if height else None,
    )


SUITES = {
    "imu-only": _sensors(),
    "+height (no localization)": _sensors(height=True),
    "imu+localization": _sensors(localization=True),
    "+height": _sensors(localization=True, height=True),
    "+base_velocity": _sensors(localization=True, base_velocity=True),
    "all": _sensors(localization=True, base_velocity=True, height=True),
}


@pytest.mark.parametrize("suite", list(SUITES))
@pytest.mark.parametrize("contacts", [True, False], ids=["contacts", "no-contacts"])
@pytest.mark.parametrize("history_steps, mimic", [(3, False), (0, False), (3, True)])
def test_derived_vocabulary_is_the_built_context(history_steps, mimic, contacts, suite):
    pytest.importorskip("mujoco")
    from protomotions.tests.support.sensor_simulator import make_sensor_test_env

    kwargs = _mimic_kwargs() if mimic else {}
    localized = SUITES[suite].localization is not None
    env = make_sensor_test_env(
        history_steps=history_steps,
        contacts=contacts,
        sensors=SUITES[suite],
        # The target observations bind the anchor position, which a sensor config
        # without localization rightly refuses.
        target_observations=localized,
        **kwargs,
    )
    try:
        assert env.simulator.contacts_enabled is contacts
        actual = set(env.context)
        # The flat test terrain has no height points: the config-derived
        # terrain namespace stands for a real Terrain, so pass None here.
        vocabulary = config_vocabulary(env.config, env.robot_config, None)
    finally:
        env.close()
    expected = set(vocabulary.expected)
    assert expected - actual == set()
    assert actual - expected - set(vocabulary.unknown) == set()
    assert ("contact_forces" in expected) is contacts
    assert ("noisy.root_pos" in expected) is localized
    assert ("noisy.ground_heights" in expected) is (
        SUITES[suite].height is not None and localized
    )
    assert ("noisy.root_vel" in expected) is (SUITES[suite].base_velocity is not None)
    # Only the built scene library knows whether it has objects and computed
    # their pointclouds, so the four scene names are undecidable from the
    # configs (and absent from this object-less double's context); with a
    # reference motion, so are the chain clips' contact labels and local
    # rotations.
    undecidable = set(SCENE_NAMES)
    if mimic:
        undecidable |= {
            "mimic.ref_state.rigid_body_contacts",
            "mimic.ref_state.local_rigid_body_rot",
        }
    assert set(vocabulary.unknown) == undecidable
    assert set(vocabulary.unknown) & actual == set()
    assert all(value is EXPECTED for value in vocabulary.expected.values())


@pytest.mark.parity_slow
@pytest.mark.skipif(
    not os.environ.get("PROTOMOTIONS_PARITY_SLOW"),
    reason="builds the MuJoCo smpl_mimic scenario; set PROTOMOTIONS_PARITY_SLOW=1",
)
def test_derived_vocabulary_is_the_built_context_of_a_real_experiment():
    """The real terrain (height points), a real motion library, a real robot
    config's contact bodies and the real MuJoCo simulator."""
    pytest.importorskip("mujoco")
    from protomotions.tests.parity import SCENARIOS, build_env, eager_mdp_kernels

    with eager_mdp_kernels():
        built = build_env(SCENARIOS["smpl_mimic"])
    try:
        env = built.env
        actual = set(env.context)
        contacts = env.simulator.contacts_enabled
        vocabulary = config_vocabulary(
            built.configs["env"], built.configs["robot"], built.configs["terrain"]
        )
    finally:
        built.env.close()
    expected = set(vocabulary.expected)
    assert expected - actual == set()
    assert actual - expected - set(vocabulary.unknown) == set()
    assert {"terrain.height_points", "terrain.height_samples"} <= expected
    assert ("contact_forces" in expected) == contacts
    assert "current.dof_forces" in expected


def test_terrain_config_decides_the_terrain_namespace():
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.envs.base_env.config import EnvConfig

    robot = SimpleNamespace(contact_bodies=None)
    without = config_vocabulary(EnvConfig(), robot, None).expected
    with_terrain = config_vocabulary(EnvConfig(), robot, TerrainConfig()).expected
    assert "terrain.height_points" not in without
    assert {"terrain.height_points", "terrain.height_samples"} <= set(with_terrain)
    # No contact bodies: none of the contact names, no sensor or history names.
    assert not any(
        name.startswith(("clean", "noisy", "historical")) for name in without
    )
    assert "contact_forces" not in without and "current.dof_forces" in without
    assert "previous_action" not in without


def test_sensor_suite_features_decide_the_sensor_names():
    from protomotions.envs.base_env.config import EnvConfig

    robot = SimpleNamespace(contact_bodies=["foot"])

    def names(sensors, history=0):
        # A history-reading observation gives the ring its depth.
        config = EnvConfig(
            sensors=sensors,
            observation_components=(
                {
                    "hist": MdpComponent.from_kernel(
                        compute_historical_reduced_coords_from_state,
                        history_steps=history,
                    )
                }
                if history
                else {}
            ),
        )
        return set(config_vocabulary(config, robot, None).expected)

    imu_only = names(_sensors())
    assert "noisy.dof_pos" in imu_only and "clean.root_rot" in imu_only
    for name in ("rigid_body_pos", "root_pos", "anchor_pos", "root_height"):
        assert f"noisy.{name}" not in imu_only
    assert "noisy.rigid_body_vel" not in imu_only
    assert "noisy.odom_disp_start" not in imu_only
    assert "noisy.ground_heights" not in imu_only
    assert "noisy_historical.dof_pos" not in imu_only  # no history
    assert "noisy.rigid_body_contacts" not in declared_paths("noisy")  # not a name

    localized = names(_sensors(localization=True), history=2)
    assert {
        "noisy.rigid_body_pos",
        "clean.root_height",
        "noisy.odom_disp_start",
    } <= localized
    assert "noisy.ground_heights" not in localized  # localization without height
    assert "noisy.rigid_body_vel" not in localized
    assert {"noisy_historical.dof_pos", "noisy_historical.raw_actions"} <= localized
    assert "noisy_historical.body_contacts" not in declared_paths("noisy_historical")
    assert "historical.body_contacts" in localized  # contacts declared

    full = names(_sensors(localization=True, base_velocity=True, height=True))
    assert {"noisy.rigid_body_vel", "noisy.root_vel", "noisy.ground_heights"} <= full
    # Sensors never carry contacts or joint forces: the names are not declared.
    assert "noisy.rigid_body_contacts" not in full and "noisy.dof_forces" not in full


# =============================================================================
# Refusals and unknowns
# =============================================================================


def _check(env_config, robot=None, terrain=None, agent=None):
    robot = robot if robot is not None else SimpleNamespace(contact_bodies=["foot"])
    return check_bindings_against_config(env_config, robot, terrain, agent)


def _reward(kernel):
    """A reward binding the kernel's own context defaults, at unit weight."""
    return MdpComponent.from_kernel(kernel, meta=RewardMeta(weight=1.0))


def test_refuses_with_the_build_checks_wording():
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.envs.terminations.rule import TerminationRule

    config = EnvConfig(
        sensors=_sensors(),
        observation_components=sensed(
            {"bad": MdpComponent.from_kernel(compute_humanoid_max_coords_observations)},
            "noisy",
        ),
        reward_components={"gt": _reward(compute_gt_rew)},
        termination_components={
            "terrain": TerminationRule(
                MdpComponent(
                    lambda height_points: height_points,
                    {"height_points": EnvContext.terrain.height_points},
                )
            )
        },
    )
    report = _check(config)
    assert report.unknown == []
    assert report.errors == [
        "observation component 'bad', argument 'body_pos': context path "
        "'noisy.rigid_body_pos' is absent: the sensor config has no localization "
        "sensor (config.sensors.localization is None)",
        "observation component 'bad', argument 'body_vel': context path "
        "'noisy.rigid_body_vel' is absent: the sensor config has no base-velocity "
        "sensor (config.sensors.base_velocity is None)",
        "observation component 'bad', argument 'ground_height': context path "
        "'noisy.ground_heights' is absent: the sensor config has no localization "
        "sensor (config.sensors.localization is None)",
        "reward component 'gt', argument 'ref_rigid_body_pos': context path "
        "'mimic.ref_state.rigid_body_pos' is absent: no control component "
        "publishes 'mimic' (config.control_components: [])",
        "termination component 'terrain', argument 'height_points': context path "
        "'terrain.height_points' is absent: the terrain has no height points",
    ]
    with pytest.raises(ValueError, match="before building anything"):
        report.raise_for_errors()


def test_config_check_reports_missing_and_misplaced_reward_meta():
    """The config-only check reports a reward with no RewardMeta, and a
    non-reward carrying one, as the build does."""
    from protomotions.envs.base_env.config import EnvConfig

    def kernel(dof_pos=EnvContext.current.dof_pos):
        return dof_pos.sum(-1)

    config = EnvConfig(
        observation_components={
            "obs_with_meta": MdpComponent.from_kernel(
                kernel, meta=RewardMeta(weight=1.0)
            )
        },
        reward_components={"rew_no_meta": MdpComponent.from_kernel(kernel)},
    )
    report = _check(config)
    assert (
        "reward component 'rew_no_meta' has no RewardMeta: pass "
        "meta=RewardMeta(weight=...) to from_kernel"
    ) in report.errors
    assert (
        "observation component 'obs_with_meta' carries a RewardMeta; only reward "
        "components take meta"
    ) in report.errors


def test_contacts_and_undeclared_names_are_refused_like_the_build():
    from protomotions.envs.base_env.config import EnvConfig

    config = EnvConfig(
        reward_components={"rate": _reward(compute_action_rate_l2)},
        observation_components={
            "undeclared": MdpComponent(
                lambda x: x, {"x": SimpleNamespace(path="nothing")}
            ),
            "contacts": MdpComponent(lambda f: f, {"f": EnvContext.contact_forces}),
        },
    )
    report = _check(config, robot=SimpleNamespace(contact_bodies=None))
    assert report.errors == [
        "observation component 'undeclared', argument 'x': context path 'nothing' "
        "is not a declared context name (see context_views.py)",
        "observation component 'contacts', argument 'f': context path "
        "'contact_forces' is absent: the robot declares no contact bodies "
        "(robot.contact_bodies is empty, so the simulator reports no contacts)",
    ]
    # The reward's ``previous_action`` is not refused: a binding to a history
    # name is what gives the ring its depth, so the configs report one slot.
    assert (
        config_vocabulary(
            config, SimpleNamespace(contact_bodies=None), None
        ).facts.history_steps
        == 1
    )


def test_data_dependent_names_are_unknown_not_refused_and_the_evaluator_is_checked():
    from protomotions.envs.terminations.rule import TerminationRule
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.envs.base_env.config import EnvConfig

    config = EnvConfig(
        control_components={"mimic": MimicControlConfig()},
        reward_components={"match": _reward(compute_contact_match_rew)},
    )
    smoothed = MdpComponent(
        lambda h: h, {"h": EnvContext.mimic.smoothed_ref_anchor_height}
    )
    evaluator = EvaluatorConfig(
        metric_components={"gt_error": MdpComponent.from_kernel(mean_body_pos_error)},
        failure_components={"smooth": TerminationRule(smoothed, 0.5)},
    )
    agent = SimpleNamespace(evaluator=evaluator)
    report = _check(config, agent=agent)
    # The smoothed height is a switch of the control config (0 here): refused
    # with the build's wording, under the evaluator manager's key.
    smoothing_off = (
        "'mimic.smoothed_ref_anchor_height' is absent: the 'mimic' control "
        "component's config does not turn on 'reference_height_smoothing_seconds' "
        "(MimicControlConfig.reference_height_smoothing_seconds is 0.0)"
    )
    assert report.errors == [
        f"evaluation component 'failure/smooth', argument 'h': context path "
        f"{smoothing_off}"
    ]
    assert report.unknown == [
        "reward component 'match', argument 'ref_contacts': context path "
        "'mimic.ref_state.rigid_body_contacts' can only be checked once the "
        "environment is built: the motion library carries 'rigid_body_contacts' "
        "only when the motion file has it"
    ]
    # With smoothing on, the name is published and the refusal goes away.
    config.control_components["mimic"] = MimicControlConfig(
        reference_height_smoothing_seconds=0.5
    )
    assert _check(config, agent=agent).errors == []

    # A component object shared between a metric and a failure rule is
    # checked once, under the metric's key (the evaluator's manager routes it
    # that way), so its refusal is not reported twice.
    config.control_components["mimic"] = MimicControlConfig()
    shared = EvaluatorConfig(
        metric_components={"smooth": smoothed},
        failure_components={"smooth_fail": TerminationRule(smoothed, 0.5)},
    )
    assert _check(config, agent=SimpleNamespace(evaluator=shared)).errors == [
        f"evaluation component 'metric/smooth', argument 'h': context path "
        f"{smoothing_off}"
    ]

    # A None component dict (an inference override clears terminations that
    # way) counts as empty; an agent config without an evaluator is fine.
    config.termination_components = None
    assert _check(config, agent=SimpleNamespace()).errors == []


def test_run_config_binding_check_raises_on_errors_and_logs_unknowns(caplog):
    from protomotions.envs.base_env.config import EnvConfig

    log = logging.getLogger("test_context_check")
    config = EnvConfig(
        control_components={"mimic": MimicControlConfig()},
        reward_components={
            "match": _reward(compute_contact_match_rew),
            "match_again": _reward(compute_contact_match_rew),
        },
    )
    robot = SimpleNamespace(contact_bodies=["foot"])
    with caplog.at_level(logging.INFO, logger="test_context_check"):
        report = run_config_binding_check(config, robot, None, None, log)
    # The report keeps one line per binding; the log gets one line per
    # unknown path, naming every component that binds it.
    assert report.errors == [] and len(report.unknown) == 2
    assert [(entry.path, entry.bound_by) for entry in report.unknown_paths] == [
        (
            "mimic.ref_state.rigid_body_contacts",
            ("reward component 'match'", "reward component 'match_again'"),
        )
    ]
    logged = [record.getMessage() for record in caplog.records]
    assert logged == [
        "config check: context path 'mimic.ref_state.rigid_body_contacts' can "
        "only be checked once the environment is built (the motion library "
        "carries 'rigid_body_contacts' only when the motion file has it); bound "
        "by reward component 'match', reward component 'match_again'"
    ]

    config.observation_components["undeclared"] = MdpComponent(
        lambda x: x, {"x": SimpleNamespace(path="nothing")}
    )
    with pytest.raises(ValueError, match="is not a declared context name"):
        run_config_binding_check(config, robot, None, None, log)


def test_masked_mimic_disabled_modality_is_refused_with_the_switch_named():
    """A binding to a modality the masked-mimic config leaves off is refused
    from the configs, naming the config field that turns it on; turning the
    switch on makes the name expected."""
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.envs.control.masked_mimic_control import (
        MaskedMimicStyleConfig,
        MaskedMimicVelocityCommandConfig,
    )

    config = EnvConfig(
        control_components={"masked_mimic": MaskedMimicControlConfig()},
        observation_components={
            "style": MdpComponent.from_kernel(
                to_float, x=EnvContext.masked_mimic.style_pose_samples
            ),
            "velocity_command": MdpComponent.from_kernel(
                compute_masked_mimic_velocity_command_obs
            ),
            "local": MdpComponent(
                lambda p: p, {"p": EnvContext.masked_mimic.local_ref_pos}
            ),
            "anchor": MdpComponent(
                lambda m: m, {"m": EnvContext.masked_mimic.anchor_rotation_mask}
            ),
        },
    )
    report = _check(config)
    switches = {
        "style": "MaskedMimicControlConfig.style is None",
        "velocity_command": "MaskedMimicControlConfig.velocity_command is None",
        "local_stream": "MaskedMimicControlConfig.local_stream is False",
        "anchor_pose_submasking": (
            "MaskedMimicControlConfig.anchor_pose_submasking is False"
        ),
    }
    assert report.errors, "every binding above is to a disabled modality"
    for line in report.errors:
        switch = next(name for name in switches if f"turn on '{name}'" in line)
        assert line.endswith(
            f"is absent: the 'masked_mimic' control component's config does not "
            f"turn on '{switch}' ({switches[switch]})"
        ), line
    assert {line.split("'")[1] for line in report.errors} == set(
        config.observation_components
    )

    config.control_components["masked_mimic"] = MaskedMimicControlConfig(
        style=MaskedMimicStyleConfig(),
        velocity_command=MaskedMimicVelocityCommandConfig(),
        local_target_condition_probability=0.5,
        anchor_pose_submasking=True,
    )
    # With every stream on, the canvas check (asked once the names check
    # out) reads the real robot's layout and agrees.
    assert _check(config, robot=_smpl_robot()).errors == []


def _smpl_robot():
    from protomotions.robot_configs.factory import robot_config

    with redirect_stdout(io.StringIO()):  # the robot factories print
        return robot_config("smpl")


def test_the_config_check_asks_every_control_about_the_bound_paths():
    """A configured control class's check_bindings refusal is reported from
    the configs, in its own words, once the name checks pass (the build's
    order)."""
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.tests.test_goal_canvas import RefusingControlConfig

    dof = MdpComponent(
        lambda d: d, {"d": EnvContext.current.dof_pos}, meta=RewardMeta(weight=1.0)
    )
    config = EnvConfig(control_components={"refusing": RefusingControlConfig()})
    assert _check(config).errors == []
    config.reward_components = {"dof": dof}
    assert _check(config).errors == [
        "refused by the test control: current.dof_pos is bound"
    ]


@dataclass
class CheckingControlConfig(ControlComponentConfig):
    """A control with one episode-ending check, bound to ``path``."""

    _target_: str = "protomotions.tests.test_context_check.CheckingControl"
    path: str = "noisy.root_pos"


class CheckingControl(ControlComponent):
    """A control whose ``terminations`` binds its config's ``path``, to show
    the config-only check covers a control's own checks."""

    def advance(self, state):
        return {}

    def publish(self, env_ids):
        return {}

    @classmethod
    def terminations(cls, config, robot_config):
        reading = resolve_path(EnvContext, config.path)
        return {"fall": TerminationRule(MdpComponent(lambda p: p, {"p": reading}))}


def test_the_config_check_covers_the_controls_own_terminations():
    """A control's episode-ending checks (``ControlComponent.terminations``,
    keyed ``<control>/<check>`` as the build's termination table keys them)
    are checked with the recipe's terminations: a check bound to a name this
    configuration will not publish is refused before the build with a recipe
    termination's wording, and the controls' ``check_bindings`` are asked
    about the checks' paths too (the build's union includes them)."""
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.tests.test_goal_canvas import RefusingControlConfig

    recipe = EnvConfig(
        termination_components={
            "fall": TerminationRule(
                MdpComponent(lambda p: p, {"p": EnvContext.noisy.root_pos})
            )
        }
    )
    control = EnvConfig(control_components={"checking": CheckingControlConfig()})
    [recipe_line] = _check(recipe).errors
    [control_line] = _check(control).errors
    assert recipe_line.startswith("termination component 'fall', argument 'p': ")
    assert "'noisy.root_pos' is absent" in recipe_line
    assert control_line == recipe_line.replace("'fall'", "'checking/fall'")

    both = EnvConfig(
        control_components={
            "refusing": RefusingControlConfig(),
            "checking": CheckingControlConfig(path="current.dof_pos"),
        }
    )
    assert _check(both).errors == [
        "refused by the test control: current.dof_pos is bound"
    ]


def test_the_config_check_refuses_a_bare_failure_component():
    """An evaluator failure must be a TerminationRule (or a ThresholdFailure
    over one); a bare metric is refused before anything is built."""
    from protomotions.agents.evaluators.config import EvaluatorConfig
    from protomotions.envs.base_env.config import EnvConfig

    evaluator = EvaluatorConfig(
        failure_components={"bad": MdpComponent.from_kernel(mean_body_pos_error)}
    )
    with pytest.raises(TypeError, match="failure 'bad' must be a TerminationRule or"):
        _check(
            EnvConfig(control_components={"mimic": MimicControlConfig()}),
            agent=SimpleNamespace(evaluator=evaluator),
        )


def test_the_amp_reference_table_is_checked_for_stale_static_keys():
    """The AMP reference-observation kernels read the motion library through
    the training component's own call, which fills their parameters from its
    runtime context and ignores a static key the kernel does not take. The
    config-only check refuses such a key by name; the inputs that call
    supplies are not bindings, so their absence is not an error here."""
    from protomotions.envs.base_env.config import EnvConfig
    from protomotions.envs.obs.humanoid_historical import (
        compute_historical_max_coords_from_motion_lib,
    )
    from protomotions.envs.rewards.meta import RewardMeta

    reference = MdpComponent(
        compute_func=compute_historical_max_coords_from_motion_lib,
        dynamic_vars={},
        static_params={"num_state_history_steps": 3, "history_steps": [1, 2, 3]},
    )
    agent = SimpleNamespace(reference_obs_components={"historical": reference})
    assert _check(EnvConfig(), agent=agent).errors == [
        "reference component 'historical': 'num_state_history_steps' is not a "
        "parameter of kernel 'compute_historical_max_coords_from_motion_lib'"
    ]
    del reference.static_params["num_state_history_steps"]
    assert _check(EnvConfig(), agent=agent).errors == []
    # Metadata placement is checked as for any non-reward table.
    reference.meta = RewardMeta(weight=1.0)
    assert _check(EnvConfig(), agent=agent).errors == [
        "reference component 'historical' carries a RewardMeta; only reward "
        "components take meta"
    ]


def test_control_component_namespaces_are_declared_and_match_the_build():
    """Every control component class declares the namespaces it publishes;
    the env's build check compares the declaration with what was published."""
    pytest.importorskip("mujoco")
    from protomotions.envs.base_env.build import validate_bindings
    from protomotions.envs.control.mimic_control import MimicControl
    from protomotions.tests.support.sensor_simulator import make_sensor_test_env

    assert MimicControl.namespaces == ("mimic",)
    env = make_sensor_test_env(**_mimic_kwargs())
    try:
        # A stale declaration is refused by the build check itself.
        env.control_manager.components["mimic"].namespaces = ("mimic", "steering")
        env._current_context = None
        with pytest.raises(
            ValueError,
            match="control component 'mimic' declares namespace 'steering' but "
            "published nothing under it",
        ):
            validate_bindings(env)
        env.control_manager.components["mimic"].namespaces = ()
        env._current_context = None
        with pytest.raises(
            ValueError,
            match="control component 'mimic' publishes namespace 'mimic' but does "
            "not declare it",
        ):
            validate_bindings(env)
    finally:
        env.close()


# =============================================================================
# Every example experiment, offline
# =============================================================================


def test_typed_overrides_land_on_the_steering_recipe():
    """One typed override per leaf kind, through the real build funnel.

    The early ``robot`` override reaches the env build; the ``float`` lands on a
    frozen ``RewardMeta`` rebuilt and reassigned through the reward MdpComponent.
    (The enum and Union kinds are exercised in ``test_config_utils``; forcing a
    non-FIXED motion mode here would fail ``motion_lib.validate`` on the recipe's
    single unsharded motion file.)
    """
    from protomotions.utils.config_builder import build_configs
    from protomotions.utils.config_utils import read_override

    module = importlib.import_module("examples.experiments.steering.mlp")
    args = fake_args(module)
    overrides = {
        "env.max_episode_length": 1500,
        "simulator.num_envs": 8,
        "simulator.headless": False,
        "robot.anchor_body_name": "pelvis",
        "env.reward_components.heading_rew.meta.weight": 3e-4,
    }
    with redirect_stdout(io.StringIO()):
        configs = build_configs(module, args, overrides)

    assert configs.env.max_episode_length == 1500
    assert configs.simulator.num_envs == 8
    assert configs.simulator.headless is False
    assert configs.robot.anchor_body_name == "pelvis"
    assert (
        read_override(configs, "env.reward_components.heading_rew.meta.weight") == 3e-4
    )


# The in-tree control component configs (each names its component class in
# ``_target_``); the tag guard below reads them to confirm every switch exists.
IN_TREE_CONTROL_CONFIGS = (
    "protomotions.envs.control.mimic_control.MimicControlConfig",
    "protomotions.envs.control.masked_mimic_control.MaskedMimicControlConfig",
    "protomotions.envs.control.masked_mimic_tasks.controller."
    "MaskedMimicTaskControlConfig",
    "protomotions.envs.control.steering_control.SteeringControlConfig",
    "protomotions.envs.control.path_follower_control.PathFollowerControlConfig",
    "protomotions.envs.control.scene_boxes_control.SceneBoxesControlConfig",
    "protomotions.envs.control.target_control.TargetControlConfig",
    "protomotions.envs.control.gpc_sft_reference_target_control."
    "GPCSFTReferenceTargetControlConfig",
)


def test_every_requires_tag_has_a_rule():
    """A ``requires=`` tag in the vocabulary is either a row of the table in
    ``context_requirements.py`` or, on a control namespace's name, a switch
    some in-tree component publishing that namespace answers for: a field of
    its config class, or a ``provides`` override that says True for a default
    config. A tag with neither would make the check raise on every
    configuration that publishes the namespace. Every row of the table is
    also explained in the list in ``context_views.py``'s docstring."""
    from dataclasses import fields

    import protomotions.envs.context_views as context_views
    from protomotions.utils.hydra_replacement import get_class

    for name in REQUIREMENTS:
        assert f"- ``{name}``:" in context_views.__doc__, name
    components = []
    for target in IN_TREE_CONTROL_CONFIGS:
        config = get_class(target)()
        components.append((get_class(config._target_), config))
    for namespace in NAMESPACES:
        for tag in namespace_requires(namespace):
            assert tag in REQUIREMENTS, (namespace, tag)
        for path, field in declared_paths(namespace).items():
            for tag in field.requires:
                if tag in REQUIREMENTS:
                    continue
                assert namespace in CONTROL_NAMESPACES, (path, tag)
                answered = [
                    type(config).__name__
                    for component, config in components
                    if namespace in component.namespaces
                    and (
                        tag in {f.name for f in fields(config)}
                        or component.provides(config, tag)
                    )
                ]
                assert answered, f"'{path}' requires '{tag}': no component answers"
    # With everything on (a sensor suite with every feature, history, a
    # terrain config, contact bodies) every namespace that is not a control
    # component's has at least one expected name.
    from protomotions.components.terrains.config import TerrainConfig
    from protomotions.envs.base_env.config import EnvConfig

    config = EnvConfig(
        sensors=SUITES["all"],
        observation_components={
            "hist": MdpComponent.from_kernel(
                compute_historical_reduced_coords_from_state, history_steps=2
            )
        },
    )
    robot = SimpleNamespace(contact_bodies=["foot"])
    vocabulary = config_vocabulary(config, robot, TerrainConfig())
    published = {
        path.split(".", 1)[0] if "." in path else "" for path in vocabulary.expected
    }
    # A namespace whose names only the built env can confirm (the scene
    # library's objects and pointclouds) is deferred, not absent.
    deferred = {path.split(".", 1)[0] for path in vocabulary.unknown if "." in path}
    for namespace in NAMESPACES:
        assert (
            namespace in CONTROL_NAMESPACES
            or namespace in published
            or namespace in deferred
        ), namespace
