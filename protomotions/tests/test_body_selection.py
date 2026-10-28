# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bodies a kernel reads: named by the recipe, resolved to indices once at build."""

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from protomotions.agents.evaluators.components import ThresholdFailure
from protomotions.agents.evaluators.manager import EvaluationManager
from protomotions.envs.body_selection import (
    BodySelection,
    resolve_body_selections,
    resolve_rule_selections,
    unresolved_selections,
)
from protomotions.envs.component_manager import ComponentManager
from protomotions.envs.context import FrozenContext
from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.envs.terminations.rule import TerminationRule
from protomotions.robot_configs.base import RobotConfig

BODIES = ("pelvis", "left_hand", "right_hand", "left_foot", "right_foot")


class FakeRobot:
    """The two attributes a selection reads, with the real group resolution."""

    kinematic_info = SimpleNamespace(body_names=list(BODIES))
    common_naming_to_robot_body_names = {
        "hands": ["left_hand", "right_hand"],
        "feet": ["left_foot", "right_foot"],
    }
    body_indices = RobotConfig.body_indices


ROBOT = FakeRobot()


def by_bodies(body_pos: Tensor, body_ids: Tensor) -> Tensor:
    return body_pos[:, body_ids].flatten(1)


def every_body(body_pos: Tensor) -> Tensor:
    return body_pos.flatten(1)


def selecting_component(selection):
    return MdpComponent(
        by_bodies,
        {"body_pos": EnvContext.current.rigid_body_pos},
        {"body_ids": selection},
    )


# --------------------------------------------------------------------------
# BodySelection
# --------------------------------------------------------------------------


def test_a_selection_is_exactly_one_kind():
    with pytest.raises(ValueError, match="exactly one of names, groups or pattern"):
        BodySelection()
    with pytest.raises(ValueError, match="exactly one of names, groups or pattern"):
        BodySelection(names=("pelvis",), groups=("hands",))
    with pytest.raises(TypeError, match="names and groups are strings"):
        BodySelection.named(1)
    with pytest.raises(TypeError, match="regular expression string"):
        BodySelection(pattern=3)


def test_named_bodies_resolve_in_the_order_written():
    assert BodySelection.named("right_hand", "pelvis").resolve(ROBOT) == [2, 0]
    with pytest.raises(ValueError, match="'nose' is not a body of FakeRobot"):
        BodySelection.named("left_hand", "nose").resolve(ROBOT)


def test_groups_resolve_through_the_robot_semantic_groups():
    assert BodySelection.group("feet", "root").resolve(ROBOT) == [3, 4, 0]
    assert BodySelection.group("all").resolve(ROBOT) == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError, match="neither a body of FakeRobot nor a semantic"):
        BodySelection.group("wings").resolve(ROBOT)


def test_a_pattern_resolves_in_body_order_and_must_match_a_whole_name():
    assert BodySelection.matching(r".*_hand").resolve(ROBOT) == [1, 2]
    assert BodySelection.matching(r"left_.*").resolve(ROBOT) == [1, 3]
    with pytest.raises(ValueError, match="pattern 'hand' matches no body"):
        BodySelection.matching("hand").resolve(ROBOT)


def test_a_selection_is_a_frozen_value():
    selection = BodySelection.group("hands")
    assert selection == BodySelection.group("hands")
    assert hash(selection) == hash(BodySelection.group("hands"))
    with pytest.raises(AttributeError):
        selection.groups = ("feet",)


# --------------------------------------------------------------------------
# Resolving a component table
# --------------------------------------------------------------------------


def test_resolve_body_selections_copies_only_the_components_holding_one():
    table = {
        "hands": selecting_component(BodySelection.group("hands")),
        "plain": MdpComponent(
            every_body, {"body_pos": EnvContext.current.rigid_body_pos}
        ),
    }
    resolved = resolve_body_selections(table, ROBOT)

    assert resolved is not table
    assert resolved["plain"] is table["plain"]
    assert resolved["hands"] is not table["hands"]
    # A table with nothing to resolve is the same mapping.
    plain_only = {"plain": table["plain"]}
    assert resolve_body_selections(plain_only, ROBOT) is plain_only
    assert torch.equal(
        resolved["hands"].static_params["body_ids"], torch.tensor([1, 2])
    )
    assert resolved["hands"].static_params["body_ids"].dtype == torch.long
    assert resolved["hands"].compute_func is by_bodies
    assert resolved["hands"].get_bindings_dict() == table["hands"].get_bindings_dict()
    assert unresolved_selections(resolved["hands"]) == {}

    # The saved configuration keeps the names.
    assert table["hands"].static_params["body_ids"] == BodySelection.group("hands")
    assert unresolved_selections(table["hands"]) == {
        "body_ids": BodySelection.group("hands")
    }

    body_pos = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    ctx = FrozenContext({"current.rigid_body_pos": body_pos}, env_ids=None, dt=0.1)
    assert torch.equal(resolved["hands"].compute(ctx), body_pos[:, [1, 2]].flatten(1))


def test_resolve_body_selections_reports_the_robot_it_resolved_against():
    table = {"typo": selecting_component(BodySelection.named("left_hnad"))}
    with pytest.raises(ValueError, match="'left_hnad' is not a body of FakeRobot"):
        resolve_body_selections(table, ROBOT)


def test_resolve_rule_selections_rebuilds_only_the_rules_with_a_selection():
    plain = MdpComponent(every_body, {"body_pos": EnvContext.current.rigid_body_pos})
    entries = {
        "feet_low": TerminationRule(
            selecting_component(BodySelection.group("feet")),
            threshold=0.5,
            fail_above=False,
            truncation=True,
        ),
        "plain": TerminationRule(plain),
    }
    resolved = resolve_rule_selections(entries, ROBOT)

    assert resolved["plain"] is entries["plain"]
    rule = resolved["feet_low"]
    assert rule is not entries["feet_low"]
    assert (rule.threshold, rule.fail_above, rule.truncation) == (0.5, False, True)
    assert torch.equal(rule.component.static_params["body_ids"], torch.tensor([3, 4]))
    assert isinstance(
        entries["feet_low"].component.static_params["body_ids"], BodySelection
    )


def test_the_manager_refuses_a_selection_that_reached_it_unresolved():
    manager = ComponentManager(torch.device("cpu"))
    ctx = FrozenContext(
        {"current.rigid_body_pos": torch.zeros(2, 5, 3)}, env_ids=None, dt=0.1
    )
    table = {"hands": selecting_component(BodySelection.group("hands"))}
    with pytest.raises(
        ValueError,
        match="observation component 'hands', parameter 'body_ids': BodySelection "
        "was not resolved",
    ):
        manager.validate_bindings(table, ctx, group="observation")
    manager.validate_bindings(
        resolve_body_selections(table, ROBOT), ctx, group="observation"
    )


def test_resolve_rule_selections_returns_the_same_table_when_nothing_resolves():
    plain = MdpComponent(every_body, {"body_pos": EnvContext.current.rigid_body_pos})
    entries = {"plain": TerminationRule(plain)}
    assert resolve_rule_selections(entries, ROBOT) is entries


def test_a_selection_serializes_as_the_names_the_recipe_wrote():
    """``to_dict`` (the saved form and the recipe snapshot) keeps the selection's
    fields; only the resolved copy carries indices."""
    component = selecting_component(BodySelection.named("left_hand", "right_hand"))
    assert component.to_dict()["body_ids"] == {
        "names": ("left_hand", "right_hand"),
        "groups": (),
        "pattern": None,
    }
    resolved = resolve_body_selections({"hands": component}, ROBOT)["hands"]
    assert resolved.to_dict()["body_ids"] == [1, 2]


def test_the_evaluation_manager_resolves_its_metric_and_failure_selections():
    """The evaluator's two tables are resolved where its manager is built, as
    ``build_env`` resolves the env's: a bare rule and a ``ThresholdFailure``
    alike, with the failure's reduction kept and the configuration's names
    untouched."""
    config = SimpleNamespace(
        metric_components={"hands": selecting_component(BodySelection.group("hands"))},
        failure_components={
            "feet": TerminationRule(
                selecting_component(BodySelection.group("feet")), 0.5
            ),
            "pelvis": ThresholdFailure(
                TerminationRule(
                    selecting_component(BodySelection.named("pelvis")), 0.5
                ),
                reduction="max",
            ),
        },
    )
    manager = EvaluationManager(config, ROBOT, 3, 2, torch.device("cpu"))

    for key, indices in (
        ("metric/hands", [1, 2]),
        ("failure/feet", [3, 4]),
        ("failure/pelvis", [0]),
    ):
        body_ids = manager.components[key].static_params["body_ids"]
        assert torch.equal(body_ids, torch.tensor(indices)), key
    assert manager._failures["pelvis"].reduction == "max"
    assert isinstance(
        config.metric_components["hands"].static_params["body_ids"], BodySelection
    )
    assert isinstance(
        config.failure_components["pelvis"].component.static_params["body_ids"],
        BodySelection,
    )


def selected_height(body_pos: Tensor, body_ids: Tensor) -> Tensor:
    return body_pos[:, body_ids, 2].mean(dim=-1)


def test_the_built_env_validates_and_runs_the_resolved_tables():
    """A selection in an observation or a reward survives the real build: the
    check at the ``check`` phase walks the managers' resolved tables, not the
    config's (whose selections stay names), and one step runs both kernels on
    the index tensors. The test double's bodies are pelvis, torso, forearm,
    hand."""
    from protomotions.envs.rewards.meta import RewardMeta
    from protomotions.tests.support.sensor_simulator import make_sensor_test_env

    observation = selecting_component(BodySelection.named("hand", "forearm"))
    reward = MdpComponent(
        selected_height,
        {"body_pos": EnvContext.current.rigid_body_pos},
        {"body_ids": BodySelection.named("hand")},
        meta=RewardMeta(weight=1.0),
    )
    env = make_sensor_test_env(
        observations={"hands": observation}, rewards={"hand_height": reward}
    )

    resolved_obs = env._observations.components["hands"].static_params["body_ids"]
    resolved_rew = env._rewards.components["hand_height"].static_params["body_ids"]
    assert resolved_obs.dtype == torch.long and resolved_rew.dtype == torch.long
    assert torch.equal(resolved_obs, torch.tensor([3, 2]))
    assert torch.equal(resolved_rew, torch.tensor([3]))
    # The configuration still names the bodies.
    assert isinstance(observation.static_params["body_ids"], BodySelection)
    assert isinstance(
        env.config.reward_components["hand_height"].static_params["body_ids"],
        BodySelection,
    )

    obs, reward_value, _, _, _ = env.step(torch.zeros(env.num_envs, 2))
    assert obs["hands"].shape == (env.num_envs, 2 * 3)
    assert reward_value.shape == (env.num_envs,)


def test_a_component_under_both_evaluator_tables_is_resolved_and_run_once():
    """A fall check a recipe logs as a metric and judges as a failure is one
    component object; the manager routes it under one key and resolves its
    selection once, so it is executed once per frame (two resolved copies would
    defeat the identity de-duplication)."""
    shared = selecting_component(BodySelection.group("feet"))
    config = SimpleNamespace(
        metric_components={"feet": shared},
        failure_components={"fell": TerminationRule(shared, 0.5)},
    )
    manager = EvaluationManager(config, ROBOT, 3, 2, torch.device("cpu"))

    assert list(manager.components) == ["metric/feet"]
    assert manager._keys["metric", "feet"] == manager._keys["failure", "fell"]
    body_ids = manager.components["metric/feet"].static_params["body_ids"]
    assert torch.equal(body_ids, torch.tensor([3, 4]))
    assert isinstance(shared.static_params["body_ids"], BodySelection)
