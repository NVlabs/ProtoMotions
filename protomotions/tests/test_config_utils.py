# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the typed override funnel and the config parsing utilities."""

import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Any, List, Optional, Union

import pytest

from protomotions.agents.common.config import (
    MLPWithConcatConfig,
    ModuleContainerConfig,
    validate_module_order,
)
from protomotions.envs.sensors.config import (
    NoNoise,
)
from protomotions.utils.config_utils import (
    apply_overrides,
    declaring_module,
    is_union,
    parse_cli_overrides,
    read_override,
    resolve_field_annotation,
)
from protomotions.utils.experiment_configs import ExperimentConfigs


class _PlainMode(Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass(frozen=True)
class _FrozenLimits:
    speed: float = 1.0

    def __post_init__(self):
        if self.speed < 0:
            raise ValueError("speed must be nonnegative")


@dataclass
class _NeedsReason:
    reason: str


@dataclass
class _Leaf:
    i: int = 0
    f: float = 0.0
    b: bool = False
    s: str = "x"
    mode: _PlainMode = _PlainMode.FAST
    opt: Optional[int] = None
    items: List[int] = field(default_factory=list)
    anyv: Any = None
    noise: Union[float, NoNoise] = 0.0


@dataclass
class _Nested:
    leaf: _Leaf = field(default_factory=_Leaf)
    limits: Optional[_FrozenLimits] = None
    needs: Optional[_NeedsReason] = None

    @property
    def derived(self) -> int:
        return 7


@dataclass
class _SiblingBad:
    """A dataclass whose ``bad`` field names a type that never resolves.

    Like a `TYPE_CHECKING`-only annotation on one field of a real config: it must
    not keep the funnel from typing ``ok``.
    """

    ok: int = 0
    bad: "DoesNotExist" = None  # noqa: F821 - an intentionally unresolvable ref


def _apply(root_obj, overrides, root="env"):
    kwargs = {name: None for name in ExperimentConfigs.ROOTS}
    kwargs[root] = root_obj
    return apply_overrides(ExperimentConfigs(**kwargs), overrides)


# --- the funnel, one test per acceptance sentence ---------------------------


def test_unknown_root_is_refused_naming_the_roots():
    with pytest.raises(
        ValueError, match=r"unknown override root 'nope'.*roots are robot"
    ):
        _apply(_Leaf(), {"nope.i": 1})


def test_agent_root_without_an_agent_config_is_refused():
    with pytest.raises(ValueError, match="this experiment has no agent config"):
        apply_overrides(
            ExperimentConfigs(None, None, None, None, None, None, None),
            {"agent.i": 1},
        )


def test_undeclared_field_is_refused():
    with pytest.raises(
        ValueError, match="'env.ghost': 'ghost' is not a declared field"
    ):
        _apply(_Leaf(), {"env.ghost": 1})


def test_a_derived_property_is_refused_with_the_derived_hint():
    with pytest.raises(
        ValueError, match="'derived' is not a declared field.*it is derived"
    ):
        _apply(_Nested(), {"env.derived": 1})


@dataclass
class _DeclaresOptional:
    value: "Optional[int]" = None  # the string form of a __future__ module


def test_an_inherited_field_resolves_in_its_declaring_module():
    # The funnel's twin of the saver's rule: a subclass in a module without the
    # imports of the declaring module still resolves the inherited field.
    fresh = types.ModuleType("protomotions_test_fresh_module_utils")
    sys.modules[fresh.__name__] = fresh
    try:

        @dataclass
        class _Child(_DeclaresOptional):
            extra: int = 1

        _Child.__module__ = fresh.__name__
        assert declaring_module(_Child, "value") == __name__
        assert declaring_module(_Child, "extra") == fresh.__name__
        assert resolve_field_annotation(_Child, "value") == Optional[int]
        obj = _Child()
        _apply(obj, {"env.value": 4})
        assert obj.value == 4
    finally:
        del sys.modules[fresh.__name__]


def test_resolve_field_annotation_returns_any_for_a_non_field():
    # A name that is not a declared field resolves to Any: the saver only asks for real
    # fields (it iterates dataclasses.fields) and the override funnel validates fields
    # with _require_field first, so this is the defensive default, not a silent miss.
    @dataclass
    class _OneField:
        a: int = 1

    assert resolve_field_annotation(_OneField, "not_a_field") is Any


def test_is_union_sees_both_spellings():
    assert is_union(Optional[int])
    assert is_union(Union[float, NoNoise])
    assert not is_union(int)
    assert not is_union(List[int])
    if sys.version_info >= (3, 10):
        assert is_union(eval("int | None"))  # a types.UnionType


@pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP 604 unions need 3.10")
def test_an_override_sets_none_through_a_pep604_optional_field():
    # The funnel's Optional check must accept ``X | None`` (a types.UnionType)
    # as it accepts ``Optional[X]`` (open item 41); the string annotation is
    # what ``from __future__ import annotations`` leaves on a config module.
    @dataclass
    class _Opt604:
        opt: "int | None" = 3

    obj = _Opt604()
    _apply(obj, {"env.opt": None})
    assert obj.opt is None
    _apply(obj, {"env.opt": 5})
    assert obj.opt == 5


def test_a_sibling_unresolvable_annotation_does_not_break_other_overrides():
    obj = _SiblingBad()
    _apply(obj, {"env.ok": 5})  # 'bad' does not resolve, but 'ok' is typed fine
    assert obj.ok == 5
    with pytest.raises(
        TypeError,
        match=r"annotation .*DoesNotExist.* of _SiblingBad.bad does not resolve",
    ):
        _apply(_SiblingBad(), {"env.bad": 1})


def test_a_none_optional_dataclass_on_the_way_is_constructed():
    nested = _Nested()
    _apply(nested, {"env.limits.speed": 2.0})
    assert nested.limits == _FrozenLimits(2.0)


def test_a_none_optional_with_required_fields_is_refused():
    nested = _Nested()
    with pytest.raises(
        ValueError, match="crosses needs which is None and _NeedsReason has required"
    ):
        _apply(nested, {"env.needs.reason": "x"})


def test_int_leaf_requires_int_and_refuses_bool():
    leaf = _Leaf()
    _apply(leaf, {"env.i": 5})
    assert leaf.i == 5
    with pytest.raises(ValueError, match=r"expected <class 'int'>.*got bool"):
        _apply(_Leaf(), {"env.i": True})


def test_float_leaf_accepts_int_and_bool_leaf_requires_bool():
    leaf = _Leaf()
    _apply(leaf, {"env.f": 3})
    assert leaf.f == 3.0 and isinstance(leaf.f, float)
    _apply(leaf, {"env.b": True})
    assert leaf.b is True
    with pytest.raises(ValueError, match=r"expected <class 'bool'>.*got int"):
        _apply(_Leaf(), {"env.b": 1})


def test_str_leaf_and_any_leaf_take_their_literal():
    leaf = _Leaf()
    _apply(leaf, {"env.s": "hi", "env.anyv": {"free": [1, 2]}})
    assert leaf.s == "hi"
    assert leaf.anyv == {"free": [1, 2]}


def test_enum_leaf_accepts_a_member_name_and_a_value():
    leaf = _Leaf()
    _apply(leaf, {"env.mode": "SLOW"})
    assert leaf.mode is _PlainMode.SLOW
    _apply(leaf, {"env.mode": "fast"})
    assert leaf.mode is _PlainMode.FAST
    with pytest.raises(ValueError, match="invalid value 'zzz' for _PlainMode"):
        _apply(_Leaf(), {"env.mode": "zzz"})


def test_list_leaf_recurses_per_element():
    leaf = _Leaf()
    _apply(leaf, {"env.items": [1, 2, 3]})
    assert leaf.items == [1, 2, 3]


def test_union_member_accepts_a_dict_literal_and_a_float():
    leaf = _Leaf()
    _apply(leaf, {"env.noise": 0.02})
    assert leaf.noise == 0.02
    _apply(leaf, {"env.noise": {"reason": "cli"}})
    assert leaf.noise == NoNoise("cli")
    with pytest.raises(TypeError):
        _apply(_Leaf(), {"env.noise": {}})  # NoNoise needs a reason


def test_a_frozen_dataclass_on_the_way_is_rebuilt_and_assigned_back():
    nested = _Nested()
    nested.limits = _FrozenLimits(1.0)
    _apply(nested, {"env.limits.speed": 4.0})
    assert nested.limits == _FrozenLimits(4.0)
    with pytest.raises(ValueError, match="speed must be nonnegative"):
        _apply(nested, {"env.limits.speed": -1.0})


def test_read_override_walks_dicts_and_attributes():
    env = SimpleNamespace(table={"k": _Leaf(i=9)})
    assert (
        read_override(
            ExperimentConfigs(None, None, None, None, None, env, None), "env.table.k.i"
        )
        == 9
    )


# --- the build-time undo check ----------------------------------------------


def _fake_experiment(bad_env_hook=None):
    """A minimal experiment module build_configs can run without a motion file."""
    import importlib
    import types

    mlp = importlib.import_module("examples.experiments.mimic.mlp")
    module = types.ModuleType("fake_undo_experiment")
    module.__file__ = "fake_undo_experiment.py"
    module.configure_robot_and_simulator = mlp.configure_robot_and_simulator
    module.terrain_config = mlp.terrain_config
    module.scene_lib_config = mlp.scene_lib_config
    module.motion_lib_config = mlp.motion_lib_config
    module.agent_config = mlp.agent_config

    def env_config(robot, args):
        if bad_env_hook is not None:
            bad_env_hook(robot)
        return mlp.env_config(robot, args)

    module.env_config = env_config
    return module


def _build_args():
    import argparse

    return argparse.Namespace(
        robot_name="g1",
        simulator="mujoco",
        headless=True,
        num_envs=1,
        experiment_name="undo",
        motion_file="motions.pt",
        scenes_file=None,
        batch_size=16,
        training_max_steps=16,
        num_mini_epochs=1,
    )


def test_a_builder_that_undoes_an_early_override_is_named():
    from protomotions.utils.config_builder import build_configs

    def undo(robot):
        robot.update_fields(anchor_body_name="left_ankle_roll_link")

    with pytest.raises(
        ValueError,
        match=r"override robot.anchor_body_name='pelvis' was undone by env_config\(\)",
    ):
        build_configs(
            _fake_experiment(undo),
            _build_args(),
            overrides={"robot.anchor_body_name": "pelvis"},
        )


def test_an_early_robot_override_reaches_the_env_build():
    from protomotions.utils.config_builder import build_configs

    configs = build_configs(
        _fake_experiment(),
        _build_args(),
        overrides={"robot.anchor_body_name": "pelvis", "env.max_episode_length": 321},
    )
    assert configs.robot.anchor_body_name == "pelvis"
    assert configs.env.max_episode_length == 321


@pytest.mark.parametrize("value, expected", [(5, 5), ([1, 3, 5], (1, 3, 5))])
def test_a_mimic_future_steps_override_lands_through_the_funnel(value, expected):
    """``future_steps: Union[int, Tuple[int, ...]]`` admits the int and the step
    list on a mimic recipe. The funnel writes after ``__post_init__``, so the
    int stays an int in the saved config and every reader normalises it
    (``MimicControl``, ``GoalCanvas``, the exporters)."""
    from protomotions.utils.config_builder import build_configs

    configs = build_configs(
        _fake_experiment(),
        _build_args(),
        overrides={"env.control_components.mimic.future_steps": value},
    )
    assert configs.env.control_components["mimic"].future_steps == expected

    with pytest.raises(ValueError, match="no member of .*future_steps"):
        build_configs(
            _fake_experiment(),
            _build_args(),
            overrides={"env.control_components.mimic.future_steps": "1,3"},
        )


def test_an_unknown_override_root_is_refused_in_the_two_pass_build():
    """The early/late split must not silently drop a typo'd root."""
    from protomotions.utils.config_builder import build_configs

    with pytest.raises(ValueError, match="unknown override root 'flonk'"):
        build_configs(_fake_experiment(), _build_args(), overrides={"flonk.bar": 1})


# --- validate_module_order, fresh and resumed -------------------------------


def _container(models):
    return ModuleContainerConfig(in_keys=["a"], out_keys=["z"], models=models)


def test_validate_module_order_passes_a_sound_container():
    good = _container(
        [MLPWithConcatConfig(in_keys=["a"], out_keys=["z"], num_out=1, layers=[])]
    )
    validate_module_order(good)  # no raise


def test_validate_module_order_raises_on_a_producer_after_its_consumer():
    bad = _container(
        [
            MLPWithConcatConfig(in_keys=["mid"], out_keys=["z"], num_out=1, layers=[]),
            MLPWithConcatConfig(in_keys=["a"], out_keys=["mid"], num_out=1, layers=[]),
        ]
    )
    with pytest.raises(ValueError, match="module ordering error"):
        validate_module_order(SimpleNamespace(nested=bad))


# --- parse_cli_overrides (unchanged behaviour) ------------------------------


def test_parse_cli_overrides_coerces_literal_values_and_keeps_strings():
    overrides = parse_cli_overrides(
        [
            "env.max_episode_length=1000",
            "simulator.headless=True",
            "robot.asset=None",
            "agent.name=gpc_prior",
            "terrain.scale=0.05",
        ]
    )
    assert overrides == {
        "env.max_episode_length": 1000,
        "simulator.headless": True,
        "robot.asset": None,
        "agent.name": "gpc_prior",
        "terrain.scale": 0.05,
    }


def test_parse_cli_overrides_rejects_missing_separator_and_empty_key():
    for value in ["env.max_episode_length", "=123"]:
        try:
            parse_cli_overrides([value])
        except ValueError as error:
            assert "Invalid override" in str(error)
        else:
            raise AssertionError(f"Expected invalid override to fail: {value}")
