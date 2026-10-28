# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tagged JSON saved form: round-trip, discovery and the per-sentence refusals.

Every value in a config bundle is tagged and rebuilt through its real constructor
(``config_schema.to_tree`` / ``from_tree``). These tests pin the grammar sentence
by sentence, round-trip every offline-buildable example recipe (emit, load, emit:
tree equality), and walk every name the emit produced to assert it resolves back to
the same class or function (the two-way discovery check).
"""

import enum
import functools
import json
import os
import sys
import types
from dataclasses import dataclass, is_dataclass
from typing import Any, List, Optional, Tuple

import pytest
import torch

from protomotions.envs.context_views import EnvContext
from protomotions.envs.mdp_component import MdpComponent
from protomotions.utils.config_schema import (
    FORMAT_VERSION,
    _unwrap_optional,
    _validate_target,
    from_tree,
    to_tree,
)
from protomotions.utils.experiment_configs import (
    ExperimentConfigs,
    load_experiment_configs,
)
from protomotions.utils.hydra_replacement import get_class


def _qual(obj) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def assert_refuses(match: str):
    """The ValueError every save/load rejection raises, matched on ``match`` -- the
    shape of every refusal test here (``with assert_refuses(...): to_tree(...)``)."""
    return pytest.raises(ValueError, match=match)


# --------------------------------------------------------------------------- #
# Module-level fixtures the saved form must resolve by name.
# --------------------------------------------------------------------------- #
class _Color(enum.Enum):
    RED = 1
    BLUE = 2


@dataclass
class _Req:
    needed: int


@dataclass
class _RaisesInPostInit:
    x: int

    def __post_init__(self):
        if self.x < 0:
            raise ValueError("x must be non-negative")


@dataclass
class _HasTable:
    table: dict


@dataclass
class _FieldNamedLikeTag:
    type: str = "a"
    name: str = "b"
    dtype: str = "c"


def _plain_fn(x):
    return x


def _kernel_with_fk(x, fk=None):
    """A kernel taking a baked-FK kernel object as a static param, like metatron's
    historical_proprioception."""
    return x


class _KernelObjectWithBareInitError:
    """A recipe kernel object whose from_init raises without a regenerate sentence."""

    @classmethod
    def from_init(cls, robot, **init):
        raise TypeError("from_init() got an unexpected keyword argument 'y'")


class _OpaqueKernel:
    """A callable object with an inspectable signature but no export protocol."""

    def __call__(self, dof_pos):
        return dof_pos


@dataclass
class _CallableDataclassKernel:
    """A callable DATACLASS kernel with no export protocol: ``is_dataclass`` is true,
    so to_tree would save it as a $type config without the _component_tree guard."""

    scale: float = 1.0

    def __call__(self, dof_pos):
        return dof_pos * self.scale


@dataclass
class _DataclassKernelObject:
    """A kernel object that is ALSO a @dataclass: it carries the export_init/from_init
    protocol, so it must save as {$kernel_object, init} even though ``is_dataclass`` is
    true (to_tree's kernel-object branch runs before its dataclass branch)."""

    scale: float = 1.0

    def __call__(self, dof_pos):
        return dof_pos * self.scale

    def export_init(self):
        return {"scale": self.scale}

    @classmethod
    def from_init(cls, robot, **init):
        return cls(**init)


class _KernelObjectThatRefusesInit:
    """A kernel object whose from_init refuses, like a drifted baked-FK init."""

    @classmethod
    def from_init(cls, robot, **init):
        raise ValueError("the saved constants do not match this code; regenerate")


# --------------------------------------------------------------------------- #
# Scalars, enum, tensor, FieldPath, function: the leaf grammar.
# --------------------------------------------------------------------------- #
def test_scalars_round_trip_as_json():
    for value in (None, True, 3, -2, "text", 1.5):
        assert to_tree(value, where="x") == value
        assert from_tree(value, Any, where="x") == value


def test_non_finite_float_refused_at_save():
    for bad in (float("inf"), float("-inf"), float("nan")):
        with assert_refuses("non-finite float"):
            to_tree(bad, where="agent.x")


def test_enum_round_trips_by_member_name():
    tree = to_tree(_Color.BLUE, where="x")
    assert tree == {"$enum": _qual(_Color), "name": "BLUE"}
    assert from_tree(tree, Any, where="x") is _Color.BLUE


def test_tensor_round_trips_with_dtype():
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    tree = to_tree(t, where="robot.default_dof_pos")
    assert tree == {"$tensor": [[1.0, 2.0], [3.0, 4.0]], "dtype": "float64"}
    back = from_tree(tree, Any, where="robot.default_dof_pos")
    assert back.dtype == torch.float64 and torch.equal(back, t)


def test_unknown_tensor_dtype_refused_on_load():
    # An unmapped dtype must fail, not silently become float32.
    tree = {"$tensor": [1.0, 2.0], "dtype": "float16"}
    with assert_refuses("dtype 'float16' is not one of"):
        from_tree(tree, Any, where="robot.default_dof_pos")


def test_fieldpath_round_trips_as_path():
    fp = EnvContext.dt
    tree = to_tree(fp, where="x")
    assert tree == {"$path": "dt"}
    assert from_tree(tree, Any, where="x").path == "dt"


def test_function_round_trips_by_qualname():
    tree = to_tree(_plain_fn, where="action.fn")
    assert tree == {"$function": _qual(_plain_fn)}
    assert from_tree(tree, Any, where="action.fn") is _plain_fn


def test_partial_refused_at_save():
    with assert_refuses("not a module-level function"):
        to_tree(functools.partial(_plain_fn, 1), where="agent.fn")


def test_a_nested_function_refused_at_save():
    def _inner(x):  # qualname carries <locals>, so it is not importable
        return x

    with assert_refuses("not a module-level function"):
        to_tree(_inner, where="agent.fn")


def test_a_plain_dict_with_a_reserved_key_refused_at_save():
    with assert_refuses(r"reserved for the saved form's tags"):
        to_tree({"$type": "sneaky"}, where="env.table")


def test_a_dataclass_field_named_like_a_tag_round_trips():
    # The "$"-prefixed tags free a dataclass to declare fields named type/name/dtype.
    obj = _FieldNamedLikeTag(type="x", name="y", dtype="z")
    tree = to_tree(obj, where="x")
    assert tree == {
        "$type": _qual(_FieldNamedLikeTag),
        "type": "x",
        "name": "y",
        "dtype": "z",
    }
    back = from_tree(tree, None, where="x")
    assert (back.type, back.name, back.dtype) == ("x", "y", "z")


def test_dict_non_string_key_refused_at_save():
    with assert_refuses("is not a string"):
        to_tree({1: "a"}, where="x")


def test_list_rebuilds_as_tuple_only_when_annotated():
    assert from_tree([1, 2], Tuple[int, int], where="x") == (1, 2)
    assert from_tree([1, 2], List[int], where="x") == [1, 2]


def test_dict_valued_field_threads_the_value_annotation():
    from typing import Dict

    rebuilt = from_tree({"a": [1, 2]}, Dict[str, Tuple[int, int]], where="x")
    assert rebuilt == {"a": (1, 2)}
    assert isinstance(rebuilt["a"], tuple)
    # An untyped dict keeps lists.
    assert from_tree({"a": [1, 2]}, Any, where="x") == {"a": [1, 2]}


def test_an_optional_container_threads_its_element_annotation_on_load():
    # Optional[List[X]] / Optional[Dict[K, V]]: the element / value annotation threads
    # through the Optional wrapper, so a field typed Optional[...] of a container
    # rebuilds its tuples as tuples, exactly as the bare container does.
    from typing import Dict

    rebuilt = from_tree([[1, 2], [3, 4]], Optional[List[Tuple[int, int]]], where="x")
    assert rebuilt == [(1, 2), (3, 4)]
    assert all(isinstance(t, tuple) for t in rebuilt)

    rebuilt = from_tree({"a": [1, 2]}, Optional[Dict[str, Tuple[int, int]]], where="x")
    assert rebuilt == {"a": (1, 2)}
    assert isinstance(rebuilt["a"], tuple)

    # Optional[Tuple[...]] still rebuilds the list as a tuple, including the variadic
    # Optional[Tuple[X, ...]] form the real config fields use.
    assert from_tree([1, 2], Optional[Tuple[int, int]], where="x") == (1, 2)
    assert from_tree([1, 2, 3], Optional[Tuple[int, ...]], where="x") == (1, 2, 3)


def test_unwrap_optional_only_strips_a_single_none_wrapper():
    from typing import Union

    assert _unwrap_optional(Optional[List[int]]) == List[int]
    assert _unwrap_optional(int) is int
    # A union of two non-None types has no single container to thread; left as is.
    assert _unwrap_optional(Union[str, int]) == Union[str, int]
    # Optional[Union[A, B]] (== Union[A, B, None]) likewise keeps the union.
    assert _unwrap_optional(Optional[Union[str, int]]) == Optional[Union[str, int]]


# --------------------------------------------------------------------------- #
# Dataclasses: tag, refusals and the load errors.
# --------------------------------------------------------------------------- #
def test_none_for_a_non_optional_field_refused_at_save():
    # This is where a hook that leaves a component table ``= None`` dies.
    with assert_refuses("does not match the declared type"):
        to_tree(_HasTable(table=None), where="env")


def test_none_allowed_for_an_optional_field():
    @dataclass
    class _Opt:
        table: Optional[dict] = None

    # A module-level class is needed to round-trip; emit alone proves the admit.
    assert to_tree(_Opt(), where="env")["table"] is None


@pytest.mark.skipif(sys.version_info < (3, 10), reason="PEP 604 unions need 3.10")
def test_none_allowed_for_a_pep604_optional_field():
    # ``X | None`` evaluates to ``types.UnionType``, not ``typing.Union``; the
    # saver's Optional check must accept both spellings (open item 41). The
    # string annotation is what ``from __future__ import annotations`` leaves
    # on every config module; the saver evals it against the module.
    @dataclass
    class _Opt604:
        table: "dict | None" = None

    assert to_tree(_Opt604(), where="env")["table"] is None


@dataclass
class _DeclaresOptional:
    # A string annotation, as ``from __future__ import annotations`` leaves on
    # every config module; ``Optional`` is imported HERE, where the field is
    # declared.
    value: "Optional[int]" = None


def test_an_inherited_field_resolves_in_its_declaring_module():
    # A subclass defined in a module that does not import Optional (the real
    # case: MaskedMimicControlConfig inheriting MimicControlConfig.future_steps,
    # whose Union / Tuple are imported in mimic_control.py only). The saver must
    # evaluate the inherited field against the declaring class's module, as
    # ``typing.get_type_hints`` does, or every such save refuses the field.
    fresh = types.ModuleType("protomotions_test_fresh_module")
    sys.modules[fresh.__name__] = fresh
    try:

        @dataclass
        class _Child(_DeclaresOptional):
            extra: int = 1

        _Child.__module__ = fresh.__name__
        assert to_tree(_Child(), where="env")["value"] is None
        assert to_tree(_Child(value=3), where="env")["value"] == 3
    finally:
        del sys.modules[fresh.__name__]


def test_unknown_field_refused_on_load():
    from protomotions.components.motion_lib import MotionLibConfig

    tree = {"$type": _qual(MotionLibConfig), "motion_file": "m", "nope": 1}
    with assert_refuses("has no field 'nope'"):
        from_tree(tree, None, where="x:motion_lib")


def test_missing_required_field_surfaces_the_constructor_error():
    tree = {"$type": _qual(_Req)}
    with assert_refuses("_Req"):
        from_tree(tree, None, where="x:obj")


def test_a_constructor_valueerror_keeps_the_bundle_prefix():
    # A __post_init__ that refuses its inputs raises ValueError, not TypeError; it
    # is still wrapped with the "bundle <where>" prefix (like a missing argument),
    # so the failure names the field, not just the bare constructor message.
    tree = {"$type": _qual(_RaisesInPostInit), "x": -1}
    with assert_refuses("bundle x:robot.*_RaisesInPostInit") as exc:
        from_tree(tree, None, where="x:robot")
    assert "regenerate" in str(exc.value)


def test_the_bundle_prefix_is_not_doubled(tmp_path):
    # load_experiment_configs passes the location as `where`, and the refusal
    # messages add the "bundle " prefix themselves, so a load-time refusal must
    # read "bundle <path>", not "bundle bundle <path>".
    path = tmp_path / "resolved_configs.json"
    configs = {root: {} for root in ExperimentConfigs.ROOTS}
    configs["robot"] = {"$type": _qual(_Req), "nope": 1}  # robot is rebuilt first
    path.write_text(
        json.dumps(
            {"format_version": FORMAT_VERSION, "protomotions": "t", "configs": configs}
        )
    )
    with pytest.raises(ValueError) as exc:
        load_experiment_configs(path)
    msg = str(exc.value)
    assert "bundle bundle" not in msg
    assert f"bundle {path}:robot" in msg


def test_a_non_dataclass_type_name_refused_on_load():
    tree = {"$type": "builtins.list"}
    with assert_refuses("not a dataclass"):
        from_tree(tree, None, where="x:obj")


# --------------------------------------------------------------------------- #
# Version header.
# --------------------------------------------------------------------------- #
def test_a_stale_format_version_refused_on_load(tmp_path):
    path = tmp_path / "resolved_configs.json"
    path.write_text(
        json.dumps(
            {"format_version": FORMAT_VERSION + 99, "protomotions": "x", "configs": {}}
        )
    )
    with assert_refuses(f"format_version {FORMAT_VERSION + 99}"):
        load_experiment_configs(path)


def test_a_versioned_bundle_without_configs_refused_on_load(tmp_path):
    path = tmp_path / "resolved_configs.json"
    path.write_text(json.dumps({"format_version": FORMAT_VERSION, "protomotions": "x"}))
    with assert_refuses("not an ExperimentConfigs bundle"):
        load_experiment_configs(path)


# --------------------------------------------------------------------------- #
# Components: kernel, bindings, params, and the kernel-object protocol.
# --------------------------------------------------------------------------- #
def test_unimportable_kernel_refused_on_load():
    tree = {
        "$kernel": "protomotions.does_not_exist.kernel_fn",
        "bindings": {},
        "params": {},
    }
    with assert_refuses("not importable or not a kernel"):
        from_tree(tree, Any, where="x:obs")


def test_a_truncated_component_subtree_refused_on_load():
    # The writer always emits "bindings" and "params" (even when empty), and
    # "init" for a kernel object. A subtree missing one is truncated or
    # hand-edited; it is refused here, naming the key, rather than defaulting
    # the absent key to {} and loading an unbound component that fails much
    # later at build with a confusing missing-binding message (B4's class of
    # versioned-but-malformed finding).
    valid = {"$kernel": _qual(_plain_fn), "bindings": {}, "params": {}}
    assert from_tree(valid, Any, where="x:obs") is not None

    for key in ("bindings", "params"):
        truncated = {k: v for k, v in valid.items() if k != key}
        with assert_refuses(f"missing.*{key}.*regenerate"):
            from_tree(truncated, Any, where="x:obs")

    # A kernel object also needs its "init" block. The missing-key check runs
    # before the kernel object is imported, so a stub name never loads.
    obj_missing_init = {"$kernel_object": "x.y.Z", "bindings": {}, "params": {}}
    with assert_refuses("missing.*init.*regenerate"):
        from_tree(obj_missing_init, Any, where="x:obs")

    # A present-but-non-object value is malformed too, not an empty default.
    non_dict = {"$kernel": _qual(_plain_fn), "bindings": "oops", "params": {}}
    with assert_refuses("bindings.*regenerate"):
        from_tree(non_dict, Any, where="x:obs")


def test_an_unknown_component_key_refused_on_load():
    # A key the writer never emits -- a misspelled "meta", a stray field -- is
    # refused naming it, not loaded. A misspelled "meta" would otherwise load
    # meta=None and fail later at build; a misspelled "required_keys" would drop
    # the check silently.
    tree = {"$kernel": _qual(_plain_fn), "bindings": {}, "params": {}, "metaa": {}}
    with assert_refuses("unknown key.*metaa.*regenerate"):
        from_tree(tree, Any, where="x:obs")


def test_a_malformed_optional_component_key_refused_on_load():
    base = {"$kernel": _qual(_plain_fn), "bindings": {}, "params": {}}
    # required_keys must be an object of string lists: a stray string value would
    # otherwise be split into characters by list(v).
    with assert_refuses("required_keys.*string lists.*regenerate"):
        from_tree({**base, "required_keys": {"bodies": "abc"}}, Any, where="x:obs")
    # meta must be a tagged object, not a scalar that would load as a bogus meta.
    with assert_refuses("meta.*must be an object.*regenerate"):
        from_tree({**base, "meta": "oops"}, Any, where="x:obs")
    # A well-formed required_keys (an object of string lists) passes this check --
    # an empty one is the common case and loads unchanged.
    assert from_tree({**base, "required_keys": {}}, Any, where="x:o") is not None


def test_a_from_init_failure_names_the_bundle_path():
    # A kernel object whose saved constants drifted raises from from_init; the
    # loader now prepends the "bundle <where>" prefix so the failure names the
    # component, not just the class, keeping from_init's own regenerate command.
    tree = {
        "$kernel_object": _qual(_KernelObjectThatRefusesInit),
        "init": {},
        "bindings": {},
        "params": {},
    }
    with assert_refuses("bundle obs.fk:.*regenerate"):
        from_tree(tree, Any, where="obs.fk", robot=None)


def test_a_kernel_object_without_from_init_is_refused_by_name():
    # A bare {"$kernel_object", "init"} naming an importable class that is not a
    # saveable kernel object (no from_init) is refused by name, not a raw
    # AttributeError on the missing classmethod.
    tree = {"$kernel_object": _qual(_OpaqueKernel), "init": {}}
    with assert_refuses("has no from_init.*not a saveable"):
        from_tree(tree, Any, where="obs.hp.params.fk", robot=None)


def test_a_bare_kernel_object_without_init_is_refused():
    # The bare-object dispatch needs init; a {"$kernel_object"} alone (no init and
    # no bindings/params to mark it a component) cannot be rebuilt and is refused
    # naming the bundle path.
    with assert_refuses("bare kernel object needs init"):
        from_tree(
            {"$kernel_object": _qual(_OpaqueKernel)}, Any, where="obs.hp.params.fk"
        )


def test_a_bare_kernel_object_with_an_unknown_key_is_refused():
    # The bare form's grammar is the tag plus init; an extra key is refused (not
    # silently dropped, the looseness #247 removed from the component form).
    tree = {"$kernel_object": _qual(_OpaqueKernel), "init": {}, "bogus": 1}
    with assert_refuses("unknown key.*bogus.*regenerate"):
        from_tree(tree, Any, where="obs.hp.params.fk", robot=None)


def test_a_bare_kernel_object_with_a_non_object_init_is_refused():
    # A list/str/null init would otherwise surface as "argument after ** must be a
    # mapping"; refuse it naming the bundle path.
    tree = {"$kernel_object": _qual(_OpaqueKernel), "init": [1, 2]}
    with assert_refuses("init must be an object.*regenerate"):
        from_tree(tree, Any, where="obs.hp.params.fk", robot=None)


def test_a_from_init_error_gets_exactly_one_regenerate_sentence():
    # A kernel object whose from_init message lacks the regenerate command gains it;
    # one whose message already carries it is not doubled. The suffix is conditional
    # on the message, shared by the kernel slot and a bare param.
    bare = {"$kernel_object": _qual(_KernelObjectWithBareInitError), "init": {}}
    with assert_refuses("unexpected keyword argument") as exc:
        from_tree(bare, Any, where="obs.hp.params.fk", robot=None)
    assert "regenerate" in str(exc.value)
    assert str(exc.value).count("regenerate") == 1

    already = {"$kernel_object": _qual(_KernelObjectThatRefusesInit), "init": {}}
    with assert_refuses("regenerate") as exc:
        from_tree(already, Any, where="obs.hp.params.fk", robot=None)
    assert str(exc.value).count("regenerate") == 1


def test_a_component_kernel_that_is_a_callable_object_refused_at_save():
    comp = MdpComponent(_OpaqueKernel(), {"dof_pos": EnvContext.current.dof_pos}, {})
    with assert_refuses("neither a module-level function nor a saveable kernel object"):
        to_tree(comp, where="obs.bad")


def test_a_component_kernel_that_is_a_callable_dataclass_refused_at_save():
    # The _component_tree guard's reason to exist: a callable DATACLASS kernel IS a
    # dataclass, so without the guard to_tree would save it as a $type config that
    # reloads as a dataclass used as a kernel. Refuse it at save instead.
    comp = MdpComponent(
        _CallableDataclassKernel(), {"dof_pos": EnvContext.current.dof_pos}, {}
    )
    with assert_refuses("neither a module-level function nor a saveable kernel object"):
        to_tree(comp, where="obs.bad")


def test_a_dataclass_kernel_object_saves_as_a_kernel_object_not_a_type():
    # A @dataclass kernel object carrying export_init/from_init must save as
    # {$kernel_object, init} (the kernel-object branch runs before the dataclass
    # branch), not as a $type config that reloads as a dataclass used as a kernel and
    # dies at the binding check with a misleading "has no field 'bindings'" message.
    comp = MdpComponent(
        _DataclassKernelObject(scale=2.0), {"dof_pos": EnvContext.current.dof_pos}, {}
    )
    tree = to_tree(comp, where="obs.k")
    assert tree["$kernel_object"] == _qual(_DataclassKernelObject)
    assert "$type" not in tree
    assert tree["init"] == {"scale": 2.0}
    back = from_tree(tree, Any, where="obs.k", robot=None)
    assert isinstance(back, MdpComponent)
    assert isinstance(back.get_compute_func(), _DataclassKernelObject)
    assert back.get_compute_func().scale == 2.0


def test_baked_anchor_component_round_trips_through_from_init():
    # The baked-FK kernel is a callable object; it persists via its export_init /
    # from_init protocol as {"$kernel_object", "init"}, with the kinematic tree
    # taken from the rebuilt robot, and computes the same observation after.
    pytest.importorskip("dm_control")
    from protomotions.envs.context_sources import sensed
    from protomotions.robot_configs.factory import robot_config
    from protomotions.utils.baked_fk import BakedAnchorFK

    robot = robot_config("g1")
    kinematic_info = robot.kinematic_info
    bodies = torch.tensor([0, 1, 2], dtype=torch.long)
    # The body-fixed anchor frame: passing current_anchor_rot=None unbinds the
    # heading tilt, so the kernel reads the joint encoders alone.
    body_fixed = MdpComponent.from_kernel(
        BakedAnchorFK(kinematic_info, robot.anchor_body_index, bodies),
        current_anchor_rot=None,
    )
    comp = sensed({"fk": body_fixed}, "noisy")["fk"]

    tree = to_tree(comp, where="obs.fk_max_coords_obs")
    assert tree["$kernel_object"].endswith("BakedAnchorFK")
    assert tree["init"] == {
        "anchor_idx": robot.anchor_body_index,
        "trackable_body_indices": [0, 1, 2],
    }

    back = from_tree(tree, Any, where="obs.fk_max_coords_obs", robot=robot)
    assert to_tree(back, where="obs.fk_max_coords_obs") == tree

    # current_anchor_rot now defaults to the heading-frame FieldPath; the raw
    # kernel's body-fixed path (the heading_frame=False intent) is reached by
    # passing None, which skips the heading tilt.
    dof_pos = torch.randn(4, kinematic_info.num_dofs)
    torch.testing.assert_close(
        comp.compute_func(dof_pos, current_anchor_rot=None),
        back.compute_func(dof_pos, current_anchor_rot=None),
    )


def test_inherited_annotation_uses_declaring_module():
    from protomotions.envs.control.masked_mimic_control import MaskedMimicControlConfig

    config = MaskedMimicControlConfig(future_steps=(1, 2, 5))
    tree = to_tree(config, where="control")
    rebuilt = from_tree(tree, Any, where="control")
    assert to_tree(rebuilt, where="control") == tree


def test_history_fk_parameter_round_trips_with_robot():
    pytest.importorskip("dm_control")
    from examples.experiments.masked_mimic.deployable_g1_observations import (
        deployable_proprioception_observations,
    )
    from protomotions.robot_configs.factory import robot_config

    robot = robot_config("g1")
    component = deployable_proprioception_observations(
        robot.kinematic_info,
        torch.tensor([0, 1, 2]),
        robot.anchor_body_index,
    )["historical_proprioception"]
    tree = to_tree(component, where="obs.history")
    assert tree["params"]["fk"]["$kernel_object"].endswith("BakedAnchorFK")
    rebuilt = from_tree(tree, Any, where="obs.history", robot=robot)
    assert to_tree(rebuilt, where="obs.history") == tree
    dof_pos = torch.randn(4, robot.kinematic_info.num_dofs)
    torch.testing.assert_close(
        component.get_params()["fk"](dof_pos, current_anchor_rot=None),
        rebuilt.get_params()["fk"](dof_pos, current_anchor_rot=None),
    )


def test_baked_target_component_round_trips_through_from_init():
    # A reduced-coords deploy recipe builds its reference-FK observation as
    # ``MdpComponent.from_kernel(BakedTargetFK(...))`` with the heading-local
    # odometer offset bound to the sensor namespace. That callable persists as
    # {"$kernel_object", "init"} exactly like the anchor kernel, and the odom
    # bindings plus future_steps travel in the component's canonicalized form,
    # so the whole component round-trips tree-equal.
    pytest.importorskip("dm_control")
    from protomotions.robot_configs.factory import robot_config
    from protomotions.utils.baked_fk import BakedTargetFK

    robot = robot_config("g1")
    bodies = torch.tensor([0, 1, 2], dtype=torch.long)
    baked = BakedTargetFK(robot.kinematic_info, robot.anchor_body_index, bodies)
    comp = MdpComponent.from_kernel(
        baked,
        odom_disp_start=EnvContext.noisy.odom_disp_start,
        odom_start_xy=EnvContext.noisy.odom_start_xy,
        odom_start_heading_inv=EnvContext.noisy.odom_start_heading_inv,
        future_steps=(1, 2),
    )

    tree = to_tree(comp, where="obs.fk_target_poses")
    assert tree["$kernel_object"].endswith("BakedTargetFK")
    assert tree["init"] == {
        "anchor_idx": robot.anchor_body_index,
        "trackable_body_indices": [0, 1, 2],
    }
    # The required odom inputs are context bindings, not pickled tensors.
    assert tree["bindings"]["odom_disp_start"] == "noisy.odom_disp_start"
    assert tree["bindings"]["odom_start_heading_inv"] == "noisy.odom_start_heading_inv"

    back = from_tree(tree, Any, where="obs.fk_target_poses", robot=robot)
    assert to_tree(back, where="obs.fk_target_poses") == tree


def test_a_baked_kernel_as_a_static_param_round_trips_with_the_robot():
    # Metatron passes one BakedAnchorFK as both the kernel of its max-coords obs
    # (covered above) AND the static ``fk=`` param of its historical-proprioception
    # obs -- the position save_experiment_configs refused before item 57. In a
    # param slot the object has no bindings of its own, so it persists as a bare
    # {"$kernel_object", "init"} nested under the component's params and rebuilds
    # through from_init with the robot threaded to the param (without that
    # threading the param's from_init would receive robot=None and crash).
    pytest.importorskip("dm_control")
    from protomotions.robot_configs.factory import robot_config
    from protomotions.utils.baked_fk import BakedAnchorFK

    robot = robot_config("g1")
    bodies = torch.tensor([0, 1, 2], dtype=torch.long)
    baked = BakedAnchorFK(robot.kinematic_info, robot.anchor_body_index, bodies)
    comp = MdpComponent(
        _kernel_with_fk, {"x": EnvContext.current.dof_pos}, {"fk": baked}
    )

    tree = to_tree(comp, where="obs.historical_proprioception")
    fk_param = tree["params"]["fk"]
    assert set(fk_param) == {"$kernel_object", "init"}  # bare: no bindings/params
    assert fk_param["$kernel_object"].endswith("BakedAnchorFK")
    assert fk_param["init"] == {
        "anchor_idx": robot.anchor_body_index,
        "trackable_body_indices": [0, 1, 2],
    }

    back = from_tree(tree, Any, where="obs.historical_proprioception", robot=robot)
    assert isinstance(back.static_params["fk"], BakedAnchorFK)
    assert to_tree(back, where="obs.historical_proprioception") == tree


def test_the_metatron_recipe_bundles_its_baked_fk_param(monkeypatch):
    # The real recipe that motivated item 57: metatron passes one BakedAnchorFK as
    # the kernel of ``fk_max_coords_obs`` AND as the static ``fk=`` param of
    # ``historical_proprioception``. Before item 57 save_experiment_configs refused
    # the param position ("BakedAnchorFK is not a module-level function") and
    # train_agent.py died at the bundle save for this recipe. Build the student
    # config offline with a stubbed expert, then round-trip every observation
    # component through the bundle grammar with the robot threaded to the param.
    pytest.importorskip("dm_control")
    from copy import deepcopy
    from types import SimpleNamespace

    from examples.experiments.metatron import g1_student as recipe
    from examples.experiments.metatron import g1_teacher as tracker
    from protomotions.robot_configs.g1 import G1RobotConfig
    from protomotions.simulator.factory import simulator_config

    args = SimpleNamespace(
        simulator="isaaclab",
        motion_file="motions.pt",
        scenes_file=None,
        batch_size=4,
        training_max_steps=64,
        expert_model_path="teacher/last.ckpt",
    )
    robot = G1RobotConfig()
    sim = simulator_config("isaaclab", robot, True, 4, "test")
    tracker.configure_robot_and_simulator(robot, sim, args)
    teacher_env = tracker.env_config(robot, args)
    teacher = SimpleNamespace(
        robot=deepcopy(robot),
        simulator=deepcopy(sim),
        env=teacher_env,
        agent=tracker.agent_config(robot, teacher_env, args),
        terrain=tracker.terrain_config(args),
    )
    monkeypatch.setattr(
        recipe, "load_experiment_configs_from_checkpoint", lambda _: teacher
    )
    recipe.configure_robot_and_simulator(robot, sim, args)
    env = recipe.env_config(robot, args)

    obs = env.observation_components
    fk = obs["historical_proprioception"].static_params["fk"]
    assert hasattr(fk, "export_init")  # the baked param position item 57 fixes
    for name, comp in obs.items():
        tree = to_tree(comp, where=f"obs.{name}")
        back = from_tree(tree, Any, where=f"obs.{name}", robot=robot)
        assert to_tree(back, where=f"obs.{name}") == tree


# --------------------------------------------------------------------------- #
# Round-trip and two-way discovery over every offline-buildable example recipe.
# --------------------------------------------------------------------------- #
_SMPL_MOTION = "examples/data/smpl_humanoid_sit_armchair.motion"
_G1_MOTION = "examples/data/g1_crouch_to_run.motion"

# The offline-buildable example recipes (CPU / MuJoCo). The GPC prior and PEFT
# recipes build a tracker from a checkpoint, and metatron.g1_student needs an
# expert checkpoint, so none of those build without a shipped model; the baked
# kernel object is covered by the kernel-object unit tests above (a synthetic
# BakedAnchorFK in the kernel and static-param slots, plus the metatron recipe
# round-trip with its expert stubbed). mlp_bm_l2c2 is the one BM recipe that
# builds offline, and its one-env MuJoCo tree emits no kernel object.
# masked_mimic/transformer.py
# builds without its optional expert and is the one recipe here whose control
# config inherits a field (MaskedMimicControlConfig.future_steps): its bundle
# could not be loaded at all until the resolvers read an inherited annotation
# in its declaring module.
RECIPES = [
    ("examples/experiments/add/mlp.py", "smpl", _SMPL_MOTION),
    ("examples/experiments/amp/mlp.py", "smpl", _SMPL_MOTION),
    ("examples/experiments/ase/mlp.py", "smpl", _SMPL_MOTION),
    ("examples/experiments/mimic/mlp.py", "smpl", _SMPL_MOTION),
    ("examples/experiments/mimic/fsq.py", "smpl", _SMPL_MOTION),
    ("examples/experiments/metatron/g1_teacher.py", "smpl", _SMPL_MOTION),
    ("examples/experiments/mimic/mlp_complex_terrain.py", "smpl", _SMPL_MOTION),
    ("examples/experiments/mimic/mlp_bm_l2c2.py", "g1", _G1_MOTION),
    ("examples/experiments/masked_mimic/transformer.py", "smpl", _SMPL_MOTION),
    ("examples/experiments/steering/mlp.py", "smpl", _SMPL_MOTION),
    ("examples/experiments/path_follower/mlp.py", "smpl", _SMPL_MOTION),
]


@functools.lru_cache(maxsize=None)
def _build(experiment: str, robot: str, motion: str) -> ExperimentConfigs:
    from pathlib import Path

    from protomotions.utils.config_builder import build_experiment

    repo_root = Path(__file__).resolve().parents[2]
    argv = [
        "--robot-name",
        robot,
        "--simulator",
        "mujoco",
        "--num-envs",
        "1",
        "--batch-size",
        "16",
        "--motion-file",
        motion,
        "--experiment-path",
        str(repo_root / experiment),
        "--experiment-name",
        "schema",
        "--training-max-steps",
        "10",
        "--headless",
        "true",
    ]
    return build_experiment(argv)


def _tagged_names(tree) -> list:
    """Every (kind, importable name) the emit produced, anywhere in the tree."""
    out = []
    if isinstance(tree, dict):
        for kind in ("$type", "$enum", "$function", "$kernel", "$kernel_object"):
            name = tree.get(kind)
            if isinstance(name, str):
                out.append((kind, name))
        for value in tree.values():
            out += _tagged_names(value)
    elif isinstance(tree, list):
        for value in tree:
            out += _tagged_names(value)
    return out


@pytest.mark.parametrize(
    "experiment,robot,motion", RECIPES, ids=[r[0].split("/")[-1][:-3] for r in RECIPES]
)
def test_recipe_bundle_round_trips(experiment, robot, motion):
    """Emit, load, emit again: the second tree equals the first for every root."""
    configs = _build(experiment, robot, motion)
    for root in ExperimentConfigs.ROOTS:
        tree = to_tree(getattr(configs, root), where=root)
        rebuilt = from_tree(tree, None, where=root, robot=configs.robot)
        assert to_tree(rebuilt, where=root) == tree, f"root {root!r} did not round-trip"


@pytest.mark.parametrize(
    "experiment,robot,motion", RECIPES, ids=[r[0].split("/")[-1][:-3] for r in RECIPES]
)
def test_recipe_bundle_names_all_resolve(experiment, robot, motion):
    """Every class and function the emit named resolves back to that same object."""
    configs = _build(experiment, robot, motion)
    for root in ExperimentConfigs.ROOTS:
        tree = to_tree(getattr(configs, root), where=root)
        for kind, name in _tagged_names(tree):
            resolved = get_class(name)
            if kind in ("$type", "$enum"):
                assert is_dataclass(resolved) or issubclass(
                    resolved, enum.Enum
                ), f"{name} ({kind}) is not a dataclass or enum"


# --------------------------------------------------------------------------- #
# A full MuJoCo rollout is bit-identical after the bundle is saved and reloaded.
# --------------------------------------------------------------------------- #
@pytest.mark.parity_slow
@pytest.mark.skipif(
    os.environ.get("PROTOMOTIONS_PARITY_SLOW") != "1",
    reason="MuJoCo scenario builds; set PROTOMOTIONS_PARITY_SLOW=1",
)
def test_smpl_mimic_rollout_survives_a_save_and_reload(monkeypatch, tmp_path):
    """The reloaded bundle drives the same rollout as the in-process configs.

    Records smpl_mimic twice through the parity harness -- once from the configs
    the build produces, once from those same configs saved to JSON and rebuilt
    through the real constructors -- and asserts the two traces are identical.
    """
    import protomotions.utils.config_builder as config_builder
    from protomotions.tests.parity import SCENARIOS, compare, record
    from protomotions.tests.parity.compare import format_result
    from protomotions.utils.experiment_configs import (
        load_experiment_configs,
        save_experiment_configs,
    )

    scenario = SCENARIOS["smpl_mimic"]
    in_process = record(scenario)

    real_build_experiment = config_builder.build_experiment

    def _build_then_reload(argv, **kwargs):
        configs = real_build_experiment(argv, **kwargs)
        return load_experiment_configs(save_experiment_configs(configs, tmp_path))

    monkeypatch.setattr(config_builder, "build_experiment", _build_then_reload)
    reloaded = record(scenario)

    result = compare(in_process, reloaded, atol=0.0)
    assert result.identical, format_result(result)


# --------------------------------------------------------------------------- #
# _target_ validation: a renamed class fails at load, with the import-order
# asymmetry for simulator backends.
# --------------------------------------------------------------------------- #
@dataclass
class _ConfigWithBogusTarget:
    _target_: str = "protomotions.utils.config_schema.NoSuchClass"
    value: int = 1


@dataclass
class _ConfigWithBogusSimTarget:
    _target_: str = "protomotions.simulator.isaacgym.simulator.NoSuchSim"
    value: int = 1


def test_validate_target_ignores_none_and_empty():
    # A config without a _target_ (getattr returns None) or with an empty one is a
    # no-op; _validate_target only checks a real path.
    _validate_target(None, "x")
    _validate_target("", "x")


def test_validate_target_refuses_a_non_path_string():
    with assert_refuses("not a module.Class path"):
        _validate_target("NotAPath", "x:agent")


def test_validate_target_refuses_a_renamed_class():
    # An importable module missing the named class -> AttributeError -> refused.
    with assert_refuses("class is gone.*regenerate"):
        _validate_target("protomotions.utils.config_schema.NoSuchClass", "x:agent")


def test_validate_target_refuses_a_renamed_protomotions_module():
    # A protomotions module that is gone -> ModuleNotFoundError under protomotions
    # -> refused (a bundle defect, not an install matter).
    with assert_refuses("renamed or moved.*regenerate"):
        _validate_target("protomotions.gone_module_xyz.Thing", "x:agent")


def test_validate_target_skips_an_uninstalled_third_party_module(caplog):
    # A non-protomotions module that is not installed is logged and skipped -- an
    # install matter (e.g. an uninstalled dependency a bundle still names), not a
    # bundle defect; no exception.
    with caplog.at_level("INFO", logger="protomotions.utils.config_schema"):
        _validate_target("totally_absent_pkg_zzz.Thing", "x:agent")
    assert "not checkable" in caplog.text


def test_validate_target_skips_simulator_targets_unconditionally():
    # A simulator target is import-order-sensitive (isaacgym/isaaclab before torch),
    # so it is never imported here even to check -- even a bogus class under
    # protomotions.simulator.* does not raise. The backend built is checked at the
    # swap and at build.
    _validate_target("protomotions.simulator.isaacgym.simulator.NoSuchSim", "x:sim")
    _validate_target("protomotions.simulator.mujoco.simulator.MujocoSimulator", "x:sim")


def test_from_tree_refuses_a_dataclass_with_a_renamed_target():
    # End to end: a config whose _target_ class is gone is refused at load through
    # the $type branch, naming the bundle path.
    tree = to_tree(_ConfigWithBogusTarget(), where="x:agent")
    with assert_refuses("class is gone.*regenerate"):
        from_tree(tree, Any, where="x:agent")


def test_from_tree_loads_a_dataclass_with_a_bogus_simulator_target():
    # End to end: a stored simulator _target_ is skipped at load (import-order), so
    # even a bogus backend class round-trips -- the sim2sim case where the stored
    # backend is never instantiated.
    tree = to_tree(_ConfigWithBogusSimTarget(), where="x:sim")
    back = from_tree(tree, Any, where="x:sim")
    assert back._target_ == "protomotions.simulator.isaacgym.simulator.NoSuchSim"
