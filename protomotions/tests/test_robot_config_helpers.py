# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for robot config parsing and factory helpers."""

from types import SimpleNamespace

import pytest
import torch

import protomotions.components.pose_lib as pose_lib
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
    abstract_names_to_body_names,
)
from protomotions.robot_configs.factory import robot_config
from protomotions.utils.config_schema import from_tree, to_tree


def _names():
    return {
        "all_left_foot_bodies": ["left_foot"],
        "all_right_foot_bodies": ["right_foot"],
        "all_left_hand_bodies": ["left_hand"],
        "all_right_hand_bodies": ["right_hand"],
        "head_body_name": ["head"],
        "torso_body_name": ["torso"],
    }


def _patch_pose_extractors(monkeypatch):
    kinematic_info = SimpleNamespace(
        body_names=["root", "torso", "head", "left_foot", "right_foot"],
        dof_names=["left_knee_joint", "right_hip_joint", "spine_joint"],
        num_dofs=3,
    )
    monkeypatch.setattr(
        pose_lib, "extract_kinematic_info", lambda mjcf_path: kinematic_info
    )
    monkeypatch.setattr(
        pose_lib,
        "extract_control_info",
        lambda mjcf_path, override_control_info=None, asset_default_joints=(): {
            "path": mjcf_path,
            "override": override_control_info,
        },
    )
    return kinematic_info


def _robot(monkeypatch, **kwargs):
    _patch_pose_extractors(monkeypatch)
    params = {
        "asset": RobotAssetConfig(asset_root="/assets", asset_file_name="robot.xml"),
        "common_naming_to_robot_body_names": _names(),
        "semantic_forward_axis_xy": [1.0, 0.0],
        "control_hz": 50,
    }
    params.update(kwargs)
    return RobotConfig(**params)


def test_control_type_and_asset_validation():
    assert ControlType.from_str("TORQUE") is ControlType.TORQUE
    assert ControlType.from_str("built_in_pd") is ControlType.BUILT_IN_PD
    with pytest.raises(ValueError, match="not a valid ControlType"):
        ControlType.from_str("velocity")

    with pytest.raises(ValueError, match="must be a valid path"):
        RobotAssetConfig(asset_file_name="robot.urdf")


def test_control_config_refuses_dict_overrides():
    """override_control_info values must be ControlInfo; a dict is refused (no from_dict)."""
    from protomotions.components.pose_lib import ControlInfo

    config = ControlConfig(override_control_info={"joint": ControlInfo(stiffness=10.0)})
    assert config.override_control_info["joint"].stiffness == 10.0

    with pytest.raises(TypeError, match="must be ControlInfo, not dict"):
        ControlConfig(override_control_info={"joint": {"stiffness": 10.0}})


def test_robot_config_post_init_resolves_anchor_defaults_and_abstract_body_names(
    monkeypatch,
):
    config = _robot(
        monkeypatch,
        anchor_body_name="torso",
        default_dof_pos={".*_knee_joint": 0.7, "spine_.*": -0.2},
        mimic_small_marker_bodies=["root", "head_body_name"],
        contact_bodies="all_left_foot_bodies",
        trackable_bodies_subset="all",
    )

    assert config.anchor_body_index == 1
    assert torch.allclose(config.default_dof_pos, torch.tensor([0.7, 0.0, -0.2]))
    assert config.number_of_actions == 3
    assert config.mimic_small_marker_bodies == ["root", "head"]
    assert config.contact_bodies == ["left_foot"]
    assert config.trackable_bodies_subset == config.kinematic_info.body_names
    assert config.control.control_info["path"] == "/assets/robot.xml"


def test_robot_config_post_init_validates_anchor_and_default_dof_length(monkeypatch):
    with pytest.raises(ValueError, match="anchor_body_name"):
        _robot(monkeypatch, anchor_body_name="missing")

    with pytest.raises(AssertionError, match="default_dof_pos length"):
        _robot(monkeypatch, default_dof_pos=[0.0, 1.0])

    with pytest.raises(AssertionError, match="must contain"):
        _robot(monkeypatch, common_naming_to_robot_body_names={})


def test_anchor_index_is_a_derived_property_that_rederives(monkeypatch):
    config = _robot(monkeypatch, anchor_body_name="torso")
    assert config.anchor_body_index == 1
    assert config.resolved_anchor_body_name == "torso"

    # Assigning the name re-derives the index and resolved name on the next read.
    config.anchor_body_name = "head"
    assert config.anchor_body_index == 2
    assert config.resolved_anchor_body_name == "head"

    config.anchor_body_name = None
    assert config.anchor_body_index == 0
    assert config.resolved_anchor_body_name == "root"

    # An unknown name is refused when the index is read.
    config.anchor_body_name = "missing"
    with pytest.raises(ValueError, match="anchor_body_name 'missing' not found"):
        _ = config.anchor_body_index


def test_anchor_index_cannot_be_assigned_directly(monkeypatch):
    config = _robot(monkeypatch, anchor_body_name="torso")
    with pytest.raises(AttributeError):
        config.anchor_body_index = 0


def test_anchor_index_self_heals_without_a_cache(monkeypatch):
    """A config unpickled from a pre-property bundle has no cache; it re-derives."""
    config = _robot(monkeypatch, anchor_body_name="torso")
    del config._anchor_cache  # a pre-D2 pickle has no _anchor_cache attribute
    assert config.anchor_body_index == 1
    assert config.resolved_anchor_body_name == "torso"


def test_update_fields_rederives_the_anchor_and_refuses_derived_fields(monkeypatch):
    config = _robot(monkeypatch, anchor_body_name="torso")
    config.update_fields(anchor_body_name="head")
    assert config.anchor_body_index == 2
    config.update_fields(anchor_body_name=None)
    assert config.anchor_body_index == 0

    # The derived index is refused, naming the field it is derived from.
    with pytest.raises(
        ValueError, match=r"anchor_body_index is derived.*\(anchor_body_name\)"
    ):
        config.update_fields(anchor_body_index=2)

    # The asset-derived init=False fields are refused too, naming the asset.
    for key in ("kinematic_info", "number_of_actions"):
        with pytest.raises(ValueError, match=rf"{key} is derived.*\(the asset\)"):
            config.update_fields(**{key: 0})


def test_body_indices_matches_abstract_expansion_and_refuses_bad_input(monkeypatch):
    config = _robot(monkeypatch)
    body_names = config.kinematic_info.body_names
    for sel in (["root"], ["all"], ["all_left_foot_bodies"], ["torso", "head"]):
        want = tuple(
            body_names.index(b) for b in abstract_names_to_body_names(sel, config)
        )
        assert config.body_indices(*sel) == want

    with pytest.raises(
        ValueError, match="is neither a body of .* nor a semantic group"
    ):
        config.body_indices("nope")
    with pytest.raises(ValueError, match="selected twice"):
        config.body_indices("root", "root")


def test_bodies_matching_returns_regex_matches_in_body_order(monkeypatch):
    config = _robot(monkeypatch)
    assert config.bodies_matching(".*_foot") == (3, 4)
    assert config.bodies_matching("torso") == (1,)
    with pytest.raises(ValueError, match="matches no body"):
        config.bodies_matching("zzz")


def test_body_indices_match_abstract_for_every_registered_robot():
    """body_indices(...) equals indexing abstract_names_to_body_names, on real robots."""
    from protomotions.robot_configs.factory import registered_robot_names

    for name in registered_robot_names():
        config = robot_config(name)
        body_names = config.kinematic_info.body_names
        selections = [["root"]] + [
            [group] for group in config.common_naming_to_robot_body_names
        ]
        for sel in selections:
            expanded = abstract_names_to_body_names(sel, config)
            if len(expanded) != len(set(expanded)):
                continue  # body_indices refuses a duplicate selection by design
            want = tuple(body_names.index(b) for b in expanded)
            assert config.body_indices(*sel) == want, (name, sel)


def test_robot_config_requires_and_normalizes_semantic_forward_axis(monkeypatch):
    _patch_pose_extractors(monkeypatch)
    params = {
        "asset": RobotAssetConfig(asset_root="/assets", asset_file_name="robot.xml"),
        "common_naming_to_robot_body_names": _names(),
        "control_hz": 50,
    }

    with pytest.raises(ValueError, match="semantic_forward_axis_xy") as exc_info:
        RobotConfig(**params)
    assert "scripts/identify_robot_facing_axis.py" in str(exc_info.value)
    assert "--view" in str(exc_info.value)

    config = _robot(monkeypatch, semantic_forward_axis_xy=[3.0, 4.0])
    assert config.semantic_forward_axis_xy == pytest.approx((0.6, 0.8))

    with pytest.raises(ValueError, match="non-zero 2D"):
        _robot(monkeypatch, semantic_forward_axis_xy=[0.0, 0.0])


def test_a_saved_robot_subtree_without_the_forward_axis_fails_to_load(monkeypatch):
    # No runtime fallback is left for a bundle that predates the field: from_tree
    # rebuilds the dataclass through its constructor, which refuses None.
    tree = to_tree(_robot(monkeypatch), where="robot")
    del tree["semantic_forward_axis_xy"]
    with pytest.raises(ValueError, match="must be declared explicitly"):
        from_tree(tree, None, where="x:robot")


def test_robot_config_update_fields_reprocesses_body_name_aliases(monkeypatch):
    config = _robot(monkeypatch, contact_bodies=None, trackable_bodies_subset=["root"])

    config.update_fields(contact_bodies=["root", "all_right_foot_bodies"])

    assert config.contact_bodies == ["root", "right_foot"]
    with pytest.raises(ValueError, match="has no field"):
        config.update_fields(missing=True)


def test_abstract_names_to_body_names_handles_none_all_lists_and_literals(monkeypatch):
    config = _robot(monkeypatch)

    assert abstract_names_to_body_names(None, config) is None
    assert (
        abstract_names_to_body_names("all", config) == config.kinematic_info.body_names
    )
    assert abstract_names_to_body_names("root", config) == ["root"]
    assert abstract_names_to_body_names("all_left_hand_bodies", config) == ["left_hand"]
    assert abstract_names_to_body_names(["root", "custom"], config) == [
        "root",
        "custom",
    ]


def test_robot_config_builds_a_registered_robot_and_applies_updates():
    """robot_config(name, **updates) builds the robot's one RobotConfig and applies updates.

    The registry's name set, the unknown-name refusal and the one-direct-base
    rule are covered in test_robot_facts_d1.py.
    """
    config = robot_config("g1", default_root_height=2.0)
    assert type(config).__name__ == "G1RobotConfig"
    assert config.default_root_height == 2.0
