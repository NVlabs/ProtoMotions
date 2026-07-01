# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for robot config parsing and factory helpers."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

import protomotions.components.pose_lib as pose_lib
import protomotions.robot_configs.base as robot_base
from protomotions.robot_configs.base import (
    ControlConfig,
    ControlType,
    RobotAssetConfig,
    RobotConfig,
    abstract_names_to_body_names,
)
from protomotions.robot_configs.factory import robot_config


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
    monkeypatch.setattr(pose_lib, "extract_kinematic_info", lambda mjcf_path: kinematic_info)
    monkeypatch.setattr(
        pose_lib,
        "extract_control_info",
        lambda mjcf_path, override_control_info=None: {"path": mjcf_path, "override": override_control_info},
    )
    return kinematic_info


def _robot(monkeypatch, **kwargs):
    _patch_pose_extractors(monkeypatch)
    params = {
        "asset": RobotAssetConfig(asset_root="/assets", asset_file_name="robot.xml"),
        "common_naming_to_robot_body_names": _names(),
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


def test_control_config_converts_dict_overrides(monkeypatch):
    class _ControlInfo:
        @staticmethod
        def from_dict(data):
            return SimpleNamespace(converted=data)

    monkeypatch.setattr(robot_base, "ControlInfo", _ControlInfo)

    config = ControlConfig(
        override_control_info={
            "joint": {"stiffness": 10.0},
            "ready": SimpleNamespace(value=1),
        }
    )

    assert config.override_control_info["joint"].converted == {"stiffness": 10.0}
    assert config.override_control_info["ready"].value == 1


def test_robot_config_post_init_resolves_anchor_defaults_and_abstract_body_names(monkeypatch):
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


def test_robot_config_update_fields_reprocesses_body_name_aliases(monkeypatch):
    config = _robot(monkeypatch, contact_bodies=None, trackable_bodies_subset=["root"])

    config.update_fields(contact_bodies=["root", "all_right_foot_bodies"])

    assert config.contact_bodies == ["root", "right_foot"]
    with pytest.raises(ValueError, match="has no field"):
        config.update_fields(missing=True)


def test_abstract_names_to_body_names_handles_none_all_lists_and_literals(monkeypatch):
    config = _robot(monkeypatch)

    assert abstract_names_to_body_names(None, config) is None
    assert abstract_names_to_body_names("all", config) == config.kinematic_info.body_names
    assert abstract_names_to_body_names("root", config) == ["root"]
    assert abstract_names_to_body_names("all_left_hand_bodies", config) == ["left_hand"]
    assert abstract_names_to_body_names(["root", "custom"], config) == ["root", "custom"]


def test_robot_config_factory_dispatches_all_robot_names_and_applies_updates(monkeypatch):
    class _FactoryConfig:
        def __init__(self):
            self.updates = {}

        def update_fields(self, **updates):
            self.updates.update(updates)

    for module_name, class_name in [
        ("protomotions.robot_configs.smpl", "SmplRobotConfig"),
        ("protomotions.robot_configs.smplx", "SMPLXRobotConfig"),
        ("protomotions.robot_configs.amp", "AMPRobotConfig"),
        ("protomotions.robot_configs.g1", "G1RobotConfig"),
        ("protomotions.robot_configs.h1_2", "H1_2RobotConfig"),
        ("protomotions.robot_configs.rigv1", "Rigv1RobotConfig"),
        ("protomotions.robot_configs.soma23", "Soma23RobotConfig"),
    ]:
        module = types.ModuleType(module_name)
        setattr(module, class_name, _FactoryConfig)
        monkeypatch.setitem(sys.modules, module_name, module)

    for name in ["smpl", "smplx", "amp", "g1", "h1_2", "rigv1", "soma23"]:
        config = robot_config(name, trackable_bodies_subset=["root"])
        assert isinstance(config, _FactoryConfig)
        assert config.updates == {"trackable_bodies_subset": ["root"]}

    with pytest.raises(ValueError, match="Invalid robot name"):
        robot_config("unknown")
