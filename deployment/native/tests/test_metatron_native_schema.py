# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the native fixed-shape binding table aligned with the pinned bundle."""

import json
from pathlib import Path
import re

import pytest

from examples.metatron_v1_contract import BOOL_INPUTS, CONTRACT, INPUT_SHAPES


def test_native_input_table_matches_python_contract_without_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    header = (
        root / "include/protomotions/deployment/metatron_v1_contract.hpp"
    ).read_text()
    assert f'"{CONTRACT}"' in header
    table = header.split("kInputs = {{", 1)[1].split("}};", 1)[0]
    rows = re.findall(r'\{"([^\"]+)", \{([^}]+)\}, (\d+), (true|false)\}', table)
    assert len(rows) == len(INPUT_SHAPES) == 39
    for (name, dims, rank, boolean), (key, shape) in zip(
        rows, INPUT_SHAPES.items(), strict=True
    ):
        assert name == key.replace(".", "_")
        assert [int(value.strip()) for value in dims.split(",")] == [1, *shape]
        assert int(rank) == len(shape) + 1
        assert (boolean == "true") == (key in BOOL_INPUTS)


def test_native_input_table_matches_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts/metatron-ep6410/manifest.json"
    if not artifact.exists():
        pytest.skip("Requires a locally supplied Metatron v1 manifest")
    manifest = json.loads(artifact.read_text())
    header = (
        root / "include/protomotions/deployment/metatron_v1_contract.hpp"
    ).read_text()
    table = header.split("kInputs = {{", 1)[1].split("}};", 1)[0]
    rows = re.findall(r'\{"([^\"]+)", \{([^}]+)\}, (\d+), (true|false)\}', table)
    assert len(rows) == len(manifest["input_names"]) == 39
    for (name, dims, rank, boolean), expected_name, shape, dtype in zip(
        rows,
        manifest["input_names"],
        manifest["input_shapes"],
        manifest["input_dtypes"],
        strict=True,
    ):
        native_shape = [int(value.strip()) for value in dims.split(",")]
        assert name == expected_name
        assert native_shape == [1, *shape]
        assert int(rank) == len(native_shape)
        assert (boolean == "true") == (dtype == "bool")


def test_action_offset_is_the_g1_standing_pose() -> None:
    """The graph output already applies this offset; native must not reapply it."""
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts/metatron-ep6410/manifest.json"
    if not artifact.exists():
        pytest.skip("Requires a locally supplied Metatron v1 manifest")
    manifest = json.loads(artifact.read_text())
    defaults = {
        "left_hip_pitch_joint": -0.312,
        "right_hip_pitch_joint": -0.312,
        "left_knee_joint": 0.669,
        "right_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "right_ankle_pitch_joint": -0.363,
        "left_elbow_joint": 0.6,
        "right_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }
    joints = manifest["robot_contract"]["joint_names"]
    offsets = manifest["action_map"]["offset"]
    scales = manifest["action_map"]["scale"]
    assert manifest["previous_action_domain"] == "raw_policy_action"
    assert manifest["pd_target_ema_alpha"] == 1.0
    assert len(joints) == len(offsets) == len(scales) == 29
    assert manifest["output_names"] == ["action", "stiffness", "damping", "raw_action"]
    for name, offset, scale in zip(joints, offsets, scales, strict=True):
        assert abs(offset - defaults.get(name, 0.0)) < 1e-6
        assert scale > 0
