# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the Metatron v1 LEAPP CLI and native hand-off without weights."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from examples.metatron_v1_contract import BOOL_INPUTS, CONTRACT, INPUT_SHAPES


def test_metatron_bundle_cli_and_native_preparation(tmp_path: Path) -> None:
    pytest.importorskip("leapp")
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper
    from tools.prepare_metatron_v1_policy import prepare

    # A synthetic identity graph with the real 39 input signatures. This tests
    # packaging/contracts, not trained policy quality or recovery behavior.
    keys = list(INPUT_SHAPES)
    names = [key.replace(".", "_") for key in keys]
    outputs = ["action", "stiffness", "damping", "raw_action"]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["noisy_dof_pos"], [name]) for name in outputs],
        "metatron-contract-fixture",
        [
            helper.make_tensor_value_info(
                name,
                TensorProto.BOOL if key in BOOL_INPUTS else TensorProto.FLOAT,
                [1, *INPUT_SHAPES[key]],
            )
            for key, name in zip(keys, names)
        ],
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 29])
            for name in outputs
        ],
    )
    model = tmp_path / "source.onnx"
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
        ),
        model,
    )
    bodies = [f"body_{i}" for i in range(33)]
    for index, name in {
        7: "left_ankle_roll_link",
        13: "right_ankle_roll_link",
        16: "torso_link",
        24: "left_rubber_hand",
        32: "right_rubber_hand",
    }.items():
        bodies[index] = name
    joints = [f"joint_{i}" for i in range(29)]
    joints[3], joints[9] = "left_knee_joint", "right_knee_joint"
    manifest = {
        "contract": CONTRACT,
        "artifact_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "input_keys": keys,
        "input_names": names,
        "input_shapes": list(INPUT_SHAPES.values()),
        "input_dtypes": ["bool" if key in BOOL_INPUTS else "float32" for key in keys],
        "output_names": outputs,
        "robot_contract": {
            "quaternion_order": "xyzw",
            "anchor_body_index": 16,
            "conditionable_body_ids": [16, 13, 7, 24, 32],
            "body_names": bodies,
            "joint_names": joints,
        },
        "previous_action_domain": "raw_policy_action",
        "action_map": {"kind": "affine", "scale": [0.25] * 29, "offset": [0.0] * 29},
    }
    sidecar = tmp_path / "source.json"
    sidecar.write_text(json.dumps(manifest))
    output = tmp_path / "bundle"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.export_metatron_v1_leapp",
            "--model",
            str(model),
            "--manifest",
            str(sidecar),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    recovery = tmp_path / "recovery.json"
    recovery.write_text('{"synthetic_fixture": true}')
    native = tmp_path / "native"
    prepare(output / "policy.yaml", None, recovery, native)
    assert (native / "model.onnx").read_bytes() == model.read_bytes()
    packaged = json.loads((native / "manifest.json").read_text())
    assert "leapp_bundle" not in packaged
    assert packaged["action_map"] == manifest["action_map"]
    assert (
        packaged["deployment_recovery_plan"]["sha256"]
        == hashlib.sha256(recovery.read_bytes()).hexdigest()
    )

    # The existing bare ONNX path still packages the same graph.
    bare = tmp_path / "bare-native"
    prepare(model, sidecar, recovery, bare)
    assert (bare / "manifest.json").read_bytes() == (
        native / "manifest.json"
    ).read_bytes()

    manifest["contract"] = "unsupported-policy"
    sidecar.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="contract"):
        prepare(model, sidecar, recovery, tmp_path / "invalid")
