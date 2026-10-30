# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The FK exporter's sidecar, its refusals and the native fixture it feeds.

``test_metatron_generic_export.py`` exports ``fk.onnx`` from the real G1 recipe
and checks it against the policy graph; this file pins the pieces that export
relies on, each on the small five-body test tree: the sidecar's own checks and
round trip, the manifest pairing refusals, the static-op and I/O checks, the
ONNX Runtime comparison, that a refused export leaves nothing on disk and never
replaces an earlier pair, the missing-checkpoint refusal, and the binary layout
the native parity bench reads (``make_metatron_fk_fixture.py``, ``PM_FK_V2``).
"""

import json
import struct
import sys
from dataclasses import replace

import numpy as np
import pytest
import torch

from deployment.export_fk_to_onnx import (
    FK_OUTPUTS,
    FkSidecar,
    _check_static,
    _validate,
    export_fk,
    write_fk_graph,
)
from protomotions.tests import test_baked_fk
from protomotions.utils.baked_fk import BakedAnchorFK

BODIES = ("root", "torso", "left", "right", "head")
JOINTS = ("j0", "j1", "j2", "j3")


@pytest.fixture
def branching_ki():
    # The five-body Y-tree of the FK tests: root -> torso -> {left, right, head}.
    return test_baked_fk.branching_ki.__wrapped__()


def _sidecar(**overrides) -> FkSidecar:
    base = dict(
        body_names=BODIES,
        joint_names=JOINTS,
        anchor_body_name="torso",
        anchor_body_index=1,
        inputs=(
            ("noisy_dof_pos", "noisy.dof_pos", (1, 4)),
            ("noisy_anchor_pos", "noisy.anchor_pos", (1, 3)),
            ("noisy_anchor_rot", "noisy.anchor_rot", (1, 4)),
        ),
        outputs=(("body_pos", (1, 5, 3)), ("body_rot", (1, 5, 4))),
        checkpoint="last.ckpt",
    )
    base.update(overrides)
    return FkSidecar(**base)


def _kernel(kinematic_info) -> BakedAnchorFK:
    return BakedAnchorFK(
        kinematic_info, anchor_idx=1, trackable_body_indices=torch.arange(5)
    )


@pytest.mark.parametrize(
    "change, message",
    [
        ({"anchor_body_index": 5}, "outside 5 bodies"),
        ({"anchor_body_name": "head"}, "is not body 1"),
        (
            {
                "inputs": (
                    ("a", "noisy.anchor_pos", (1, 3)),
                    ("d", "noisy.dof_pos", (1, 4)),
                    ("r", "noisy.anchor_rot", (1, 4)),
                )
            },
            "in that order",
        ),
        (
            {
                "inputs": (
                    ("d", "noisy.dof_pos", (1, 3)),
                    ("a", "noisy.anchor_pos", (1, 3)),
                    ("r", "noisy.anchor_rot", (1, 4)),
                )
            },
            "has shape \\(1, 3\\), not \\(1, 4\\)",
        ),
        ({"outputs": (("pos", (1, 5, 3)), ("rot", (1, 5, 4)))}, "FK outputs must be"),
        ({"outputs": (("body_pos", (1, 4, 3)), ("body_rot", (1, 5, 4)))}, "not \\(1, 5, 3\\)"),
    ],
)
def test_sidecar_refuses_an_inconsistent_record(change, message):
    with pytest.raises(ValueError, match=message):
        _sidecar(**change)


def test_sidecar_round_trips_and_refuses_a_foreign_file(tmp_path):
    sidecar = _sidecar()
    assert FkSidecar.from_dict(sidecar.to_dict()) == sidecar
    sidecar.to_yaml(tmp_path / "fk.yaml")
    assert FkSidecar.from_yaml(tmp_path / "fk.yaml") == sidecar
    assert sidecar.input_names == ("noisy_dof_pos", "noisy_anchor_pos", "noisy_anchor_rot")
    assert sidecar.output_shapes == {"body_pos": (1, 5, 3), "body_rot": (1, 5, 4)}
    with pytest.raises(ValueError, match="not an FK sidecar"):
        FkSidecar.from_dict({**sidecar.to_dict(), "type": "unified_pipeline"})
    with pytest.raises(ValueError, match="schema 2"):
        FkSidecar.from_dict({**sidecar.to_dict(), "schema": 2})


def _manifest(path, **changes):
    content = {
        "robot_contract": {
            "body_names": list(BODIES),
            "joint_names": list(JOINTS),
            "anchor_body_index": 1,
        },
        "input_names": [
            "noisy_dof_pos",
            "noisy_anchor_pos",
            "noisy_anchor_rot",
            "current_action",
        ],
    }
    for key, value in changes.items():
        if key == "input_names":
            content[key] = value
        else:
            content["robot_contract"][key] = value
    path.write_text(json.dumps(content))
    return path


def test_check_manifest_refuses_a_policy_the_graph_does_not_belong_to(tmp_path):
    sidecar = _sidecar()
    sidecar.check_manifest(_manifest(tmp_path / "ok.json"))
    cases = [
        ({"body_names": list(BODIES[::-1])}, "body order differs"),
        ({"joint_names": ["j0", "j1", "j3", "j2"]}, "joint order differs"),
        ({"anchor_body_index": 0}, "anchor body differs"),
        (
            {"input_names": ["clean_dof_pos", "noisy_anchor_pos", "noisy_anchor_rot"]},
            r"inputs \['noisy_dof_pos'\] are not inputs of the policy manifest",
        ),
    ]
    for change, message in cases:
        with pytest.raises(ValueError, match=message):
            sidecar.check_manifest(_manifest(tmp_path / "bad.json", **change))


def test_check_static_refuses_data_dependent_ops_and_foreign_io():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    sidecar = _sidecar()
    x = helper.make_tensor_value_info("noisy_dof_pos", TensorProto.FLOAT, [1, 4])
    nz = helper.make_tensor_value_info("body_pos", TensorProto.INT64, [2, None])
    model = helper.make_model(
        helper.make_graph([helper.make_node("NonZero", ["noisy_dof_pos"], ["body_pos"])], "nz", [x], [nz]),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    with pytest.raises(ValueError, match=r"contains \['NonZero'\]"):
        _check_static(model, sidecar)
    y = helper.make_tensor_value_info("body_pos", TensorProto.FLOAT, [1, 4])
    model = helper.make_model(
        helper.make_graph([helper.make_node("Identity", ["noisy_dof_pos"], ["body_pos"])], "id", [x], [y]),
        opset_imports=[helper.make_opsetid("", 17)],
    )
    with pytest.raises(ValueError, match="differs from the sidecar"):
        _check_static(model, sidecar)
    assert onnx is not None


def test_write_fk_graph_matches_the_kernel_and_refuses_a_disagreement(tmp_path, branching_ki):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    fk = _kernel(branching_ki)
    sidecar = _sidecar()
    onnx_path = write_fk_graph(fk, branching_ki, sidecar, tmp_path, validate=True, samples=4)
    assert onnx_path == tmp_path / "fk.onnx"
    assert FkSidecar.from_yaml(tmp_path / "fk.yaml") == sidecar
    assert not (tmp_path / "fk.onnx.partial").exists()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dof_pos = torch.tensor([[0.3, -0.2, 0.5, 1.0]])
    anchor_pos = torch.tensor([[0.1, 0.2, 0.9]])
    anchor_rot = torch.nn.functional.normalize(torch.tensor([[0.1, 0.2, 0.3, 0.9]]), dim=-1)
    got = session.run(
        list(FK_OUTPUTS),
        dict(zip(sidecar.input_names, (dof_pos.numpy(), anchor_pos.numpy(), anchor_rot.numpy()))),
    )
    with torch.no_grad():
        expected = fk.world_body_poses(dof_pos, anchor_pos, anchor_rot)
    for value, truth in zip(got, expected):
        np.testing.assert_allclose(value, truth.numpy(), atol=1e-6)

    class Shifted:
        def world_body_poses(self, dof_pos, anchor_pos, anchor_rot):
            pos, rot = fk.world_body_poses(dof_pos, anchor_pos, anchor_rot)
            return pos + 1e-3, rot

    with pytest.raises(ValueError, match="body_pos differs from the kernel"):
        _validate(onnx_path, Shifted(), branching_ki, sidecar, samples=1)


def test_refused_export_leaves_nothing_and_keeps_an_earlier_pair(tmp_path, branching_ki, monkeypatch):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    import deployment.export_fk_to_onnx as module

    fk = _kernel(branching_ki)

    def refuse(*args, **kwargs):
        raise ValueError("refused for the test")

    monkeypatch.setattr(module, "_validate", refuse)
    with pytest.raises(ValueError, match="refused for the test"):
        write_fk_graph(fk, branching_ki, _sidecar(), tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == []

    monkeypatch.undo()
    first = _sidecar(checkpoint="first.ckpt")
    write_fk_graph(fk, branching_ki, first, tmp_path)
    graph_bytes = (tmp_path / "fk.onnx").read_bytes()
    monkeypatch.setattr(module, "_validate", refuse)
    with pytest.raises(ValueError, match="refused for the test"):
        write_fk_graph(fk, branching_ki, _sidecar(checkpoint="second.ckpt"), tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["fk.onnx", "fk.yaml"]
    assert (tmp_path / "fk.onnx").read_bytes() == graph_bytes
    assert FkSidecar.from_yaml(tmp_path / "fk.yaml") == first


def test_export_fk_refuses_a_missing_checkpoint(tmp_path):
    (tmp_path / "resolved_configs.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="last.ckpt does not exist"):
        export_fk(str(tmp_path / "last.ckpt"), str(tmp_path / "fk"))
    assert not (tmp_path / "fk").exists()


def test_fixture_writer_records_the_v2_layout_the_parity_bench_reads(tmp_path, branching_ki, monkeypatch):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from deployment.native.tools import make_metatron_fk_fixture

    fk = _kernel(branching_ki)
    sidecar = _sidecar()
    onnx_path = write_fk_graph(fk, branching_ki, sidecar, tmp_path, validate=False)
    fixture = tmp_path / "fixture.bin"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_metatron_fk_fixture.py",
            "--onnx", str(onnx_path),
            "--sidecar", str(tmp_path / "fk.yaml"),
            "--output", str(fixture),
            "--samples", "3",
        ],
    )
    make_metatron_fk_fixture.main()

    raw = fixture.read_bytes()
    # Header: a 12-byte magic (``PM_FK_V2`` padded with NULs) and a little-endian
    # uint32 sample count; then per sample the kernel's inputs in argument order
    # followed by its outputs, all float32: dof_pos[4], anchor_pos[3],
    # anchor_rot[4], body_pos[5,3], body_rot[5,4].
    assert raw[:12] == b"PM_FK_V2\0\0\0\0"
    (count,) = struct.unpack_from("<I", raw, 12)
    assert count == 3
    widths = (4, 3, 4, 5 * 3, 5 * 4)
    record = sum(widths) * 4
    assert len(raw) == 16 + count * record

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for index in range(count):
        values = np.frombuffer(raw, dtype="<f4", count=sum(widths), offset=16 + index * record)
        parts = np.split(values, np.cumsum(widths)[:-1])
        dof_pos, anchor_pos, anchor_rot, body_pos, body_rot = (part[None] for part in parts)
        assert np.all(np.abs(dof_pos) <= 0.5)
        assert abs(float(np.linalg.norm(anchor_rot)) - 1.0) < 1e-5
        from_graph = session.run(
            list(FK_OUTPUTS),
            dict(zip(sidecar.input_names, (dof_pos, anchor_pos, anchor_rot))),
        )
        np.testing.assert_array_equal(body_pos.reshape(1, 5, 3), from_graph[0])
        np.testing.assert_array_equal(body_rot.reshape(1, 5, 4), from_graph[1])
        with torch.no_grad():
            expected = fk.world_body_poses(
                torch.from_numpy(dof_pos.copy()),
                torch.from_numpy(anchor_pos.copy()),
                torch.from_numpy(anchor_rot.copy()),
            )
        np.testing.assert_allclose(body_pos.reshape(1, 5, 3), expected[0].numpy(), atol=1e-5)
        np.testing.assert_allclose(body_rot.reshape(1, 5, 4), expected[1].numpy(), atol=1e-5)
    assert replace(sidecar, checkpoint="other") != sidecar
