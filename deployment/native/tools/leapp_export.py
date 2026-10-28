# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a deployment wrapper as a LEAPP bundle that Isaac Deploy can load.

The whole pipeline (raw context -> observations -> model -> action processor)
is one LEAPP node. Sensor/action histories stay external inputs, as in the
plain ONNX export: the runtime updates them from the command it applied.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import torch

from deployment.native.examples.policy_artifact import (
    NATIVE_OUTPUT_NAMES,
    manifest_tensor_specs,
)

if TYPE_CHECKING:
    from deployment.contract import DeploymentContract

NODE_NAME = "policy"
OUTPUT_NAMES = NATIVE_OUTPUT_NAMES


def _checked_document(
    model: Path,
    ports: dict[str, list[dict]],
    expected_sha256: str,
    *,
    dynamic_batch: bool = False,
) -> dict:
    """Check declared ports against ONNX and describe one external-input node."""
    import leapp
    import onnx
    from google.protobuf.descriptor import FieldDescriptor

    model_bytes = model.read_bytes()
    digest = hashlib.sha256(model_bytes).hexdigest()
    if digest != expected_sha256:
        raise ValueError("ONNX checksum differs from its contract")
    graph = onnx.load(model, load_external_data=False)

    # Bundle integrity must cover weights too. This exporter deliberately only
    # accepts self-contained ONNX; external tensor files need their own hashes.
    def check_tensors(message):
        if (
            isinstance(message, onnx.TensorProto)
            and message.data_location == onnx.TensorProto.EXTERNAL
        ):
            raise ValueError("LEAPP packaging requires a self-contained ONNX model")
        for field, value in message.ListFields():
            if field.message_type is not None:
                for child in (
                    value if field.label == FieldDescriptor.LABEL_REPEATED else [value]
                ):
                    check_tensors(child)

    check_tensors(graph)
    onnx.checker.check_model(graph)
    # Expand-based PD gains can leave a symbolic joint width in the initial
    # trace. Infer it from the graph, without modifying the packaged bytes.
    graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True, data_prop=True)
    node = {}
    for direction, graph_ports in (
        ("input", graph.graph.input),
        ("output", graph.graph.output),
    ):
        descriptions = ports[direction]
        names = [entry["name"] for entry in descriptions]
        if len(set(names)) != len(names) or len(names) != len(graph_ports):
            raise ValueError(f"{direction} metadata lengths or names differ")
        if set(names) != {port.name for port in graph_ports}:
            raise ValueError(f"ONNX {direction} names differ from its contract")
        if direction == "input" and names != [port.name for port in graph_ports]:
            raise ValueError("ONNX input order differs from its contract")
        expected = {entry["name"]: entry for entry in descriptions}
        descriptions = []
        # LEAPP's inference manager associates outputs positionally. Preserve
        # ONNX output order here even if the runtime manifest binds by name.
        for port in graph_ports:
            tensor = port.type.tensor_type
            dtype = str(onnx.helper.tensor_dtype_to_np_dtype(tensor.elem_type))
            entry = expected[port.name]
            dimensions = tensor.shape.dim
            shape_matches = len(dimensions) == len(entry["shape"]) and all(
                (dynamic_batch and i == 0 and bool(dim.dim_param))
                or (dim.HasField("dim_value") and dim.dim_value == size)
                for i, (dim, size) in enumerate(zip(dimensions, entry["shape"]))
            )
            if not shape_matches or dtype != entry["dtype"]:
                raise ValueError(
                    f"ONNX {direction} {port.name} shape/dtype differs from its contract"
                )
            descriptions.append(entry)
        node[f"{direction}s"] = descriptions
    node["parameters"] = {
        "model_path": "policy.onnx",
        "backend": "onnx",
        "sha256sum": digest,
        "md5sum": hashlib.md5(model_bytes, usedforsecurity=False).hexdigest(),
    }
    return {
        "models": {NODE_NAME: node},
        "pipeline": {
            "inputs": {NODE_NAME: [p.name for p in graph.graph.input]},
            "outputs": {NODE_NAME: [p.name for p in graph.graph.output]},
            "data_flow": {},
            "feedback_flow": {},
        },
        "system information": {
            "leapp version": leapp.__version__,
            "leapp config version": leapp.__config_version__,
        },
    }


def _write_bundle(model: Path, document: dict, output: Path) -> Path:
    import yaml

    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(model, output / "policy.onnx")
    bundle = output / "policy.yaml"
    bundle.write_text(yaml.safe_dump(document, sort_keys=False))
    return bundle


def _manifest_ports(manifest: dict, direction: str) -> list[dict]:
    """Native v1 metadata excludes the batch axis; LEAPP includes it."""
    inputs, outputs = _semantics(manifest["robot_contract"]["joint_names"])
    semantics = inputs if direction == "input" else outputs
    result = manifest_tensor_specs(manifest, direction)
    for entry in result:
        key = entry["protomotions_key"]
        entry["type"] = "tensor"
        meaning = semantics.get(key, {})
        if "kind" in meaning:
            entry["kind"] = meaning["kind"].value
        if "element_names" in meaning:
            entry["element_names"] = [meaning["element_names"]]
    return result


def export_onnx_bundle(model: Path, manifest: dict, output: Path) -> Path:
    """Package an existing fixed-batch native graph without changing its bytes."""
    from deployment.native.examples.policy_artifact import validate_leapp_bundle

    ports = {
        direction: _manifest_ports(manifest, direction)
        for direction in ("input", "output")
    }
    document = _checked_document(model, ports, manifest["artifact_sha256"])
    bundle = _write_bundle(model, document, output)
    metadata = copy.deepcopy(manifest)
    metadata["leapp_bundle"] = bundle.name
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    validate_leapp_bundle(bundle, metadata)
    return bundle


def _pipeline_ports(contract: DeploymentContract) -> dict[str, list[dict]]:
    """Keep all context, provider and history semantics from the typed contract.

    LEAPP standard kinds describe physical quantities only. Goals, masks and raw
    actions retain their ProtoMotions kinds instead of being mislabeled sensors.
    """
    from leapp.utils.enums import InputKindEnum, OutputKindEnum

    kinds = {
        "input": {
            "joint_pos": InputKindEnum.JOINT_POSITION,
            "joint_vel": InputKindEnum.JOINT_VELOCITY,
            "body_pos": InputKindEnum.BODY_POSITION,
            "body_rot": InputKindEnum.BODY_ROTATION,
            "anchor_rot": InputKindEnum.BODY_ROTATION,
            "root_local_ang_vel": InputKindEnum.BODY_ANGULAR_VELOCITY,
        },
        "output": {
            "joint_pos_targets": OutputKindEnum.JOINT_POSITION,
            "stiffness_targets": OutputKindEnum.KP,
            "damping_targets": OutputKindEnum.KD,
        },
    }
    document = contract.to_dict()
    result = {}
    for direction in ("input", "output"):
        result[direction] = []
        for spec in document[f"policy_{direction}s"]:
            entry = {
                "name": spec["name"],
                "shape": spec["shape"],
                "dtype": spec["dtype"] if direction == "input" else "float32",
                "type": "tensor",
                "protomotions_key": spec["key"],
                "protomotions": spec,
            }
            if spec["kind"] in kinds[direction]:
                entry["kind"] = kinds[direction][spec["kind"]].value
            if "element_names" in spec:
                entry["element_names"] = spec["element_names"]
            if "joint_names" in spec:
                entry["element_names"] = [spec["joint_names"]]
            result[direction].append(entry)
    return result


def export_pipeline_bundle(
    model: Path, contract: DeploymentContract, output: Path
) -> Path:
    """Package the generic pipeline, preserving its actual inputs and outputs.

    The ONNX may have a dynamic batch axis. LEAPP describes the contract's
    concrete deployment batch size without rewriting that graph. Host-side
    histories stay explicit inputs, not automatic network-output feedback.
    """
    document = _checked_document(
        model, _pipeline_ports(contract), contract.onnx_sha256, dynamic_batch=True
    )
    bundle = _write_bundle(model, document, output)
    contract.to_yaml(output / "deployment_contract.yaml")
    load_pipeline_contract(bundle)
    return bundle


def load_pipeline_contract(bundle: Path) -> DeploymentContract:
    """Validate graph/interface agreement before binding a generic LEAPP host."""
    import yaml
    from deployment.contract import DeploymentContract
    from deployment.native.examples.policy_artifact import bundle_model_path

    contract = DeploymentContract.from_yaml(bundle.parent / "deployment_contract.yaml")
    expected = _checked_document(
        bundle_model_path(bundle),
        _pipeline_ports(contract),
        contract.onnx_sha256,
        dynamic_batch=True,
    )
    actual = yaml.safe_load(bundle.read_text())
    for key in ("models", "pipeline"):
        if actual[key] != expected[key]:
            raise ValueError(f"LEAPP {key} differs from deployment contract")
    return contract


def _semantics(joint_names: Sequence[str]) -> tuple[dict, dict]:
    """Standard LEAPP kinds for the inputs/outputs Isaac Deploy binds directly."""
    from leapp.utils.enums import InputKindEnum, OutputKindEnum

    joints = {"element_names": list(joint_names)}
    inputs = {
        "noisy.dof_pos": {"kind": InputKindEnum.JOINT_POSITION, **joints},
        "noisy.dof_vel": {"kind": InputKindEnum.JOINT_VELOCITY, **joints},
        "noisy.root_local_ang_vel": {"kind": InputKindEnum.BODY_ANGULAR_VELOCITY},
        "noisy.anchor_rot": {"kind": InputKindEnum.BODY_ROTATION},
    }
    outputs = {
        "action": {"kind": OutputKindEnum.JOINT_POSITION, **joints},
        "stiffness": {"kind": OutputKindEnum.KP, **joints},
        "damping": {"kind": OutputKindEnum.KD, **joints},
        "raw_action": joints,  # fed back by the runtime as the previous action
    }
    return inputs, outputs


def export_leapp_bundle(
    wrapper: torch.nn.Module,
    keys: Sequence[str],
    input_names: Sequence[str],
    samples: Sequence[Sequence[torch.Tensor]],
    output_dir: Path,
    graph_name: str,
    joint_names: Sequence[str],
    opset: int = 17,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> Path:
    """Trace ``wrapper`` on ``samples`` and write a validated LEAPP bundle.

    ``keys`` are the ProtoMotions context keys and ``input_names`` their ONNX
    names, in the wrapper's argument order. The first sample is the one whose
    Python branches are captured; LEAPP checks the export against all of them.
    Returns the bundle YAML path.
    """
    import leapp
    from leapp import TensorSemantics, annotate
    from leapp.leapp_graph.datatypes import is_traced_type

    if not samples or not keys:
        raise ValueError("LEAPP export requires inputs and at least one sample")
    if any(len(sample) != len(keys) for sample in samples) or len(keys) != len(
        input_names
    ):
        raise ValueError("every sample needs one tensor per input key")
    if len(set(keys)) != len(keys) or len(set(input_names)) != len(input_names):
        raise ValueError("LEAPP input keys and names must be unique")
    input_semantics, output_semantics = _semantics(joint_names)
    traced_names = None
    leapp.start(name=graph_name, save_path=str(output_dir), max_cached_io=len(samples))
    try:
        for sample in samples:
            traced = annotate.input_tensors(
                NODE_NAME,
                [
                    TensorSemantics(
                        name,
                        value,
                        extra={"protomotions_key": key},
                        **input_semantics.get(key, {}),
                    )
                    for key, name, value in zip(keys, input_names, sample)
                ],
            )
            if len(keys) == 1:
                traced = (traced,)
            with torch.no_grad():
                values = wrapper(*traced)
                if len(values) != len(OUTPUT_NAMES):
                    raise ValueError(
                        "wrapper must return action, stiffness, damping, raw_action"
                    )
                outputs = dict(zip(OUTPUT_NAMES, values))
            if traced_names is None:
                # Gains that don't depend on the inputs are not traced; LEAPP
                # exports them as constant outputs. Decide once, on the tracing
                # pass: later passes only cache values for validation.
                traced_names = [n for n in OUTPUT_NAMES if is_traced_type(outputs[n])]
                if not {"action", "raw_action"} <= set(traced_names):
                    raise ValueError("action outputs must be computed from the inputs")
            static = [
                TensorSemantics(n, outputs[n].detach().clone(), **output_semantics[n])
                for n in OUTPUT_NAMES
                if n not in traced_names
            ]
            annotate.output_tensors(
                NODE_NAME,
                [
                    TensorSemantics(n, outputs[n], **output_semantics[n])
                    for n in traced_names
                ],
                static_outputs=static or None,
                export_with="onnx-torchscript",
                backend_params={"opset_version": opset},
            )
    finally:
        leapp.stop()
    leapp.compile_graph(
        visualize=False, validate=True, rtol=rtol, atol=atol, strict=True
    )
    return output_dir / graph_name / f"{graph_name}.yaml"
