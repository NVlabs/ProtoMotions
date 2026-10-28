# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a policy artifact: a bare ONNX file or a LEAPP bundle.

A LEAPP bundle holds the ONNX
model, ``<name>.yaml`` (the LEAPP contract Isaac Deploy loads) and
``manifest.json`` (the contract this runtime consumes). The two contracts are
cross-checked before any control cycle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml

NATIVE_OUTPUT_NAMES = ("action", "stiffness", "damping", "raw_action")


def manifest_tensor_specs(
    manifest: dict, direction: Literal["input", "output"]
) -> list[dict]:
    """Describe native ports, including the fixed float32 per-joint outputs.

    The native manifest declares input shapes/dtypes, but only output names.
    Every output is a batch-one joint vector; this is the runtime's contract,
    not a dimension guessed from an arbitrary graph.
    """
    names = manifest[f"{direction}_names"]
    if direction == "input":
        shapes = manifest["input_shapes"]
        dtypes = manifest["input_dtypes"]
        keys = manifest["input_keys"]
    else:
        if set(names) != set(NATIVE_OUTPUT_NAMES):
            raise ValueError("native output names differ from the action contract")
        shapes = [[len(manifest["robot_contract"]["joint_names"])]] * len(names)
        dtypes = ["float32"] * len(names)
        keys = names
    if (
        not (len(keys) == len(names) == len(shapes) == len(dtypes))
        or len(set(keys)) != len(keys)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"{direction} metadata lengths or names differ")
    return [
        {"name": name, "shape": [1, *shape], "dtype": dtype, "protomotions_key": key}
        for name, shape, dtype, key in zip(names, shapes, dtypes, keys)
    ]


def bundle_model_path(bundle_yaml: Path) -> Path:
    """Path of the single ``policy`` model referenced by a LEAPP bundle YAML."""
    models = yaml.safe_load(bundle_yaml.read_text()).get("models", {})
    if list(models) != ["policy"]:
        raise ValueError(
            f"expected one 'policy' model in {bundle_yaml}, found {list(models)}"
        )
    relative = Path(models["policy"]["parameters"]["model_path"])
    model = bundle_yaml.parent / relative
    if relative.is_absolute() or not model.resolve().is_relative_to(
        bundle_yaml.parent.resolve()
    ):
        raise ValueError("LEAPP model_path must stay inside the bundle")
    return model


def resolve_policy_artifact(
    artifact: Path, manifest: Path | None = None
) -> tuple[Path, Path]:
    """Return ``(onnx_model, manifest)`` for an ONNX file or a LEAPP bundle YAML.

    The manifest defaults to the ``.json`` sidecar of an ONNX file, or to
    ``manifest.json`` next to a bundle YAML.
    """
    if artifact.suffix not in (".yaml", ".yml"):
        return artifact, manifest or artifact.with_suffix(".json")
    manifest = manifest or artifact.parent / "manifest.json"
    validate_leapp_bundle(artifact, json.loads(manifest.read_text()))
    return bundle_model_path(artifact), manifest


def validate_leapp_bundle(bundle_yaml: Path, manifest: dict) -> None:
    """Reject a bundle whose YAML inputs/outputs disagree with the manifest."""
    if manifest.get("leapp_bundle") != bundle_yaml.name:
        raise ValueError(f"manifest leapp_bundle does not reference {bundle_yaml.name}")
    node = yaml.safe_load(bundle_yaml.read_text())["models"]["policy"]
    model = bundle_model_path(bundle_yaml)
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if (
        digest != manifest["artifact_sha256"]
        or digest != node["parameters"]["sha256sum"]
    ):
        raise ValueError("LEAPP model checksum differs from YAML or manifest")
    for direction in ("input", "output"):
        items = node[f"{direction}s"]
        specs = manifest_tensor_specs(manifest, direction)
        names = [spec["name"] for spec in specs]
        actual_names = [item["name"] for item in items]
        if len(items) != len(specs):
            raise ValueError(f"LEAPP {direction} metadata lengths differ")
        if len(set(actual_names)) != len(items):
            raise ValueError(f"duplicate LEAPP {direction} names")
        # Inputs retain argument order; outputs are consumed by name because
        # LEAPP appends constant gains after the computed outputs.
        if (
            actual_names != names
            if direction == "input"
            else set(actual_names) != set(names)
        ):
            raise ValueError(f"LEAPP {direction} names differ from manifest")
        by_name = {item["name"]: item for item in items}
        for spec in specs:
            item = by_name[spec["name"]]
            # These robot runtimes consume batch-one policies. Do not silently
            # discard a changed batch dimension during the contract check.
            if list(item["shape"]) != spec["shape"] or item["dtype"] != spec["dtype"]:
                raise ValueError(
                    f"LEAPP {direction} {spec['name']} shape/dtype differs from manifest"
                )
            if (
                direction == "input"
                and item["protomotions_key"] != spec["protomotions_key"]
            ):
                raise ValueError(
                    f"LEAPP input {spec['name']} context key differs from manifest"
                )
