# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one fixed-shape FP32 TensorRT engine for the scene policy.

The native launchers reject a mismatched engine I/O contract. The temporary
output is renamed only after TensorRT has returned a complete serialized plan.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import tensorrt as trt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, help="qualified graph-safe ONNX candidate")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workspace-gib", type=int, default=2)
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=range(6),
        default=0,
        help="TensorRT tactic-search level (0 keeps the fast-build baseline)",
    )
    args = parser.parse_args()
    artifact = args.artifact_dir
    original = artifact / "model.onnx"
    onnx = args.onnx or original
    manifest = json.loads((artifact / "manifest.json").read_text())
    if manifest["contract"] != "protomotions.masked_mimic.g1.v5_raw_context_scene":
        raise ValueError("expected the 39-input scene policy contract")
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if digest != manifest["artifact_sha256"]:
        raise ValueError("ONNX digest does not match the pinned manifest")
    if args.onnx and onnx.resolve() != (artifact / "model.graphsafe.onnx").resolve():
        raise ValueError(
            "only the separately qualified graph-safe candidate is accepted"
        )
    output = args.output or artifact / "model.fp32.plan"
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(
            "engine or partial build already exists; inspect it first"
        )

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network, logger)
    start = time.monotonic()
    print(f"TensorRT {trt.__version__}: parsing {onnx}", flush=True)
    if not onnx_parser.parse_from_file(str(onnx)):
        details = "\n".join(
            str(onnx_parser.get_error(i)) for i in range(min(8, onnx_parser.num_errors))
        )
        raise RuntimeError(f"ONNX parse failed:\n{details}")
    expected = set(manifest["input_names"])
    actual = {network.get_input(i).name for i in range(network.num_inputs)}
    if len(expected) != 39 or actual != expected:
        raise ValueError("parsed TensorRT inputs differ from the 39-input manifest")
    if {network.get_output(i).name for i in range(network.num_outputs)} != set(
        manifest["output_names"]
    ):
        raise ValueError("parsed TensorRT outputs differ from the manifest")
    print(
        f"parsed layers={network.num_layers} elapsed={time.monotonic()-start:.1f}s",
        flush=True,
    )
    config = builder.create_builder_config()
    config.builder_optimization_level = args.optimization_level
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace_gib * (1 << 30)
    )
    print(
        f"building FP32 plan with optimization level {args.optimization_level}; "
        "this stage may take time",
        flush=True,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT did not produce a serialized engine")
    output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("xb") as handle:
        handle.write(plan)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(output)
    print(
        f"ready {output} bytes={output.stat().st_size} elapsed={time.monotonic()-start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
