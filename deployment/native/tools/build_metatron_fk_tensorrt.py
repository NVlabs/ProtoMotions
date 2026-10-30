# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile an exported FK graph to a TensorRT FP32 plan.

The graph is ``fk.onnx`` from ``deployment/export_fk_to_onnx.py``: the policy's
own ``BakedAnchorFK`` kernel traced on its own, inputs the joint angles and the
anchor pose, outputs the world body positions and rotations, batch 1. Its
sidecar ``fk.yaml`` gives the expected tensor names and shapes; with
``--manifest`` the sidecar is also checked against the policy manifest (body
order, joint order, anchor, and that the FK inputs are inputs of the policy) so
the plan is never paired with a policy it does not belong to.
"""

import argparse
import os
from pathlib import Path
import time

import tensorrt as trt

from deployment.export_fk_to_onnx import FkSidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="fk.onnx")
    parser.add_argument("--sidecar", type=Path, required=True, help="its fk.yaml")
    parser.add_argument("--output", type=Path, required=True, help="the plan to write")
    parser.add_argument("--manifest", type=Path, default=None, help="policy manifest")
    args = parser.parse_args()

    if (
        args.output.exists()
        or args.output.with_suffix(args.output.suffix + ".partial").exists()
    ):
        raise FileExistsError("FK plan or partial build already exists")

    sidecar = FkSidecar.from_yaml(args.sidecar)
    if args.manifest is not None:
        sidecar.check_manifest(args.manifest)
    expected_inputs = sidecar.input_shapes
    expected_outputs = sidecar.output_shapes

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    onnx_parser = trt.OnnxParser(network, logger)
    start = time.monotonic()
    print(f"TensorRT {trt.__version__}: parsing FK {args.onnx}", flush=True)
    if not onnx_parser.parse_from_file(str(args.onnx)):
        errors = "\n".join(
            str(onnx_parser.get_error(i)) for i in range(min(8, onnx_parser.num_errors))
        )
        raise RuntimeError(f"FK ONNX parse failed:\n{errors}")
    if network.num_inputs != len(expected_inputs) or network.num_outputs != len(
        expected_outputs
    ):
        raise ValueError(
            f"FK plan must have the sidecar's {len(expected_inputs)} inputs and "
            f"{len(expected_outputs)} outputs"
        )
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        if (
            tensor.name not in expected_inputs
            or tuple(tensor.shape) != expected_inputs[tensor.name]
            or tensor.dtype != trt.float32
        ):
            raise ValueError(
                f"unexpected FK input: {tensor.name} {tensor.shape} {tensor.dtype}"
            )
    for index in range(network.num_outputs):
        tensor = network.get_output(index)
        if (
            tensor.name not in expected_outputs
            or tuple(tensor.shape) != expected_outputs[tensor.name]
            or tensor.dtype != trt.float32
        ):
            raise ValueError(
                f"unexpected FK output: {tensor.name} {tensor.shape} {tensor.dtype}"
            )
    print(
        f"FK parsed layers={network.num_layers} inputs={list(sidecar.input_names)} "
        f"anchor={sidecar.anchor_body_name} elapsed={time.monotonic() - start:.1f}s",
        flush=True,
    )

    config = builder.create_builder_config()
    config.builder_optimization_level = 0
    config.max_aux_streams = 0
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    print(
        "building static FP32 FK plan; TF32 off, optimization level 0, no auxiliary streams",
        flush=True,
    )
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT did not produce an FK engine")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    try:
        with partial.open("xb") as handle:
            handle.write(plan)
            handle.flush()
            os.fsync(handle.fileno())
        partial.replace(args.output)
    except BaseException:
        # A failed write must not leave a partial that blocks every later run.
        partial.unlink(missing_ok=True)
        raise
    print(
        f"FK TensorRT plan ready {args.output} bytes={args.output.stat().st_size} "
        f"elapsed={time.monotonic() - start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
