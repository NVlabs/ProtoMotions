# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile the fixed-shape, three-input G1 FK ONNX model to TensorRT FP32."""

import argparse
import os
from pathlib import Path
import time

import tensorrt as trt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.with_suffix(args.output.suffix + ".partial").exists()
    ):
        raise FileExistsError("FK plan or partial build already exists")
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
    expected_inputs = {"root_pos": (1, 3), "root_rot": (1, 4), "dof_pos": (1, 29)}
    expected_outputs = {"body_pos": (1, 33, 3), "body_rot": (1, 33, 4)}
    if network.num_inputs != 3 or network.num_outputs != 2:
        raise ValueError("FK plan must have three inputs and two outputs")
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
        f"FK parsed layers={network.num_layers} elapsed={time.monotonic()-start:.1f}s",
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
    with partial.open("xb") as handle:
        handle.write(plan)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(args.output)
    print(
        f"FK TensorRT plan ready {args.output} bytes={args.output.stat().st_size} "
        f"elapsed={time.monotonic()-start:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
