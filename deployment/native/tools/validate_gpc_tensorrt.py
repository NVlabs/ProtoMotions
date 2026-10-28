# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference-parity checker for native GPC TensorRT engines."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import tensorrt as trt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.gpc_tensorrt_benchmark import (  # noqa: E402 -- add deployment to sys.path first
    CUDA_DEVICE_TO_HOST,
    CUDA_HOST_TO_DEVICE,
    Cuda,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--plugin-library", type=Path)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=2e-5)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    plugin_handle = None
    if args.plugin_library is not None:
        plugin_handle = ctypes.CDLL(str(args.plugin_library), mode=ctypes.RTLD_GLOBAL)

    reference = ort.InferenceSession(
        str(args.reference_onnx), providers=["CPUExecutionProvider"]
    )
    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"could not deserialize TensorRT engine: {args.engine}")
    context = engine.create_execution_context()
    cuda = Cuda()
    stream = ctypes.c_void_p()
    cuda.check(cuda.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    host_ptrs: dict[str, ctypes.c_void_p] = {}
    device_ptrs: dict[str, ctypes.c_void_p] = {}
    for item in manifest["inputs"]:
        shape = tuple(item["shape"])
        nbytes = int(np.prod(shape)) * 4
        host_ptrs[item["name"]] = cuda.malloc_host(nbytes)
        device_ptrs[item["name"]] = cuda.malloc(nbytes)
        context.set_tensor_address(item["name"], int(device_ptrs[item["name"]].value))
    host_output_ptr = cuda.malloc_host(29 * 4)
    device_output = cuda.malloc(29 * 4)
    context.set_tensor_address("action", int(device_output.value))
    output_view = np.ctypeslib.as_array(
        (ctypes.c_float * 29).from_address(host_output_ptr.value)
    )
    rng = np.random.default_rng(184250)
    maximum_error = 0.0
    failed_cases = 0
    try:
        for case in range(args.cases):
            values: dict[str, np.ndarray] = {}
            for item in manifest["inputs"]:
                shape = tuple(item["shape"])
                name = item["name"]
                if name == "gumbel_noise":
                    values[name] = (
                        -np.log(-np.log(rng.uniform(1e-6, 1.0 - 1e-6, shape)))
                    ).astype(np.float32)
                else:
                    scale = 0.7 if name == "reduced_coords_obs" else 0.3
                    values[name] = rng.normal(0.0, scale, shape).astype(np.float32)
            expected = reference.run(["action"], values)[0]
            for name, value in values.items():
                ctypes.memmove(host_ptrs[name].value, value.ctypes.data, value.nbytes)
                cuda.check(
                    cuda.lib.cudaMemcpyAsync(
                        device_ptrs[name],
                        host_ptrs[name],
                        value.nbytes,
                        CUDA_HOST_TO_DEVICE,
                        stream,
                    ),
                    f"H2D {name}",
                )
            if not context.execute_async_v3(int(stream.value)):
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            cuda.check(
                cuda.lib.cudaMemcpyAsync(
                    host_output_ptr, device_output, 29 * 4, CUDA_DEVICE_TO_HOST, stream
                ),
                "D2H action",
            )
            cuda.check(cuda.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            actual = output_view.copy().reshape(1, 29)
            error = float(np.max(np.abs(actual - expected)))
            maximum_error = max(maximum_error, error)
            if not np.isfinite(actual).all() or error > args.threshold:
                failed_cases += 1
                print(
                    json.dumps(
                        {
                            "case": case,
                            "max_abs_action_error": error,
                            "actual_finite": bool(np.isfinite(actual).all()),
                        }
                    ),
                    flush=True,
                )
        report = {
            "status": "ok" if failed_cases == 0 else "failed",
            "sampling_mode": manifest["sampling_mode"],
            "cases": args.cases,
            "threshold": args.threshold,
            "max_abs_action_error": maximum_error,
            "failed_cases": failed_cases,
            "reference_onnx": str(args.reference_onnx),
            "engine": str(args.engine),
        }
        print(json.dumps(report, indent=2), flush=True)
        return 0 if failed_cases == 0 else 2
    finally:
        for pointer in device_ptrs.values():
            cuda.lib.cudaFree(pointer)
        for pointer in host_ptrs.values():
            cuda.lib.cudaFreeHost(pointer)
        cuda.lib.cudaFree(device_output)
        cuda.lib.cudaFreeHost(host_output_ptr)
        cuda.lib.cudaStreamDestroy(stream)
        del plugin_handle


if __name__ == "__main__":
    raise SystemExit(main())
