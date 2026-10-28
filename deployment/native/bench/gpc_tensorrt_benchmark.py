# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and benchmark fixed-shape GPC TensorRT graphs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt


CUDA_SUCCESS = 0
CUDA_HOST_TO_DEVICE = 1
CUDA_DEVICE_TO_HOST = 2


class Cuda:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so")
        self.lib.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMallocHost.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.lib.cudaMallocHost.restype = ctypes.c_int
        self.lib.cudaFreeHost.argtypes = [ctypes.c_void_p]
        self.lib.cudaFreeHost.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.cudaStreamBeginCapture.restype = ctypes.c_int
        self.lib.cudaStreamEndCapture.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.cudaStreamEndCapture.restype = ctypes.c_int
        self.lib.cudaGraphInstantiateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        self.lib.cudaGraphInstantiateWithFlags.restype = ctypes.c_int
        self.lib.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cudaGraphLaunch.restype = ctypes.c_int
        self.lib.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphExecDestroy.restype = ctypes.c_int
        self.lib.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphDestroy.restype = ctypes.c_int

    def check(self, code: int, operation: str) -> None:
        if code != CUDA_SUCCESS:
            raise RuntimeError(f"{operation} failed with CUDA error {code}")

    def malloc(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), size), "cudaMalloc")
        return pointer

    def malloc_host(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self.check(
            self.lib.cudaMallocHost(ctypes.byref(pointer), size), "cudaMallocHost"
        )
        return pointer


def load_plugin_library(path: Path | None):
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"TensorRT plugin library not found: {path}")
    # Keep the handle alive for the full builder/runtime lifetime. GLOBAL is
    # required so TensorRT can resolve the registered creator from the shared
    # object when parsing and deserializing the plan.
    return ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("contract") != "protomotions.gpc.g1.unconditional.v1":
        raise ValueError(f"unexpected GPC contract: {manifest.get('contract')}")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise ValueError("unexpected GPC output contract")
    output = outputs[0]
    if (
        output.get("name") != "action"
        or output.get("shape") != [1, 29]
        or output.get("dtype") != "float32"
    ):
        raise ValueError("unexpected GPC output contract")
    return manifest


def build_engine(
    model: Path,
    manifest_path: Path,
    engine_path: Path,
    workspace_mib: int,
    optimization_level: int,
    tf32: bool,
    fp16: bool,
) -> None:
    manifest = load_manifest(manifest_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    # Dynamo exports large GPC weights as ONNX external data. Parsing from the
    # filename lets TensorRT resolve the sidecar next to the model.
    if not parser.parse_from_file(str(model)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT parse failed:\n{errors}")
    expected_inputs = [item["name"] for item in manifest["inputs"]]
    if [
        network.get_input(i).name for i in range(network.num_inputs)
    ] != expected_inputs:
        raise RuntimeError("TensorRT input names do not match manifest")
    if network.num_outputs != 1 or network.get_output(0).name != "action":
        raise RuntimeError("TensorRT output names do not match manifest")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib << 20)
    config.builder_optimization_level = optimization_level
    config.max_aux_streams = 0
    if tf32:
        config.set_flag(trt.BuilderFlag.TF32)
    else:
        config.clear_flag(trt.BuilderFlag.TF32)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(int(value) for value in tensor.shape)
        profile.set_shape(tensor.name, shape, shape, shape)
    config.add_optimization_profile(profile)
    print(
        f"building {model} inputs={network.num_inputs} nodes={network.num_layers} "
        f"trt={trt.__version__}",
        flush=True,
    )
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build returned no engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    metadata = {
        "model_sha256": sha256(model),
        "manifest_sha256": sha256(manifest_path),
        "engine_sha256": sha256(engine_path),
        "contract": manifest["contract"],
        "sampling_mode": manifest["sampling_mode"],
        "top_p": manifest.get("top_p"),
        "temperature": manifest["temperature"],
        "builder_optimization_level": optimization_level,
        "tf32": tf32,
        "fp16": fp16,
        "max_aux_streams": config.max_aux_streams,
        "tensorrt_version": trt.__version__,
    }
    engine_path.with_suffix(engine_path.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def percentile(values: list[int], fraction: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction * 100.0))


def benchmark(
    model: Path,
    manifest_path: Path,
    engine_path: Path,
    warmup: int,
    samples: int,
    use_cuda_graph: bool,
) -> dict:
    manifest = load_manifest(manifest_path)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("could not deserialize TensorRT engine")
    context = engine.create_execution_context()
    cuda = Cuda()
    stream = ctypes.c_void_p()
    cuda.check(cuda.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    host_inputs: dict[str, np.ndarray] = {}
    host_ptrs: dict[str, ctypes.c_void_p] = {}
    device_inputs: dict[str, ctypes.c_void_p] = {}
    host_output = np.empty((1, 29), dtype=np.float32)
    host_output_ptr = cuda.malloc_host(host_output.nbytes)
    device_output = cuda.malloc(host_output.nbytes)
    graph = ctypes.c_void_p()
    graph_exec = ctypes.c_void_p()
    rng = np.random.default_rng(184250)
    try:
        for item in manifest["inputs"]:
            name = item["name"]
            shape = tuple(item["shape"])
            host = np.zeros(shape, dtype=np.float32)
            if name == "gumbel_noise":
                host[...] = -np.log(-np.log(rng.uniform(1e-6, 1.0 - 1e-6, shape)))
            host_inputs[name] = host
            host_ptrs[name] = cuda.malloc_host(host.nbytes)
            ctypes.memmove(host_ptrs[name], host.ctypes.data, host.nbytes)
            device_inputs[name] = cuda.malloc(host.nbytes)
            context.set_tensor_address(name, int(device_inputs[name].value))
        context.set_tensor_address("action", int(device_output.value))

        def enqueue() -> None:
            for name, host in host_inputs.items():
                cuda.check(
                    cuda.lib.cudaMemcpyAsync(
                        device_inputs[name],
                        host_ptrs[name],
                        host.nbytes,
                        CUDA_HOST_TO_DEVICE,
                        stream,
                    ),
                    "input copy",
                )
            # TensorRT 10.x Python exposes the v3 enqueue API as
            # ``execute_async_v3``. The C++ API calls the same operation
            # enqueueV3; do not use the C++ spelling here because it makes the
            # Python benchmark version-dependent.
            stream_handle = int(stream.value)
            if hasattr(context, "execute_async_v3"):
                ok = context.execute_async_v3(stream_handle)
            elif hasattr(context, "enqueue_v3"):
                ok = context.enqueue_v3(stream_handle)
            else:
                raise RuntimeError("TensorRT binding has no v3 async execution API")
            if not ok:
                raise RuntimeError("TensorRT v3 execution failed")
            cuda.check(
                cuda.lib.cudaMemcpyAsync(
                    host_output_ptr,
                    device_output,
                    host_output.nbytes,
                    CUDA_DEVICE_TO_HOST,
                    stream,
                ),
                "output copy",
            )

        if use_cuda_graph:
            for _ in range(3):
                enqueue()
                cuda.check(cuda.lib.cudaStreamSynchronize(stream), "warmup synchronize")
            cuda.check(cuda.lib.cudaStreamBeginCapture(stream, 2), "begin capture")
            enqueue()
            cuda.check(
                cuda.lib.cudaStreamEndCapture(stream, ctypes.byref(graph)),
                "end capture",
            )
            cuda.check(
                cuda.lib.cudaGraphInstantiateWithFlags(
                    ctypes.byref(graph_exec), graph, 0
                ),
                "graph instantiate",
            )

        def run_once() -> int:
            if "gumbel_noise" in host_inputs:
                host_inputs["gumbel_noise"][...] = -np.log(
                    -np.log(
                        rng.uniform(1e-6, 1.0 - 1e-6, host_inputs["gumbel_noise"].shape)
                    )
                )
            start = time.perf_counter_ns()
            if use_cuda_graph:
                cuda.check(cuda.lib.cudaGraphLaunch(graph_exec, stream), "graph launch")
            else:
                enqueue()
            cuda.check(cuda.lib.cudaStreamSynchronize(stream), "stream synchronize")
            return time.perf_counter_ns() - start

        for _ in range(warmup):
            run_once()
        durations = [run_once() for _ in range(samples)]
        report = {
            "status": "ok",
            "contract": manifest["contract"],
            "sampling_mode": manifest["sampling_mode"],
            "top_p": manifest.get("top_p"),
            "provider": (
                f"TensorRT {trt.__version__} + CUDA graph"
                if use_cuda_graph
                else f"TensorRT {trt.__version__} + CUDA stream"
            ),
            "model_sha256": sha256(model),
            "engine_sha256": sha256(engine_path),
            "samples": samples,
            "latency_us": {
                "mean": float(np.mean(durations) / 1000.0),
                "p50": percentile(durations, 0.50) / 1000.0,
                "p99": percentile(durations, 0.99) / 1000.0,
            },
            "output_finite": bool(np.isfinite(host_output).all()),
        }
        return report
    finally:
        if graph_exec.value:
            cuda.lib.cudaGraphExecDestroy(graph_exec)
        if graph.value:
            cuda.lib.cudaGraphDestroy(graph)
        for pointer in device_inputs.values():
            cuda.lib.cudaFree(pointer)
        for pointer in host_ptrs.values():
            cuda.lib.cudaFreeHost(pointer)
        cuda.lib.cudaFree(device_output)
        cuda.lib.cudaFreeHost(host_output_ptr)
        cuda.lib.cudaStreamDestroy(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--workspace-mib", type=int, default=4096)
    parser.add_argument("--builder-optimization-level", type=int, default=3)
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="opt into TF32 tactics (not recommended for discrete GPC parity)",
    )
    parser.add_argument("--no-tf32", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--fp16", action="store_true", help="enable TensorRT FP16 tactics"
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--plugin-library",
        type=Path,
        help="shared TensorRT plugin library (required by the top-p graph)",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    if manifest["sampling_mode"] == "topp" and args.plugin_library is None:
        raise ValueError(
            "top-p requires --plugin-library so TensorRT can load TopPFilter"
        )
    # Keep the library loaded while TensorRT uses its registered plugins.
    _plugin_handle = load_plugin_library(args.plugin_library)
    if args.build or not args.engine.exists():
        build_engine(
            args.model,
            args.manifest,
            args.engine,
            args.workspace_mib,
            args.builder_optimization_level,
            args.tf32 and not args.no_tf32,
            args.fp16,
        )
    if args.skip_benchmark:
        print(f"TensorRT engine ready: {args.engine}", flush=True)
        return 0
    report = benchmark(
        args.model,
        args.manifest,
        args.engine,
        args.warmup,
        args.samples,
        not args.no_cuda_graph,
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
