# GPC TensorRT deployment

This is code-only support. No GPC checkpoint, ONNX weights, engine, or model
download is included. Provide your own compatible checkpoint and saved config.

The GPC exporter produces one fixed-shape graph for each sampling mode:

- `argmax`: deterministic token selection; two inputs.
- `gumbel`: Gumbel-max selection with externally supplied noise; three inputs.
- `topp`: exact top-p filtering followed by Gumbel-max selection; three inputs.

The random noise is generated outside TensorRT and supplied through a persistent
input buffer. CUDA graphs remain valid because the shapes and memory addresses
do not change. This is Gumbel-max categorical sampling, not an approximation that
turns discrete tokens into a continuous softmax action.

GPC uses FP32 TensorRT with TF32 disabled by default. TF32 can change a near-tied
logit enough to select a different discrete token, so every engine must pass a
100-case reference-parity check before deployment. The Python policy adapter enforces
`NVIDIA_TF32_OVERRIDE=0` and separates its `_notf32` cache automatically.

## Build the native targets

Use TensorRT headers and libraries from the same runtime family that will
deserialize the plans. The CUDA toolkit used to compile the plugin must be
compatible with the CUDA runtime used at launch.

```bash
cmake -S . -B build-clang-gpc \
  -DPROTOMOTIONS_BUILD_GPC_TENSORRT=ON \
  -DTENSORRT_ROOT=/path/to/TensorRT \
  -DTENSORRT_LIBRARY_ROOT=/path/to/tensorrt_libs \
  -DCUDA_RUNTIME_ROOT=/path/to/cuda_runtime \
  -DCUDA_TOOLKIT_INCLUDE_ROOT=/path/to/cuda-toolkit
cmake --build build-clang-gpc --target gpc_tensorrt_native_benchmark -j2
```

Build the exact top-p plugin when top-p is needed. Point CMake at the CUDA
12.8 toolkit (or the toolkit matching the target runtime); do not silently use a
different CUDA major/minor toolchain.

```bash
cmake -S . -B build-clang-gpc-plugin \
  -DPROTOMOTIONS_BUILD_GPC_TENSORRT=ON \
  -DPROTOMOTIONS_BUILD_GPC_TRT_PLUGIN=ON \
  -DCUDA_NVCC_EXECUTABLE=/path/to/cuda-12.8/bin/nvcc \
  -DCUDA_CUDART_ROOT=/path/to/cuda-12.8/lib64 \
  -DTENSORRT_ROOT=/path/to/TensorRT \
  -DTENSORRT_LIBRARY_ROOT=/path/to/tensorrt_libs \
  -DCUDA_RUNTIME_ROOT=/path/to/cuda_runtime \
  -DCUDA_TOOLKIT_INCLUDE_ROOT=/path/to/cuda-12.8
cmake --build build-clang-gpc-plugin --target protomotions_gpc_plugins -j2
```

## One-command preparation

`tools/prepare_gpc_policy.py` exports the selected modes, builds their native
plans, checks them against the PyTorch/ONNX reference, and records native CUDA
graph timings. It requires the top-p plugin if `topp` is selected.

```bash
python tools/prepare_gpc_policy.py \
  --experiment-dir /path/to/experiment \
  --output-dir artifacts/gpc_policy \
  --plugin-library build-clang-gpc-plugin/libprotomotions_gpc_plugins.so \
  --native-benchmark build-clang-gpc/gpc_tensorrt_native_benchmark
```

Preparation defaults to TF32-disabled FP32 engines. `--tf32` is available only
as an explicit experiment and should not be used for categorical GPC deployment
without a larger parity/behavior qualification.

Each mode is written below the output directory with `unified_pipeline.onnx`,
`unified_pipeline.json`, `unified_pipeline.plan`, parity output, and
`native_timing.txt`. The top-p directory also contains
`reference_pipeline.onnx`, which is the unmodified full-sort graph used for
numerical validation.

## Native timing boundary

The native benchmark times persistent-buffer H2D, TensorRT execution, D2H,
stream synchronization, and action validation. It excludes sensor-contract
construction and random-noise generation, which belong immediately outside this
policy boundary in the control loop.

```bash
# argmax
./build-clang-gpc/gpc_tensorrt_native_benchmark \
  artifacts/gpc_policy/argmax/unified_pipeline.plan 1000 \
  --mode argmax --cuda-graphs

# Gumbel-max
./build-clang-gpc/gpc_tensorrt_native_benchmark \
  artifacts/gpc_policy/gumbel/unified_pipeline.plan 1000 \
  --mode gumbel --cuda-graphs

# exact top-p, here using the configured p value from the manifest
./build-clang-gpc/gpc_tensorrt_native_benchmark \
  artifacts/gpc_policy/topp/unified_pipeline.plan 1000 \
  --mode topp --cuda-graphs \
  --plugin-library build-clang-gpc-plugin/libprotomotions_gpc_plugins.so
```

## Integration issue ledger

These are recorded because they affect future policy integration and should be
checked by preparation tooling rather than rediscovered at deployment time.

1. Dynamo exports may place weights in an adjacent ONNX external-data file.
   TensorRT must parse the ONNX by path, with that sidecar present; reading only
   `read_bytes()` is not sufficient for an ONNX build.
2. TensorRT's built-in TopK limit on this runtime is below GPC's vocabulary size
   of 6561. Truncating the vocabulary would change top-p semantics, so the exact
   full sort/filter is implemented as a TensorRT plugin.
3. The deployment-only `TopPFilter` node is intentionally outside the standard
   ONNX operator set. ONNX checker failure is expected; plugin registration and
   TensorRT deserialization are the validation points.
4. The plugin is registered in the default TensorRT namespace. The ONNX node,
   manifest, and creator must use the same namespace.
5. The plugin uses dynamic shared memory and must opt in at enqueue time.
6. A plugin compiled with CUDA 13.1 produced PTX that the CUDA 12.8 runtime
   rejected. Compile the plugin with the runtime-compatible toolkit and explicit
   `sm_120` code generation.
7. TensorRT plan serialization is runtime-version-specific. A plan serialized
   with version tag 239 cannot be loaded by a runtime expecting tag 240 even
   when both report TensorRT 10.x. The builder and native launcher must use the
   same TensorRT runtime package.
8. Python TensorRT calls use `execute_async_v3`; the native API is `enqueueV3`.
   Qualification must exercise the native executable, not only the Python
   wrapper.
9. Exact top-p costs more than argmax/Gumbel because it performs a full
   vocabulary sort at each of eight autoregressive steps. This is expected
   algorithmic cost, not a CPU fallback.
10. Parity is mandatory: compare native TensorRT outputs against the reference
    graph before recording latency or deploying a plan. The preparation helper
    runs 100 cases by default.
11. The exported GPC graph applies ProtoMotions' `bm_pd_action` transform
    before returning the 29-DOF command. The previous-action input is therefore
    also in processed joint-position-target space; adapters must not scale the
    output a second time.
