# Deployment qualification and benchmarks

Run from `deployment/` with its Python import path configured as described in
the [deployment README](../README.md). Model-dependent checks require locally
provided artifacts; none of these tools should send hardware commands.

| Tool | Boundary checked |
| --- | --- |
| `runtime_benchmark` | Core runtime overhead with a synthetic backend/controller |
| `mailbox_benchmark` | Latest-only mailbox overhead |
| `metatron_v1_inference_benchmark` | Native TensorRT policy inference |
| `metatron_v1_parity` | Native outputs against recorded reference feeds |
| `metatron_v1_fk_parity` | FK engine (joints + anchor pose in) against ONNX Runtime reference poses |
| `metatron_v1_rest_dump`, `metatron_v1_task_dump` | Native rest/task fixtures for Python parity tools |
| `gpc_tensorrt_benchmark.py` | Python TensorRT sampling-mode benchmark |
| `gpc_tensorrt_native_benchmark` | Native sampling inference, transfers and validation |

Build the core benchmarks with the normal CMake build. Policy benchmarks need
their optional TensorRT/CUDA targets and a compatible user-provided model.
See [Metatron v1](../docs/metatron_v1_native.md) and
[GPC](../docs/gpc_tensorrt.md) for preparation and commands.

Record GPU, driver, CUDA/TensorRT versions, model hash, precision, sampling mode,
warm-up and sample counts alongside results. Distinguish inference-only timing
from complete sensor-to-command latency. Neither proves physical safety.
