# Metatron v1 native qualification

The policy measurements below are historical upstream results for the pinned
epoch-6410 model. The FK measurements and native build checks were rerun on
2026-10-29 against the published FK-rebound package. Use the dedicated
Metatron v1 launcher and requalify GPU-local plans. Physical Unitree operation
remains unqualified; simulator success does not establish hardware safety.

## Action contract and the offset regression

The exported `action` output is a **processed joint-position target**. The
`raw_action` output is the network action before the exported action module.
For this pinned artifact the output relation is
`action[i] = action_map.offset[i] + action_map.scale[i] * raw_action[i]`.
The motor command uses `action[i]` directly. Applying the offset again would
be a defect. The native controller checks that relation on every inference,
and startup validates joint order, action-map size, and input/output layout.

Only after the backend accepts a position command does the controller update
history. It converts the accepted, potentially limited joint target to the
raw domain as `(accepted[i] - offset[i]) / scale[i]`. A repeated transport
hold does not create a fictitious policy step. This matches the action module
and historical-input ordering in the current ProtoMotions main branch, rather than treating
`raw_action` as a motor target.

## Inference and FK parity

The pinned source ONNX is unchanged. The CUDA-graph-compatible policy ONNX
replaces only the two masked-positive-square-root subgraphs whose Boolean
indexing creates dynamic shapes. The rewrite is value-identical to the source
ONNX on 18 reference feeds. The TensorRT 10.9 FP32 CUDA-graph plan produced
these worst absolute output differences against ONNX Runtime CPU on those feeds:

| Output | Maximum difference |
| --- | ---: |
| Processed action | 0.001754 rad |
| Raw action | 0.005002 normalized units |
| Stiffness and damping | 0 |
| Internal affine relation | 1.20e-7 rad |

The FK engine runs `fk.onnx`, the small graph that
`deployment/export_fk_to_onnx.py` exports from the same `BakedAnchorFK` kernel
object the policy graph (`unified_pipeline.onnx`) traces, rebuilt from the same
config bundle beside the checkpoint: it takes the 29 measured joints and the
synchronized torso anchor pose (`noisy_dof_pos`, `noisy_anchor_pos`,
`noisy_anchor_rot`) and returns all 33 body positions and xyzw rotations
(`body_pos`, `body_rot`) in the robot's body order. The input names are the
policy graph's names for the same paths, so the runtime feeds one tensor under
one name to both graphs. The policy graph itself keeps its four control
outputs and computes the body poses internally; it is the very kernel the FK
graph evaluates, so there is one FK implementation and nothing to drift, and
it is not a simulator-privileged state source. The exporter fixes the batch
axis to one, refuses data-dependent ops and checks ONNX Runtime against the
kernel on random joint angles and anchor poses; the plan builder, with
`--manifest`, checks the sidecar's body order, joint order and anchor body
against the policy manifest. **Disable TF32 when building its TensorRT
plan**: on the previous standalone FK graph default TF32 caused up to
0.000481 m error in a 20-pose test. On 2026-10-29, the published package's
`fk.onnx` was compiled with TensorRT 10.9.0.34 (FP32, TF32 disabled) and
tested on an RTX 5090. Across 20 deterministic poses, with CUDA graph capture
enabled, the maximum errors against ONNX Runtime were **1.79e-7 m** for body
positions and **2.38e-7** per quaternion component. The same maxima held
without CUDA graphs. The native FK test checks all 33 body positions and rotations and handles
equivalent quaternion signs (`body_rot` comes from a matrix-to-quaternion
conversion, so `q` and `-q` are the same rotation). It runs offline; it does
not send hardware commands.

To build and qualify the FK engine from an export made with a matching
ProtoMotions installation, first export the FK graph beside the checkpoint
(from the repository root, `python deployment/export_fk_to_onnx.py
--checkpoint <dir>/inference_last.ckpt` writes `<dir>/compiled_models/fk.onnx`
and `fk.yaml`; `--output` picks another directory). The checkpoint must be
one whose deployable goal binds the FK kernel. The package pinned in
`data/pretrained_models/metatron/huggingface.json` now does; fetch it with
`python scripts/download_metatron.py`. The downloader refuses to overwrite
an older package, so move the old `G1/` directory aside before downloading.
Then, from `deployment/native`:

```bash
python tools/build_metatron_fk_tensorrt.py \
  --onnx /path/to/export/compiled_models/fk.onnx \
  --sidecar /path/to/export/compiled_models/fk.yaml \
  --manifest artifacts/metatron-ep6410/manifest.json \
  --output artifacts/metatron-ep6410/fk.fp32.plan
python tools/make_metatron_fk_fixture.py \
  --onnx /path/to/export/compiled_models/fk.onnx \
  --sidecar /path/to/export/compiled_models/fk.yaml \
  --output /tmp/metatron_fk_fixture.bin --samples 20
./build-clang/metatron_v1_fk_parity \
  artifacts/metatron-ep6410/fk.fp32.plan \
  /tmp/metatron_fk_fixture.bin --cuda-graphs
```

Both tools read `fk.onnx` and its sidecar `fk.yaml` as exported; neither
rewrites the graph. The sidecar records the body names, the joint names, the
anchor body (name and index), the input and output names with their shapes and
the checkpoint file name; the plan builder, given `--manifest`, checks the body
order, the joint order, the anchor and that the FK inputs are inputs of the
policy, so an FK graph is never paired with a policy it does not belong to. A fixture written by the previous tool (magic
`PM_FK_V1`) is refused by the parity binary, which expects `PM_FK_V2`
(joints, anchor pose, body positions, body rotations).

The plan must be loaded with the same TensorRT major/minor build family used
to create it. An engine load or graph-capture failure is a hard qualification
failure, not a reason to silently switch to another provider.

The 2026-10-29 check also built the native tree with Clang 18.1.3, Metatron
native, MuJoCo and Unitree SDK2 enabled: the 16 CTest tests that configuration
registers all passed, including the SDK2 compile and G1 sensor-contract
checks. Three tests are registered only when their files exist on disk (the
MuJoCo sensor test needs the prepared G1 MJCF; the recovery and manifest tests
need `artifacts/metatron-ep6410/recovery_plan.json` and `manifest.json`) and
were not part of that run. The build compiles the FK engine, rest/task paths
and launcher; it does not qualify physical robot operation or
rebuild/requalify the separate policy TensorRT plan.
