# Metatron v1 native qualification

These are historical upstream measurements for the pinned epoch-6410 model,
not results rerun by the source integration tests. Use the dedicated Metatron
v1 launcher and requalify GPU-local plans. Physical Unitree operation remains
unqualified; simulator success does not establish hardware safety.

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

The separate FK ONNX is exported from the same ProtoMotions kinematic
hierarchy and accepts only synchronized root pose and 29 measured joints.
It is not a simulator-privileged state source. **Disable TF32 when building
its TensorRT plan**: default TF32 caused up to 0.000481 m error in a 20-pose
test, while no-TF32 matched the ONNX reference to 2.39e-7 m and 1.79e-7
quaternion component across arbitrary orientations, with CUDA graph capture enabled. The native FK test
checks all 33 body positions and rotations and handles equivalent quaternion
signs. It runs offline; it does not send hardware commands.

To reproduce the FK export and qualification using a matching ProtoMotions
installation, choose an XML whose body order matches the manifest:

```bash
python tools/export_metatron_fk.py \
  --xml /path/to/g1_holo_compat.xml \
  --manifest artifacts/metatron-ep6410/manifest.json \
  --output artifacts/metatron-ep6410/fk.onnx
python tools/build_metatron_fk_tensorrt.py \
  --onnx artifacts/metatron-ep6410/fk.onnx \
  --output artifacts/metatron-ep6410/fk.fp32.plan
python tools/make_metatron_fk_fixture.py \
  --onnx artifacts/metatron-ep6410/fk.onnx \
  --manifest artifacts/metatron-ep6410/manifest.json \
  --output /tmp/metatron_fk_fixture.bin
./build-clang/metatron_v1_fk_parity \
  artifacts/metatron-ep6410/fk.fp32.plan \
  /tmp/metatron_fk_fixture.bin --cuda-graphs
```

The plan must be loaded with the same TensorRT major/minor build family used
to create it. An engine load or graph-capture failure is a hard qualification
failure, not a reason to silently switch to another provider.
