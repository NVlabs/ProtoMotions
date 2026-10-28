# Metatron Export Verification

The release has two intentionally different paths:

- **Scene-capable artifacts:** mapped PyTorch weights and the maintainer-provided
  39-input ONNX. The host must supply scene measurements.
- **Generic re-export:** the saved native preset emits dummy scene tokens with
  visibility always false. The resulting 31-input graph has no scene inputs.

The checks below verify that distinction without introducing a scene provider,
new scene export logic, or checkpoint compatibility code.

## Artifacts Tested

Verification date: 2026-10-27. These are actual trained weights, not the
reduced-width models used by the fast exporter tests.

| Artifact | SHA-256 |
| --- | --- |
| Native `metatron/G1/inference_last.ckpt` (download with `python scripts/download_metatron.py`) | `e5fc2e47a8f9ee9b96a674cabbf236b3c510f887427645a14bc1779b10f0b43a` |
| Pinned epoch-6410 scene-capable `model.onnx` | `7180e82cae7ec04ff7766c262d430f57573d93678e025d8a71916885072a0cd4` |

The packaged checkpoint contains 278 tensors. Its encoder/prior/trunk contain
14/27/6 modules. Shared frozen time-normalization statistics were copied into
separate stream processors; the learned numeric values were not retrained.
The existing mapped checkpoint is loaded strictly. No paired-output processor
or second remapping is needed for export.

## Empty-scene Re-export

The generic exporter produced a new ONNX and LEAPP bundle directly from the
packaged checkpoint and JSON config. The LEAPP graph is byte-identical to that
new ONNX.

We replayed 32 recorded input samples, including two resets and edited applied
actions. All four outputs were compared: raw actions, PD targets, stiffness
and damping. The new host's body poses were reconstructed from the same joint,
IMU and localization readings as the original graph, not simulator body truth.
An encoder pre-hook raised if the privileged encoder was called; none was.

| Comparison | Maximum absolute raw-action difference |
| --- | --- |
| New CPU ONNX versus pinned ONNX, scenes masked | `6.35e-6` |
| LEAPP CUDA, FP32 versus CPU PyTorch, scenes masked | `2.38e-6` |
| PyTorch CUDA versus new CPU ONNX, batch 3 | `3.76e-6` |

CPU PyTorch, ONNX Runtime and actual LEAPP `InferenceManager.run_policy` agreed
within `atol=rtol=1e-4`. The generic graph also passed batches 1 and 3. The pinned
native graph remains batch one.

## Scene-capable Release

Separately, 20 cases tested one visible box, two visible boxes, nonzero object
position/yaw estimation offsets, and hidden boxes at five recorded robot states.
The original scene preprocessing supplied corner features to the mapped PyTorch
prior. The pinned ONNX performed that preprocessing inside its graph. This
comparison does not use or add a native ProtoMotions scene provider.

| Comparison | Maximum absolute raw-action difference |
| --- | --- |
| Mapped CPU PyTorch versus pinned scene ONNX | `8.94e-6` |
| Mapped CPU PyTorch versus scene ONNX through LEAPP CUDA, FP32 | `4.41e-6` |
| Mapped CPU versus CUDA PyTorch, scene cases | `3.70e-6` |

The largest CPU PyTorch/ONNX PD-target difference was `3.79e-6` radians; gains
matched exactly. Visible boxes changed the policy actions in every visible
case. With masks false, changing the corner geometry had zero effect on the
PyTorch actions. LEAPP packaging retained the original scene-capable ONNX bytes.

For these FP32 checks, `NVIDIA_TF32_OVERRIDE=0` was set before launch. LEAPP
0.7.2 automatically selected CUDA. Its default TF32 run exceeded the same
`1e-4` tolerance (maximum observed discrepancy `1.15e-3` in the empty-scene
replay). We disabled TF32 for the comparison; no tolerance or weight was changed.

## Regression Coverage and Limits

`protomotions/tests/test_metatron_pretrained.py` checks the saved dummy-scene
bindings, scene-token dtypes and masks, and strict packaged-weight loading.
With LFS payloads present, it probes visible/hidden scene responses and runs
64 headless MuJoCo playback steps with a midpoint reset. The scene probe is at
the model-input boundary; the physical rollout still uses the empty-scene preset.

`protomotions/tests/test_metatron_generic_export.py` covers generic export,
LEAPP execution, dynamic batches, goal masks and applied-action history with
small models. `deployment/native/tests/test_leapp_bundle.py` covers native graph
packaging and validation.

These results establish numerical agreement and preserved scene conditioning.
They do **not** establish sitting success, closed-loop furniture interaction,
new training support, physical-robot safety, or a newly qualified TensorRT
engine. The native launcher still exposes idle, steering and recovery, not a
scene-interaction task. Use a matching scene-capable host for the pinned graph.
