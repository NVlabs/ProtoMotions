# Metatron Export Verification

The release has two intentionally different paths:

- **Scene-capable artifacts:** mapped PyTorch weights and the maintainer-provided
  39-input ONNX. The host must supply scene measurements.
- **Generic re-export:** the saved configuration reads the scene from the
  host through the `scene_boxes` stream (published empty in the simulator by
  `SceneBoxesControl`). The pinned package exports 37 inputs, the eight
  `scene_boxes_*` inputs among them, and four control outputs, with no
  host-supplied body poses.

The checks below verify that the generic graph agrees with the maintainer's
graph, scene inputs included, without a scene provider in ProtoMotions or
checkpoint compatibility code.

## Artifacts Tested

Scene-revision verification date: 2026-10-29. These are actual trained
weights, not reduced-width test models. The package is pinned to Hub revision
[`ada816723d62c194c82ffd8fab97a3d8a001f94d`](https://huggingface.co/nvidia/Metatron-v1/commit/ada816723d62c194c82ffd8fab97a3d8a001f94d),
the second revision after the FK rebind
([`a571317ebab6abfe1bc55e3a51a6619439f06c79`](https://huggingface.co/nvidia/Metatron-v1/commit/a571317ebab6abfe1bc55e3a51a6619439f06c79)).
Only `G1/experiment_config.py`, `G1/resolved_configs.json`, `G1/MODEL_CARD.md`
and their three entries in the functional Hub index `config.json` changed. The
index preserves the newer external-data browser artifacts as well as the
original demo/racer records. Its verified size/hash are recorded in
`data/pretrained_models/metatron/huggingface.json`. The weights, launch
config and browser graphs are unchanged, and old Hub revisions remain
addressable.

The saved goal now binds `BakedAnchorFK` to joint readings and the torso anchor
pose. Body poses are no longer policy inputs; the policy retains its four
control outputs. A host needing body poses outside the call runs `fk.onnx`,
exported from the same kernel by `deployment/export_fk_to_onnx.py`.

| Artifact | SHA-256 |
| --- | --- |
| Native `metatron/G1/inference_last.ckpt` (download with `python scripts/download_metatron.py`) | `e5fc2e47a8f9ee9b96a674cabbf236b3c510f887427645a14bc1779b10f0b43a` |
| Pinned epoch-6410 scene-capable `model.onnx` | `7180e82cae7ec04ff7766c262d430f57573d93678e025d8a71916885072a0cd4` |

The packaged checkpoint contains 278 tensors. Its encoder/prior/trunk contain
14/27/6 modules. Shared frozen time-normalization statistics were copied into
separate stream processors; the learned numeric values were not retrained.
The existing mapped checkpoint is loaded strictly. No paired-output processor
or second remapping is needed for export.

## Published scene-revision checks (2026-10-29)

The saved configuration adds `"scene_boxes": SceneBoxesControlConfig()` (one
object, two boxes: the network's scene-token capacity) and binds the scene
pair through `configure_observations(robot_cfg, cfg, scene_boxes=True)`, so
`scene_box_corners` is `compute_fk_scene_box_corners` on the raw joint
readings, the anchor pose and the eight `scene_boxes.*` names with the goal's
`BakedAnchorFK`, and `scene_box_mask` is visible AND valid. Regenerating the
bundle from that recipe with the package's own `config.yaml` arguments gives
19 differing JSON leaves against the FK-rebind bundle: those two observations,
the new control and the nonfunctional package-version header. The published
config's SHA-256 is
`7fb1e13884292976746647593451541905cf2018849643ddb57f9d814baddc3d`.

`deployment/export_to_onnx.py` on the downloaded package gives 37 inputs: the
FK-rebind export's 29 inputs other than `current_dof_pos` (which only sized the
dummy scene tokens and has no reader now) plus the eight `scene_boxes_*`
inputs, with the maintainer graph's names, trailing dims and dtypes (`[1, 2,
3]` centers and extents, BOOL `[1, 2]` valid and `[1, 1]` visible, `[1, 3]`
and `[1, 4]` object pose, `[3]` position offset, scalar yaw offset). The
exporter's own check: raw action `2.03e-6`, joint targets `8.94e-7`, gains
exact.

The new graph was compared with the pinned 39-input `demo/model.onnx` on 64
MuJoCo playback states (a full reset at step 31), the same scene fed to both
graphs in four cases placed around the robot:

| Case | Raw action | Joint targets | Gains | Boxes move the action by |
| --- | ---: | ---: | --- | ---: |
| hidden | `6.20e-6` | `2.74e-6` | exact | 0 |
| one box | `7.15e-6` | `3.10e-6` | exact | 2.69 |
| two boxes | `1.11e-5` | `3.58e-6` | exact | 2.58 |
| two boxes, pose bias | `8.11e-6` | `4.53e-6` | exact | 2.55 |

The placeholder control's empty scene and an explicitly hidden scene give
identical actions. Driving the environment with the exported graph's actions
(600 headless steps with a reset, and 400 steps with two visible boxes) kept
the robot up from the crouched start, pelvis between 0.50 and 0.86 m.

## Published FK-rebind checks (2026-10-29)

Regenerating the saved bundle from its unchanged recipe reproduced the
published configuration, apart from the nonfunctional package-version header
(`3.1` versus `source`). The only functional change from the old bundle is the
`fk_masked_mimic_target_poses` binding. The published config's SHA-256 is
`3919e9f4a30072cd708995293d3301388b4248a4c39a9604668b5dcc532bc5e0`, the same
bytes that were verified before the upload.

Both exporters ran on the downloaded published package. The policy graph has
30 inputs and the four control outputs; the FK graph has 727 nodes, three
inputs (`noisy_dof_pos`, `noisy_anchor_pos`, `noisy_anchor_rot`) and two
outputs (`body_pos [1,33,3]`, `body_rot [1,33,4]`). The anchor is `torso_link`
(index 16). ONNX Runtime matched the FK kernel within `2.38e-7`.

A 64-step MuJoCo playback, with a full reset at step 31, compared the new goal
observation against the old kernel fed the sensor runtime's body poses. The
maximum difference was `5.37e-7`. FK versus runtime positions differed by
`1.53e-5 m` near world coordinate 180 m (float32 spacing at that scale);
rotation error `1 - abs(q1 · q2)` was `2.38e-7`.

The new graph and unchanged 39-input browser graph were also compared on
64 rollout states with scenes masked. Shared named inputs were identical;
the old graph's root inputs came from the simulator, and its nine action slots
were packed as the current action followed by eight historical slots.

| Pinned / new output | Maximum absolute difference |
| --- | ---: |
| `raw_action` / `actions` | `7.63e-6` |
| `action` / `joint_pos_targets` | `3.70e-6` rad |
| `stiffness` / `stiffness_targets` | `0` |
| `damping` / `damping_targets` | `0` |

The pre-upload check of the same configuration on 2026-10-28 measured
`6.44e-6` raw action and `2.80e-6` joint targets on the same comparison. Both are below `1e-4`;
byte-identical graph exports across different toolchains are not required.
The new export's own PyTorch/ONNX check measured `2.62e-6` raw action and
`1.13e-6` joint targets, with identical gains.

The [native qualification](metatron_v1_native_qualification.md) records the
TensorRT FK GPU parity and the native compile/CTest results. These checks do
not exercise scene-conditioned interaction or requalify the policy plan.

## Earlier empty-scene re-export (2026-10-27)

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

## Scene-capable release checks (2026-10-27)

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

`protomotions/tests/test_metatron_pretrained.py` checks the published scene
bindings (the `scene_boxes` control, the corners kernel on the goal's FK kernel
and the raw readings, the mask kernel), the raw-sensor FK goal binding, and
strict packaged-weight loading.
With LFS payloads present, it probes visible/hidden scene responses and runs
64 headless MuJoCo playback steps with a midpoint reset. The scene probe is at
the model-input boundary; the physical rollout still runs the control's empty scene.

`protomotions/tests/test_metatron_generic_export.py` covers generic export,
LEAPP execution, dynamic batches, goal masks and applied-action history with
small models. `deployment/native/tests/test_leapp_bundle.py` covers native graph
packaging and validation.

These results establish numerical agreement and preserved scene conditioning.
They do **not** establish sitting success, closed-loop furniture interaction,
new training support, physical-robot safety, or a newly qualified TensorRT
policy engine. Only the separate FK engine was requalified on 2026-10-29.
The native launcher still exposes idle, steering and recovery, not a
scene-interaction task. Use a matching scene-capable host for the pinned graph.
