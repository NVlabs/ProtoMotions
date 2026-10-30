# Metatron Through the Generic Exporter

This path exports Metatron from its saved JSON config bundle and checkpoint,
including the public `examples/experiments/metatron/g1_student.py` recipe and
the `data/pretrained_models/metatron/G1` checkpoint fetched with
`python scripts/download_metatron.py`. It runs the learned
**prior and decoder**, with
zero latent noise. It does not run the privileged encoder, load the teacher, or
start a simulator.

```bash
python deployment/export_to_onnx.py \
  --checkpoint results/my_metatron/last.ckpt \
  --output results/my_metatron/export
```

The output is `unified_pipeline.onnx` plus `unified_pipeline.yaml`. The exporter
loads weights strictly and checks every graph output against PyTorch using ONNX
Runtime. The graph includes observations, the prior/decoder and action
processing. Its outputs are exactly the four control outputs: raw `actions`,
`joint_pos_targets`, `stiffness_targets` and `damping_targets`. Every output is
a command; nothing has to be skipped. When the recipe binds an FK kernel, as the
public `g1_student.py` does, the graph computes the robot's current body poses
inside the goal observation from the joint angles and the anchor pose; the poses
are neither graph inputs nor graph outputs. A host that needs the body poses
outside the policy call runs a second, small graph, `fk.onnx`, which
`deployment/export_fk_to_onnx.py` exports from the same kernel object the policy
traces (described below). A plain tracker binds no FK kernel, so it has no FK
graph to export.

Add `--leapp` to also write a LEAPP bundle with the same graph and complete typed
contract under `export/leapp/`. See [LEAPP export](leapp.md) for loading it and
binding the host-owned histories. This does not convert the graph to the fixed
39-input native v1 interface.

## Inputs and Ownership

The YAML lists the actual graph inputs, their shapes, dtypes and providers.
Noisy sensor names are preserved: `noisy.*` does not become `current.*`.
`current.*` means simulator truth, not an on-robot sensor estimate.

The exporter is told who will feed the graph. The default, `--host robot`, is
the sensor definition plus the goal streams and the policy's own past actions;
a binding to the simulator's body state is refused, naming the kind and the
host that could feed it. `--host simulation` is for a game engine, a browser
MuJoCo or a sim-to-sim check: it also accepts the simulator's own body poses,
velocities and ground heights under `current.*` (the `simulated_*` kinds), and
the YAML records `host: simulation`. The deployable Metatron recipe is a robot
export; export it with the default even while testing it in a simulator.

The `masked_mimic` block is the trained control's `GoalCanvas`: body order,
anchor, future slots, style frame count and enabled modalities. A binding to a
disabled modality fails at export. Do not infer body order from a robot diagram
or sort the names. The graph's body-pose tensors use `robot.body_names`; packed
goal masks and style features use `masked_mimic.conditionable_bodies`.

Here `E` is batch size, `K` masked future slots, `N` all robot bodies, `B`
conditionable bodies and `F` style frames. Units are metres, seconds and radians;
quaternions are XYZW. Every goal input has provider `masked_mimic`, supplied by
the host's task controller, not its sensor estimator.

| Context input under `masked_mimic.` | Shape | Meaning |
| --- | --- | --- |
| `ref_pos`, `ref_rot` | `[E,K,N,3/4]` | Global target poses in the placed world frame |
| `local_ref_pos`, `local_ref_rot` | `[E,K,N,3/4]` | Also raw poses, **not pre-localized**; the graph removes each target's anchor XY and heading |
| `time_offsets`, `local_time_offsets` | `[E,K]` | Seconds until each target, not absolute timestamps |
| `target_poses_masks`, `local_target_poses_masks` | `[E,K]` | Frame visibility |
| `target_bodies_masks`, `local_target_bodies_masks` | `[E,K*B*2]` | Flattened time, body, translation/rotation visibility |
| `anchor_translation_mask` | `[E,K,2]` | Anchor XY and Z visibility |
| `anchor_rotation_mask` | `[E,K,2]` | Full rotation and heading visibility; full rotation takes precedence |
| `style_pose_samples` | `[E,F*B*9]` | Position plus six tangent/normal rotation features per body |
| `style_velocity_samples` | `[E,F*B*6]` | Linear then angular velocity per body |
| `style_condition_masks` | `[E,F,2*B+1]` | Per-body translation/rotation bits, then one velocity bit |
| `style_frame_masks` | `[E,F]` | Style frame visibility |
| `average_velocity` | `[E,2]` | World-frame anchor XY velocity, m/s |
| `velocity_command_mask` | `[E,1]` | Velocity-command visibility |

All masks are boolean, with **true meaning visible**. Style poses use each
sample's own anchor XY/heading and lowest body height; style velocities use
its anchor-heading frame. The host supplies already masked style features,
as the training publisher does. `compute_style_pose_samples`,
`compute_style_velocity_samples` and `apply_style_condition_masks` specify
these operations in `protomotions/envs/obs/masked_mimic.py`.

The sensor-only recipe compares global target and current pose heights relative
to their respective lowest body origins. That is not terrain height. The host
supplies only its raw readings: the joint positions, the IMU orientation and
the anchor position from localization (`noisy.dof_pos`, `noisy.anchor_rot`,
`noisy.anchor_pos`). The graph computes the current body poses from them with
the recipe's FK kernel; grounding and goal conversion happen inside the graph
too. Body poses are not graph inputs of this robot export.

## History Timing

Immediately before policy call `t`, the measured state is `s(t)`:

| Input | Contents |
| --- | --- |
| `current_action` | Applied raw action `a(t-1)` |
| `raw_action_history[:,j]` when bound | `a(t-j-1)` |
| `noisy_historical.<measurement>[:,j]` | Measurement from `s(t-j-1)` |
| `noisy_historical.raw_actions[:,j]` | Its incoming action `a(t-j-2)` |

Thus the first past state pairs with the action **entering** it, not the action
leaving it. Reset repeats the first measured state and zeros the command history.
The full ring is supplied; selection such as `[8,4,3,2,1]` stays in the graph.

`deployment.input_history.InputHistory` implements this schedule with NumPy:
Use this helper for the generic Metatron contract. The tracker-specific
`ActionFeedback`/`action_history_entries` readers in `tracker_inputs.py` only
handle the older tracker history keys, not `current_action` or measurement history.

```python
history = InputHistory(contract.inputs)
history.reset(first_measurements)
# Each policy call:
context = {**measurements, **task_goals, **history.inputs()}
feed = {spec.onnx_name: context[spec.key] for spec in contract.inputs}
outputs = session.run(None, feed)
# Apply the command, including any controller edits, then record what was used.
history.commit(measurements, applied_outputs)
```

`applied_outputs` uses `actions` for raw commands and `joint_pos_targets` for
processed commands when those histories are bound. Convert any target-space
controller edit back to raw-action coordinates before recording raw history.
Do not record a rejected action or update history at the faster physics rate.

## Separate From the Existing Demo Bundle

This is **not** the fixed 39-input Metatron v1 manifest. It has different input
and output names and computes the body poses inside the graph from the raw
readings. The existing native v1 runtime, policy preparation and demo assets are
unchanged. That runtime runs FK on the current measurement before every policy
call: its goal canvas starts from the measured pose, and steering and recovery
edit it. It gets those poses from `fk.onnx`, the policy's own FK kernel traced
on its own, rebuilt from the same config bundle beside the checkpoint:

```bash
python deployment/export_fk_to_onnx.py \
  --checkpoint results/my_metatron/inference_last.ckpt
```

This writes `compiled_models/fk.onnx` and `fk.yaml` beside the checkpoint;
`--output` picks another directory. The inputs are `noisy_dof_pos` `[1,29]`
(one value per G1 joint), `noisy_anchor_pos` `[1,3]` and `noisy_anchor_rot`
`[1,4]` (xyzw). They are the policy graph's names for the same paths, so a host
feeds one tensor under one name to both graphs. The batch is fixed to 1 because
the plan serves one robot. The outputs are `body_pos` `[1,N,3]` and `body_rot`
`[1,N,4]` (xyzw) in `robot.body_names` order. The exporter checks ONNX Runtime
against the kernel on random joint angles within the limits and random anchor
poses, refuses data-dependent ops (`NonZero`, `If`, `Loop`, `Scan`), refuses a
bundle whose actor observations bind no FK kernel ("nothing to export") and
refuses two observations whose kernels or paths differ. The sidecar `fk.yaml`
records the body names, the joint names, the anchor body (name and index), the
input and output names with their shapes and the checkpoint file name.

Because `fk.onnx` is the policy's kernel, there is one FK implementation and the
two graphs agree by construction. The previous standalone FK exporter re-ran
pose_lib's forward kinematics on the MJCF with its own quaternion conversion, a
second FK implementation that could drift from the policy's kernel. The policy's
output list stays homogeneous.

```bash
python deployment/native/tools/build_metatron_fk_tensorrt.py \
  --onnx fk.onnx --sidecar fk.yaml --output fk.plan --manifest manifest.json
python deployment/native/tools/make_metatron_fk_fixture.py \
  --onnx fk.onnx --sidecar fk.yaml --output fixture.bin --samples 20
```

The first command compiles `fk.onnx` to a static FP32 TensorRT plan and, with
`--manifest`, checks the sidecar's body order, joint order and anchor against
the policy manifest's `robot_contract` (a mismatch is refused with "the FK
graph's body order differs from the manifest"). The second writes the parity
fixture. `body_rot` comes from a matrix-to-quaternion conversion, so `q` and
`-q` are the same rotation; a comparison must allow the sign. See the
[native qualification guide](metatron_v1_native_qualification.md). Do not put
the generic policy graph into that launcher without a matching host
implementation.

### Scene Support: Released Artifacts Versus Re-export

| Path | Scene behavior |
| --- | --- |
| Released PyTorch weights | Keep the trained scene-token modules; a host can supply visible scene features at the model boundary. |
| Maintainer-provided 39-input ONNX | Keeps original scene preprocessing and scene inputs. LEAPP packaging preserves those exact bytes. |
| Generic re-export of the packaged checkpoint, or of any configuration that binds the `scene_boxes` stream (`SceneBoxesControl` in `control_components`, `deployable_scene_box_observations` in the recipe) | Exports the eight `scene_boxes_*` inputs of the maintainer-provided graph (centres, extents, object pose, validity, visibility, the two pose-bias offsets) with the `scene` provider; the corners are computed inside the graph in the root heading frame, the root pose from the FK kernel, so no body pose is an input. A host with no scene feeds the empty scene (every box invalid and hidden). The pinned package revision binds it. |
| Fresh public `g1_student.py` recipe | Does not include a scene-token stream. |

To re-export the packaged policy:

```bash
python deployment/export_to_onnx.py \
  --checkpoint data/pretrained_models/metatron/G1/inference_last.ckpt \
  --output results/metatron-generic/export --leapp
```

This does not modify the separately released ONNX, whose host contract differs
(the simulator's root pose and a nine-slot action history instead of
`current_action` and the eight-slot history). It does not implement SAMP
training; the `scene_boxes` stream above is the exporter's scene provider, fed
by the host. The maintainer-provided scene-capable artifact stays the one for
its paired browser and native hosts.

The fresh public recipe has 29 inputs and four outputs. The packaged checkpoint
re-exports with 37: those 29 plus the eight scene inputs. The package pinned in
`data/pretrained_models/metatron/huggingface.json` binds the FK goal observation
(`fk_masked_mimic_target_poses`) to raw joint readings and the torso anchor pose.
Run `python scripts/download_metatron.py` to fetch that revision. If an older
download is present, move its `G1/` directory aside first: the downloader
deliberately refuses to overwrite different local files. Older saved bundles
binding `noisy.rigid_body_pos`/`noisy.rigid_body_rot` remain refused at export;
the weights did not change. Neither count is a feature-vector width. The
eight additional inputs are the host-supplied object geometry described above;
they do not change the sensor-estimated proprioception.

Tests cover JSON config reconstruction, strict checkpoint loading, disabled
modalities, boolean masks, history/reset parity against `StatePublisher`,
PyTorch/ONNX agreement at batch sizes one and three, the in-graph FK kernel's
world poses against the sensor runtime's poses, the policy graph's output list
being the four control outputs, and `fk.onnx` against the kernel the policy
traces (equal to float rounding, inputs named as the policy names the same
paths, no data-dependent ops, a sidecar that refuses a manifest whose body
order differs, and a plain tracker refused with nothing to export). This
validates the export contract, not real-robot safety or recovery performance.

See [released-policy verification](metatron_export_verification.md) for the
actual checkpoint and pre-exported ONNX comparison, including scene inputs.
