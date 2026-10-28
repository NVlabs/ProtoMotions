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
loads weights strictly and checks all four outputs against PyTorch using ONNX
Runtime. The graph includes observations, the prior/decoder and action
processing. Outputs are raw `actions`, `joint_pos_targets`, `stiffness_targets`
and `damping_targets`.

Add `--leapp` to also write a LEAPP bundle with the same graph and complete typed
contract under `export/leapp/`. See [LEAPP export](leapp.md) for loading it and
binding the host-owned histories. This does not convert the graph to the fixed
39-input native v1 interface.

## Inputs and Ownership

The YAML lists the actual graph inputs, their shapes, dtypes and providers.
Noisy sensor names are preserved: `noisy.*` does not become `current.*`.
`current.*` means simulator truth, not an on-robot sensor estimate.

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
supplies reconstructed `noisy.rigid_body_pos` and `noisy.rigid_body_rot` from its
joint/IMU/localization readings; grounding and goal conversion happen inside
the graph.

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
and output names and uses full sensor-reconstructed body poses. The existing
native v1 runtime, policy preparation, FK graph and demo assets are unchanged.
Do not put this graph into that launcher without a matching host implementation.

### Scene Support: Released Artifacts Versus Re-export

| Path | Scene behavior |
| --- | --- |
| Released PyTorch weights | Keep the trained scene-token modules; a host can supply visible scene features at the model boundary. |
| Maintainer-provided 39-input ONNX | Keeps original scene preprocessing and scene inputs. LEAPP packaging preserves those exact bytes. |
| Generic re-export of the packaged checkpoint | Uses the saved dummy scene observations: zero corner tokens and masks always false. No scene inputs are exported. |
| Fresh public `g1_student.py` recipe | Does not include a scene-token stream. |

For a **scene-disabled** re-export of the packaged policy:

```bash
python deployment/export_to_onnx.py \
  --checkpoint data/pretrained_models/metatron/G1/inference_last.ckpt \
  --output results/metatron-scene-disabled/export --leapp
```

This does not remove scene weights from the PyTorch checkpoint or modify the
separately released ONNX. It does not implement a scene provider or SAMP training.
Use the maintainer-provided scene-capable artifact with a matching host for
object interaction; do not replace it with this re-export.

The packaged checkpoint re-exports with 31 inputs, including `current.dof_pos`
as the dummy-scene functions' batch/dtype source. The fresh public recipe has
30 inputs. Neither count is a feature-vector width. The additional tensor does
not supply object geometry or change the sensor-estimated proprioception.

Tests cover JSON config reconstruction, strict checkpoint loading, disabled
modalities, boolean masks, history/reset parity against `StatePublisher`, and
PyTorch/ONNX agreement at batch sizes one and three. This validates the export
contract, not real-robot safety or recovery performance.

See [released-policy verification](metatron_export_verification.md) for the
actual checkpoint and pre-exported ONNX comparison, including scene inputs.
