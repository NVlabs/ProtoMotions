# Metatron v1 policy contract

Metatron v1 uses a dedicated native TensorRT runtime with the exact 39-input
G1 contract. Its serialized identifier is
`protomotions.masked_mimic.g1.v5_raw_context_scene`; this identifier and the
exported tensor names are deliberately unchanged to keep existing artifacts
loadable. They are not the product name or a separate supported policy family.

The graph contains the deterministic prior and actuator mapping. Runtime task
goals use a separate FK engine. The native launcher supports idle, steering and
recovery. It does not currently expose tracking clips, object interaction,
style selection or Pico task control. Python modules provide offline reference
and contract checks, not a second live launcher.

## Prepare artifacts

The pinned candidate is epoch 6410, ONNX SHA-256
`7180e82cae7ec04ff7766c262d430f57573d93678e025d8a71916885072a0cd4`. These
weights and the recovery plan are NVIDIA-internal and are not distributed with
this repository or checked into Git. GPC likewise ships no model download
helper.

For your own compatible export, use the source revision matching the saved
checkpoint/config, then validate and package the artifacts:

```bash
python -m tools.prepare_metatron_v1_policy \
  --model /path/to/model.onnx \
  --manifest /path/to/manifest.json \
  --recovery-plan /path/to/recovery_plan.json \
  --output artifacts/my-metatron-policy
```

Build GPU-local policy/FK engines and run using the
[Metatron v1 native guide](metatron_v1_native.md). Match robot geometry,
inertial properties, PD gains, joint ordering and action mapping; matching
input dimensions alone is insufficient.

You can also [export this ONNX/manifest pair as a LEAPP bundle](leapp.md)
without retracing or changing the model bytes. Native preparation accepts the
bundle YAML through `--model` and finds its manifest automatically.

The pinned ONNX keeps its scene inputs. The corresponding released PyTorch
checkpoint also retains the scene-token weights. In contrast, generic
ProtoMotions re-export uses the checkpoint's native preset, which always emits
dummy scene tokens with false masks. That export is scene-disabled and has a
different input contract; it is not a replacement for this pinned graph. See
[export paths](metatron_generic_export.md#scene-support-released-artifacts-versus-re-export)
and [verification evidence](metatron_export_verification.md).

## State and action timing

At observation `s(t)`, `raw_action_history[0]` is incoming action `a(t-1)`.
Eight historical rows pair `s(t-k)` with `a(t-k-1)`, for `k=1..8`. State and
action histories therefore have different lengths: eight and nine. Reset
repeats the first measured state and fills incoming actions with zeros.

Inference does not advance history. `MetatronController::on_command_accepted`
converts the accepted PD target through the inverse affine action map and
commits the pre-step sensor sample. Thus command limiting is reflected in the
next observation. Repeated transport holds do not add fictitious policy steps;
unacknowledged commands prevent another inference.

## Goals and recovery

Idle and stopped steering use local calibrated body poses and zero-velocity
style, not a lock to an unobservable global position. Requested heading remains
global. Moving steering includes sparse future hand targets. Scene-object
inputs remain masked until a localization integration supplies measurements.

Recovery owns goals while active. It uses IMU tilt and FK-relative torso height
above the lowest body origin, not measured ground contact. Depending on pose,
it selects a prone prompt, seated get-up segment or height ramp, then requires
upright settling before returning control. The recovery plan must match the
model manifest. Do not treat recovery phase progression as successful recovery.

## Qualification

Source tests pin masks, sensor/history timing, accepted-action mapping, task
clearing and safety. `tools/check_metatron_history.py` can replay exported
reference traces; task inputs in that trace are replayed, not regenerated.
The separate rest/task parity tools check native goal construction against the
offline Python reference. See the [native qualification notes](metatron_v1_native_qualification.md)
for historical upstream measurements and reproduction commands.

Historical fixtures do not establish unseen-fall recovery rates or physical
robot safety. Requalify GPU-local plans and the complete control loop before
operator-supervised hardware checks.
