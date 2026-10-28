# LEAPP export and deployment

[LEAPP](https://github.com/nvidia-isaac/leapp) describes an inference pipeline
with an ONNX graph and a YAML interface: tensor names, shapes, dtypes, semantic
roles and joint ordering. The matching ProtoMotions contract travels with the
graph: a typed YAML sidecar for the generic exporter, or the JSON manifest for
the fixed 39-input native Metatron v1 runtime. These are different interfaces;
LEAPP packaging does not convert one into the other.

## Export a main-next checkpoint

From the repository root, install `leapp==0.7.2` alongside the ONNX export
dependencies, then run:

```bash
python deployment/export_to_onnx.py \
  --checkpoint results/my_metatron/last.ckpt \
  --output results/my_metatron/export \
  --leapp
```

This loads the saved JSON config bundle and weights, exports the deterministic
prior/decoder (not the privileged encoder), and checks PyTorch/ONNX parity.
Alongside the usual `unified_pipeline.onnx` and YAML contract, it writes:

```text
leapp/
  policy.onnx
  policy.yaml
  deployment_contract.yaml
```

The LEAPP graph is byte-identical to the ONNX export. Its inputs come from the
actual observation bindings, not a fixed input count. The current public G1
Metatron recipe has 30 inputs, including global/local targets, style, velocity,
anchor submasks, sensor estimates and history. It has no scene-box inputs. The
existing scene demo below has 39 inputs; neither count is a feature-vector width.
Re-exporting the native `metatron/G1` checkpoint (download with
`python scripts/download_metatron.py`) instead has 31 inputs:
its two dummy scene tokens are always masked, with `current.dof_pos` supplying
their batch/dtype. That graph is scene-disabled even though its source weights
retain scene conditioning. See the [artifact distinction](metatron_generic_export.md#scene-support-released-artifacts-versus-re-export).

Each LEAPP port retains its complete ProtoMotions declaration, including
provider, body/joint ordering and history offsets. The sidecar also retains the
goal canvas, robot, controller and sensor contract. Goals and raw actions do not
receive misleading physical-sensor or joint-target labels.

Validate the bundle before binding a host to it:

```python
from leapp import InferenceManager
from deployment.input_history import InputHistory
from deployment.native.tools.leapp_export import load_pipeline_contract

contract = load_pipeline_contract(bundle_path)  # pathlib.Path to policy.yaml
manager = InferenceManager(str(bundle_path))
history = InputHistory(contract.inputs)
history.reset(first_measurements)

# At each control tick, measurements and task_goals are keyed by context path.
context = {**measurements, **task_goals, **history.inputs()}
outputs = manager.run_policy({
    f"policy/{spec.onnx_name}": context[spec.key] for spec in contract.inputs
})
commands = {
    spec.key: outputs[f"policy/{spec.onnx_name}"] for spec in contract.outputs
}
# The host applies commands, including safety/controller edits, and records
# the corresponding applied raw actions and PD targets, not unmodified guesses.
history.commit(measurements, applied_commands)
```

The four outputs are raw `actions`, `joint_pos_targets`, `stiffness_targets`
and `damping_targets`. Only joint targets and gains get LEAPP actuation kinds.
Histories stay host-owned: there is no automatic output feedback in the LEAPP
graph. Use the [Metatron timing table](metatron_generic_export.md#history-timing)
to preserve the incoming action for each measured state.

`load_pipeline_contract` checks hashes, input/output shape and dtype, ordering,
and agreement between the LEAPP ports and the typed sidecar. A dynamic batch
axis is allowed; other dimensions must resolve from the ONNX graph. This is a
generic host interface, not an adapter for the fixed native v1 executable.

## Metatron v1: Package a Scene-capable Graph

Run from `deployment/`, with that directory on `PYTHONPATH` as described in the
[README](../README.md). Install the optional export dependencies:

```bash
# "leapp" here is NVIDIA's github.com/nvidia-isaac/leapp, not PyPI's unrelated
# Red Hat tool; install version 0.7.2 from that repository.
pip install onnx 'leapp @ git+https://github.com/nvidia-isaac/leapp@0.7.2'
python -m tools.export_metatron_v1_leapp \
  --model /path/to/model.onnx \
  --manifest /path/to/manifest.json \
  --output artifacts/metatron-v1-leapp
```

The destination must not already exist. It contains:

```text
metatron-v1-leapp/
  policy.onnx
  policy.yaml
  manifest.json
```

This command accepts the reviewed Metatron v1 39-input contract only. It checks
the source checksum and ONNX port names/order, batch-one shapes and dtypes,
then packages the **unchanged ONNX bytes**. It rejects external tensor-data
files. The manifest declares input shapes/dtypes and output names; the native
action contract defines all four outputs as float32 per-joint vectors. No
extra output metadata fields need to be added to existing policy manifests.
Joint positions/velocities, body orientation/angular velocity, PD target
positions and gains receive LEAPP semantic kinds. Joint names retain training
order; raw context paths remain available as `protomotions_key` annotations.
The serialized `masked_mimic.*` keys are unchanged wire identifiers, not an
additional supported deployment family.

This is artifact export, **not checkpoint retracing**. Produce and qualify the
source ONNX using the source revision matching its checkpoint and saved config.
No weights are downloaded by this command, and no model weights are committed.
Existing qualification applies only because the graph bytes do not change;
packaging alone is not evidence of correct task control or hardware safety.
Use the maintainer-provided scene-capable ONNX for this path, not the generic
scene-disabled re-export of the packaged checkpoint.

## Native TensorRT hand-off

Pass the bundle YAML instead of an ONNX path:

```bash
python -m tools.prepare_metatron_v1_policy \
  --model artifacts/metatron-v1-leapp/policy.yaml \
  --recovery-plan /path/to/recovery_plan.json \
  --output artifacts/metatron-v1-native
```

Preparation checks both descriptions and their model checksums, validates the
Metatron v1 contract and materializes the native `model.onnx`, `manifest.json`
and recovery-plan package. Plain ONNX plus `--manifest` still works. Continue
with the [native guide](metatron_v1_native.md) to build and qualify GPU-local
policy and FK TensorRT engines. LEAPP does not replace TensorRT optimization,
graph-safety qualification, the native safety layer or the robot backend.
The native binary consumes engines/JSON, not YAML directly.

For offline LEAPP replay, `leapp.InferenceManager(".../policy.yaml")` loads the
same bundle. Supply every input as `policy/<onnx_input_name>`; outputs use the
same prefix. YAML output order follows the ONNX graph, while native validation
binds outputs by name. Constant gains may therefore appear after raw actions.

## Exporting a new wrapper

`tools.leapp_export.export_leapp_bundle(...)` retains the annotation-based
export API from the deployment repository. Pass a PyTorch wrapper, ordered
context keys/ONNX names, representative input samples and joint names. The
wrapper must return `(action, stiffness, damping, raw_action)` and contain the
complete observation/model/action-processing path. It exports one `policy`
node and validates it against the supplied samples. The caller supplies the
matching JSON robot/action/history manifest; the helper cannot infer those
semantics from tensors.

For this tracing path, set `PYTORCH_JIT=0` **before importing PyTorch or the
observation kernels**. Otherwise scripted rotation helpers are opaque to LEAPP.
Only the executed Python branches are traced: validate unseen samples spanning
task modes, masks and reset boundaries. Do not interpret a small tracing sample
set as full policy parity. The byte-preserving Metatron bundle command does not
trace and does not require disabling JIT.

Histories remain runtime-owned inputs. Metatron v1 has eight sensor-history
rows and nine incoming-action rows; advance them only after command acceptance,
including any limiting applied to the command. LEAPP does not automatically
supply task goals, localization, scene measurements, or hardware safety.

GPC's code-only tooling remains separate. Its sampling/plugin contracts are
not automatically converted by this Metatron command. No new tracker runtime
or checkpoint-compatibility code is introduced.

## Verification

```bash
python -m pytest tests/test_leapp_bundle.py
```

The tests exercise real LEAPP tracing and unseen-input inference, unchanged
ONNX packaging, output-order parity, plain ONNX resolution, YAML/manifest
mismatch rejection, checksum/path validation and native preparation. They use
small synthetic models, not published checkpoints. The generic checkpoint path
is additionally exercised from the repository root by
`protomotions/tests/test_metatron_generic_export.py`: real recipe construction,
strict checkpoint loading, masks, and PyTorch/ONNX/LEAPP agreement at batch
sizes one and three. LEAPP-dependent tests skip
when the optional dependency is absent. Target GPU and physical-robot
qualification remain separate requirements.

The released checkpoint and pinned scene-capable ONNX are also covered by
[artifact-level verification](metatron_export_verification.md). For strict
FP32 comparisons on NVIDIA GPUs, launch the replay with
`NVIDIA_TF32_OVERRIDE=0`: LEAPP 0.7.2 selects CUDA when available, and its default
TF32 arithmetic can differ from CPU FP32 beyond `1e-4`. This is a precision
setting, not a model or tolerance change.
