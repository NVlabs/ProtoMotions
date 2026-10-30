# Exporting policies with LEAPP

ProtoMotions uses [LEAPP](https://nvidia-isaac.github.io/leapp/) to export
policies for deployment. LEAPP (Lightweight Export Annotations for Policy
Pipelines) exports the policy code you already run (observation processing, the
network and action processing) as a single ONNX model plus YAML metadata that
names every input and output: its role, joint order and timing.

## Why LEAPP

For ProtoMotions users this means:

- **Deploy across different runtimes.** LEAPP provides a standardized contract
  that describes the policy, so the policy can run on different runtimes. For
  example, ProtoMotions provides a very lightweight runtime, and
  [Isaac ROS Deploy](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_deploy/index.html)
  has runtimes that run LEAPP bundles as part of ROS 2 nodes or `ros2_control`.
  [AGILE WBC](https://github.com/nvidia-isaac/WBC-AGILE/blob/main/docs/source/deploy-policies.md)
  and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/policy_deployment/05_leapp/exporting_policies_with_leapp.html)
  export the same contract.
- **Deploy the same computation you trained.** Observation ordering,
  normalization, forward kinematics and action scaling are part of the exported
  graph, so a runtime can't drift from training. With `--leapp-graph-history`,
  the sensor and action histories run inside the graph too.
- **Check exports before trusting them.** LEAPP compares the exported model
  with PyTorch on recorded samples, including the state updates.

To learn more, see the LEAPP docs:
[getting started](https://nvidia-isaac.github.io/leapp/getting_started.html),
[the generated configs](https://nvidia-isaac.github.io/leapp/generated_configs.html)
and [running a bundle](https://nvidia-isaac.github.io/leapp/leapp_runtime.html).

## Export a policy

### Install LEAPP

Install LEAPP from GitHub (PyPI's `leapp` is an unrelated tool). It is pinned to
a commit until the next LEAPP release. LEAPP's repository stores documentation
images in Git LFS, which the Python package does not need:

```bash
GIT_LFS_SKIP_SMUDGE=1 pip install onnx \
  'leapp @ git+https://github.com/nvidia-isaac/leapp@fe7c334dcecb43224f26107f14d75b7805ebc4a7'
```

### Export a checkpoint

From the repository root, download the Metatron G1 checkpoint with
`python scripts/download_metatron.py`, then:

```bash
python deployment/export_to_onnx.py \
  --checkpoint data/pretrained_models/metatron/G1/inference_last.ckpt \
  --output results/metatron_export \
  --leapp
```

This loads the checkpoint's saved config bundle and weights, exports the
deterministic prior/decoder, checks PyTorch/ONNX parity and writes:

```text
results/metatron_export/
  unified_pipeline.onnx
  unified_pipeline.yaml
  leapp/
    policy.onnx               # identical to unified_pipeline.onnx
    policy.yaml               # LEAPP interface
    deployment_contract.yaml  # ProtoMotions contract: robot, goals, histories
```

Keep the `leapp/` directory together: `policy.yaml` names `policy.onnx` by
relative path, and `deployment_contract.yaml` is the ProtoMotions contract
written beside them. In this bundle the sensor and action histories are inputs that
the runtime maintains, for example with `deployment/input_history.py`; see the
[history timing](metatron_generic_export.md#history-timing).
`load_pipeline_contract()` reads this bundle's `deployment_contract.yaml`.

Both bundles return the four commands (`actions`, `joint_pos_targets`,
`stiffness_targets`, `damping_targets`); every output is a command, so the host
applies them all. The robot's body poses are not inputs of either bundle. The
policy computes them inside the graph, from the joint readings and the torso anchor pose, with
the forward-kinematics kernel bound in the saved configuration, and does not
return them. A host that needs those poses outside the policy call, as the
native Metatron runtime does before each call, exports the same kernel as a
second small graph with
`python deployment/export_fk_to_onnx.py --checkpoint data/pretrained_models/metatron/G1/inference_last.ckpt`,
which writes `compiled_models/fk.onnx` and its sidecar `fk.yaml` beside the
checkpoint. Its inputs carry the policy graph's names for the same paths
(`noisy_dof_pos`, `noisy_anchor_pos`, `noisy_anchor_rot`), so a host feeds one
tensor under one name to both graphs; the [native guide](metatron_v1_native.md)
covers compiling it to a TensorRT plan.

### Graph-owned histories

`--leapp-graph-history` writes a second bundle, `leapp_graph_history/policy/`,
in which the graph owns the histories as LEAPP feedback state, so every LEAPP
runtime runs the same history update. LEAPP traces the policy for this bundle,
so it needs TorchScript disabled:

```bash
PYTORCH_JIT=0 python deployment/export_to_onnx.py \
  --checkpoint data/pretrained_models/metatron/G1/inference_last.ckpt \
  --output results/metatron_export \
  --leapp-graph-history
```

LEAPP cannot see inside `@torch.jit.script` functions such as the rotation
helpers used by forward kinematics, so their results would be frozen at the
first traced sample; the exporter refuses to trace with TorchScript enabled.
`PYTORCH_JIT=0` also changes the node structure of `unified_pipeline.onnx`
(same numerics), so add `--leapp` to the same run only if that is acceptable.

The history inputs disappear from the interface; per tick the runtime supplies
the measurements and goals as before, plus:

- `applied_actions`: the raw action actually applied on the previous tick,
  after any safety or controller edits;
- `command_accepted`: false if that command was dropped, so the history does
  not advance.

On reset, a LEAPP runtime restores the bundle's initial values. A runtime that
never edits commands passes the previous `actions` output and `true`.

This bundle has a fixed batch size of 1 and no `deployment_contract.yaml`, since
its interface differs from the contract's. The robot, control and goal metadata
are in `unified_pipeline.yaml`, which the exporter writes to the output
directory one level above the bundle; copy it along with the bundle.
LEAPP keeps only the inputs whose values reach an output, so an input the
graph reads for its shape alone is absent from this bundle (the FK-rebind
revision of the packaged G1 re-export had one, `current_dof_pos`, which only
sized its dummy scene tokens; the pinned revision has none). A runtime that
builds its input dictionary from the `leapp/` bundle's contract can leave such
a key out; LEAPP's `run_policy` ignores keys the graph does not take.

### Check the export

The exporter validates both bundles against PyTorch while it runs. With
`onnxruntime-gpu` installed, LEAPP runs its check on the GPU when one is
visible, and the GPU's default TF32 arithmetic differs from CPU FP32 by about
`1e-3`; for strict comparisons, export with `NVIDIA_TF32_OVERRIDE=0`. The
tests are:

```bash
python -m pytest protomotions/tests/test_metatron_generic_export.py \
  protomotions/tests/test_leapp_graph_history.py
```

The released checkpoint is also covered by the
[artifact-level verification](metatron_export_verification.md).

## Native Metatron v1 runtime

The native runtime runs the fixed 39-input, scene-capable Metatron v1 graph.
LEAPP packages that graph without changing its bytes, and the native preparation
tool accepts the bundle in place of the ONNX file. Run from `deployment/native/`,
with that directory on `PYTHONPATH` as described in the
[deployment README](../../README.md):

```bash
python -m tools.export_metatron_v1_leapp \
  --model /path/to/model.onnx \
  --manifest /path/to/manifest.json \
  --output artifacts/metatron-v1-leapp

python -m tools.prepare_metatron_v1_policy \
  --model artifacts/metatron-v1-leapp/policy.yaml \
  --recovery-plan /path/to/recovery_plan.json \
  --output artifacts/metatron-v1-native
```

Use the maintainer-provided scene-capable ONNX here, not a re-export of the
downloaded `metatron/G1` checkpoint: that re-export takes the same eight scene
inputs from the host, but its host contract differs (no simulator root pose;
the current action and the eight-slot history as separate inputs), so the
launcher's 39-input manifest does not describe it. Continue with the
[native guide](metatron_v1_native.md) to build the TensorRT engines;
`python -m pytest tests/test_leapp_bundle.py` covers this packaging.
