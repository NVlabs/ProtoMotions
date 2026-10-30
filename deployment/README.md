# ProtoMotions deployment

---

> **Status: experimental prototyping.** Everything in this directory except the
> top-level Python modules listed under *Supported* below is prototyping code,
> kept in the repository as reference material for people building their own
> deployment path. It is not supported, its APIs, CMake targets and file layouts
> can change or disappear without notice, it is outside the project's
> compatibility promises, and continuous integration does not build or run it
> (only the Python tests under `protomotions/tests` run in CI; they cover the
> exporters and the three files under `native/` named below, not the
> runtime). Several examples and tools talk to real hardware
> (a Unitree G1 through the vendor SDK, a Pico controller over the network) and
> will move a robot: read them before running them.

### Supported

The top-level Python modules are the supported export path, and CI tests them
through `protomotions/tests`:

- `contract.py`: the `DeploymentContract` sidecar every exported model ships with
- `export_to_onnx.py`: the exporter
  (`python deployment/export_to_onnx.py --checkpoint <dir>/inference_last.ckpt`;
  add `--host simulation` for a policy that reads the simulator's own body state)
- `export_fk_to_onnx.py`: the FK exporter, which writes `fk.onnx` and `fk.yaml`
  from the same kernel object the policy graph traces
  (`python deployment/export_fk_to_onnx.py --checkpoint <dir>/inference_last.ckpt`)
- `sample_context.py`, `tracker_inputs.py`, `input_history.py`,
  `metatron_policy.py`, `motion_utils.py`, `state_utils.py`: the exporter's
  inputs and helpers
- `test_tracker_mujoco.py`: a manual MuJoCo check of an exported tracker (run by
  hand, not by CI)

Some files under `native/` are imported from outside it, so they are not free
to change or disappear even though the tree they sit in is experimental:
`export_to_onnx.py --leapp` writes its bundle through
`native/tools/leapp_export.py`, which imports `native/examples/policy_artifact.py`,
and `--leapp-graph-history` through `native/tools/leapp_graph_history.py`;
CI's pytest run imports both LEAPP modules
(`protomotions/tests/test_metatron_generic_export.py`,
`test_leapp_graph_history.py`), `native/examples/metatron_v1_contract.py`
(`test_metatron_generic_export.py`, `test_scene_boxes.py`) and
`native/tools/make_metatron_fk_fixture.py` (`test_fk_exporter.py`).

### Experimental

- `native/include/`, `native/src/`, `native/tests/`, `native/CMakeLists.txt`,
  `native/cmake/`: a standalone C++20 runtime that loads an exported ONNX model
  and its sidecar and drives a simulator or a robot backend
- `native/tools/`: export and conversion utilities for specific policies
  (Metatron v1, GPC, LEAPP)
- `native/examples/`: end-to-end runs, including operator input and robot backends
- `native/bench/`, `native/docs/`, `native/assets/`, `native/scripts/`: timing
  harnesses, design notes, small example assets, and the installation helper
  `install.sh`

Style presets under `native/assets/styles` are feature vectors derived from an internal
motion-capture clip. Nothing under *Experimental* is a release artifact. For a
supported way to run an exported policy, use the exporter above and the shipped
sidecar.

---

This directory contains both **policy export tools** and a **standalone C++20
deployment runtime**. Train in ProtoMotions, export a model with its input/action
contract, qualify it in simulation, then connect the same controller to a robot
backend. Training does not depend on the C++ build or the robot SDKs.

The runtime was imported from NVIDIA's internal deployment repository; the
integration changes are listed in [SOURCE.md](native/SOURCE.md). The existing tracker
exporter and helpers at this directory's top level remain in place.

## Start here

- Build and test the core using the commands below; no robot or GPU is needed.
- For the current 39-input Metatron v1 policy, read
  [policy preparation](native/docs/metatron_v1_policy.md) and
  [native simulation/deployment](native/docs/metatron_v1_native.md).
- For GPC fixed-shape sampling engines, read [GPC TensorRT](native/docs/gpc_tensorrt.md).

GPC is **code-only**: no checkpoint, ONNX weights, TensorRT plan, download URL,
or pretrained model is distributed. Supply your own compatible model and config.
The existing top-level tracker utilities are unchanged; this runtime does not
add a BeyondMimic controller or launcher integration.

**Hardware safety:** a passing simulator test is not physical qualification.
The launchers start stopped/damping; never add `--arm` automatically. Qualify
joint order, units, gains, limits, sensor freshness, command watchdog and the
physical stop control with the robot secured. Metatron v1 uses X for damping
and Y for revive into idle; see its guide for the complete control mapping.

## What is supported

| Path | Implementation | Scope and limits |
| --- | --- | --- |
| Core runtime | C++ runtime, safety supervisor, controller manager, C ABI | Backend-neutral; vendor SDKs and inference libraries are optional |
| MuJoCo | Native backend and asynchronous viewer | Simulation, smoke tests and sensor/action contract validation |
| Unitree G1 | Optional SDK2 backend and G1 launchers | 29 body DOFs; requires SDK2/CycloneDDS and operator qualification |
| Metatron v1 | Dedicated native TensorRT policy engine + FK engine built from the policy's exported `fk.onnx` | Exact 39-input G1 contract; launcher exposes idle, steering and recovery |
| GPC unconditional prior | Python adapter, fixed-shape TensorRT runner and benchmark | Specific 29-DOF, 64-observation, eight-token contract; argmax, Gumbel-max and exact top-p |
| Operator input | Linux evdev Xbox, optional SDL2 and Pico/XRoboToolkit interfaces | Availability and bindings depend on the launcher; Pico SDK is external |

The interfaces are extensible, but copying a SOMA or arbitrary policy
artifact into a G1 launcher does **not** add support for that robot or contract.
GPC steering and generic native AMP adapters are extension opportunities, not
shipped plug-and-play paths. Recorded upstream timing/qualification results in
the detailed guides describe their original machines and artifacts, not a
certification of this checkout or your hardware.

## How the code works

```text
robot SDK / MuJoCo -> RobotBackend -> RobotState
                                       |
operator InputSource -> task goals -> Controller / policy adapter
                                       |
                              ActuatorCommand
                                       |
                     SafetySupervisor -> RobotBackend
```

The backend owns transport and sensor conversion; the controller owns policy
and task meaning. `DeploymentRuntime` owns lifecycle and validates state and
commands before sending them. `ControllerManager` switches controllers at cycle
boundaries. `FixedRateScheduler` owns timing separately from policy inference.
Invalid, stale or unsupported state/commands and transport failures enter the
safety path, not a silent fallback to arbitrary previous actions. See
[architecture](native/docs/architecture.md) and [packet-gap safety](native/docs/packet_gap_safety.md).

Exported graphs own observation processing, normalization, network inference,
action postprocessing and the forward kinematics of the robot's own body. A
robot host feeds raw sensor readings (joint positions, IMU orientation, the
anchor position from localization), task goals and histories; it never feeds
body poses (a simulation host may, see *Which host* below), and the policy
graph returns exactly its four control outputs
(`actions`, `joint_pos_targets`, `stiffness_targets`, `damping_targets`), every
one of them a command. A host that needs the body poses *outside* the policy
call, as the Metatron v1 native loop does when it runs FK on the current
measurement before every call to build its task goals, runs a second small
graph, `fk.onnx`. `export_fk_to_onnx.py` traces it from the same `BakedAnchorFK`
kernel object the policy graph traces, rebuilt from the same config bundle
beside the checkpoint. The previous standalone FK exporter re-ran forward
kinematics on the MJCF with its own quaternion conversion, a second
implementation that could drift from the policy's kernel; `fk.onnx` *is* the
policy's kernel, so there is one FK implementation and the two graphs agree by
construction. Its inputs `noisy_dof_pos`, `noisy_anchor_pos` and
`noisy_anchor_rot` (batch 1) carry the policy graph's names for the same paths,
so a host feeds one tensor under one name to both graphs; its outputs are
`body_pos` and `body_rot` (`xyzw`, the robot's body order); and its sidecar
`fk.yaml` records the body order, joint order and anchor so the native tools
can check it against the policy manifest.

**Which host.** `export_to_onnx.py --host` names what will feed the graph, and
the exporter refuses a binding that host cannot feed:

- `robot` (the default): what a robot has, whichever namespace a recipe reads
  it from: joint readings, the IMU orientation, a localization fix for the
  anchor, the policy's own past actions, the goal streams and, for a
  scene-conditioned policy, the scene boxes the host describes (the `scene`
  provider; the shipped G1 tracker binds its inputs under `current.*`, the
  Metatron recipe under `noisy.*`; the rule is per kind, not per namespace).
  The robot's body poses are
  computed inside the graph by the recipe's FK kernel, never fed. A policy
  meant for a robot is exported this way even while it is being tested in a
  simulator.
- `simulation`: additionally the simulator's own body state under `current.*`
  (every body's pose and velocity, the ground height under the anchor; the
  `simulated_*` kinds in `contract.py`), for a game engine or any harness that
  runs its own simulator and feeds its body state in place of sensors. The
  sidecar then records `host: simulation`. The shipped SOMA tracker
  (`motion_tracker/soma-bones`) is such a policy: its observations read the
  simulator's body state, so it exports only with `--host simulation`. The
  repository's reference tracker host (`test_tracker_mujoco.py`) feeds robot
  inputs only and refuses a simulation sidecar up front; the host for such a
  graph is the engine's own.

| Directory/file | Responsibility | Status |
| --- | --- | --- |
| `export_to_onnx.py`, `export_fk_to_onnx.py`, `tracker_inputs.py`, `contract.py` | Tracker export, [generic Metatron export](native/docs/metatron_generic_export.md) and the FK graph `fk.onnx` | Supported |
| `test_tracker_mujoco.py`, `motion_utils.py`, `state_utils.py` | Existing tracker playback and shared helpers | Supported |
| `native/include/protomotions/deployment/`, `native/src/` | Native interfaces, runtime, safety, robot/input and inference adapters | Experimental |
| `native/examples/` | Launchers and shared policy/task adapters | Experimental |
| `native/tools/` | Export, engine preparation, mapping/parity and input diagnostics | Experimental |
| `native/tests/` | Native and Python contract tests plus small fixtures | Experimental |
| `native/bench/` | Offline parity and latency programs | Experimental |
| `native/scripts/` | The explicit installation helper, `install.sh`. The Metatron package download is not here: it is `scripts/download_metatron.py` at the repository root, which writes to `data/pretrained_models/metatron/G1`. | Experimental |

## Build and test

### Core: no GPU or robot SDK

Requires CMake >=3.20, a C++20 compiler, and `nlohmann/json` headers (for example,
Ubuntu's `nlohmann-json3-dev`). Commands here start at the **ProtoMotions root**:

```bash
cmake -S deployment/native -B deployment/native/build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build deployment/native/build -j4
ctest --test-dir deployment/native/build --output-on-failure
```

For a non-system JSON installation, add
`-DNLOHMANN_JSON_INCLUDE_DIR=/path/to/prefix/include`. Nothing is downloaded by
CMake. To install the core into an explicit prefix:

```bash
bash deployment/native/scripts/install.sh --prefix /path/to/deployment-install
```

CMake consumers can use `find_package(ProtoMotionsDeployment CONFIG REQUIRED)`
and link `ProtoMotions::Deployment`. The core C ABI is version 2; rebuild clients
against matching headers. This CMake project is built separately; installing the
ProtoMotions Python package does not build or install it.

### Optional backends and inference

Add the relevant options to a separate build directory:

| CMake option | Required external dependencies |
| --- | --- |
| `PROTOMOTIONS_BUILD_MUJOCO=ON` | `MUJOCO_ROOT` with MuJoCo headers/library |
| `PROTOMOTIONS_BUILD_UNITREE_SDK2=ON` | `UNITREE_SDK2_ROOT`, optionally `UNITREE_DDS_ROOT` |
| `PROTOMOTIONS_BUILD_METATRON_NATIVE=ON` | TensorRT and CUDA roots below; Metatron v1 launcher also requires MuJoCo, Unitree SDK2 and evdev in the current build |
| `PROTOMOTIONS_BUILD_GPC_TENSORRT=ON` | TensorRT and CUDA; optional `PROTOMOTIONS_BUILD_GPC_TRT_PLUGIN=ON` for exact top-p |
| `PROTOMOTIONS_BUILD_EVDEV_INPUT=ON` | Linux; enabled by default on Linux |
| `PROTOMOTIONS_BUILD_SDL2_INPUT=ON` | SDL2 development files and pkg-config |
| `PROTOMOTIONS_BUILD_PICO_INPUT=ON` | `XROBOTOOLKIT_ROOT` external SDK |

GPU targets use `TENSORRT_ROOT`, optional `TENSORRT_LIBRARY_ROOT`,
`CUDA_RUNTIME_ROOT`, and `CUDA_TOOLKIT_INCLUDE_ROOT`. Despite its name, the last
variable is the toolkit **prefix containing `include/`**, not `include/` itself.
Use headers and libraries matching the target runtime. The imported top-p plugin
build currently targets **sm_120**; other GPUs need an explicitly adjusted and
qualified CUDA build, not just a copied engine. See the per-policy build guides.

### Python tools and tests

Use a Python environment with the matching ProtoMotions dependencies and install
[requirements-deployment.txt](native/requirements-deployment.txt). For CUDA/TensorRT ORT,
use a compatible `onnxruntime-gpu` installation **instead of** the CPU ORT wheel.
Export additionally needs PyTorch and ONNX; TensorRT preparation needs its Python
bindings. No runtime launcher installs dependencies for you.

The imported tools retain their standalone `examples.*` / `tools.*` imports,
resolved from `deployment/native` on `PYTHONPATH`. For commands in the detailed
runtime guides, use this working directory and path order so they resolve the
runtime's own `native/examples/` rather than ProtoMotions' training `examples/`:

```bash
cd deployment/native
export PROTOMOTIONS_ROOT="$(cd ../.. && pwd)"
export PYTHONPATH="$PWD:$PROTOMOTIONS_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests -q
```

Tests loading the G1 MJCF require its Git LFS mesh payloads. Fetch the matching
assets first; LFS pointer text is not a usable mesh. Private policy artifacts,
vendor SDKs and GPU plans are not bundled. Artifact-dependent checks cannot
qualify a model when those artifacts are absent.

## Map a trained policy into deployment

1. Keep the checkpoint and its saved config bundle together. The current
   exporter, `export_to_onnx.py`, reads the JSON bundle (`resolved_configs.json`);
   older native artifacts were exported from `resolved_configs*.pt`. Export with
   the **training-source revision that
   can load that artifact**, not an arbitrary newer checkout. The GPC exporter
   has fixed architecture/API assumptions; its historical model classes must
   exist in the selected source.
2. Metatron v1 and GPC use ONNX graphs plus JSON manifests. Metatron v1 also
   supports [LEAPP bundle export and native preparation](native/docs/leapp.md), preserving
   the qualified ONNX graph and adding its YAML interface. See the respective
   preparation guides. Carry external ONNX weight sidecars with the model.
   No GPC weights are supplied or downloaded by this repository.
   For the generic Metatron recipe, `export_to_onnx.py --leapp` writes the
   graph, LEAPP interface and typed deployment contract together, and
   `--leapp-graph-history` writes a LEAPP bundle whose graph owns the
   histories. The bundle writers are `native/tools/leapp_export.py` and
   `native/tools/leapp_graph_history.py`, exercised by `protomotions/tests`.
   That recipe's actual input set is not the fixed
   39-input native v1 interface;
   see [generic Metatron export](native/docs/metatron_generic_export.md).
3. Bind inputs by **semantic name, shape, dtype and time convention**. Check the
   ordered joint/body lists, the torso anchor pose the host derives from the
   pelvis IMU and the waist joints, quaternion
   order (`xyzw` at the deployment boundary), units, history lengths, reset fill
   and policy period. Matching tensor dimensions alone is insufficient.
4. Check the output action domain. A processed position target must not receive
   the action offset/scale twice. Raw-action history and processed-target history
   differ across contracts; preserve the manifest's convention and update only
   at the appropriate accepted-command boundary. Metatron v1 explicitly
   reconstructs raw history from the accepted target via its affine action map.
5. Compare exported inference against the training reference, including resets,
   multiple goal modes and sequential history. Then test a closed-loop MuJoCo
   rollout, before any supervised physical trial.

For a new robot, implement `RobotBackend` and define its capabilities, joint
mapping, sensor frames, limits and emergency output. For a new policy, implement
`Controller` (or `ArtifactController` with a `PolicyRuntime`) and keep its
contract/task logic there. For a new operator device, implement `InputSource`
and wire `InputRouter` at startup. Reuse runtime/safety code rather than adding
policy-specific branches to it. Each extension needs focused contract tests.

## TensorRT optimization and qualification

Treat optimization as a validated artifact pipeline:

```text
checkpoint + saved config -> ONNX + manifest -> parity
   -> target-machine TensorRT plan -> native parity -> warmed latency -> rollout
```

- Start with FP32. Build plans on the intended GPU/runtime; `.plan` files and
  ORT TensorRT caches are different artifacts and are not generally portable.
- For Metatron, `native/tools/rewrite_metatron_graphsafe.py` and
  `native/tools/validate_metatron_graphsafe.py` prepare/check the graph-safe candidate;
  `native/tools/build_metatron_tensorrt.py` exposes `--optimization-level` (0–5).
  Build the FK engine from `fk.onnx` and its sidecar `fk.yaml` with
  `native/tools/build_metatron_fk_tensorrt.py`, which compiles a static FP32
  plan and, given `--manifest`, checks the sidecar's body order, joint order
  and anchor against the policy manifest's `robot_contract`, refusing a
  mismatch. `native/tools/make_metatron_fk_fixture.py` writes the parity
  fixture from the same two files; `body_rot` comes from a matrix-to-quaternion
  conversion, so a parity check must accept `q` and `-q` as the same rotation.
  Follow the
  [Metatron v1 native guide](native/docs/metatron_v1_native.md) for complete commands.
- For GPC, `native/tools/prepare_gpc_policy.py` exports, builds, validates and benchmarks
  the fixed-shape sampling modes. Exact top-p requires the matching plugin.
  Keep TF32 disabled for the baseline: small logit changes can flip discrete
  tokens even when floating-point errors look small.
- CUDA graphs reduce launch overhead after warm-up using persistent buffers
  and fixed shapes/addresses. Sampling noise stays an explicit input. Enable
  capture only for contracts that support it; compare graph and stream paths.
- Test FP16/BF16 as separate candidates, not assumed-safe upgrades. Re-run
  the family-specific parity tools before accepting a lower-precision engine.
- Measure warmed native H2D/inference/D2H timing and full state-to-command
  latency separately. Include sensor freshness and command deadlines in the
  latter. Low policy-only latency is not a hardware safety or stability result.

Keep generated models, plans, caches, recordings and vendor SDKs out of Git.
The pinned Metatron v1 weights are NVIDIA-internal and are not distributed with
this repository; they are not needed to build the core or run the source-only
tests.

## Further reading

[Architecture](native/docs/architecture.md) ·
[Metatron v1 contract](native/docs/metatron_v1_policy.md) ·
[Metatron v1 native runtime](native/docs/metatron_v1_native.md) ·
[GPC](native/docs/gpc_tensorrt.md) · [Benchmarks](native/bench/README.md)
