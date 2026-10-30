# Metatron v1 native deployment

The native launcher consumes the pinned `protomotions.masked_mimic.g1.v5_raw_context_scene` 39-input contract. It uses one C++ `MetatronTask`, history boundary, TensorRT policy, and TensorRT FK engine for both MuJoCo and Unitree. Only the `RobotBackend` changes. The ONNX graph already contains the actuator offset and scale: the launcher sends its applied action directly and converts it back to raw action only for accepted-command history. It does not add the offset twice.

The pinned launcher bundle is
`artifacts/metatron-ep6410/{model.onnx,manifest.json,recovery_plan.json}`.
Build `model.graphsafe.fp32.plan` and `fk.fp32.plan` for the local GPU with
the repository's TensorRT preparation tooling; plans are GPU/runtime-specific.

The G1 package fetched by `python scripts/download_metatron.py` now binds the
FK kernel in its saved configuration. From the repository root, run
`python deployment/export_fk_to_onnx.py --checkpoint
data/pretrained_models/metatron/G1/inference_last.ckpt` to write
`compiled_models/fk.onnx` and `fk.yaml` beside the checkpoint; `--output` picks
another directory. Move an older downloaded `G1/` directory aside before
downloading the new revision. See [export verification](metatron_export_verification.md)
for the revision and measured agreement; the weights and launcher policy ONNX
are unchanged.

The generic policy graph computes body poses internally and keeps its four
control outputs. The native runtime also needs those poses before each call,
because its goal canvas starts from the measured pose. The separate `fk.onnx`
traces the same `BakedAnchorFK` kernel from the packaged config.
`python tools/build_metatron_fk_tensorrt.py --onnx fk.onnx --sidecar fk.yaml
--output fk.fp32.plan --manifest manifest.json` compiles a static FP32 plan
and checks body order, joint order, anchor and input names against the launcher
manifest (see the [qualification guide](metatron_v1_native_qualification.md)).
This does not make the generic 37-input policy graph interchangeable with the
launcher's 39-input graph.

Build with the optional Metatron native, MuJoCo, Unitree SDK2, and evdev
components enabled, then run `ctest --test-dir build-clang --output-on-failure`.

Prepare the simulation MJCF once; the source MJCF remains unchanged:

```bash
python tools/prepare_metatron_mujoco_xml.py \
  --source /path/to/g1_holo_compat.xml \
  --output artifacts/metatron-ep6410/g1_holo_compat_sim.xml
```

The preparation installs position actuators and a floor, as the simulator requires. The policy's PD gains override the XML gains on each command. A visualizer is optional and runs on a separate thread; it cannot supply observations to the policy.

```bash
./build-clang/metatron_v1_deploy_cpp \
  --backend mujoco \
  --xml artifacts/metatron-ep6410/g1_holo_compat_sim.xml \
  --engine artifacts/metatron-ep6410/model.graphsafe.fp32.plan \
  --fk-engine artifacts/metatron-ep6410/fk.fp32.plan \
  --manifest artifacts/metatron-ep6410/manifest.json \
  --recovery-plan artifacts/metatron-ep6410/recovery_plan.json \
  --xbox-device /dev/input/by-id/YOUR_XBOX-event-joystick \
  --cuda-graphs --viewer --glfw-lib /path/to/libglfw.so
```

The same binary takes `--backend unitree --net-if YOUR_VERIFIED_INTERFACE --arm`, omitting `--xml`, `--viewer`, and `--glfw-lib`. The FK plan already contains the robot kinematics, so an MJCF path is not used on hardware. It requires an explicit `--arm` but still starts in damping hold. Physical G1 operation is **not yet qualified** by the local tests; perform operator-supervised checks of sensor freshness, odometry, safety stop, and stance before reviving it. Do not run it merely because the simulator passes.

Before hardware launch, run the subscriber-only preflight. It does not release MotionSwitcher or create motor publishers, and does not require `--arm`:

```bash
./build-clang/unitree_sensor_probe --net-if YOUR_VERIFIED_INTERFACE --participant-index 0 --seconds 15
```

Require progressing LowState ticks, fresh LowState and odometry ages (both below 100 ms at completion), and no CRC or backward-tick errors. The launcher also sets CycloneDDS participant index 0 by default; `--dds-participant-index N` must agree with the probe and any interface-scoped firewall rules. An existing `CYCLONEDDS_URI` takes precedence, so verify its participant index. Do not disable the host firewall for deployment. If the preflight fails, do not arm; diagnose missing LowState versus odometry from the per-second counters. Startup reports their separate counts and ages, and runtime reports the specific validation or hold failure.

The runtime holds only the last validated position target through a short repeated-sensor gap (up to 60 ms), without advancing the policy or action history. LowState and odometry older than 100 ms are rejected, and longer gaps latch a fault requiring an explicit Y after sensor recovery. These limits must be requalified if the robot's measured topic rates differ substantially.

Controls: Both backends start in continuously refreshed damping hold. X always enters damping hold and takes priority while held; after releasing X, Y revives into idle. Pressing Y again while active leaves the current mode and policy history unchanged. A enters steering; RB returns to idle. Left stick sets a heading relative to the current robot heading while held; releasing it keeps the last global heading. Right stick commands planar motion relative to that heading, capped at 1 m/s. Idle and stopped steering condition on an upright, locally calibrated body pose, not an unobservable global lock. Recovery overrides the task when measured torso height or tilt indicates a fall and returns to the task after the rise/settle criteria. This first native launcher does not yet expose object interaction, tracking clips, style selection, or Pico controls.

Local qualification: all 19 CTests pass. The rest pose matched the Python reference within 1.2e-7 m; five idle/steering target fixtures passed at 2e-5 tolerance. A 130-cycle native MuJoCo run executed idle, moving steering, and idle with no safety fault and torso root height around 0.76 m. A separate prone-start fixture entered recovery, handed back to idle by cycle 200, and held approximately 0.764 m through cycle 475. These are fixtures, not a recovery success rate across randomized poses. The native policy-only benchmark measured about 2.06 ms median inference on the qualification GPU; this is not a measured end-to-end physical robot latency. Direct C++ policy and FK parity reports are in the repository's native qualification tools.

## Optimized FP32 plan

The local TensorRT 10.9 plan `model.graphsafe.opt1.fp32.plan` is a separate
tactic-search build; it does not replace `model.graphsafe.fp32.plan`. Rebuild
it on another GPU/runtime with the same graph-safe ONNX and manifest:

```bash
python -u tools/build_metatron_tensorrt.py \
  --artifact-dir artifacts/metatron-ep6410 \
  --onnx artifacts/metatron-ep6410/model.graphsafe.onnx \
  --output artifacts/metatron-ep6410/model.graphsafe.opt1.fp32.plan \
  --optimization-level 1
```

For the launch command above, substitute the optimized filename in `--engine`
after validating it on the target machine. The native CUDA graph now captures
the pinned host-to-device and device-to-host transfers with inference. On the
local RTX 5090 Laptop GPU, 1,500 warmed native calls measured 1.393 ms p50 /
1.693 ms p95 with the optimized plan, versus 2.153 ms / 2.649 ms for the
level-0 plan in the same runtime. The candidate passed 18 recorded action
parity feeds (max applied-action error 0.00121 rad), all 19 CTests, and a
155-cycle scripted MuJoCo idle/steering/safety run. These are not live-robot
end-to-end latency or physical-safety qualifications.

The optional simulator fixture command `--scripted-sim --steps N` is prohibited for Unitree by the launcher. It starts with ten damping cycles, drives Y at cycle 10, A at 40, repeats Y at 60, selects idle with RB at 100, damps with X at 120, revives with Y at 130, and repeats Y at 140. It is intended solely for repeatable MuJoCo control-loop checks.
