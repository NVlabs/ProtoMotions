# Runtime architecture

The live control path is native C++. Python tools prepare assets and qualify
contracts offline. Training does not link to the deployment runtime.

```text
RobotBackend -> RobotState -> Controller -> ActuatorCommand
     ^                                         |
     +------------ SafetySupervisor <----------+
```

`DeploymentRuntime` owns startup, control ordering and the fault latch.
`ControllerManager` switches controllers at cycle boundaries. Backends own
transport, state conversion and safe outputs; policies own observations and
task meaning. `FixedRateScheduler` owns timing separately from inference.

## Extension boundaries

| Extension | Implement | Keep out of the shared core |
| --- | --- | --- |
| Robot or simulator | `RobotBackend` | Vendor packets, joint ordering, frame conversion |
| Policy | `Controller`, or `ArtifactController` with `PolicyRuntime` | Input histories, task goals, inference, action interpretation |
| Operator device | `InputSource` with `InputRouter` bindings | Device event decoding and calibration |

Use one owner per state transition. New policies should reuse lifecycle,
command validation and safety rather than introducing another control loop.
Each extension needs tests for reset, dimensions, units, stale state and stop.
The interfaces are extensible; a new robot still needs an explicit mapping and
qualification. The provided Metatron v1 adapter is specific to G1.

## Metatron v1

`metatron_v1::MetatronTask` assembles idle, steering and recovery goals.
`MetatronController` uses the same sensor/history and accepted-command boundary
for MuJoCo and Unitree. `Engine` executes the fixed-shape TensorRT policy;
`FkEngine` separately computes body poses for task construction. The policy
graph owns learned inference, observation processing and actuator mapping.

The manifest pins 39 named inputs, their shapes/dtypes, joint/body ordering and
the affine action map. A processed target goes directly to command validation,
without another offset or scale. Only an accepted command advances history;
limited targets are converted back to raw-action space before being recorded.
The controller rejects another inference while its last command is pending.

The artifact's serialized identifier and `masked_mimic.*` tensor keys are
preserved verbatim. These are the Metatron v1 model ABI, not additional runtime
implementations. Renaming them requires re-exporting the model and manifest.

Scene-object inputs remain hidden without a localization integration. The
launcher currently exposes idle, steering and recovery, not object interaction,
clip tracking, style selection or Pico teleoperation.

## State and backend contracts

The deployment boundary uses xyzw quaternions. Unitree's torso IMU and a
pelvis-rooted FK carrier are different frames; the sensor adapter explicitly
accounts for waist kinematics rather than treating them interchangeably.
State samples and histories must be synchronized, not independently read from
buffers that can change during a control cycle.

The optional Unitree SDK2 backend validates packets and freshness, maps G1's
29 motors, and emits checked commands. `unitree_sensor_probe` provides a
subscriber-only preflight. MuJoCo implements the same `RobotBackend` interface;
its command gains temporarily override the XML baseline and omitted gains
restore that baseline. This keeps emergency damping from permanently changing
the next policy command.

## Safety and timing

Runtime validation covers finite values, dimensions, capabilities, freshness,
command/gain limits and read/send failures. Hard failures latch a fault and
refresh backend emergency damping/hold output until explicit operator revival.
Applications must configure finite gain caps for position-gain backends.

Metatron v1 starts stopped in damping. X has stop priority; Y revives into idle.
Real hardware also requires explicit `--arm`. A simulation result is not
permission or qualification to operate a physical robot.

The generic repeated-state grace defaults to zero. Metatron v1 permits a
bounded hold of the last validated position command, without advancing policy
history; see [sensor-gap safety](packet_gap_safety.md). This is not balance
control. Invalid state and operator-stop events still fault immediately.

Warm accelerator engines before arming. Measure sensor-to-command latency
separately from inference-only timing. Viewer work belongs off the control
thread; latest-only snapshots may drop visual frames without duplicating a
policy step. Hardware deadlines still require target-machine qualification.

## GPC code-only support

GPC provides export/preparation tools, an ONNX Runtime policy adapter and native
TensorRT runner. Fixed-shape categorical sampling supports argmax, Gumbel-max
and exact top-p; the latter uses a CUDA plugin. No GPC weights or downloader
are supplied. See [GPC TensorRT](gpc_tensorrt.md) for the contract and parity gate.
