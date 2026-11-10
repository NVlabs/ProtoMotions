# Metatron v1 native: live metrics and hand teleoperation

Two optional additions to the native launcher. Neither changes the policy
contract, the safety supervisor or the accepted-command history boundary.

## Live metrics

The launcher publishes one JSON object per 50 Hz control cycle over UDP to
`127.0.0.1:9870` (datagrams are dropped, never awaited, when the socket is
full) and, with `--metrics-file run/metrics.jsonl`, to a file for a record.
When a person is at the controls (`--xbox-device` or `--pico`) and a display
exists, the launcher starts the dashboard itself, as a separate process, and
closes it when it exits; scripted runs never get a window. The dashboard's
Python is set at configure time (`PROTOMOTIONS_DASHBOARD_PYTHON`, a Python with
numpy and matplotlib). Keys are `group/name`; vectors appear as `name/0`, `name/1`, ...

| Group | Keys | Source |
| --- | --- | --- |
| policy | `policy/name` (string) | manifest policy name, or checkpoint basename with parent; no full host paths |
| names | `joint/names` (string, comma-joined, every frame), `limits/nearest_joint` (string) | the manifest's robot contract, in policy order, which is also the motor order on hardware; the viewer labels every per-joint and per-motor metric with them and names the hottest motor, the joint nearest its limit and the joints with the most velocity jitter |
| joints | `joint/q/i`, `joint/dq/i`, `joint/torque/i`, `joint/jerk/i`, `joint/jerk_rms` | backend state; torque from MuJoCo actuator force or Unitree motor feedback; jerk by finite differences of the reported velocity (rad/s³), RMS over joints |
| limits | `limits/joints_near_limit` (within about 3 degrees), `limits/min_margin_rad` | joint ranges of the robot MJCF (`--xml`) |
| power | `power/mechanical_w` = sum of abs(torque x velocity), `power/energy_j` | launcher |
| timing | `cycle/compute_ms` (the whole control cycle: sensor read, observation build, policy engine, command send), its stages `cycle/read_ms`, `cycle/observe_ms` (task goals and the sensor FK engine), `cycle/policy_ms` (the TensorRT policy alone), `cycle/send_ms` (the transport; in MuJoCo this is where the 20 ms of physics is stepped, so it dominates in simulation), `cycle/state_age_ms`, `cycle/held`, `cycle/result` | runtime and launcher |
| communication | `comm/missed_states`; Unitree `comm/lowstate_packets`, `comm/crc_errors`, `comm/lowstate_age_ms`, `comm/odometry_age_ms` | runtime and backend sensor status |
| pose | `pose/root`, `pose/root_quat` (xyzw), `pose/torso_z`, `pose/tilt_deg` | backend state; enough to re-pose the robot for a replay render |
| task identity | `task/name`, `task/mode_name` (strings) | the task; titles the viewer's task tab and mode readout |
| task | `task/heading_error_deg`, `task/speed_requested`, `task/speed_measured`, `task/velocity_error_x`, `task/velocity_error_y`, `task/support_slip_m`, `task/hand_error_left_m`, `task/hand_error_right_m`, `task/mode`, `task/recovery_phase` | `MetatronTask::metrics()` |
| teleop | `support/*`, `goal/left/*`, `goal/right/*`, `ik/offset_left_rad`, `ik/offset_right_rad`, `ik/solve_ms`, `task/torso_height_goal_m`, `task/torso_height_m` | task and arm IK |

Both TensorRT engines run as captured CUDA graphs, always; there is no flag.
On an RTX 5090 with the fp32 ep6410 plan a MuJoCo cycle measures about 2.6 ms:
1.7 ms policy engine, 0.3 ms observation build, 0.55 ms physics step. The
policy time is kernel execution at batch one, not launch overhead; a lower-
precision plan is the remaining lever.

A task contributes by filling its `Metrics` struct each `build()`; the launcher
copies it into the frame. Adding a metric is one `frame.set("task/name", value)`.

`python tools/live_metrics_viewer.py --udp 0.0.0.0:9870` opens a dark,
read-only matplotlib dashboard. Its header identifies the policy, stream state
and cycle. Five clickable tabs separate the data:

- **Overview:** heading, speed, hand errors, torso height and tilt (separate
  metre/degree plots), mechanical power and maximum temperatures.
- **Joints:** positions, velocities, torques and limit margins.
- **Thermal:** two maximum-temperature trends and paired winding/driver bars
  for all 29 motors, indexed in robot motor order. Hatched placeholders mean
  unavailable or stale sensors, not zero degrees.
- **Timing & comms:** compute time, sensor ages, communication errors, held
  cycles and received packets.
- **Teleop:** hand error, IK correction and solve time, foot slip, re-plants,
  torso targets and world-space hand goals.

Only the selected tab has chart artists and redraws at 10 Hz. The reader keeps
all numeric histories regardless of tab; switching tabs restores their recent
history immediately. The viewer runs separately from the controller and cannot
send commands.

The top bar carries the NVIDIA mark, the policy name, the task and mode, a
link-state pill (LIVE, RECORDED, STALE, WAITING) and the ProtoMotions
wordmark. Under the tab strip a row of stat tiles shows what an operator
watches whatever tab is up: mode, compute time, hand error, tracking error,
hottest motor (amber and red at the display thresholds) and joint jerk RMS.
Each chart prints its latest values in its bottom-right corner. Tabs are a
strip of labels with the active one underlined; a tab shows up to six charts
a page, two columns, paged with the arrows at the bottom. A chart
appears once its data arrives and stays off the page while every channel on it
has only ever read zero. Overview, Thermal and Timing & comms always have a
tab; Joints and the task tab appear once one of their keys has arrived, and
stay for the rest of the stream (a dropout reads UNAVAILABLE or STALE, never
disappears). A stream without the temperature family shows the Thermal tab
with "no temperature telemetry" rather than hiding it. The launcher emits the temperature family (`motor/*`
temperatures, `motor/temperature_count`, `motor/temperature_valid`) only on
hardware or when a reading exists, so a MuJoCo record has no Thermal tab while a
robot whose sensor reads nothing still has one, marked UNAVAILABLE. Older
simulation records that carry a zero count keep a Thermal tab, unavailable.

The task tab is titled from `task/name` and has a stable internal id (`task`).
A curated preset exists for the exact name `Metatron v1`: its "overview" panels
(heading, speed, hand errors, torso target) join the overview tab, its "task"
panels lead the task tab. Presets are presentation and known units only. Every
numeric key under `task/`, `goal/`, `ik/` and `support/` that the preset does
not consume still gets a fallback panel: the numeric components of a vector
share one (`stem/#` patterns match only numeric final segments, so
`goal/left/enabled` never lands on a metres axis), every other key has its own,
labelled only when the suffix is unambiguous (`_m`, `_deg`, `_rad`, `_ms`).
Equal units are never merged on a guess. A task with another name gets the
fallback layout alone, with no viewer change; a record without `task/name` is
titled "Task" and shows the raw mode number. Within a tab a panel appears once
it has data; a tab requested without data says so rather than borrowing charts.
Per-motor thermal bars count the motors the stream names or counts, never a
constant.

`--file run/metrics.jsonl --screenshot out.png` renders every visible tab as
`out_<tab>.png` (`overview`, `task`, `joints`, `thermal`, `timing`); `out.png`
remains the overview for existing commands. `--tab thermal` selects the initial
tab and, with `--screenshot`, saves only that tab to the requested filename.
Recorded views are labelled **RECORDED**: temperature freshness uses the frame's
recorded state ages, not time spent rendering. Live views also count time since
receipt. String metadata is retained for the header, not plotted as a numeric
signal. Finally, `tools/render_metrics_replay.py` turns the same record into an
MP4 through MuJoCo (kinematic re-posing, no policy re-run) with goal and
support markers.

### Temperature display

The viewer has a motor-temperature panel and a warning visible on every tab. It accepts
`motor/max_winding_c`, `motor/max_driver_c`, and per-motor `motor/winding_c/i`
and `motor/driver_c/i` in Celsius. Amber shading starts at 85°C; red shading,
bold hot traces and a **HIGH TEMPERATURE** banner start at 90°C, matching the
launcher's existing high-temperature warning entry threshold. Override the
display bands with `--temperature-caution-c` and `--temperature-high-c`.
These are visual warnings, not manufacturer-certified operating limits or
safety interlocks; the viewer cannot alter motor commands. The 85°C visual band
does not change the launcher's separate warning hysteresis.

Missing or nonfinite temperatures show **UNAVAILABLE**, not zero. No new frame
for 0.5 seconds, or a hardware/state age of at least 0.5 seconds, shows
**STALE** even when fresh UDP frames continue arriving. Invalid supplied state
ages are also stale. A frame with `motor/temperature_valid=0` but
some finite sensors shows **PARTIAL telemetry** and still highlights any high
readings. Old winding-only recordings explicitly say driver data is unavailable.
MuJoCo supplies no physical motor-temperature measurement.

For a UI-only demonstration with all 29 motors (synthetic values, not robot measurements):

```bash
python tools/live_metrics_viewer.py --file tests/fixtures/temperature_metrics.jsonl \
  --title "SYNTHETIC temperature UI test — not robot measurements" \
  --screenshot temperature_ui.png
```

## Hand teleoperation

Mode `kTeleop` of `MetatronTask` adds hand targets on top of steering:

1. Steering keeps driving the torso and feet. A stationary command plants both
   feet and holds them as world targets; the stand/walk transition uses speed
   hysteresis (enter below 0.03 m/s, leave at 0.07 m/s). The planted pair
   defines the **support frame**: centre at the feet midpoint on the floor,
   yaw at planting.
2. Operator hand offsets (Pico: position relative to the operator's root in
   the headset-yaw frame, height above the operator's floor) are mapped into
   the support frame by `hand_goal_from_operator` (`metatron_v1_teleop.hpp`),
   scaled and clamped to a workspace box, and written to the policy's hand
   slots as world positions. The mapping is anchored at the hand-over: on the
   first planted cycle after B turns teleop on, each goal is offset so it
   coincides with the robot's measured hand, and the operator's motion from
   then on moves it. Until then the hands keep their pose, so nothing jumps
   when teleop starts; B off and on re-anchors. Hand rotation goes the same
   way: the anchor stores, per hand, the rotation from the controller's world
   orientation to the hand's measured one, so the goal holds the hand's frame
   at the hand-over and a wrist turn in the world turns the goal by the same
   world rotation. Rotation goals are given only with the anchor (an
   unanchored controller orientation says nothing about the hand's) and were
   first exercised in MuJoCo on 2026-11-05, not on the robot.
3. Each Pico grip is a hand clutch. Releasing it holds that hand's last
   position and rotation relative to the feet; moving the controller while
   released does not move the goal. Re-gripping aligns the controller to that
   held goal before subsequent motion, with independent left/right state.
   While walking, hand goals remain active in the measured feet frame rather
   than snapping to the steering-rest pose. Steering without teleop keeps the
   rest hands slightly forward/outward of the pelvis.
4. `ArmIkCorrection` (the native port of the browser demo's `arm_ik.ts`)
   adds a bounded per-arm correction to the policy's PD targets: a damped
   least-squares step from the measured hand pose error over the seven
   shoulder-to-hand joints, solved as a box QP whose bounds already contain the
   soft joint range (90% of the range), the offset budget (0.025 rad, or 0.05
   raw-action units when smaller) and the rate limit (0.05 rad/s). The offset is
   re-solved from the current error each cycle, so it never integrates; edited
   joints are inverted through the manifest's affine map so the policy's action
   history records what the motors received. It uses `--xml` for kinematics
   only; the model is never stepped. It is on by default; `--no-hand-ik`
   selects policy-only hand tracking for comparison.
5. The torso target follows the lowest hand goal while the feet are planted:
   0.45 m above that hand, never below 0.55 m and never above the upright rest
   height, ramped at 0.4 m/s because a stepped torso target makes the policy
   step with it. Without this a floor goal asks the policy to fold at the waist
   while standing tall, and the hands stop 14 cm short. The hand workspace box
   reaches down to 0.05 m above the floor.
6. The headset height commands the torso too: the Pico adapter publishes the
   headset height above the operator's floor, and the robot torso is sent to
   the same fraction of its upright height as the headset is of the operator's
   standing height (the running maximum of the headset height). The lower of
   this and the hand-derived target wins, with the same clamp and ramp. In a
   script a segment's `torso` field plays the same role. The headset's heights
   are only floor-relative when its tracking origin is on the floor; with an
   eye-level origin shoulder-height hands read near zero, every goal clamps to
   the floor and the robot crouches after them. `--operator-height m` (the
   operator's standing headset height) fixes this without touching the headset:
   the reading on the B press that turns teleop on, taken standing upright,
   sets the floor offset for every height afterwards and the standing height.
7. Prompt pre-emphasis (the web demo's prompt feedback, variant V3 of the
   harness study) prompts each hand past its goal by the remaining error: gain
   1, velocity damping 0.15 s, bias capped at 0.12 m, slewed at 0.1 m/s with a
   0.15 s lag, and the prompt never below a hand resting on the floor. The bias
   restarts when a goal jumps more than 5 cm (a new destination) and is dropped
   while walking; the direct hand goal remains active. The final prompt is
   clamped to the hand workspace. Hand error metrics are always against the real goal;
   `task/prompt_bias_*_m` report the bias length. Always on: mean hold error
   over the native scenarios 6.5 cm against 8.9 cm without it.
8. The staged recovery's fall test (torso tilt past 75 degrees or torso below
   0.5 m) treats a deliberate bow as intentional while a hand is commanded
   below 0.35 m and for one second after, provided the torso stays above
   0.45 m and short of horizontal. A torso that sinks still triggers recovery.

Controls: A steering, B hand teleop on/off while steering, RB idle, X damping,
Y revive. On Pico, each grip clutches that arm and each finger trigger controls
the matching Dex-3 hand; finger commands require fresh validated hand state.
Without Pico, the Dex-3 hands stay in damping hold. The optional
`--operator-height` is in metres. Presses are latched by the input sources (`InputState::pressed`),
so a tap shorter than one 20 ms control cycle still counts; before, the
launcher compared 50 Hz snapshots and a quick LB tap could be swallowed.
When presses still seem to go missing, run the launcher with
`PROTOMOTIONS_INPUT_TRACE=1` (one line per button change with the raw mask,
the pressed mask and the task mode) next to `tools/xbox_raw_events.py`, an
independent reader of the same evdev device. A press in the raw stream but
not in the trace is the launcher's fault; a press in neither never left the
pad. The PowerA wired pad (20d6:207b) reports button releases late, seconds
late at times, and the kernel then filters the next presses of that button as
duplicates: the symptom is a bumper that needs several taps. Use another pad
for operator runs. The launcher also warns on a kernel event-queue overrun
(`SYN_DROPPED`), the one way it can itself lose a press. The same bindings serve Xbox, Pico (`--pico`, needs a build with
`PROTOMOTIONS_BUILD_PICO_INPUT`) and the simulation script.

## Motion tracking

`--motions examples/tracking_motions.json` loads the browser demo's reference
motions (`web_demo/scripts/extract_tracking_motions.py` layout; the deployment
bundle carries `dancing`, `kick` and `spinning`, the front roll is left out).
LB starts the selected motion and stops a playing one; D-pad left/right and
the dashboard select, and selecting never interrupts a playing motion.
A motion ends in idle, standing where it finished.

The robot has no global position worth trusting, so the reference is
re-anchored every cycle: its torso xy at the current time is placed on the
robot's torso xy, and every other body and every future slot keeps its offset
from that torso. Heading and floor height are aligned once when the motion
starts; rotations are then the reference's. The five slots look 20, 40, 80,
160 and 320 ms ahead (the demo's horizons) with the conditionable bodies fully
visible, and the average-velocity command is the reference torso's over the
next half second. Metrics: `tracking/motion`, `tracking/selected`,
`tracking/available`, `tracking/elapsed_s`, `tracking/duration_s`,
`tracking/progress`, `tracking/mean_body_error_m` (conditionable bodies
against the re-anchored reference), `tracking/anchor_height_error_m` (the
anchor's xy error is zero by construction), `tracking/mean_rotation_error_deg`
and `tracking/ref/<i>` (reference positions the replay renderer draws).

### Settings from the dashboard

The dashboard writes operator settings to a small JSON file,
`/dev/shm/protomotions_teleop_settings_<uid>.json` (the launcher names it and
passes it with `--settings-file` when it starts the dashboard), replaced
atomically on every change so the launcher never reads half a document:

```json
{"version": 3, "max_speed_mps": 0.5, "max_yaw_rate_rps": 1.2, "selected_motion": "dancing"}
```

The launcher polls the file's modification time once a cycle and parses it
only on change. Which numeric keys exist, and their ranges, are declared by
the owner of each tunable, not by the dashboard: `MetatronTask::settings_schema()`
lists each key with its default (what runs when the file says nothing), the
largest value its qualification covered, and a hard bound. The launcher sends
the schema on the stream as `settings/schema` (JSON) and the dashboard builds
its sliders from it, with a tick at the default and a red band past the
qualified value, so an operator can pull beyond what was qualified and sees
that they did. Today the task declares `max_speed_mps` (default and qualified
1 m/s, hard bound 5 m/s, which `set_steering` also refuses past) and
`max_yaw_rate_rps` (default and qualified a quarter turn per second, hard
bound a full turn in two seconds). The file can set any value up to the hard
bound; one malformed field rejects the whole document and the current settings
stay, with the reason reported once on the console and on the stream as
`settings/error`. The applied values go out every frame as `settings/version`
and `settings/<key>`, shown beside each slider (red past qualified) so the
dashboard displays what the launcher runs with rather than what it last wrote.
`max_speed_mps` scales the right stick's planar command; `max_yaw_rate_rps`
bounds how fast the heading goal turns toward where the left stick points.
Settings are state, not events: they survive a dashboard restart, and nothing
in the file starts, stops or switches a motion. A launcher start removes the
file, so every run begins from the task's defaults rather than whatever the
previous operator had pulled the caps to. `selected_motion` only names
what LB plays; a playing motion runs to its end or until LB. A dashboard
started by hand on a record (`--file`) has no settings file and no Settings
tab. In a script a `track` segment with a `motion` name plays the same path.

## Simulation qualification

`--teleop-script file.json` replaces the operator in MuJoCo with timed
segments (`idle`, `steer` with heading and velocity, `teleop` with per-hand
forward/lateral/height in the support frame). Disturbances for qualification,
simulation only: `--sim-push time,fx,fy,fz[,duration]` (world-frame force on
`torso_link`, repeatable), `--sim-friction mu` (floor), `--sim-hand-mass kg`
(added to each rubber hand, inertia scaled). `tools/teleop_qualification.py`
runs the grid and scores each record: upright at the end, hand error at hold,
settle time, planted-foot slip, recovery time after a push, energy.

There is one shipped configuration and no flags to choose another; the
comparisons that selected it are in the deployment report. A passing
simulation grid is not physical qualification; the teleop mode and the arm
correction were not exercised on the physical G1 yet.
