// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include "protomotions/deployment/policy.hpp"
#include "protomotions/deployment/input.hpp"
#include "protomotions/deployment/unitree_input.hpp"
#include "protomotions/deployment/latest_mailbox.hpp"
#include "protomotions/deployment/runtime.hpp"
#include "protomotions/deployment/scheduler.hpp"
#include "protomotions/deployment/unitree_sdk2_backend.hpp"

using namespace protomotions::deployment;

namespace {

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      std::cerr << "check failed: " #condition << " at " << __FILE__ << ":" \
                << __LINE__ << '\n';                                           \
      return 1;                                                                 \
    }                                                                           \
  } while (false)

class FakeBackend final : public RobotBackend {
 public:
  enum class ReadMode {
    kValid,
    kStale,
    kNonFinite,
    kWrongDimension,
    kInvalidOrientation,
    kWrongTorque,
  };

  FakeBackend() {
    caps.dof_count = 2;
    caps.supports_position = true;
    caps.supports_anchor_reset = true;
  }

  const RobotCapabilities& capabilities() const noexcept override { return caps; }

  bool read_state(RobotState& state) override {
    if (throw_on_read) throw std::runtime_error("read failure");
    if (read_delay.count() > 0) std::this_thread::sleep_for(read_delay);
    if (!healthy) return false;
    state.timestamp = std::chrono::steady_clock::now();
    state.sequence = zero_sequence ? 0 : (repeat_sequence ? 1 :
                     (freeze_sequence ? sequence : ++sequence));
    state.dof_position.assign(read_mode == ReadMode::kWrongDimension ? 1 : 2,
                              0.0);
    state.dof_velocity.assign(2, 0.0);
    if (read_mode == ReadMode::kStale) {
      state.timestamp -= std::chrono::milliseconds(200);
    } else if (read_mode == ReadMode::kNonFinite) {
      state.dof_position[0] = std::nan("");
    } else if (read_mode == ReadMode::kInvalidOrientation) {
      state.base_orientation_xyzw = {0.0, 0.0, 0.0, 0.0};
    } else if (read_mode == ReadMode::kWrongTorque) {
      state.dof_torque.assign(1, 0.0);
    }
    return true;
  }

  bool send_command(const ActuatorCommand& command) override {
    if (throw_on_send) throw std::runtime_error("send failure");
    sent = command.values.size() == 2;
    last_command = command;
    ++send_count;
    return send_ok;
  }

  bool reset_anchor_frame() override {
    if (throw_on_reset) throw std::runtime_error("reset failure");
    return healthy && reset_ok;
  }
  bool emergency_stop() noexcept override {
    stopped = true;
    return stop_ok;
  }

  RobotCapabilities caps;
  ReadMode read_mode = ReadMode::kValid;
  bool healthy = true;
  bool send_ok = true;
  bool sent = false;
  bool stopped = false;
  bool stop_ok = true;
  bool repeat_sequence = false;
  bool freeze_sequence = false;
  int send_count = 0;
  ActuatorCommand last_command;
  bool zero_sequence = false;
  std::uint64_t sequence = 0;
  bool reset_ok = true;
  std::chrono::milliseconds read_delay{0};
  bool throw_on_read = false;
  bool throw_on_send = false;
  bool throw_on_reset = false;
};

ActuatorCommand position_command(const RobotState&) {
  ActuatorCommand out;
  out.mode = ControlMode::kPosition;
  out.values = {0.1, -0.1};
  return out;
}

ActuatorCommand torque_command(const RobotState&) {
  ActuatorCommand out;
  out.mode = ControlMode::kTorque;
  out.values = {0.1, -0.1};
  return out;
}

ActuatorCommand invalid_command(const RobotState&) {
  ActuatorCommand out = position_command({});
  out.values[0] = 3.0;
  return out;
}

SafetySupervisor safety() {
  SafetyLimits limits;
  limits.max_abs_command = 2.0;
  return SafetySupervisor(limits);
}

SafetySupervisor tight_safety() {
  SafetyLimits limits;
  limits.max_abs_command = 2.0;
  limits.max_cycle_interval = std::chrono::milliseconds(1);
  return SafetySupervisor(limits);
}

class FakePolicyRuntime final : public PolicyRuntime {
 public:
  bool load(const PolicyArtifactConfig& config) override {
    if (throw_on_load_) throw std::runtime_error("provider load failure");
    loaded_ = load_ok_ && config.backend == PolicyBackend::kCallback &&
              !config.artifact_path.empty();
    return loaded_;
  }
  bool loaded() const noexcept override { return loaded_; }
  ActuatorCommand step(const RobotState&) override { return position_command({}); }
  void stop() noexcept override {
    loaded_ = false;
    stop_called_ = true;
  }

  bool load_ok_ = true;
  bool throw_on_load_ = false;
  bool stop_called_ = false;

 private:
  bool loaded_ = false;
};

class HeartbeatInputSource final : public InputSource {
 public:
  std::string_view name() const noexcept override { return "heartbeat"; }
  InputDevice device() const noexcept override { return InputDevice::kXbox; }

  bool poll(InputState& state) override {
    state.timestamp = std::chrono::steady_clock::now();
    state.sequence = 1;
    state.buttons = xbox_button::kA;
    state.connected = true;
    return true;
  }
};

bool faulted_cycle(FakeBackend& backend, PolicyStep callback,
                   CycleResult expected,
                   SafetySupervisor supervisor = safety()) {
  const auto read_mode = backend.read_mode;
  const bool throw_on_read = backend.throw_on_read;
  const bool healthy = backend.healthy;
  backend.read_mode = FakeBackend::ReadMode::kValid;
  backend.throw_on_read = false;
  backend.healthy = true;
  ControllerManager manager;
  manager.add(std::make_unique<CallbackController>("test", std::move(callback)));
  DeploymentRuntime runtime(backend, manager, std::move(supervisor));
  if (!runtime.start()) return false;
  if (!runtime.request_controller("test")) return false;
  backend.read_mode = read_mode;
  backend.throw_on_read = throw_on_read;
  backend.healthy = healthy;
  return runtime.run_cycle(std::chrono::steady_clock::now()) == expected &&
         runtime.lifecycle() == LifecycleState::kFault && backend.stopped;
}

}  // namespace

class AckController final : public Controller {
 public:
  std::string_view name() const noexcept override { return "ack"; }
  bool start(const RobotState&) override { return true; }
  ActuatorCommand step(const RobotState& state) override {
    return position_command(state);
  }
  void on_command_accepted(const ActuatorCommand&) override { ++accepted; }
  void stop() noexcept override {}
  int accepted = 0;
};

int main() {
  // Incoming raw-action history advances only after an accepted new command.
  {
    FakeBackend backend;
    ControllerManager manager;
    auto controller = std::make_unique<AckController>();
    auto* observer = controller.get();
    CHECK(manager.add(std::move(controller)));
    SafetyLimits limits;
    limits.max_abs_command = 2.0;
    limits.repeated_state_grace = std::chrono::milliseconds(60);
    DeploymentRuntime runtime(backend, manager, SafetySupervisor(limits));
    CHECK(runtime.start() && runtime.request_controller("ack"));
    CHECK(runtime.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kSent);
    CHECK(observer->accepted == 1);
    backend.freeze_sequence = true;
    CHECK(runtime.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kWaitingForState);
    CHECK(observer->accepted == 1);
    backend.freeze_sequence = false;
    backend.send_ok = false;
    CHECK(runtime.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kBackendFault);
    CHECK(observer->accepted == 1);
  }

  // Packet jitter must not turn a finite position command into a damping
  // latch. No controller/history advancement is allowed for repeated samples.
  {
    FakeBackend b;
    ControllerManager m;
    int calls = 0;
    m.add(std::make_unique<CallbackController>("gap", [&](const RobotState& s) {
      ++calls;
      return position_command(s);
    }));
    SafetyLimits l;
    l.max_abs_command = 2;
    l.repeated_state_grace = std::chrono::milliseconds(60);
    DeploymentRuntime r(b, m, SafetySupervisor(l));
    CHECK(r.start());
    CHECK(r.request_controller("gap"));
    // Duplicate immediately after startup stays safely idle, no inference.
    b.freeze_sequence = true;
    CHECK(r.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kWaitingForState);
    CHECK(calls == 0 && r.lifecycle() == LifecycleState::kSafeIdle);
    b.freeze_sequence = false;
    CHECK(r.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kSent);
    const auto stamp = r.last_state().timestamp;
    const auto command_stamp = b.last_command.timestamp;
    b.stopped = false;
    b.freeze_sequence = true;
    for (const int delay : {20, 40}) {
      CHECK(r.run_cycle(stamp + std::chrono::milliseconds(delay)) == CycleResult::kWaitingForState);
      CHECK(calls == 1 && !b.stopped && r.lifecycle() == LifecycleState::kActive);
      CHECK(b.last_command.timestamp == command_stamp);
      CHECK(b.last_command.values == position_command({}).values);
      CHECK(!r.last_cycle_timing().valid());
    }
    // Missing sequence numbers are allowed; only repeated states are paused.
    b.freeze_sequence = false;
    b.sequence += 5;
    CHECK(r.run_cycle(stamp + std::chrono::milliseconds(50)) == CycleResult::kSent);
    CHECK(calls == 2 && r.fault_count() == 0 && r.held_cycle_count() == 3);
    // Repeated callbacks have new fake timestamps: these MUST NOT renew grace.
    b.freeze_sequence = true;
    CHECK(r.run_cycle(r.last_state().timestamp + std::chrono::milliseconds(61)) == CycleResult::kSafetyFault);
    CHECK(calls == 2 && b.stopped && r.lifecycle() == LifecycleState::kFault);
    CHECK(r.last_fault_reason().find("hold") != std::string::npos);
    b.freeze_sequence = false;
    CHECK(r.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kSafeStop);
    CHECK(calls == 2);  // no automatic re-arm after sustained loss
  }
  for (int failure = 0; failure < 6; ++failure) {
    FakeBackend b;
    b.caps.supports_torque = true;
    ControllerManager m;
    m.add(std::make_unique<CallbackController>("gap", failure == 4 ? torque_command : position_command));
    SafetyLimits l;
    l.max_abs_command = 2;
    l.repeated_state_grace = std::chrono::milliseconds(60);
    DeploymentRuntime r(b, m, SafetySupervisor(l));
    CHECK(r.start() && r.request_controller("gap"));
    CHECK(r.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kSent);
    b.freeze_sequence = true;
    if (failure == 0) b.read_mode = FakeBackend::ReadMode::kNonFinite;
    if (failure == 1) --b.sequence;
    if (failure == 2) b.send_ok = false;
    if (failure == 3) b.read_mode = FakeBackend::ReadMode::kStale;
    if (failure == 5) r.emergency_stop();
    CHECK(r.run_cycle(std::chrono::steady_clock::now()) != CycleResult::kWaitingForState);
    CHECK(r.lifecycle() == LifecycleState::kFault);
  }
  // A packet arriving after an observed gap's budget cannot auto-rescue it.
  {
    FakeBackend b;
    ControllerManager m;
    m.add(std::make_unique<CallbackController>("gap", position_command));
    SafetyLimits l;
    l.max_abs_command = 2;
    l.repeated_state_grace = std::chrono::milliseconds(60);
    DeploymentRuntime r(b, m, SafetySupervisor(l));
    CHECK(r.start() && r.request_controller("gap"));
    CHECK(r.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kSent);
    const auto stamp = r.last_state().timestamp;
    b.freeze_sequence = true;
    CHECK(r.run_cycle(stamp + std::chrono::milliseconds(20)) == CycleResult::kWaitingForState);
    b.freeze_sequence = false;
    CHECK(r.run_cycle(stamp + std::chrono::milliseconds(60)) == CycleResult::kSafetyFault);
    CHECK(r.last_fault_reason() == "sensor gap exceeded bounded hold deadline");
  }
  CHECK(dex3_position_motor_mode() == 1U);
  const UnitreeSdk2BackendConfig unitree_defaults;
  CHECK(unitree_defaults.align_born_place);
  CHECK(!unitree_defaults.sensor_only);
  CHECK(unitree_defaults.max_odometry_age == std::chrono::milliseconds(0));
  CHECK(unitree_defaults.command_period_seconds == 0.002);
  CHECK((unitree_defaults.root_position_without_odometry ==
         std::array<double, 3>{0.0, 0.0, 0.0}));
  CHECK(unitree_defaults.dex3_stiffness[0] == 2.0);
  CHECK(unitree_defaults.dex3_stiffness[1] == 1.0);
  CHECK(unitree_defaults.dex3_damping[0] == 0.5);
  CHECK(unitree_defaults.dex3_damping[1] == 0.2);
  CHECK(unitree_defaults.dex3_emergency_damping == 0.1);

  auto keyboard = std::make_unique<BufferedInputSource>(
      "keyboard", InputDevice::kKeyboard);
  auto* keyboard_ptr = keyboard.get();
  InputRouter input_router;
  CHECK(input_router.add_source(std::move(keyboard)));
  CHECK(!input_router.add_source(
      std::make_unique<BufferedInputSource>("keyboard", InputDevice::kKeyboard)));
  CHECK(input_router.bind("keyboard", 1U << 0, "idle"));
  CHECK(input_router.bind("keyboard", (1U << 1) | (1U << 2), "steering"));
  InputState input_state;
  input_state.connected = true;
  input_state.buttons = 1U << 0;
  CHECK(keyboard_ptr->publish(input_state));
  const auto idle_event = input_router.poll();
  CHECK(idle_event &&
        idle_event->type == InputEvent::Type::kControllerSwitch &&
        idle_event->controller == "idle");
  CHECK(!input_router.poll());
  input_state.sequence = 2;
  input_state.buttons = (1U << 1) | (1U << 2);
  CHECK(keyboard_ptr->publish(input_state));
  const auto steering_event = input_router.poll();
  CHECK(steering_event &&
        steering_event->type == InputEvent::Type::kControllerSwitch &&
        steering_event->controller == "steering");
  input_state.sequence = 3;
  input_state.buttons = 1U << 1;
  CHECK(keyboard_ptr->publish(input_state));
  CHECK(!input_router.poll());

  auto future_source = std::make_unique<BufferedInputSource>(
      "future", InputDevice::kKeyboard);
  auto* future_source_ptr = future_source.get();
  CHECK(input_router.add_source(std::move(future_source)));
  InputState future_state;
  future_state.connected = true;
  future_state.sequence = 1;
  CHECK(future_source_ptr->publish(future_state));
  CHECK(!input_router.poll());
  future_state.sequence = 2;
  future_state.timestamp = std::chrono::steady_clock::now() +
                           std::chrono::seconds(1);
  CHECK(future_source_ptr->publish(future_state));
  const auto future_event = input_router.poll();
  CHECK(future_event &&
        future_event->type == InputEvent::Type::kEmergencyStop);
  CHECK(!future_source_ptr->publish(future_state));

  CHECK(input_router.bind_emergency_stop("keyboard", xbox_button::kX));
  CHECK(input_router.bind_revive("keyboard", xbox_button::kY));
  CHECK(input_router.bind_speed_limit("keyboard", xbox_button::kDpadLeft, 0));
  input_state.sequence = 4;
  input_state.buttons = xbox_button::kX;
  CHECK(keyboard_ptr->publish(input_state));
  const auto stop_event = input_router.poll();
  CHECK(stop_event && stop_event->type == InputEvent::Type::kEmergencyStop);
  input_state.sequence = 5;
  input_state.buttons = xbox_button::kY;
  CHECK(keyboard_ptr->publish(input_state));
  const auto revive_event = input_router.poll();
  CHECK(revive_event && revive_event->type == InputEvent::Type::kRevive);
  input_state.sequence = 6;
  input_state.buttons = xbox_button::kDpadLeft;
  CHECK(keyboard_ptr->publish(input_state));
  const auto speed_event = input_router.poll();
  CHECK(speed_event && speed_event->type == InputEvent::Type::kSpeedLimit &&
        speed_event->speed_level == 0);

  // A held controller state must refresh its liveness timestamp even when its
  // sequence does not change; otherwise the source timeout would falsely
  // emergency-stop a healthy operator input.
  InputRouter heartbeat_router(std::chrono::milliseconds(5));
  CHECK(heartbeat_router.add_source(std::make_unique<HeartbeatInputSource>()));
  CHECK(heartbeat_router.bind("heartbeat", xbox_button::kA, "idle"));
  const auto heartbeat_event = heartbeat_router.poll();
  CHECK(heartbeat_event &&
        heartbeat_event->type == InputEvent::Type::kControllerSwitch &&
        heartbeat_event->controller == "idle");
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(!heartbeat_router.poll());

  auto xbox = std::make_unique<BufferedInputSource>("xbox", InputDevice::kXbox);
  auto* xbox_ptr = xbox.get();
  CHECK(input_router.add_source(std::move(xbox)));
  CHECK(input_router.bind_emergency_stop("xbox", xbox_button::kX));
  // Safety edges are global: an emergency stop from a later source wins over
  // an ordinary controller request from an earlier source in the same poll.
  input_state.sequence = 7;
  input_state.buttons = 1U << 0;
  CHECK(keyboard_ptr->publish(input_state));
  InputState xbox_state;
  xbox_state.sequence = 1;
  xbox_state.connected = true;
  xbox_state.buttons = xbox_button::kX;
  CHECK(xbox_ptr->publish(xbox_state));
  const auto global_stop_event = input_router.poll();
  CHECK(global_stop_event &&
        global_stop_event->type == InputEvent::Type::kEmergencyStop);
  input_state.sequence = 8;
  input_state.timestamp = std::chrono::steady_clock::now() -
                          std::chrono::milliseconds(1000);
  input_state.connected = true;
  CHECK(xbox_ptr->publish(input_state));
  const auto stale_input_event = input_router.poll();
  CHECK(stale_input_event &&
        stale_input_event->type == InputEvent::Type::kEmergencyStop);
  input_state.sequence = 9;
  input_state.timestamp = std::chrono::steady_clock::now();
  input_state.connected = true;
  CHECK(xbox_ptr->publish(input_state));
  CHECK(!input_router.poll());
  input_state.sequence = 10;
  input_state.connected = false;
  CHECK(xbox_ptr->publish(input_state));
  const auto disconnect_event = input_router.poll();
  CHECK(disconnect_event &&
        disconnect_event->type == InputEvent::Type::kEmergencyStop);

  // The runtime must reject an unbounded command limit before arming rather
  // than waiting for the first controller cycle to fault.
  FakeBackend unbounded_backend;
  ControllerManager unbounded_manager;
  CHECK(unbounded_manager.add(
      std::make_unique<CallbackController>("idle", position_command)));
  DeploymentRuntime unbounded_runtime(unbounded_backend, unbounded_manager,
                                      SafetySupervisor{});
  CHECK(!unbounded_runtime.start());
  CHECK(unbounded_runtime.lifecycle() == LifecycleState::kFault);

  std::array<std::uint8_t, 40> remote{};
  remote[3] = (1U << 0) | (1U << 4) | (1U << 5) | (1U << 6) |
               (1U << 7);  // A plus all four Unitree D-pad bits.
  float lx = 0.25F;
  std::memcpy(remote.data() + 4, &lx, sizeof(lx));
  const auto decoded = decode_unitree_remote(remote, 4);
  CHECK(decoded.connected);
  CHECK((decoded.buttons & (1U << 0)) != 0);
  CHECK((decoded.buttons & xbox_button::kDpadUp) != 0);
  CHECK((decoded.buttons & xbox_button::kDpadRight) != 0);
  CHECK((decoded.buttons & xbox_button::kDpadDown) != 0);
  CHECK((decoded.buttons & xbox_button::kDpadLeft) != 0);
  CHECK(std::abs(decoded.axes[0] - 0.25F) < 1e-6F);
  remote[2] = 1U << 4;  // Unitree R2 remains bindable in the normalized map.
  const auto trigger_decoded = decode_unitree_remote(remote, 5);
  CHECK((trigger_decoded.buttons & xbox_button::kRightTrigger) != 0);
  float invalid_axis = 2.0F;
  std::memcpy(remote.data() + 4, &invalid_axis, sizeof(invalid_axis));
  bool rejected_axis = false;
  try {
    (void)decode_unitree_remote(remote, 6);
  } catch (const std::invalid_argument&) {
    rejected_axis = true;
  }
  CHECK(rejected_axis);
  std::memcpy(remote.data() + 4, &lx, sizeof(lx));
  UnitreeControllerInputSource unitree_source;
  CHECK(unitree_source.update(remote, 1));
  CHECK(!unitree_source.update(remote, 1));
  CHECK(unitree_source.update(remote, 2));

  RobotCapabilities invalid_capabilities;
  CHECK(!safety().validate_capabilities(invalid_capabilities));
  invalid_capabilities.dof_count = 2;
  CHECK(!safety().validate_capabilities(invalid_capabilities));

  FakeBackend default_gain_backend;
  default_gain_backend.caps.has_default_position_gains = true;
  default_gain_backend.caps.default_stiffness = 40.0;
  default_gain_backend.caps.default_damping = 1.0;
  SafetyLimits missing_gain_limits;
  missing_gain_limits.max_abs_command = 2.0;
  SafetySupervisor missing_gain_safety(missing_gain_limits);
  ControllerManager missing_gain_manager;
  CHECK(missing_gain_manager.add(
      std::make_unique<CallbackController>("missing-gain", position_command)));
  DeploymentRuntime missing_gain_runtime(default_gain_backend,
                                         missing_gain_manager,
                                         missing_gain_safety);
  CHECK(!missing_gain_runtime.start());
  CHECK(missing_gain_runtime.lifecycle() == LifecycleState::kFault);
  CHECK(!missing_gain_safety.validate_command(
      position_command({}), default_gain_backend.capabilities()));
  SafetyLimits gain_limits;
  gain_limits.max_abs_command = 2.0;
  gain_limits.max_abs_stiffness = 20.0;
  gain_limits.max_abs_damping = 1.0;
  SafetySupervisor gain_safety(gain_limits);
  CHECK(!gain_safety.validate_command(position_command({}),
                                      default_gain_backend.capabilities()));
  gain_limits.max_abs_stiffness = 40.0;
  gain_safety = SafetySupervisor(gain_limits);
  CHECK(gain_safety.validate_command(position_command({}),
                                     default_gain_backend.capabilities()));

  LatestMailbox<RobotState> mailbox;
  CHECK(!mailbox.latest());
  auto mailbox_state = std::make_shared<const RobotState>();
  CHECK(mailbox.publish(mailbox_state, 7));
  CHECK(mailbox.latest() == mailbox_state);
  CHECK(mailbox.latest_sequence() == 7);
  CHECK(!mailbox.publish(mailbox_state, 6));
  CHECK(!mailbox.publish(nullptr, 8));
  CHECK(mailbox.publish(mailbox_state, 8));
  const auto snapshot = mailbox.snapshot();
  CHECK(snapshot.value == mailbox_state);
  CHECK(snapshot.sequence == 8);
  mailbox.clear();
  CHECK(!mailbox.latest());
  CHECK(!mailbox.publish(mailbox_state, 8));
  CHECK(mailbox.publish(mailbox_state, 9));

  FakeBackend backend;
  ControllerManager manager;
  CHECK(manager.add(std::make_unique<CallbackController>("idle", position_command)));
  CHECK(!manager.add(std::make_unique<CallbackController>("idle", position_command)));

  DeploymentRuntime runtime(backend, manager, safety());
  CHECK(runtime.start());
  CHECK(runtime.reset_anchor_frame());
  CHECK(runtime.request_controller("idle"));
  CHECK(runtime.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kSent);
  CHECK(runtime.lifecycle() == LifecycleState::kActive);
  CHECK(backend.sent);
  CHECK(runtime.last_cycle_timing().valid());
  CHECK(runtime.last_cycle_timing().state_to_command().count() >= 0);
  CHECK(runtime.last_cycle_timing().state_to_send() >=
        runtime.last_cycle_timing().state_to_command());

  FakeBackend replay;
  replay.repeat_sequence = true;
  ControllerManager replay_manager;
  CHECK(replay_manager.add(
      std::make_unique<CallbackController>("replay", position_command)));
  DeploymentRuntime replay_runtime(replay, replay_manager, safety());
  CHECK(replay_runtime.start());
  CHECK(replay_runtime.request_controller("replay"));
  // start() consumes sequence 1 as the barrier. A backend that replays that
  // sample immediately must fault instead of treating it as a fresh frame.
  CHECK(replay_runtime.run_cycle(std::chrono::steady_clock::now()) ==
        CycleResult::kSafetyFault);
  CHECK(replay_runtime.lifecycle() == LifecycleState::kFault);
  CHECK(replay_runtime.revive());
  CHECK(replay_runtime.run_cycle(std::chrono::steady_clock::now()) ==
        CycleResult::kSafetyFault);
  CHECK(replay_runtime.lifecycle() == LifecycleState::kFault);

  FakeBackend start_failure;
  start_failure.stop_ok = false;
  ControllerManager start_failure_manager;
  DeploymentRuntime start_failure_runtime(start_failure,
                                           start_failure_manager, safety());
  CHECK(!start_failure_runtime.start());
  CHECK(start_failure_runtime.lifecycle() == LifecycleState::kFault);

  CHECK(manager.add(std::make_unique<CallbackController>("second", position_command)));
  CHECK(runtime.request_controller("second"));
  CHECK(runtime.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kSent);
  CHECK(manager.active_name() == "second");
  CHECK(!runtime.start());

  FakeBackend stop_failure_backend;
  ControllerManager stop_failure_manager;
  DeploymentRuntime stop_failure_runtime(stop_failure_backend,
                                         stop_failure_manager, safety());
  CHECK(stop_failure_runtime.start());
  stop_failure_backend.stop_ok = false;
  CHECK(!stop_failure_runtime.stop());
  CHECK(stop_failure_runtime.lifecycle() == LifecycleState::kFault);

  runtime.emergency_stop();
  CHECK(runtime.lifecycle() == LifecycleState::kFault);
  CHECK(runtime.revive());
  CHECK(runtime.lifecycle() == LifecycleState::kSafeIdle);
  CHECK(runtime.run_cycle(std::chrono::steady_clock::now()) ==
        CycleResult::kNoController);

  backend.healthy = false;
  CHECK(runtime.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kNoState);
  CHECK(runtime.lifecycle() == LifecycleState::kFault);
  CHECK(backend.stopped);
  backend.stopped = false;
  CHECK(runtime.run_cycle(std::chrono::steady_clock::now()) ==
        CycleResult::kSafeStop);
  CHECK(backend.stopped);

  FakeBackend stale;
  stale.read_mode = FakeBackend::ReadMode::kStale;
  CHECK(faulted_cycle(stale, position_command, CycleResult::kSafetyFault));
  FakeBackend nonfinite;
  nonfinite.read_mode = FakeBackend::ReadMode::kNonFinite;
  CHECK(faulted_cycle(nonfinite, position_command, CycleResult::kSafetyFault));
  FakeBackend wrong_dimension;
  wrong_dimension.read_mode = FakeBackend::ReadMode::kWrongDimension;
  CHECK(faulted_cycle(wrong_dimension, position_command,
                      CycleResult::kSafetyFault));
  FakeBackend invalid_orientation;
  invalid_orientation.read_mode = FakeBackend::ReadMode::kInvalidOrientation;
  CHECK(faulted_cycle(invalid_orientation, position_command,
                      CycleResult::kSafetyFault));
  FakeBackend wrong_torque;
  wrong_torque.read_mode = FakeBackend::ReadMode::kWrongTorque;
  CHECK(faulted_cycle(wrong_torque, position_command,
                      CycleResult::kSafetyFault));
  FakeBackend unsupported;
  CHECK(faulted_cycle(unsupported, torque_command, CycleResult::kSafetyFault));
  FakeBackend invalid;
  CHECK(faulted_cycle(invalid, invalid_command, CycleResult::kSafetyFault));
  FakeBackend send_failure;
  send_failure.send_ok = false;
  CHECK(faulted_cycle(send_failure, position_command,
                      CycleResult::kBackendFault));
  FakeBackend throwing_read;
  throwing_read.throw_on_read = true;
  CHECK(faulted_cycle(throwing_read, position_command,
                     CycleResult::kBackendFault));
  FakeBackend throwing_send;
  throwing_send.throw_on_send = true;
  CHECK(faulted_cycle(throwing_send, position_command,
                     CycleResult::kBackendFault));

  FakeBackend reset_failure;
  ControllerManager reset_manager;
  reset_manager.add(std::make_unique<CallbackController>("reset", position_command));
  DeploymentRuntime reset_runtime(reset_failure, reset_manager, safety());
  CHECK(reset_runtime.start());
  CHECK(reset_runtime.request_controller("reset"));
  CHECK(reset_runtime.run_cycle(std::chrono::steady_clock::now()) == CycleResult::kSent);
  reset_failure.reset_ok = false;
  CHECK(!reset_runtime.reset_anchor_frame());
  CHECK(reset_runtime.lifecycle() == LifecycleState::kFault);
  FakeBackend throwing_reset;
  throwing_reset.throw_on_reset = true;
  ControllerManager throwing_reset_manager;
  throwing_reset_manager.add(
      std::make_unique<CallbackController>("reset", position_command));
  DeploymentRuntime throwing_reset_runtime(throwing_reset,
                                           throwing_reset_manager, safety());
  CHECK(throwing_reset_runtime.start());
  CHECK(!throwing_reset_runtime.reset_anchor_frame());
  CHECK(throwing_reset_runtime.lifecycle() == LifecycleState::kFault);

  FakeBackend throwing_backend;
  CHECK(faulted_cycle(throwing_backend,
                      [](const RobotState&) -> ActuatorCommand {
                        throw std::runtime_error("controller failure");
                      },
                      CycleResult::kBackendFault));
  FakeBackend slow_backend;
  ControllerManager slow_manager;
  slow_manager.add(std::make_unique<CallbackController>(
      "slow", [](const RobotState& state) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return position_command(state);
      }));
  DeploymentRuntime slow_runtime(slow_backend, slow_manager, tight_safety());
  CHECK(slow_runtime.start());
  CHECK(slow_runtime.request_controller("slow"));
  CHECK(slow_runtime.run_cycle(std::chrono::steady_clock::now()) ==
        CycleResult::kSafetyFault);
  CHECK(slow_runtime.lifecycle() == LifecycleState::kFault);
  FakeBackend slow_read_backend;
  slow_read_backend.read_delay = std::chrono::milliseconds(2);
  CHECK(faulted_cycle(slow_read_backend, position_command,
                      CycleResult::kSafetyFault, tight_safety()));

  FakeBackend zero_sequence_backend;
  ControllerManager zero_sequence_manager;
  zero_sequence_manager.add(std::make_unique<CallbackController>(
      "test", position_command));
  DeploymentRuntime zero_sequence_runtime(zero_sequence_backend,
                                           zero_sequence_manager, safety());
  CHECK(zero_sequence_runtime.start());
  CHECK(zero_sequence_runtime.request_controller("test"));
  zero_sequence_backend.zero_sequence = true;
  CHECK(zero_sequence_runtime.run_cycle(std::chrono::steady_clock::now()) ==
        CycleResult::kSafetyFault);
  CHECK(zero_sequence_runtime.lifecycle() == LifecycleState::kFault);

  PolicyArtifactConfig config;
  config.artifact_path = "memory://test";
  CHECK(policy_backend_from_name("tensorrt") == PolicyBackend::kTensorRt);
  CHECK(!policy_backend_from_name("unknown"));
  ControllerPtr artifact = std::make_unique<ArtifactController>(
      "artifact", config, std::make_unique<FakePolicyRuntime>());
  RobotState state;
  CHECK(artifact->start(state));
  artifact->stop();

  auto failing_runtime = std::make_unique<FakePolicyRuntime>();
  auto* failing_runtime_ptr = failing_runtime.get();
  failing_runtime_ptr->load_ok_ = false;
  ControllerPtr failing_artifact = std::make_unique<ArtifactController>(
      "failing-artifact", config, std::move(failing_runtime));
  CHECK(!failing_artifact->start(state));
  CHECK(failing_runtime_ptr->stop_called_);

  auto throwing_runtime = std::make_unique<FakePolicyRuntime>();
  auto* throwing_runtime_ptr = throwing_runtime.get();
  throwing_runtime_ptr->throw_on_load_ = true;
  ControllerPtr throwing_artifact = std::make_unique<ArtifactController>(
      "throwing-artifact", config, std::move(throwing_runtime));
  bool load_threw = false;
  try {
    (void)throwing_artifact->start(state);
  } catch (const std::runtime_error&) {
    load_threw = true;
  }
  CHECK(load_threw);
  CHECK(!throwing_runtime_ptr->stop_called_);

  auto sensor_runtime = std::make_unique<FakePolicyRuntime>();
  PolicyArtifactConfig sensor_config = config;
  sensor_config.requires_odometry = true;
  ControllerPtr sensor_artifact = std::make_unique<ArtifactController>(
      "sensor-artifact", sensor_config, std::move(sensor_runtime));
  CHECK(!sensor_artifact->start(state));

  FixedRateScheduler scheduler(std::chrono::microseconds(10));
  CHECK(!scheduler.wait());
  const auto scheduler_start = std::chrono::steady_clock::now();
  scheduler.reset(scheduler_start);
  scheduler.advance(scheduler_start + std::chrono::microseconds(25));
  CHECK(scheduler.overruns() == 2);
  FixedRateScheduler invalid_scheduler(std::chrono::microseconds(0));
  CHECK(!invalid_scheduler.valid());
  invalid_scheduler.reset(std::chrono::steady_clock::now());
  CHECK(!invalid_scheduler.wait());
  return 0;
}
