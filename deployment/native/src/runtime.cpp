// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/runtime.hpp"

#include <algorithm>
#include <exception>

namespace protomotions::deployment {

bool DeploymentRuntime::start() {
  if (lifecycle_ == LifecycleState::kFault ||
      lifecycle_ == LifecycleState::kActive) {
    return false;
  }
  const auto fail_start = [this](const std::string& reason) {
    fault(reason.c_str());
    return false;
  };
  RobotState startup_state;
  try {
    std::string startup_reason;
    if (!safety_.validate_capabilities(backend_.capabilities(), &startup_reason))
      return fail_start("startup capabilities: " + startup_reason);
    if (!backend_.read_state(startup_state))
      return fail_start("startup backend returned no complete state");
    if (!safety_.validate_state(startup_state, backend_.capabilities(),
                                std::chrono::steady_clock::now(), &startup_reason))
      return fail_start("startup state: " + startup_reason);
    if (startup_state.sequence == 0)
      return fail_start("startup sensor sequence is zero");
  } catch (const std::exception& error) {
    return fail_start(std::string("startup exception: ") + error.what());
  } catch (...) {
    return fail_start("unknown startup exception");
  }
  controllers_.stop();
  if (!backend_.emergency_stop()) {
    // The failed stop was already attempted above; latch the state without a
    // second transport call.
    lifecycle_ = LifecycleState::kFault;
    last_fault_reason_ = "backend rejected startup safe hold";
    ++fault_count_;
    return false;
  }
  // The startup read is the first sample in this control stream. Keep it as
  // the sequence barrier so a latest-only transport cannot replay that exact
  // sample immediately after arming and have it accepted as new data.
  clear_cycle_tracking(false);
  last_state_ = startup_state;
  last_state_sequence_ = startup_state.sequence;
  has_state_sequence_ = true;
  lifecycle_ = LifecycleState::kSafeIdle;
  return true;
}

bool DeploymentRuntime::stop() noexcept {
  controllers_.stop();
  bool stopped = false;
  try {
    stopped = backend_.emergency_stop();
  } catch (...) {
  }
  clear_cycle_tracking(true);
  lifecycle_ = stopped ? LifecycleState::kStopped : LifecycleState::kFault;
  if (!stopped) ++fault_count_;
  return stopped;
}

void DeploymentRuntime::emergency_stop() noexcept {
  fault();
}

bool DeploymentRuntime::revive() noexcept {
  if (lifecycle_ != LifecycleState::kFault) return false;
  controllers_.stop();
  try {
    // Keep the backend in its known safe output state while the operator
    // chooses the next controller. Re-arm does not silently select a policy.
    if (!backend_.emergency_stop()) return false;
  } catch (...) {
    return false;
  }
  clear_cycle_tracking(false);
  // Keep the sequence anchor across re-arm. A revive is not a new sensor
  // stream: accepting an old/replayed sample here would let stale data reach
  // the controller after an emergency stop.
  lifecycle_ = LifecycleState::kSafeIdle;
  return true;
}

bool DeploymentRuntime::reset_anchor_frame() {
  if (lifecycle_ == LifecycleState::kFault ||
      lifecycle_ == LifecycleState::kStopped) {
    return false;
  }
  controllers_.stop();
  try {
    if (!backend_.emergency_stop()) {
      fault();
      return false;
    }
    // Some transports (notably Unitree SDK2) do not own a policy/world
    // anchor. The controller restart below is the reset boundary for those
    // adapters; only call the backend hook when it advertises one.
    if (backend_.capabilities().supports_anchor_reset &&
        !backend_.reset_anchor_frame()) {
      fault();
      return false;
    }
  } catch (...) {
    fault();
    return false;
  }
  lifecycle_ = LifecycleState::kSafeIdle;
  // Anchor reset changes policy coordinates, not necessarily sensor-stream
  // freshness. Clear the sequence barrier only for a backend that explicitly
  // guarantees that its reset hook restarts the state sequence (MuJoCo does).
  clear_cycle_tracking(backend_.capabilities().resets_state_sequence);
  return true;
}

bool DeploymentRuntime::request_controller(const std::string& name) {
  if (lifecycle_ == LifecycleState::kFault ||
      lifecycle_ == LifecycleState::kStopped) {
    return false;
  }
  return controllers_.request_switch(name);
}

CycleResult DeploymentRuntime::run_cycle(
    std::chrono::steady_clock::time_point now) {
  if (lifecycle_ == LifecycleState::kFault) {
    // Keep the safe output alive at the backend's control rate. Physical
    // transports generally require repeated packets, and MuJoCo must keep
    // stepping the damping controller instead of appearing frozen.
    try {
      return backend_.emergency_stop() ? CycleResult::kSafeStop
                                       : CycleResult::kBackendFault;
    } catch (...) {
      return CycleResult::kBackendFault;
    }
  }
  if (lifecycle_ == LifecycleState::kStopped) {
    return CycleResult::kBackendFault;
  }
  try {
    const auto cycle_begin = std::chrono::steady_clock::now();
    RobotState state;
    if (!backend_.read_state(state)) {
      fault("backend returned no complete state");
      return CycleResult::kNoState;
    }
    CycleTiming timing;
    timing.state_ready = std::chrono::steady_clock::now();
    // The backend may timestamp a sample a few microseconds after the caller
    // captured its cycle deadline. Validate against a current monotonic time so
    // a healthy freshly-read sample is not rejected as being from the future.
    const auto validation_now = std::max(now, std::chrono::steady_clock::now());
    if (validation_now - cycle_begin > safety_.max_cycle_interval()) {
      fault("state read exceeded cycle deadline");
      return CycleResult::kSafetyFault;
    }
    std::string state_reason;
    if (!safety_.validate_state(state, backend_.capabilities(), validation_now,
                                &state_reason)) {
      fault(state_reason.c_str());
      return CycleResult::kSafetyFault;
    }
    // Sequence zero is reserved for an uninitialized sample. Every concrete
    // backend must publish a positive monotonic sequence before actuation.
    if (state.sequence == 0) {
      fault("uninitialized sensor sequence");
      return CycleResult::kSafetyFault;
    }
    if (has_state_sequence_ && state.sequence < last_state_sequence_) {
      fault("sensor sequence moved backwards");
      return CycleResult::kSafetyFault;
    }
    if (waiting_for_state_ &&
        validation_now - last_state_.timestamp >= safety_.repeated_state_grace()) {
      fault("sensor gap exceeded bounded hold deadline");
      return CycleResult::kSafetyFault;
    }
    if (has_state_sequence_ && state.sequence == last_state_sequence_) {
      return hold_repeated_state(validation_now);
    }
    last_state_sequence_ = state.sequence;
    has_state_sequence_ = true;
    if (last_cycle_time_.time_since_epoch().count() != 0 &&
        validation_now - last_cycle_time_ > safety_.max_cycle_interval()) {
      fault("control loop interval exceeded deadline");
      return CycleResult::kSafetyFault;
    }
    auto command = controllers_.step(state);
    if (!command) {
      // Safe idle is an intentional damping state (startup and revive), not a
      // transport fault. Refresh it every cycle: hardware publishes a
      // continuous hold, and MuJoCo must advance under the same damping
      // command rather than freezing the scene until a controller starts.
      if (lifecycle_ != LifecycleState::kSafeIdle) {
        fault("active controller unavailable");
        return CycleResult::kNoController;
      }
      if (!backend_.emergency_stop()) {
        fault("backend rejected safe-idle hold");
        return CycleResult::kBackendFault;
      }
      last_state_ = std::move(state);
      last_timing_ = {};
      waiting_for_state_ = false;
      return CycleResult::kNoController;
    }
    std::string command_reason;
    if (!safety_.validate_command(*command, backend_.capabilities(), &command_reason)) {
      fault(command_reason.c_str());
      return CycleResult::kSafetyFault;
    }
    const auto command_ready_check = std::chrono::steady_clock::now();
    if (command_ready_check - validation_now > safety_.max_cycle_interval()) {
      fault("policy inference exceeded cycle deadline");
      return CycleResult::kSafetyFault;
    }
    command->timestamp = command_ready_check;
    // Keep this marker immediately before the transport call. The timestamp
    // assignment above is part of the state-ready -> command-ready work.
    timing.command_ready = std::chrono::steady_clock::now();
    if (!backend_.send_command(*command)) {
      fault("backend rejected policy command");
      return CycleResult::kBackendFault;
    }
    controllers_.on_command_accepted(*command);
    timing.send_done = std::chrono::steady_clock::now();
    last_state_ = std::move(state);
    last_timing_ = timing;
    last_cycle_time_ = timing.command_ready;
    last_command_ = *command;
    has_position_command_ = command->mode == ControlMode::kPosition;
    waiting_for_state_ = false;
    lifecycle_ = LifecycleState::kActive;
    return CycleResult::kSent;
  } catch (const std::exception& error) {
    fault(error.what());
    return CycleResult::kBackendFault;
  } catch (...) {
    fault("unknown control cycle exception");
    return CycleResult::kBackendFault;
  }
}

CycleResult DeploymentRuntime::hold_repeated_state(
    std::chrono::steady_clock::time_point now) {
  // Age is measured from the LAST ACCEPTED sample, never a repeated callback's
  // timestamp. Repeating a packet cannot renew either deadline.
  if (safety_.repeated_state_grace().count() <= 0 ||
      now < last_state_.timestamp ||
      now - last_state_.timestamp >= safety_.repeated_state_grace() ||
      (last_cycle_time_.time_since_epoch().count() != 0 &&
       now - last_cycle_time_ > safety_.max_cycle_interval())) {
    fault("repeated sensor sequence exceeded hold budget (or hold disabled)");
    return CycleResult::kSafetyFault;
  }
  bool accepted = false;
  if (lifecycle_ == LifecycleState::kSafeIdle) {
    accepted = backend_.emergency_stop();
  } else if (has_position_command_ &&
             safety_.validate_state(last_state_, backend_.capabilities(), now) &&
             safety_.validate_command(last_command_, backend_.capabilities())) {
    // Retain the last validated position target and gains. Do not extrapolate,
    // replay torque/velocity commands, run inference, or refresh its timestamp.
    accepted = backend_.send_command(last_command_);
  } else {
    fault("sensor gap cannot hold this control mode safely");
    return CycleResult::kSafetyFault;
  }
  if (!accepted) {
    fault("backend rejected bounded hold command");
    return CycleResult::kBackendFault;
  }
  waiting_for_state_ = true;
  ++held_cycle_count_;
  last_timing_ = {};
  return CycleResult::kWaitingForState;
}

void DeploymentRuntime::fault(const char* reason) noexcept {
  try { last_fault_reason_ = reason; } catch (...) {}
  controllers_.stop();
  try {
    (void)backend_.emergency_stop();
  } catch (...) {
  }
  clear_cycle_tracking(false);
  lifecycle_ = LifecycleState::kFault;
  ++fault_count_;
}

void DeploymentRuntime::clear_cycle_tracking(bool clear_sequence) noexcept {
  last_cycle_time_ = {};
  last_timing_ = {};
  has_position_command_ = false;
  waiting_for_state_ = false;
  if (clear_sequence) {
    has_state_sequence_ = false;
    last_state_sequence_ = 0;
  }
}

}  // namespace protomotions::deployment
