// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>

#include "protomotions/deployment/backend.hpp"
#include "protomotions/deployment/controller_manager.hpp"
#include "protomotions/deployment/safety.hpp"

namespace protomotions::deployment {

enum class CycleResult {
  kSent,
  // The runtime is fault-latched, so no controller is evaluated. The backend
  // accepted the safe damping/hold command for this cycle.
  kSafeStop,
  kNoState,
  kNoController,
  kSafetyFault,
  kBackendFault,
  // No inference or history advancement. A bounded position/safe-idle hold
  // was accepted while waiting for the next sensor sequence.
  kWaitingForState,
};

// Timestamps for the comparable control-cycle boundary. `state_ready` is
// captured immediately after RobotBackend::read_state returns; `command_ready`
// is captured immediately before RobotBackend::send_command is called. The
// interval between them is the state-read -> action-send work being compared
// across deployment frameworks. `send_done` is retained for transport cost.
struct CycleTiming {
  std::chrono::steady_clock::time_point state_ready{};
  std::chrono::steady_clock::time_point command_ready{};
  std::chrono::steady_clock::time_point send_done{};

  bool valid() const noexcept {
    return state_ready.time_since_epoch().count() != 0 &&
           command_ready >= state_ready && send_done >= command_ready;
  }
  std::chrono::nanoseconds state_to_command() const noexcept {
    return command_ready - state_ready;
  }
  std::chrono::nanoseconds state_to_send() const noexcept {
    return send_done - state_ready;
  }
};

class DeploymentRuntime {
 public:
  DeploymentRuntime(RobotBackend& backend, ControllerManager& controllers,
                    SafetySupervisor safety = SafetySupervisor(SafetyLimits{}))
      : backend_(backend), controllers_(controllers), safety_(std::move(safety)) {}

  bool start();
  // Returns false when the backend rejected the safe output. A failed stop
  // leaves the runtime faulted so callers cannot mistake it for a clean halt.
  bool stop() noexcept;
  void emergency_stop() noexcept;
  // Re-arms a faulted runtime in safe idle. No controller is selected; the
  // operator must explicitly request one before actuation resumes. While
  // faulted, run_cycle() continues sending the backend's safe command so a
  // robot transport is not left with a one-shot damping packet.
  bool revive() noexcept;
  bool reset_anchor_frame();
  bool request_controller(const std::string& name);
  CycleResult run_cycle(std::chrono::steady_clock::time_point now);

  LifecycleState lifecycle() const noexcept { return lifecycle_; }
  std::uint64_t fault_count() const noexcept { return fault_count_; }
  const std::string& last_fault_reason() const noexcept { return last_fault_reason_; }
  std::uint64_t held_cycle_count() const noexcept { return held_cycle_count_; }
  const RobotState& last_state() const noexcept { return last_state_; }
  const CycleTiming& last_cycle_timing() const noexcept { return last_timing_; }

 private:
  // Clear timing measurements without discarding the latest-only sequence
  // barrier unless the backend explicitly reports a sequence restart. A
  // revive, startup, and ordinary anchor reset must not accept a replayed
  // transport sample as a fresh state frame.
  void clear_cycle_tracking(bool clear_sequence) noexcept;
  void fault(const char* reason = "operator emergency stop") noexcept;
  CycleResult hold_repeated_state(std::chrono::steady_clock::time_point now);

  RobotBackend& backend_;
  ControllerManager& controllers_;
  SafetySupervisor safety_;
  RobotState last_state_;
  CycleTiming last_timing_;
  ActuatorCommand last_command_;
  bool has_position_command_ = false;
  bool waiting_for_state_ = false;
  std::uint64_t held_cycle_count_ = 0;
  std::string last_fault_reason_;
  std::chrono::steady_clock::time_point last_cycle_time_{};
  std::uint64_t last_state_sequence_ = 0;
  bool has_state_sequence_ = false;
  LifecycleState lifecycle_ = LifecycleState::kStopped;
  std::uint64_t fault_count_ = 0;
};

}  // namespace protomotions::deployment
