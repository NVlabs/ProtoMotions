// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <iostream>

#include "protomotions/deployment/runtime.hpp"

using namespace protomotions::deployment;

namespace {

class DemoBackend final : public RobotBackend {
 public:
  DemoBackend() {
    capabilities_.dof_count = 2;
    capabilities_.supports_position = true;
    capabilities_.supports_anchor_reset = true;
    capabilities_.has_projected_gravity = true;
  }

  const RobotCapabilities& capabilities() const noexcept override {
    return capabilities_;
  }

  bool read_state(RobotState& state) override {
    state.timestamp = std::chrono::steady_clock::now();
    state.sequence = ++sequence_;
    state.dof_position.assign(capabilities_.dof_count, 0.0);
    state.dof_velocity.assign(capabilities_.dof_count, 0.0);
    return true;
  }

  bool send_command(const ActuatorCommand& command) override {
    last_command_ = command;
    return true;
  }

  bool reset_anchor_frame() override { return true; }

  bool emergency_stop() noexcept override {
    stopped_ = true;
    return true;
  }

 private:
  RobotCapabilities capabilities_;
  ActuatorCommand last_command_;
  std::uint64_t sequence_ = 0;
  bool stopped_ = false;
};

ActuatorCommand hold_pose(const RobotState&) {
  ActuatorCommand command;
  command.values = {0.0, 0.0};
  return command;
}

}  // namespace

int main() {
  DemoBackend backend;
  ControllerManager controllers;
  controllers.add(std::make_unique<CallbackController>("idle", hold_pose));

  SafetyLimits limits;
  limits.max_abs_command = 2.0;
  DeploymentRuntime runtime(backend, controllers, SafetySupervisor(limits));
  if (!runtime.start()) return 1;
  if (!runtime.request_controller("idle")) return 1;
  const auto now = std::chrono::steady_clock::now();
  const auto result = runtime.run_cycle(now);
  std::cout << "cycle=" << (result == CycleResult::kSent ? "sent" : "failed")
            << " controller=" << controllers.active_name() << '\n';
  runtime.stop();
  return result == CycleResult::kSent ? 0 : 1;
}
