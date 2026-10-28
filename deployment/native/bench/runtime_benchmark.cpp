// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

#include "protomotions/deployment/runtime.hpp"

using namespace protomotions::deployment;

namespace {

class FakeBackend final : public RobotBackend {
 public:
  explicit FakeBackend(std::size_t dof_count) {
    capabilities_.dof_count = dof_count;
    capabilities_.supports_position = true;
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

  bool send_command(const ActuatorCommand&) override { return true; }
  bool reset_anchor_frame() override { return true; }
  bool emergency_stop() noexcept override { return true; }

 private:
  RobotCapabilities capabilities_;
  std::uint64_t sequence_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
  const int samples = argc > 1 ? std::max(1000, std::atoi(argv[1])) : 10000;
  FakeBackend backend(29);
  ControllerManager manager;
  manager.add(std::make_unique<CallbackController>("benchmark", [](const RobotState&) {
    ActuatorCommand command;
    command.values.assign(29, 0.0);
    return command;
  }));
  SafetyLimits limits;
  limits.max_abs_command = 2.0;
  DeploymentRuntime runtime(backend, manager, SafetySupervisor(limits));
  if (!runtime.start()) return 1;
  if (!runtime.request_controller("benchmark")) return 1;
  std::vector<double> durations;
  std::vector<double> state_to_command;
  std::vector<double> state_to_send;
  durations.reserve(samples);
  state_to_command.reserve(samples);
  state_to_send.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    const auto start = std::chrono::steady_clock::now();
    if (runtime.run_cycle(start) != CycleResult::kSent) return 2;
    const auto& timing = runtime.last_cycle_timing();
    if (!timing.valid()) return 3;
    state_to_command.push_back(timing.state_to_command().count() / 1000.0);
    state_to_send.push_back(timing.state_to_send().count() / 1000.0);
    durations.push_back(static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count()) /
                       1000.0);
  }
  std::sort(durations.begin(), durations.end());
  auto percentile = [&durations](double p) {
    return durations[static_cast<std::size_t>(p * (durations.size() - 1))];
  };
  auto percentile_of = [](std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>(p * (values.size() - 1))];
  };
  std::cout << "{\"status\":\"ok\",\"backend\":\"protomotions-runtime\","
               "\"p50_us\":"
            << percentile(0.50) << ",\"p95_us\":" << percentile(0.95)
            << ",\"p99_us\":" << percentile(0.99) << ",\"samples\":"
            << samples << ",\"boundary\":\"after RobotBackend::read_state -> before RobotBackend::send_command\","
               "\"state_to_command_p50_us\":"
            << percentile_of(state_to_command, 0.50)
            << ",\"state_to_command_p95_us\":"
            << percentile_of(state_to_command, 0.95)
            << ",\"state_to_command_p99_us\":"
            << percentile_of(state_to_command, 0.99)
            << ",\"state_to_send_p50_us\":"
            << percentile_of(state_to_send, 0.50)
            << ",\"state_to_send_p95_us\":"
            << percentile_of(state_to_send, 0.95)
            << ",\"state_to_send_p99_us\":"
            << percentile_of(state_to_send, 0.99)
            << ",\"failures\":0,\"stale_state_rejections\":0,"
               "\"command_send_failures\":0}\n";
  return 0;
}
