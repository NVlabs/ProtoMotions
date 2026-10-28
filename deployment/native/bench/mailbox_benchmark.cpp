// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "protomotions/deployment/latest_mailbox.hpp"
#include "protomotions/deployment/runtime.hpp"

using namespace protomotions::deployment;

namespace {

struct Summary {
  std::vector<double> state_to_command_us;
  std::vector<double> state_to_send_us;
  std::uint64_t dropped = 0;
  std::uint64_t publish_failures = 0;
};

class BaseBackend : public RobotBackend {
 public:
  BaseBackend() {
    caps_.dof_count = 29;
    caps_.supports_position = true;
  }
  const RobotCapabilities& capabilities() const noexcept override { return caps_; }
  bool send_command(const ActuatorCommand&) override { return true; }
  bool reset_anchor_frame() override { return true; }
  bool emergency_stop() noexcept override { return true; }

 protected:
  RobotCapabilities caps_;
};

class DirectBackend final : public BaseBackend {
 public:
  bool read_state(RobotState& state) override {
    state.sequence = ++sequence_;
    state.timestamp = std::chrono::steady_clock::now();
    state.dof_position.assign(caps_.dof_count, 0.0);
    state.dof_velocity.assign(caps_.dof_count, 0.0);
    return true;
  }

 private:
  std::uint64_t sequence_ = 0;
};

class LatestBackend final : public BaseBackend {
 public:
  explicit LatestBackend(int producer_burst) : producer_burst_(producer_burst) {}

  bool publish() {
    bool success = true;
    for (int i = 0; i < producer_burst_; ++i) {
      auto state = std::make_shared<RobotState>();
      state->sequence = ++published_;
      state->timestamp = std::chrono::steady_clock::now();
      state->dof_position.assign(caps_.dof_count, 0.0);
      state->dof_velocity.assign(caps_.dof_count, 0.0);
      if (!mailbox_.publish(std::move(state), published_)) {
        ++publish_failures_;
        success = false;
      }
    }
    return success;
  }

  bool read_state(RobotState& state) override {
    const auto snapshot = mailbox_.snapshot();
    if (!snapshot.value) return false;
    if (consumed_ != 0 && snapshot.sequence > consumed_ + 1) {
      dropped_ += snapshot.sequence - consumed_ - 1;
    }
    consumed_ = snapshot.sequence;
    state = *snapshot.value;
    return true;
  }

  std::uint64_t dropped() const noexcept { return dropped_; }
  std::uint64_t publish_failures() const noexcept { return publish_failures_; }

 private:
  LatestMailbox<RobotState> mailbox_;
  int producer_burst_;
  std::uint64_t published_ = 0;
  std::uint64_t consumed_ = 0;
  std::uint64_t dropped_ = 0;
  std::uint64_t publish_failures_ = 0;
};

ActuatorCommand idle_command(const RobotState&) {
  ActuatorCommand command;
  command.values.assign(29, 0.0);
  return command;
}

Summary run(DeploymentRuntime& runtime, int samples) {
  Summary summary;
  summary.state_to_command_us.reserve(samples);
  summary.state_to_send_us.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    if (runtime.run_cycle(std::chrono::steady_clock::now()) != CycleResult::kSent) {
      std::exit(2);
    }
    const auto& timing = runtime.last_cycle_timing();
    summary.state_to_command_us.push_back(
        timing.state_to_command().count() / 1000.0);
    summary.state_to_send_us.push_back(timing.state_to_send().count() / 1000.0);
  }
  return summary;
}

double percentile(std::vector<double> values, double p) {
  std::sort(values.begin(), values.end());
  return values[static_cast<std::size_t>(p * (values.size() - 1))];
}

void print_summary(const char* name, const Summary& summary, int samples,
                   int producer_burst) {
  std::cout << "{\"mode\":\"" << name << "\",\"producer_burst\":"
            << producer_burst << ",\"samples\":" << samples
            << ",\"state_to_command_p50_us\":"
            << percentile(summary.state_to_command_us, 0.50)
            << ",\"state_to_command_p95_us\":"
            << percentile(summary.state_to_command_us, 0.95)
            << ",\"state_to_command_p99_us\":"
            << percentile(summary.state_to_command_us, 0.99)
            << ",\"state_to_send_p50_us\":"
            << percentile(summary.state_to_send_us, 0.50)
            << ",\"state_to_send_p95_us\":"
            << percentile(summary.state_to_send_us, 0.95)
            << ",\"state_to_send_p99_us\":"
            << percentile(summary.state_to_send_us, 0.99)
            << ",\"dropped_samples\":" << summary.dropped
            << ",\"publish_failures\":" << summary.publish_failures
            << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  const int samples = argc > 1 ? std::max(1000, std::atoi(argv[1])) : 10000;
  const int producer_burst = argc > 2 ? std::max(1, std::atoi(argv[2])) : 2;
  SafetyLimits limits;
  limits.max_abs_command = 2.0;

  DirectBackend direct_backend;
  ControllerManager direct_manager;
  direct_manager.add(std::make_unique<CallbackController>("idle", idle_command));
  DeploymentRuntime direct(direct_backend, direct_manager, SafetySupervisor(limits));
  if (!direct.start() || !direct.request_controller("idle")) return 1;
  Summary direct_summary = run(direct, samples);
  print_summary("direct", direct_summary, samples, 1);

  LatestBackend latest_backend(producer_burst);
  ControllerManager latest_manager;
  latest_manager.add(std::make_unique<CallbackController>("idle", idle_command));
  DeploymentRuntime latest(latest_backend, latest_manager, SafetySupervisor(limits));
  // Startup validates one complete sample before entering SafeIdle.  Publish
  // it explicitly so the benchmark exercises the same lifecycle as a real
  // asynchronous adapter rather than failing before the timed loop begins.
  if (!latest_backend.publish()) return 1;
  if (!latest.start() || !latest.request_controller("idle")) return 1;
  Summary latest_summary;
  latest_summary.state_to_command_us.reserve(samples);
  latest_summary.state_to_send_us.reserve(samples);
  for (int i = 0; i < samples; ++i) {
    if (!latest_backend.publish()) return 3;
    if (latest.run_cycle(std::chrono::steady_clock::now()) != CycleResult::kSent) return 2;
    const auto& timing = latest.last_cycle_timing();
    latest_summary.state_to_command_us.push_back(timing.state_to_command().count() / 1000.0);
    latest_summary.state_to_send_us.push_back(timing.state_to_send().count() / 1000.0);
  }
  latest_summary.dropped = latest_backend.dropped();
  latest_summary.publish_failures = latest_backend.publish_failures();
  print_summary("latest_only", latest_summary, samples, producer_burst);
  return 0;
}
