// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/c_api.h"

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "protomotions/deployment/controller_manager.hpp"
#include "protomotions/deployment/runtime.hpp"

namespace {

using namespace protomotions::deployment;

class CBackend final : public RobotBackend {
 public:
  CBackend(size_t dof_count, pm_runtime_callbacks_t callbacks)
      : dof_count_(dof_count), callbacks_(callbacks) {
    capabilities_.dof_count = dof_count;
    capabilities_.supports_position = true;
    // The C ABI exposes these optional raw sensor slots to the host callback.
    // A callback may leave any slot absent, but if it marks one present the
    // runtime must not reject it merely because the generic wrapper has no
    // vendor-specific capability discovery step.
    capabilities_.has_odometry = true;
    capabilities_.has_projected_gravity = true;
    capabilities_.has_vision = true;
    capabilities_.has_lidar = true;
    capabilities_.supports_anchor_reset = true;
    capabilities_.resets_state_sequence = true;
  }

  const RobotCapabilities& capabilities() const noexcept override {
    return capabilities_;
  }

  bool read_state(RobotState& state) override {
    state.dof_position.resize(dof_count_);
    state.dof_velocity.resize(dof_count_);
    pm_state_output_t output{
        state.dof_position.data(),
        state.dof_velocity.data(),
        state.base_position.data(),
        state.base_orientation_xyzw.data(),
        state.projected_gravity.data(),
        state.operator_input_data.data(),
        state.operator_input_data.size(),
        nullptr,
        0,
        nullptr,
        0,
        dof_count_,
        0,
        0,
        0,
        0,
        0,
    };
    try {
      if (!callbacks_.read_state ||
          !callbacks_.read_state(callbacks_.user, &output, &sequence_) ||
          output.timestamp_ns == 0) {
        return false;
      }
    } catch (...) {
      return false;
    }
    state.timestamp = std::chrono::steady_clock::time_point(
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::nanoseconds(output.timestamp_ns)));
    state.sequence = sequence_;
    state.has_odometry = output.has_odometry != 0;
    state.has_projected_gravity = output.has_projected_gravity != 0;
    state.has_vision = output.has_vision != 0;
    state.has_lidar = output.has_lidar != 0;
    if (state.has_vision && output.vision_data && output.vision_size > 0) {
      state.vision_data.assign(output.vision_data,
                               output.vision_data + output.vision_size);
    }
    if (state.has_lidar && output.lidar_data && output.lidar_size > 0) {
      state.lidar_data.assign(output.lidar_data,
                              output.lidar_data + output.lidar_size);
    }
    return true;
  }

  bool send_command(const ActuatorCommand& command) override {
    try {
      return callbacks_.send_command &&
             callbacks_.send_command(
                 callbacks_.user, command.values.data(),
                 command.stiffness.empty() ? nullptr : command.stiffness.data(),
                 command.damping.empty() ? nullptr : command.damping.data(),
                 command.armature.empty() ? nullptr : command.armature.data(),
                 dof_count_);
    } catch (...) {
      return false;
    }
  }

  bool reset_anchor_frame() override {
    try {
      if (!callbacks_.reset_anchor_frame ||
          !callbacks_.reset_anchor_frame(callbacks_.user)) {
        return false;
      }
      // The callback owns the sensor stream sequence during normal reads.
      // Anchor reset also resets that source (MuJoCo does this explicitly),
      // so reset the C ABI barrier with it; otherwise the first post-reset
      // frame replays sequence 1 and the runtime correctly faults as stale.
      sequence_ = 0;
      return true;
    } catch (...) {
      return false;
    }
  }

  bool emergency_stop() noexcept override {
    try {
      return callbacks_.emergency_stop &&
             callbacks_.emergency_stop(callbacks_.user) != 0;
    } catch (...) {
      return false;
    }
  }

 private:
  size_t dof_count_;
  pm_runtime_callbacks_t callbacks_;
  RobotCapabilities capabilities_;
  std::uint64_t sequence_ = 0;
};

class CController final : public Controller {
 public:
  CController(std::string name, size_t dof_count, pm_step_controller_fn step,
              void* user)
      : name_(std::move(name)),
        dof_count_(dof_count),
        step_(step),
        user_(user) {}

  std::string_view name() const noexcept override { return name_; }
  bool start(const RobotState&) override { return step_ != nullptr; }

  ActuatorCommand step(const RobotState& state) override {
    ActuatorCommand command;
    command.mode = ControlMode::kPosition;
    command.values.resize(dof_count_);
    command.stiffness.resize(dof_count_);
    command.damping.resize(dof_count_);
    command.armature.resize(dof_count_);
    pm_state_view_t view{
        state.dof_position.data(),
        state.dof_velocity.data(),
        state.base_position.data(),
        state.base_orientation_xyzw.data(),
        state.projected_gravity.data(),
        state.operator_input_data.data(),
        state.operator_input_data.size(),
        state.vision_data.data(),
        state.vision_data.size(),
        state.lidar_data.data(),
        state.lidar_data.size(),
        dof_count_,
        state.sequence,
        state.has_odometry ? 1 : 0,
        state.has_projected_gravity ? 1 : 0,
        state.has_vision ? 1 : 0,
        state.has_lidar ? 1 : 0,
    };
    pm_command_output_t output{
        command.values.data(), command.stiffness.data(), command.damping.data(),
        command.armature.data(), dof_count_, 0, 0, 0};
    if (!step_ || !step_(user_, &view, &output)) {
      command.values.clear();
    } else {
      if (!output.has_stiffness) command.stiffness.clear();
      if (!output.has_damping) command.damping.clear();
      if (!output.has_armature) command.armature.clear();
    }
    return command;
  }

  void stop() noexcept override {}

 private:
  std::string name_;
  size_t dof_count_;
  pm_step_controller_fn step_;
  void* user_;
};

}  // namespace

struct pm_runtime {
  std::unique_ptr<CBackend> backend;
  protomotions::deployment::ControllerManager controllers;
  protomotions::deployment::DeploymentRuntime runtime;

  pm_runtime(size_t dof_count, pm_runtime_callbacks_t callbacks,
             double max_abs_command, int max_cycle_interval_ms,
             double max_abs_stiffness, double max_abs_damping)
      : backend(std::make_unique<CBackend>(dof_count, callbacks)),
        runtime(*backend, controllers, make_safety(max_abs_command,
                                                    max_cycle_interval_ms,
                                                    max_abs_stiffness,
                                                    max_abs_damping)) {}

  ~pm_runtime() { runtime.stop(); }

  static SafetySupervisor make_safety(double max_abs_command,
                                      int max_cycle_interval_ms,
                                      double max_abs_stiffness,
                                      double max_abs_damping) {
    SafetyLimits limits;
    limits.max_abs_command = max_abs_command;
    limits.max_cycle_interval = std::chrono::milliseconds(max_cycle_interval_ms);
    limits.max_abs_stiffness = max_abs_stiffness;
    limits.max_abs_damping = max_abs_damping;
    return SafetySupervisor(limits);
  }
};

extern "C" int pm_runtime_abi_version() {
  return PROTOMOTIONS_DEPLOYMENT_C_ABI_VERSION;
}

extern "C" pm_runtime_t* pm_runtime_create(
    size_t dof_count, pm_runtime_callbacks_t callbacks, double max_abs_command,
    int max_cycle_interval_ms, double max_abs_stiffness,
    double max_abs_damping) {
  if (dof_count == 0 || !callbacks.read_state || !callbacks.send_command ||
      !callbacks.reset_anchor_frame || !callbacks.emergency_stop ||
      !std::isfinite(max_abs_command) || max_abs_command <= 0.0 ||
      max_cycle_interval_ms <= 0 || !std::isfinite(max_abs_stiffness) ||
      max_abs_stiffness < 0.0 || !std::isfinite(max_abs_damping) ||
      max_abs_damping < 0.0) {
    return nullptr;
  }
  try {
    return new pm_runtime(dof_count, callbacks, max_abs_command,
                          max_cycle_interval_ms, max_abs_stiffness,
                          max_abs_damping);
  } catch (...) {
    return nullptr;
  }
}

extern "C" void pm_runtime_destroy(pm_runtime_t* runtime) { delete runtime; }

extern "C" int pm_runtime_add_controller(pm_runtime_t* runtime,
                                           const char* name,
                                           pm_step_controller_fn step,
                                           void* user) {
  if (!runtime || !name || !*name || !step) return 0;
  try {
    return runtime->controllers.add(std::make_unique<CController>(
               name, runtime->backend->capabilities().dof_count, step, user))
               ? 1
               : 0;
  } catch (...) {
    return 0;
  }
}

extern "C" int pm_runtime_start(pm_runtime_t* runtime) {
  return runtime && runtime->runtime.start() ? 1 : 0;
}

extern "C" int pm_runtime_stop(pm_runtime_t* runtime) {
  if (!runtime) return 0;
  return runtime->runtime.stop() ? 1 : 0;
}

extern "C" int pm_runtime_reset_anchor_frame(pm_runtime_t* runtime) {
  return runtime && runtime->runtime.reset_anchor_frame() ? 1 : 0;
}

extern "C" int pm_runtime_request_controller(pm_runtime_t* runtime,
                                               const char* name) {
  if (!runtime || !name) return 0;
  try {
    return runtime->runtime.request_controller(name) ? 1 : 0;
  } catch (...) {
    return 0;
  }
}

extern "C" int pm_runtime_run_cycle(pm_runtime_t* runtime) {
  if (!runtime) return 0;
  const auto result = runtime->runtime.run_cycle(
      std::chrono::steady_clock::now());
  // A safe-stop cycle is a successful backend command too; the lifecycle
  // query remains available to tell callers that the controller is faulted.
  return result == CycleResult::kSent || result == CycleResult::kSafeStop ||
         result == CycleResult::kWaitingForState ? 1
                                                                            : 0;
}

extern "C" const char* pm_runtime_last_fault_reason(const pm_runtime_t* runtime) {
  return runtime ? runtime->runtime.last_fault_reason().c_str() : "null runtime";
}

extern "C" void pm_runtime_emergency_stop(pm_runtime_t* runtime) {
  if (runtime) runtime->runtime.emergency_stop();
}

extern "C" int pm_runtime_revive(pm_runtime_t* runtime) {
  return runtime && runtime->runtime.revive() ? 1 : 0;
}

extern "C" uint64_t pm_runtime_last_state_to_command_ns(
    const pm_runtime_t* runtime) {
  if (!runtime) return 0;
  return static_cast<uint64_t>(
      runtime->runtime.last_cycle_timing().state_to_command().count());
}

extern "C" int pm_runtime_lifecycle(const pm_runtime_t* runtime) {
  if (!runtime) return -1;
  return static_cast<int>(runtime->runtime.lifecycle());
}
