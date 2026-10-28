// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/safety.hpp"

#include <algorithm>
#include <cmath>

namespace protomotions::deployment {
namespace {

template <typename Container>
bool finite_and_bounded(const Container& values, double bound) {
  for (const double value : values) {
    if (!std::isfinite(value) || std::abs(value) > bound) return false;
  }
  return true;
}

void set_reason(std::string* reason, const char* message) {
  if (reason) *reason = message;
}

bool gain_within_limit(bool present, double value, double limit) {
  return !present ||
         (std::isfinite(value) && value >= 0.0 && std::isfinite(limit) &&
          limit >= 0.0 && value <= limit);
}

}  // namespace

bool SafetySupervisor::validate_capabilities(
    const RobotCapabilities& capabilities, std::string* reason) const {
  if (limits_.repeated_state_grace.count() < 0 ||
      limits_.repeated_state_grace > limits_.max_state_age ||
      limits_.repeated_state_grace > limits_.max_cycle_interval) {
    set_reason(reason, "sensor hold grace must fit within freshness and cycle limits");
    return false;
  }
  if (capabilities.dof_count == 0 ||
      (!capabilities.supports_position && !capabilities.supports_torque &&
       !capabilities.supports_velocity)) {
    set_reason(reason, "backend must expose a DOF and a supported control mode");
    return false;
  }
  if (!gain_within_limit(capabilities.has_default_position_gains,
                         capabilities.default_stiffness,
                         limits_.max_abs_stiffness) ||
      !gain_within_limit(capabilities.has_default_position_gains,
                         capabilities.default_damping,
                         limits_.max_abs_damping)) {
    set_reason(reason, "backend default gains require finite safety caps");
    return false;
  }
  if (limits_.require_finite_command_limit &&
      (!std::isfinite(limits_.max_abs_command) ||
       limits_.max_abs_command < 0.0)) {
    set_reason(reason, "finite command limit is not configured");
    return false;
  }
  return true;
}

bool SafetySupervisor::validate_state(
    const RobotState& state, const RobotCapabilities& capabilities,
    std::chrono::steady_clock::time_point now, std::string* reason) const {
  if (state.dof_position.size() != capabilities.dof_count ||
      state.dof_velocity.size() != capabilities.dof_count) {
    set_reason(reason, "state dof dimension mismatch");
    return false;
  }
  if (!state.dof_torque.empty() &&
      state.dof_torque.size() != capabilities.dof_count) {
    set_reason(reason, "state torque dimension mismatch");
    return false;
  }
  if (!state.finite()) {
    set_reason(reason, "state contains non-finite data");
    return false;
  }
  if ((state.has_odometry && !capabilities.has_odometry) ||
      (state.has_projected_gravity && !capabilities.has_projected_gravity) ||
      (state.has_vision && !capabilities.has_vision) ||
      (state.has_lidar && !capabilities.has_lidar)) {
    set_reason(reason, "state advertises unsupported sensor data");
    return false;
  }
  if ((state.has_vision && state.vision_data.empty()) ||
      (state.has_lidar && state.lidar_data.empty())) {
    set_reason(reason, "advertised sensor payload is empty");
    return false;
  }
  if (state.timestamp.time_since_epoch().count() == 0 ||
      now < state.timestamp || now - state.timestamp > limits_.max_state_age) {
    set_reason(reason, "state is stale");
    return false;
  }
  return true;
}

bool SafetySupervisor::validate_command(
    const ActuatorCommand& command, const RobotCapabilities& capabilities,
    std::string* reason) const {
  if (limits_.require_finite_command_limit ||
      std::isfinite(limits_.max_abs_command)) {
    if (!std::isfinite(limits_.max_abs_command) || limits_.max_abs_command < 0.0) {
      set_reason(reason, "finite command limit is not configured");
      return false;
    }
  }
  if (!capabilities.supports(command.mode)) {
    set_reason(reason, "control mode is not supported by backend");
    return false;
  }
  const auto sized_or_empty = [dof_count = capabilities.dof_count](
                                  const auto& values) {
    return values.empty() || values.size() == dof_count;
  };
  const auto finite_limit_or_empty = [](const auto& values, double limit) {
    return values.empty() || (std::isfinite(limit) && limit >= 0.0);
  };
  if (command.values.size() != capabilities.dof_count ||
      !sized_or_empty(command.stiffness) || !sized_or_empty(command.damping) ||
      !sized_or_empty(command.armature) ||
      !finite_limit_or_empty(command.stiffness, limits_.max_abs_stiffness) ||
      !finite_limit_or_empty(command.damping, limits_.max_abs_damping) ||
      !finite_limit_or_empty(command.armature, limits_.max_abs_armature) ||
      !finite_and_bounded(command.values, limits_.max_abs_command) ||
      !finite_and_bounded(command.stiffness, limits_.max_abs_stiffness) ||
      !finite_and_bounded(command.damping, limits_.max_abs_damping) ||
      !finite_and_bounded(command.armature,
                          limits_.max_abs_armature) ||
      (command.stiffness.empty() &&
       !gain_within_limit(capabilities.has_default_position_gains,
                          capabilities.default_stiffness,
                          limits_.max_abs_stiffness)) ||
      (command.damping.empty() &&
       !gain_within_limit(capabilities.has_default_position_gains,
                          capabilities.default_damping,
                          limits_.max_abs_damping))) {
    set_reason(reason, "command contains invalid values or dimensions");
    return false;
  }
  return true;
}

}  // namespace protomotions::deployment
