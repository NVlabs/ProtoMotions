// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace protomotions::deployment {

enum class ControlMode { kPosition, kTorque, kVelocity };

enum class LifecycleState { kStopped, kSafeIdle, kActive, kFault };

struct RobotCapabilities {
  std::size_t dof_count = 0;
  bool supports_position = false;
  bool supports_torque = false;
  bool supports_velocity = false;
  bool has_odometry = false;
  bool has_projected_gravity = false;
  bool has_vision = false;
  bool has_lidar = false;
  // Some transports fill omitted command gains with a known per-DOF default.
  // Expose that default so the safety layer can apply configured caps before
  // the backend substitutes it.
  bool has_default_position_gains = false;
  double default_stiffness = 0.0;
  double default_damping = 0.0;
  // Whether reset_anchor_frame() has a transport-side frame to reset. Policy
  // controllers still restart at a reset boundary when this is false. Keep
  // this at the end so existing positional aggregate initializers retain
  // their meaning.
  bool supports_anchor_reset = false;
  // A reset hook may also restart the transport's state sequence (MuJoCo
  // does). Only backends that guarantee this may clear the runtime's
  // latest-only sequence barrier at reset.
  bool resets_state_sequence = false;

  bool supports(ControlMode mode) const noexcept {
    switch (mode) {
      case ControlMode::kPosition:
        return supports_position;
      case ControlMode::kTorque:
        return supports_torque;
      case ControlMode::kVelocity:
        return supports_velocity;
    }
    return false;
  }
};

struct RobotState {
  std::chrono::steady_clock::time_point timestamp{};
  std::uint64_t sequence = 0;
  std::vector<double> dof_position;
  std::vector<double> dof_velocity;
  std::vector<double> dof_torque;
  // Optional telemetry in hardware DOF order, Celsius: winding, driver.
  // Empty means unavailable (e.g. simulation), not zero temperature. This is
  // not a policy input or an automatic safety-stop condition.
  std::vector<std::array<double, 2>> motor_temperature;
  // Explicit root and torso-anchor positions keep policy geometry out of the
  // control loop's simulator or language adapters.
  std::array<double, 3> base_position{};
  std::array<double, 3> root_position{};
  std::array<double, 3> anchor_position{};
  // For the G1 policy, this is the direct torso-IMU angular velocity (the
  // pelvis/root carrier used by qualification FK is a separate adapter value).
  std::array<double, 3> base_angular_velocity{};
  // RPY uses the same frame and quaternion convention as base_orientation.
  std::array<double, 3> base_rpy{};
  // The deployment contract uses xyzw quaternions. For the G1 policy this is
  // the direct torso-IMU orientation (not a pelvis carrier orientation).
  std::array<double, 4> base_orientation_xyzw{0.0, 0.0, 0.0, 1.0};
  std::array<double, 4> root_orientation_xyzw{0.0, 0.0, 0.0, 1.0};
  std::array<double, 3> root_angular_velocity{};
  std::array<double, 3> projected_gravity{0.0, 0.0, -1.0};
  // Optional raw operator-controller payload. The transport owns its format;
  // an input adapter may decode it without making the runtime vendor-aware.
  std::array<std::uint8_t, 40> operator_input_data{};
  bool has_operator_input = false;
  bool has_odometry = false;
  bool has_anchor_position = false;
  bool has_root_kinematics = false;
  bool has_projected_gravity = false;
  bool has_vision = false;
  bool has_lidar = false;
  // Raw payloads are transport-owned. Their interpretation stays in the
  // generated ProtoMotions policy artifact.
  std::vector<float> vision_data;
  std::vector<float> lidar_data;
  // World-frame planar velocity from the optional odometry source. This is
  // kept separate from joint/root kinematics because it is an operator-input
  // limiter, not part of the policy's raw sensor contract.
  std::array<double, 3> odometry_velocity{};

  bool finite() const noexcept;
};

struct ActuatorCommand {
  ControlMode mode = ControlMode::kPosition;
  std::vector<double> values;
  std::vector<double> stiffness;
  std::vector<double> damping;
  std::vector<double> armature;
  std::chrono::steady_clock::time_point timestamp{};

  bool finite() const noexcept;
};

struct SafetyLimits {
  std::chrono::milliseconds max_state_age{100};
  // A real robot must provide an explicit command limit. Leaving this at
  // infinity is rejected by default rather than silently weakening safety.
  double max_abs_command = std::numeric_limits<double>::infinity();
  double max_abs_stiffness = std::numeric_limits<double>::infinity();
  double max_abs_damping = std::numeric_limits<double>::infinity();
  double max_abs_armature = std::numeric_limits<double>::infinity();
  std::chrono::milliseconds max_cycle_interval{100};
  // Opt-in bounded position hold for a repeated, otherwise valid sensor sample.
  // Zero preserves strict behavior. Never extends max_state_age or cycle deadline.
  std::chrono::milliseconds repeated_state_grace{0};
  bool require_finite_command_limit = true;
};

}  // namespace protomotions::deployment
