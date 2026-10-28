// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include "protomotions/deployment/metatron_v1_contract.hpp"
#include "protomotions/deployment/types.hpp"

namespace protomotions::deployment::metatron_v1 {
namespace detail {
inline std::array<float, 4> normalized_xyzw(const std::array<double, 4>& q) {
  double squared = 0.0;
  for (double value : q) squared += value * value;
  if (!std::isfinite(squared) || squared < 1e-12)
    throw std::invalid_argument("scene sensor quaternion is invalid");
  const double inverse_norm = 1.0 / std::sqrt(squared);
  return {static_cast<float>(q[0] * inverse_norm),
          static_cast<float>(q[1] * inverse_norm),
          static_cast<float>(q[2] * inverse_norm),
          static_cast<float>(q[3] * inverse_norm)};
}
}  // namespace detail

// Both backends publish the same RobotState fields. In particular, the ONNX
// root angular-velocity input is the pelvis/root sample, while anchor rotation
// and position are the synchronized torso sample. Never substitute one IMU
// for the other or apply a second pelvis-to-torso transform here.
inline SensorReading reading_from_robot_state(const RobotState& state) {
  if (state.dof_position.size() != kDofCount ||
      state.dof_velocity.size() != kDofCount ||
      !state.has_anchor_position || !state.has_root_kinematics ||
      !state.finite()) {
    throw std::invalid_argument("scene policy requires 29 joints and complete synchronized root/torso state");
  }
  SensorReading reading;
  for (std::size_t index = 0; index < kDofCount; ++index) {
    reading.dof_pos[index] = static_cast<float>(state.dof_position[index]);
    reading.dof_vel[index] = static_cast<float>(state.dof_velocity[index]);
  }
  for (std::size_t axis = 0; axis < 3; ++axis) {
    reading.root_local_ang_vel[axis] =
        static_cast<float>(state.root_angular_velocity[axis]);
    reading.anchor_pos[axis] = static_cast<float>(state.anchor_position[axis]);
    reading.root_pos[axis] = static_cast<float>(state.root_position[axis]);
  }
  reading.anchor_rot = detail::normalized_xyzw(state.base_orientation_xyzw);
  reading.root_rot = detail::normalized_xyzw(state.root_orientation_xyzw);
  return reading;
}
}  // namespace protomotions::deployment::metatron_v1
