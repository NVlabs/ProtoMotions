// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/types.hpp"

namespace protomotions::deployment {
namespace {

bool finite_array(const auto& values) {
  for (const double value : values) {
    if (!std::isfinite(value)) return false;
  }
  return true;
}

}  // namespace

bool RobotState::finite() const noexcept {
  const double orientation_norm =
      base_orientation_xyzw[0] * base_orientation_xyzw[0] +
      base_orientation_xyzw[1] * base_orientation_xyzw[1] +
      base_orientation_xyzw[2] * base_orientation_xyzw[2] +
      base_orientation_xyzw[3] * base_orientation_xyzw[3];
  return finite_array(dof_position) && finite_array(dof_velocity) &&
         finite_array(dof_torque) && finite_array(base_position) &&
         finite_array(root_position) && finite_array(anchor_position) &&
         finite_array(base_angular_velocity) && finite_array(base_rpy) &&
         finite_array(base_orientation_xyzw) && orientation_norm > 1e-8 &&
         std::abs(orientation_norm - 1.0) < 1e-3 &&
         finite_array(root_orientation_xyzw) &&
         finite_array(root_angular_velocity) &&
         finite_array(projected_gravity) &&
         finite_array(odometry_velocity) &&
         finite_array(vision_data) && finite_array(lidar_data);
}

bool ActuatorCommand::finite() const noexcept {
  return finite_array(values) && finite_array(stiffness) &&
         finite_array(damping) && finite_array(armature);
}

}  // namespace protomotions::deployment
