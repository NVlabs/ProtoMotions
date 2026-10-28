// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_sensor.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace protomotions::deployment;
using namespace protomotions::deployment::metatron_v1;

int main() {
  RobotState state;
  state.dof_position.assign(29, 0.0);
  state.dof_velocity.assign(29, 0.0);
  state.has_anchor_position = true;
  state.has_root_kinematics = true;
  state.root_position = {1.0, 2.0, 0.3};
  state.anchor_position = {1.1, 2.2, 0.8};
  state.root_orientation_xyzw = {0.0, 0.0, 0.0, 2.0};
  state.base_orientation_xyzw = {0.0, 0.0, std::sqrt(0.5), std::sqrt(0.5)};
  state.root_angular_velocity = {0.1, 0.2, 0.3};
  state.base_angular_velocity = {3.0, 2.0, 1.0};
  const SensorReading reading = reading_from_robot_state(state);
  assert(reading.root_pos[2] == 0.3F);
  assert(reading.anchor_pos[2] == 0.8F);
  assert(reading.root_rot[3] == 1.0F);
  assert(std::abs(reading.anchor_rot[2] - std::sqrt(0.5F)) < 1e-6F);
  assert(reading.root_local_ang_vel[0] == 0.1F);
  assert(reading.root_local_ang_vel[0] != state.base_angular_velocity[0]);
  state.has_anchor_position = false;
  try {
    (void)reading_from_robot_state(state);
    return 1;
  } catch (const std::invalid_argument&) {
  }
  return 0;
}
