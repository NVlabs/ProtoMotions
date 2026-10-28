// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_sensor.hpp"
#include "protomotions/deployment/mujoco_backend.hpp"

#include <cassert>
#include <cmath>

using namespace protomotions::deployment;
using namespace protomotions::deployment::metatron_v1;

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  MuJoCoBackendConfig config;
  config.xml_path = argv[1];
  config.policy_anchor_body_name = "torso_link";
  config.substeps_per_command = 20;
  MuJoCoBackend simulation(config);
  RobotState state;
  assert(simulation.read_state(state));
  assert(state.has_root_kinematics && state.has_anchor_position);
  assert(state.has_odometry && state.has_projected_gravity);
  assert(state.dof_position.size() == kDofCount);
  const auto reading = reading_from_robot_state(state);
  assert(std::abs(reading.root_pos[2] - state.root_position[2]) < 1e-5);
  assert(std::abs(reading.anchor_pos[2] - state.anchor_position[2]) < 1e-5);
  ActuatorCommand hold;
  hold.mode = ControlMode::kPosition;
  hold.values = state.dof_position;
  hold.stiffness.assign(kDofCount, 0.0);
  hold.damping.assign(kDofCount, 8.0);
  assert(simulation.send_command(hold));
  assert(simulation.read_state(state));
  assert(state.finite());
  return 0;
}
