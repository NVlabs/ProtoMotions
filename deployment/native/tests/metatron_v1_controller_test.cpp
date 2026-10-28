// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_controller.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace protomotions::deployment;
using namespace protomotions::deployment::metatron_v1;

int main() {
  std::array<float, kDofCount> scale{};
  std::array<float, kDofCount> offset{};
  scale.fill(0.5F);
  offset.fill(-0.2F);
  int calls = 0;
  MetatronController controller(
      ActionMap(scale, offset),
      [&calls](Inputs& inputs, const RobotState&) {
        if (calls == 0) inputs.booleans(16)[0] = 1;
      },
      [&calls](const Inputs& inputs) {
        if (calls == 0) {
          assert(inputs.floats(30)[0] == 0.0F);
          assert(inputs.booleans(16)[0] == 1);
        } else {
          assert(inputs.booleans(16)[0] == 0);  // Prior goal was cleared.
          assert(std::abs(inputs.floats(30)[0] - 0.5F) < 1e-6F);
          assert(inputs.floats(28)[0] == 0.0F); // Two calls back.
          assert(inputs.floats(26)[0] == 0.3F); // Pre-step state, not new state.
        }
        ++calls;
        Outputs outputs;
        outputs.raw_action.fill(1.0F);  // This must not go to the motors.
        outputs.action.fill(0.3F);      // Already processed by the ONNX graph.
        outputs.stiffness.fill(40.0F);
        outputs.damping.fill(2.0F);
        return outputs;
      });
  RobotState state;
  state.dof_position.assign(kDofCount, 0.3);
  state.dof_velocity.assign(kDofCount, 0.0);
  state.has_anchor_position = true;
  state.has_root_kinematics = true;
  assert(controller.start(state));
  ActuatorCommand first = controller.step(state);
  assert(first.values[0] == static_cast<double>(0.3F));
  assert(first.values[0] != 1.0);
  try {
    (void)controller.step(state);
    return 1;  // An unacknowledged action must not advance history.
  } catch (const std::logic_error&) {
  }
  // The transport/safety layer accepted a changed target, so history must
  // hold its inverse-affine raw value, not the unused network raw output.
  first.values[0] = 0.05;
  controller.on_command_accepted(first);
  state.dof_position[0] = 0.4;
  (void)controller.step(state);
  controller.stop();
  assert(controller.start(state));
  return 0;
}
