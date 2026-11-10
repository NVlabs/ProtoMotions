// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_goals.hpp"

#include <array>
#include <cassert>
#include <cmath>

using namespace protomotions::deployment::metatron_v1;

int main() {
  std::array<float, 33 * 3> positions{};
  std::array<float, 33 * 4> rotations{};
  std::array<float, 225> style{};
  positions[16 * 3 + 2] = 0.79F;
  for (int body = 0; body < 33; ++body) rotations[body * 4 + 3] = 1.0F;
  style[2] = 0.79F;
  MetatronGoals goals(positions, rotations, style);
  Inputs inputs;
  inputs.booleans(16)[1] = 1;
  inputs.booleans(17)[0] = 1;
  inputs.booleans(2)[1] = 1;
  goals.apply(inputs, "steering", false);
  assert(inputs.booleans(16)[1] == 1);  // Requested heading stays global.
  assert(inputs.booleans(17)[0] == 1);
  assert(inputs.booleans(7)[1] == 0);   // Roll/pitch not requested locally.

  inputs.reset();
  inputs.booleans(16)[1] = 1;
  inputs.booleans(17)[0] = 1;
  inputs.booleans(2)[0] = 1;  // Full torso rotation for planted teleop.
  inputs.booleans(2)[1] = 1;  // Keep the world heading goal too.
  inputs.floats(11)[16 * 4] = 0.1F;
  inputs.floats(11)[16 * 4 + 3] = 0.995F;
  goals.apply(inputs, "teleop", false);
  assert(inputs.booleans(7)[1] == 1);
  assert(inputs.booleans(16)[1] == 1);
  assert(std::abs(inputs.floats(6)[16 * 4] - 0.1F) < 1e-6F);

  inputs.reset();
  goals.apply(inputs, "idle", true);
  assert(inputs.floats(5)[16 * 3 + 2] == 0.79F);
  assert(inputs.floats(5)[33 * 3 + 16 * 3 + 2] == 0.79F);
  assert(inputs.booleans(7)[0] == 1);
  assert(inputs.booleans(7)[1] == 1);
  assert(inputs.booleans(7)[3] == 0);  // Feet have position, not rotation.
  assert(inputs.booleans(8)[4] == 1);
  assert(inputs.floats(9)[0] == 0.1F);
  assert(inputs.floats(9)[4] == 0.5F);
  assert(inputs.booleans(12)[0] == 1);
  assert(inputs.booleans(19)[0] == 1);
  assert(inputs.floats(4)[0] == 0.0F);

  inputs.reset();
  inputs.booleans(16)[0] = 1;
  goals.apply(inputs, "odometry", true);
  assert(inputs.booleans(16)[0] == 1);
  assert(inputs.booleans(7)[0] == 0);

  // Steering rest hands are clear of the pelvis, but Pico teleop hands are
  // passed through unchanged, including while a velocity command is present.
  inputs.reset();
  inputs.floats(10)[24 * 3] = 0.1F;
  inputs.floats(10)[32 * 3] = 0.2F;
  inputs.booleans(16)[6] = inputs.booleans(16)[8] = 1;
  inputs.booleans(3)[1] = 1;
  inputs.floats(4)[0] = 0.5F;
  inputs.booleans(19)[0] = 1;
  goals.apply(inputs, "steering", false);
  assert(std::abs(inputs.floats(5)[24 * 3] - 0.16F) < 1e-6F);
  assert(std::abs(inputs.floats(5)[24 * 3 + 1] - 0.05F) < 1e-6F);
  assert(std::abs(inputs.floats(5)[32 * 3] - 0.26F) < 1e-6F);
  assert(std::abs(inputs.floats(5)[32 * 3 + 1] + 0.05F) < 1e-6F);

  inputs.reset();
  inputs.floats(10)[24 * 3] = 0.1F;
  inputs.floats(10)[32 * 3] = 0.2F;
  inputs.booleans(16)[6] = inputs.booleans(16)[8] = 1;
  inputs.booleans(3)[1] = 1;
  inputs.floats(4)[0] = 0.5F;
  inputs.booleans(19)[0] = 1;
  goals.apply(inputs, "teleop", false);
  assert(inputs.booleans(7)[6] && inputs.booleans(7)[8]);
  assert(std::abs(inputs.floats(5)[24 * 3] - 0.1F) < 1e-6F);
  assert(std::abs(inputs.floats(5)[32 * 3] - 0.2F) < 1e-6F);
  assert(inputs.floats(4)[0] == 0.5F && inputs.booleans(19)[0]);
  return 0;
}
