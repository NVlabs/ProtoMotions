// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_goals.hpp"

#include <array>
#include <cassert>

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
  return 0;
}
