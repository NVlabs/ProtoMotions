// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_contract.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace protomotions::deployment::metatron_v1;

int main() {
  static_assert(kInputs.size() == 39);
  static_assert(kInputs[5].count() == 5 * 33 * 3);
  static_assert(kInputs[30].count() == 9 * 29);
  static_assert(kInputs[38].count() == 1);
  Inputs inputs;
  assert(inputs.floats(34)[3] == 1.0F);
  assert(inputs.booleans(37)[0] == 0);
  SensorReading initial;
  initial.dof_pos[0] = 1.0F;
  initial.anchor_rot[0] = 0.2F;
  inputs.prepare(initial);
  assert(inputs.floats(26)[0] == 1.0F);
  assert(inputs.floats(26)[29] == 1.0F);
  assert(inputs.floats(30)[0] == 0.0F);
  std::array<float, kDofCount> accepted{};
  accepted[0] = 0.5F;
  inputs.commit(initial, accepted);
  SensorReading next = initial;
  next.dof_pos[0] = 2.0F;
  inputs.prepare(next);
  assert(inputs.floats(22)[0] == 2.0F);
  assert(inputs.floats(26)[0] == 1.0F);
  assert(inputs.floats(30)[0] == 0.5F);
  assert(inputs.floats(28)[0] == 0.0F);
  // A transport hold must not advance history: two preparations are identical.
  inputs.prepare(next);
  assert(inputs.floats(30)[0] == 0.5F);
  inputs.commit(next, accepted);
  inputs.prepare(next);
  assert(inputs.floats(28)[0] == 0.5F);
  inputs.reset();
  inputs.prepare(initial);
  assert(inputs.floats(30)[0] == 0.0F);
  assert(inputs.floats(26)[0] == 1.0F);
  try {
    inputs.booleans(22);
    return 1;
  } catch (const std::out_of_range&) {
  }
  return 0;
}
