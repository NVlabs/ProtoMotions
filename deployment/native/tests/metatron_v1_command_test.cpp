// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_command.hpp"

#include <cassert>

using namespace protomotions::deployment::metatron_v1;

int main() {
  Outputs outputs;
  outputs.raw_action.fill(0.0F);
  outputs.action.fill(-0.312F);  // Nonzero default-pose offset.
  outputs.stiffness.fill(80.0F);
  outputs.damping.fill(4.0F);
  const auto command = command_from_outputs(outputs);
  assert(command.values.size() == kDofCount);
  assert(command.values[0] == static_cast<double>(outputs.action[0]));
  assert(command.values[0] != static_cast<double>(outputs.raw_action[0]));
  assert(command.stiffness[0] == 80.0);
  assert(command.damping[0] == 4.0);
  return 0;
}
