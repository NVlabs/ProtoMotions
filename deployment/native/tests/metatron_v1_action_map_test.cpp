// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_action_map.hpp"

#include <array>
#include <cassert>
#include <stdexcept>

using namespace protomotions::deployment::metatron_v1;

int main() {
  std::array<float, kDofCount> scale{};
  std::array<float, kDofCount> offset{};
  scale.fill(0.5F);
  offset.fill(-0.2F);
  const ActionMap map(scale, offset);
  std::array<float, kDofCount> accepted{};
  accepted.fill(0.3F);
  assert(map.raw_from_applied(accepted)[0] == 1.0F);
  std::array<float, kDofCount> raw{};
  raw.fill(1.0F);
  assert(map.matches_output(accepted, raw));
  accepted[0] -= 0.2F;  // A second default-pose offset is not acceptable.
  assert(!map.matches_output(accepted, raw));
  scale[0] = 0.0F;
  try {
    const ActionMap invalid(scale, offset);
    return 1;
  } catch (const std::invalid_argument&) {
  }
  return 0;
}
