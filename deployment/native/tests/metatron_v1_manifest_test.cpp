// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_manifest.hpp"

#include <cassert>
#include <cmath>
#include <string>

using namespace protomotions::deployment::metatron_v1;

int main(int argc, char** argv) {
  assert(argc == 2);
  const auto manifest = Manifest::load(argv[1]);
  assert(std::abs(manifest.offset[0] - (-0.312F)) < 1e-6F);
  assert(std::abs(manifest.scale[0] - 0.5475465F) < 1e-6F);
  std::array<float, kDofCount> standing_target = manifest.offset;
  const auto raw = manifest.action_map().raw_from_applied(standing_target);
  for (float value : raw) assert(std::abs(value) < 1e-6F);
  return 0;
}
