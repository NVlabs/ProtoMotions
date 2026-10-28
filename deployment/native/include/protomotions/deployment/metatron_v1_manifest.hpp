// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <string>

#include "protomotions/deployment/metatron_v1_action_map.hpp"

namespace protomotions::deployment::metatron_v1 {

// The manifest is part of the policy. Reject a plausible-looking but wrong
// 39-input artifact before it can drive the same 29 physical motors.
struct Manifest {
  std::array<float, kDofCount> scale{};
  std::array<float, kDofCount> offset{};

  static Manifest load(const std::string& path);
  ActionMap action_map() const { return ActionMap(scale, offset); }
};

}  // namespace protomotions::deployment::metatron_v1
