// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cmath>

namespace protomotions::deployment {

// Pico grips clutch arm pose; the independent finger triggers command Dex-3.
struct Dex3TriggerControl {
  static constexpr std::array<float, 7> kOpen{};
  // Right-thumb flexion is not neutral at zero on this hand. These targets
  // stay inside the manually observed open/closed range.
  static constexpr std::array<float, 7> kOpenRight{
      0.0F, 0.60F, -0.10F, 0.0F, 0.0F, 0.0F, 0.0F};
  static constexpr std::array<float, 7> kCloseLeft{
      0.0F, 0.163F, 0.875F, -0.785F, -0.875F, -0.785F, -0.875F};
  static constexpr std::array<float, 7> kCloseRight{
      0.0F, -0.65F, -1.55F, 0.785F, 0.875F, 0.785F, 0.875F};

  std::array<bool, 2> closed{};

  void reset() noexcept { closed = {}; }

  void update(float left_trigger, float right_trigger) noexcept {
    closed[0] = pressed(left_trigger);
    closed[1] = pressed(right_trigger);
  }

  const std::array<float, 7>& target(int side) const noexcept {
    return side == 0 ? (closed[0] ? kCloseLeft : kOpen)
                     : (closed[1] ? kCloseRight : kOpenRight);
  }

 private:
  static bool pressed(float trigger) noexcept {
    return std::isfinite(trigger) && trigger >= 0.5F;
  }
};

}  // namespace protomotions::deployment
