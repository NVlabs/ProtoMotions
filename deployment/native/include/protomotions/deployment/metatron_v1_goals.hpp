// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <algorithm>
#include <cstddef>
#include <string_view>

#include "protomotions/deployment/metatron_v1_contract.hpp"

namespace protomotions::deployment::metatron_v1 {

// Port of examples/metatron_goals.py. Both hardware and MuJoCo launchers
// should invoke this same object after building their task targets.
class MetatronGoals final {
 public:
  MetatronGoals(const std::array<float, 33 * 3>& rest_positions,
             const std::array<float, 33 * 4>& rest_rotations,
             const std::array<float, 225>& rest_style)
      : positions_(rest_positions), rotations_(rest_rotations), style_(rest_style) {}

  void apply(Inputs& inputs, std::string_view mode, bool standing) const {
    if (mode == "odometry") return;
    auto& global_masks = inputs.booleans(16);
    auto& local_masks = inputs.booleans(7);
    auto& local_pose_masks = inputs.booleans(8);
    auto& global_pose_masks = inputs.booleans(17);
    auto& anchor_translation = inputs.booleans(3);
    auto& anchor_rotation = inputs.booleans(2);
    std::copy(global_masks.begin(), global_masks.end(), local_masks.begin());
    for (std::size_t slot = 0; slot < 5; ++slot) {
      local_masks[slot * 10] &= anchor_translation[slot * 2 + 1];
      local_masks[slot * 10 + 1] &= anchor_rotation[slot * 2];
      const auto first = local_masks.begin() + slot * 10;
      const bool local_visible = std::any_of(first, first + 10,
                                            [](std::uint8_t value) { return value != 0; });
      local_pose_masks[slot] = static_cast<std::uint8_t>(
          global_pose_masks[slot] != 0 && local_visible);
      const auto heading = static_cast<std::uint8_t>(
          global_masks[slot * 10 + 1] != 0 && anchor_rotation[slot * 2 + 1] != 0);
      std::fill(global_masks.begin() + slot * 10,
                global_masks.begin() + (slot + 1) * 10, 0U);
      global_masks[slot * 10 + 1] = heading;
      anchor_translation[slot * 2] = 0;
      anchor_translation[slot * 2 + 1] = 0;
      anchor_rotation[slot * 2] = 0;
      global_pose_masks[slot] = static_cast<std::uint8_t>(
          global_pose_masks[slot] != 0 && heading != 0);
    }
    inputs.floats(5) = inputs.floats(10);
    inputs.floats(6) = inputs.floats(11);
    inputs.floats(9) = inputs.floats(18);
    if (standing) idle(inputs);
  }

 private:
  void idle(Inputs& inputs) const {
    auto& local_positions = inputs.floats(5);
    auto& local_rotations = inputs.floats(6);
    auto& masks = inputs.booleans(7);
    auto& pose_masks = inputs.booleans(8);
    auto& times = inputs.floats(9);
    constexpr std::array<float, 5> offsets = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F};
    for (std::size_t slot = 0; slot < 5; ++slot) {
      std::copy(positions_.begin(), positions_.end(),
                local_positions.begin() + slot * positions_.size());
      std::copy(rotations_.begin(), rotations_.end(),
                local_rotations.begin() + slot * rotations_.size());
      for (std::size_t body = 0; body < 5; ++body) {
        masks[slot * 10 + body * 2] = 1;
        masks[slot * 10 + body * 2 + 1] = static_cast<std::uint8_t>(body == 0);
      }
      pose_masks[slot] = 1;
      times[slot] = offsets[slot];
    }
    std::copy(style_.begin(), style_.end(), inputs.floats(14).begin());
    std::fill(inputs.floats(15).begin(), inputs.floats(15).end(), 0.0F);
    std::fill(inputs.booleans(12).begin(), inputs.booleans(12).end(), 1U);
    std::fill(inputs.booleans(13).begin(), inputs.booleans(13).end(), 1U);
    std::fill(inputs.floats(4).begin(), inputs.floats(4).end(), 0.0F);
    inputs.booleans(19)[0] = 1;
  }

  std::array<float, 33 * 3> positions_;
  std::array<float, 33 * 4> rotations_;
  std::array<float, 225> style_;
};
}  // namespace protomotions::deployment::metatron_v1
