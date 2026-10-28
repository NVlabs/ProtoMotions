// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/steering_style.hpp"
#include "crawling_style_data.hpp"

#include <algorithm>
#include <cmath>

namespace protomotions::deployment {
const CrawlingStylePreset& crawling_style_preset() noexcept {
  return kCrawlingStylePreset;
}

std::array<float, 150> crawling_style_velocity(
    float heading, const std::array<float, 2>& world_velocity,
    bool velocity_active) noexcept {
  auto result = kCrawlingStylePreset.velocity;
  if (!velocity_active) return result;
  const float c = std::cos(heading), s = std::sin(heading);
  const float x = c * world_velocity[0] + s * world_velocity[1];
  const float y = -s * world_velocity[0] + c * world_velocity[1];
  for (std::size_t frame = 0; frame < 5; ++frame) {
    const float dx = x - result[frame * 30];
    const float dy = y - result[frame * 30 + 1];
    for (std::size_t body = 0; body < 5; ++body) {
      result[frame * 30 + body * 6] += dx;
      result[frame * 30 + body * 6 + 1] += dy;
    }
  }
  return result;
}

void crawling_pose_masks(std::span<std::uint8_t, 50> bodies,
                         std::span<std::uint8_t, 5> poses,
                         std::span<std::uint8_t, 10> translation,
                         std::span<std::uint8_t, 10> rotation,
                         std::span<float, 5> times) noexcept {
  for (std::size_t frame = 0; frame < 5; ++frame) {
    const bool keep = bodies[frame * 10] && translation[frame * 2] && poses[frame];
    std::fill_n(bodies.begin() + frame * 10, 10, 0U);
    bodies[frame * 10] = keep;
    translation[frame * 2] = keep;
    translation[frame * 2 + 1] = 0U;
    rotation[frame * 2] = rotation[frame * 2 + 1] = 0U;
    poses[frame] = keep;
    if (!keep) times[frame] = 0.0F;
  }
}
}  // namespace protomotions::deployment
