// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace protomotions::deployment {
struct CrawlingStylePreset {
  std::array<float, 225> pose;
  std::array<float, 150> velocity;
};

const CrawlingStylePreset& crawling_style_preset() noexcept;
std::array<float, 150> crawling_style_velocity(
    float heading, const std::array<float, 2>& world_velocity,
    bool velocity_active) noexcept;
// Preserve only existing torso-XY objectives; crawl supplies the posture.
void crawling_pose_masks(std::span<std::uint8_t, 50> bodies,
                         std::span<std::uint8_t, 5> poses,
                         std::span<std::uint8_t, 10> translation,
                         std::span<std::uint8_t, 10> rotation,
                         std::span<float, 5> times) noexcept;
}  // namespace protomotions::deployment
