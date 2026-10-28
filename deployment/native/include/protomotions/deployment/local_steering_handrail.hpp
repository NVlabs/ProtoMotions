// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>

namespace protomotions::deployment {

struct LocalSteeringFrame {
  std::array<std::array<float, 3>, 5> positions{};
  float yaw = 0.0F;
};

// One future clean-gait frame, ordered torso, right/left foot, left/right hand.
// The relative template is expressed in the calibrated torso heading frame.
inline LocalSteeringFrame local_steering_frame(
    const std::array<std::array<float, 3>, 5>& relative,
    const std::array<float, 3>& anchor, float current_heading,
    float target_heading, const std::array<float, 2>& velocity,
    const std::array<float, 6>& hands, float phase, float swing_scale,
    float time) noexcept {
  constexpr float pi = 3.14159265358979323846F;
  const auto wrap = [](float angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
  };
  const float speed = std::hypot(velocity[0], velocity[1]);
  const float move = speed > 0.05F ? std::atan2(velocity[1], velocity[0]) : current_heading;
  const float cadence = std::min(2.0F, 0.83F + 0.93F * speed);
  const float amplitude = (0.13F + 0.065F * speed) * swing_scale;
  LocalSteeringFrame result;
  result.yaw = current_heading + std::clamp(wrap(target_heading-current_heading), -2.0F*time, 2.0F*time);
  result.positions[0] = {anchor[0]+velocity[0]*time, anchor[1]+velocity[1]*time, anchor[2]};
  const float c = std::cos(result.yaw), s = std::sin(result.yaw);
  for (std::size_t index = 1; index < 5; ++index) {
    auto offset = relative[index];
    if (index <= 2) {
      const float rel = wrap(move-result.yaw);
      offset[1] = (index == 1 ? -1.0F : 1.0F) *
          (0.07F + 0.09F*std::abs(std::sin(rel))*swing_scale);
      const float future_phase = phase + 2.0F*pi*cadence*time + (index == 2 ? pi : 0.0F);
      offset[0] += amplitude*std::cos(future_phase)*std::cos(rel);
      offset[1] += amplitude*std::cos(future_phase)*std::sin(rel);
      offset[2] += 0.08F*swing_scale*std::max(0.0F, -std::sin(future_phase));
    } else {
      for (std::size_t axis = 0; axis < 3; ++axis) offset[axis] += hands[(index-3)*3+axis];
    }
    result.positions[index] = {
        result.positions[0][0] + c*offset[0]-s*offset[1],
        result.positions[0][1] + s*offset[0]+c*offset[1],
        result.positions[0][2] + offset[2]};
  }
  return result;
}

// Position-only reference clutch. Released hands retain their desired offset;
// the slew limiter continues approaching it, including after grip release.
class LocalSteeringPicoHands {
 public:
  using Hands = std::array<std::array<float, 3>, 2>;
  void latch(const Hands& world, float heading) noexcept {
    reference_ = world;
    heading_ = heading;
    desired_ = {};
    command_ = {};
    base_ = {};
    engaged_ = {};
  }
  std::array<float, 6> update(const Hands& world,
                              const std::array<float, 2>& grips,
                              float dt) noexcept {
    constexpr std::array<float, 3> lo{-0.15F, -0.22F, -0.10F};
    constexpr std::array<float, 3> hi{0.45F, 0.22F, 0.55F};
    std::array<float, 6> result{};
    const float c = std::cos(-heading_), s = std::sin(-heading_);
    for (std::size_t hand = 0; hand < 2; ++hand) {
      const bool held = grips[hand] > 0.5F;
      if (held && !engaged_[hand]) {
        reference_[hand] = world[hand];
        base_[hand] = command_[hand];
      }
      engaged_[hand] = held;
      if (held) {
        const float x = world[hand][0] - reference_[hand][0];
        const float y = world[hand][1] - reference_[hand][1];
        const std::array<float, 3> delta{c*x-s*y, s*x+c*y, world[hand][2]-reference_[hand][2]};
        for (std::size_t axis = 0; axis < 3; ++axis) {
          desired_[hand][axis] = std::clamp(base_[hand][axis] + 0.65F * delta[axis], lo[axis], hi[axis]);
        }
      }
      std::array<float, 3> delta{};
      float norm = 0.0F;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        delta[axis] = desired_[hand][axis] - command_[hand][axis];
        norm += delta[axis] * delta[axis];
      }
      norm = std::sqrt(norm);
      const float scale = norm > 1.5F * dt ? 1.5F * dt / norm : 1.0F;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        command_[hand][axis] += scale * delta[axis];
        result[hand * 3 + axis] = command_[hand][axis];
      }
    }
    return result;
  }
 private:
  float heading_ = 0.0F;
  Hands reference_{}, base_{}, desired_{}, command_{};
  std::array<bool, 2> engaged_{};
};
}  // namespace protomotions::deployment
