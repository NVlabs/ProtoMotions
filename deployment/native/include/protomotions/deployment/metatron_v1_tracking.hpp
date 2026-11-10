// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "protomotions/deployment/metatron_v1_contract.hpp"
#include "protomotions/deployment/metatron_v1_recovery.hpp"
#include "protomotions/deployment/motion_library.hpp"

namespace protomotions::deployment::metatron_v1 {

struct TrackingMetrics {
  float mean_body_error_m = 0.0F;         // conditionable bodies, against the re-anchored reference
  float anchor_height_error_m = 0.0F;     // the anchor's xy is re-anchored by construction; its height is not
  float mean_rotation_error_deg = 0.0F;
};

// Plays a reference motion into the policy's five target slots. The robot has
// no global position worth trusting, so each cycle the reference is translated
// so that its torso xy at the current time coincides with the robot's torso xy:
// every other body, and every future slot, keeps its offset from that torso.
// Heading and floor height are aligned once, when the motion starts, and the
// rotations are then taken from the reference as they are.
class TrackingPlayer final {
 public:
  static constexpr std::array<float, 5> kHorizons = {0.02F, 0.04F, 0.08F, 0.16F, 0.32F};

  explicit TrackingPlayer(const MotionLibrary& library);

  void start(std::size_t motion, const RecoveryPose& measured);
  void stop() noexcept;
  void advance(float dt_seconds) noexcept;

  bool active() const noexcept { return active_; }
  const MotionLibrary& library() const noexcept { return library_; }
  std::size_t motion() const noexcept { return motion_; }
  float elapsed_seconds() const noexcept { return elapsed_; }
  float duration_seconds() const;
  bool complete() const;

  // Fills ref_pos/ref_rot for every body and slot, the global target masks for
  // the conditionable bodies, the anchor masks, time offsets and the average
  // velocity command; the local tensors mirror the global ones unmasked.
  void write_inputs(Inputs& inputs, const RecoveryPose& measured) const;
  // Reference pose at ``time_seconds`` re-anchored to the robot's current torso
  // xy (positions body_count*3, rotations body_count*4).
  void reference(float time_seconds, const RecoveryPose& measured,
                 std::span<float> positions, std::span<float> rotations) const;
  TrackingMetrics metrics(const RecoveryPose& measured) const;

 private:
  const MotionLibrary& library_;
  bool active_ = false;
  std::size_t motion_ = 0;
  float elapsed_ = 0.0F;
  float yaw_delta_ = 0.0F;      // robot heading minus reference heading at the start
  float floor_offset_ = 0.0F;   // robot floor minus reference floor at the start
  mutable std::vector<float> pos_, rot_, now_pos_, now_rot_;
};

}  // namespace protomotions::deployment::metatron_v1
