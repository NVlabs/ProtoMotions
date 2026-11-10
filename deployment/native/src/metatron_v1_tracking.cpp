// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_tracking.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace protomotions::deployment::metatron_v1 {
namespace {

constexpr float kPi = 3.14159265358979323846F;
constexpr float kVelocityHorizon = 0.5F;  // the policy's average-velocity command looks this far ahead

float yaw_of(const float* q) {
  return std::atan2(2.0F * (q[3] * q[2] + q[0] * q[1]), 1.0F - 2.0F * (q[1] * q[1] + q[2] * q[2]));
}

float lowest_z(std::span<const float> positions) {
  float lowest = std::numeric_limits<float>::infinity();
  for (std::size_t i = 2; i < positions.size(); i += 3) lowest = std::min(lowest, positions[i]);
  return lowest;
}

}  // namespace

TrackingPlayer::TrackingPlayer(const MotionLibrary& library) : library_(library) {
  if (library_.body_count() != 33)
    throw std::invalid_argument("tracking motions must describe the policy's 33 bodies");
  if (library_.conditionable_body_ids().size() != 5)
    throw std::invalid_argument("tracking motions must name the policy's five conditionable bodies");
  pos_.assign(33 * 3, 0.0F);
  rot_.assign(33 * 4, 0.0F);
  now_pos_.assign(33 * 3, 0.0F);
  now_rot_.assign(33 * 4, 0.0F);
}

void TrackingPlayer::start(std::size_t motion, const RecoveryPose& measured) {
  if (motion >= library_.motion_count()) throw std::out_of_range("tracking motion index");
  library_.sample(motion, 0.0F, pos_, rot_);
  const int anchor = library_.anchor_body_index();
  // Heading and floor are aligned once: the reference then plays in the
  // robot's frame of the moment it started, which is what an operator expects
  // of "do this motion here, facing this way".
  yaw_delta_ = yaw_of(measured.rotations.data() + anchor * 4) - yaw_of(rot_.data() + anchor * 4);
  floor_offset_ = lowest_z(measured.positions) - lowest_z(pos_);
  motion_ = motion;
  elapsed_ = 0.0F;
  active_ = true;
}

void TrackingPlayer::stop() noexcept {
  active_ = false;
  elapsed_ = 0.0F;
}

void TrackingPlayer::advance(float dt_seconds) noexcept {
  if (!active_ || !std::isfinite(dt_seconds) || dt_seconds <= 0.0F) return;
  elapsed_ = std::min(duration_seconds(), elapsed_ + dt_seconds);
}

float TrackingPlayer::duration_seconds() const {
  return active_ ? library_.motion_length(motion_) : 0.0F;
}

bool TrackingPlayer::complete() const {
  return active_ && elapsed_ >= duration_seconds();
}

void TrackingPlayer::reference(float time_seconds, const RecoveryPose& measured,
                               std::span<float> positions, std::span<float> rotations) const {
  if (!active_) throw std::logic_error("no tracking motion is playing");
  const int anchor = library_.anchor_body_index();
  const float t_now = std::clamp(elapsed_, 0.0F, duration_seconds());
  const float t = std::clamp(time_seconds, 0.0F, duration_seconds());
  // The reference torso at the current time is the point that coincides with
  // the robot's torso; a future slot keeps its offset from that point.
  library_.sample(motion_, t_now, now_pos_, now_rot_);
  const float ax = now_pos_[anchor * 3], ay = now_pos_[anchor * 3 + 1];
  const float rx = measured.positions[anchor * 3], ry = measured.positions[anchor * 3 + 1];
  library_.sample(motion_, t, positions, rotations);
  const float c = std::cos(yaw_delta_), s = std::sin(yaw_delta_);
  const float qz = std::sin(0.5F * yaw_delta_), qw = std::cos(0.5F * yaw_delta_);
  for (std::size_t b = 0; b < 33; ++b) {
    const float dx = positions[b * 3] - ax, dy = positions[b * 3 + 1] - ay;
    positions[b * 3] = rx + c * dx - s * dy;
    positions[b * 3 + 1] = ry + s * dx + c * dy;
    positions[b * 3 + 2] += floor_offset_;
    // yaw quaternion (0,0,qz,qw) times the reference rotation, xyzw
    const float x = rotations[b * 4], y = rotations[b * 4 + 1], z = rotations[b * 4 + 2], w = rotations[b * 4 + 3];
    rotations[b * 4] = qw * x - qz * y;
    rotations[b * 4 + 1] = qw * y + qz * x;
    rotations[b * 4 + 2] = qw * z + qz * w;
    rotations[b * 4 + 3] = qw * w - qz * z;
  }
}

void TrackingPlayer::write_inputs(Inputs& inputs, const RecoveryPose& measured) const {
  if (!active_) return;
  const auto& conditionable = library_.conditionable_body_ids();
  auto& global_masks = inputs.booleans(16);
  for (std::size_t slot = 0; slot < kHorizons.size(); ++slot) {
    reference(elapsed_ + kHorizons[slot], measured, pos_, rot_);
    std::copy(pos_.begin(), pos_.end(), inputs.floats(10).begin() + slot * 33 * 3);
    std::copy(rot_.begin(), rot_.end(), inputs.floats(11).begin() + slot * 33 * 4);
    // The browser demo's "global" profile: every conditionable body visible in
    // position and rotation, the anchor fully specified, the heading taken from
    // the reference rather than held.
    for (std::size_t c = 0; c < conditionable.size(); ++c) {
      global_masks[slot * 10 + c * 2] = 1;
      global_masks[slot * 10 + c * 2 + 1] = 1;
    }
    inputs.booleans(17)[slot] = 1;
    inputs.floats(18)[slot] = kHorizons[slot];
    inputs.booleans(3)[slot * 2] = 1;
    inputs.booleans(3)[slot * 2 + 1] = 1;
    inputs.booleans(2)[slot * 2] = 1;
    inputs.booleans(2)[slot * 2 + 1] = 0;
  }
  inputs.floats(5) = inputs.floats(10);
  inputs.floats(6) = inputs.floats(11);
  inputs.floats(9) = inputs.floats(18);
  // Average anchor velocity over the next half second, in the robot's frame.
  const int anchor = library_.anchor_body_index();
  const float ahead = std::min(duration_seconds(), elapsed_ + kVelocityHorizon);
  const float window = std::max(library_.motion_dt(motion_), ahead - elapsed_);
  reference(elapsed_, measured, now_pos_, now_rot_);
  const float x0 = now_pos_[anchor * 3], y0 = now_pos_[anchor * 3 + 1];
  reference(ahead, measured, pos_, rot_);
  inputs.floats(4)[0] = (pos_[anchor * 3] - x0) / window;
  inputs.floats(4)[1] = (pos_[anchor * 3 + 1] - y0) / window;
  inputs.booleans(19)[0] = 1;
}

TrackingMetrics TrackingPlayer::metrics(const RecoveryPose& measured) const {
  TrackingMetrics out;
  if (!active_) return out;
  reference(elapsed_, measured, pos_, rot_);
  const auto& bodies = library_.conditionable_body_ids();
  for (const int body : bodies) {
    const float* p = measured.positions.data() + body * 3;
    const float* q = measured.rotations.data() + body * 4;
    out.mean_body_error_m += std::sqrt((p[0] - pos_[body * 3]) * (p[0] - pos_[body * 3]) +
                                       (p[1] - pos_[body * 3 + 1]) * (p[1] - pos_[body * 3 + 1]) +
                                       (p[2] - pos_[body * 3 + 2]) * (p[2] - pos_[body * 3 + 2]));
    const float dot = std::min(1.0F, std::abs(q[0] * rot_[body * 4] + q[1] * rot_[body * 4 + 1] +
                                              q[2] * rot_[body * 4 + 2] + q[3] * rot_[body * 4 + 3]));
    out.mean_rotation_error_deg += 2.0F * std::acos(dot) * 180.0F / kPi;
  }
  out.mean_body_error_m /= static_cast<float>(bodies.size());
  out.mean_rotation_error_deg /= static_cast<float>(bodies.size());
  const int anchor = library_.anchor_body_index();
  out.anchor_height_error_m = std::abs(measured.positions[anchor * 3 + 2] - pos_[anchor * 3 + 2]);
  return out;
}

}  // namespace protomotions::deployment::metatron_v1
