// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_recovery.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace protomotions::deployment::metatron_v1 {
namespace {
constexpr std::array<float, 5> kOffsets = {0.2F, 0.4F, 0.6F, 0.8F, 1.0F};

std::array<float, 4> slerp(const float* a, const float* b, float fraction) {
  std::array<float, 4> right{b[0], b[1], b[2], b[3]};
  float dot = 0.0F;
  for (int axis = 0; axis < 4; ++axis) dot += a[axis] * right[axis];
  if (dot < 0.0F) {
    for (float& value : right) value = -value;
    dot = -dot;
  }
  dot = std::clamp(dot, 0.0F, 1.0F);
  const float angle = std::acos(dot);
  float left_weight = 1.0F - fraction;
  float right_weight = fraction;
  if (angle > 1.0e-5F) {
    const float inv = 1.0F / std::sin(angle);
    left_weight = std::sin((1.0F - fraction) * angle) * inv;
    right_weight = std::sin(fraction * angle) * inv;
  }
  std::array<float, 4> result{};
  float norm_squared = 0.0F;
  for (int axis = 0; axis < 4; ++axis) {
    result[axis] = left_weight * a[axis] + right_weight * right[axis];
    norm_squared += result[axis] * result[axis];
  }
  const float inv_norm = 1.0F / std::sqrt(norm_squared);
  for (float& value : result) value *= inv_norm;
  return result;
}

float height_from_positions(const std::vector<float>& values, std::size_t slot) {
  const std::size_t base = slot * 33 * 3;
  float lowest = std::numeric_limits<float>::infinity();
  for (std::size_t body = 0; body < 33; ++body) {
    lowest = std::min(lowest, values[base + body * 3 + 2]);
  }
  return values[base + 16 * 3 + 2] - lowest;
}
}  // namespace

StagedRecovery::StagedRecovery(
    const std::string& plan_path,
    const std::vector<std::string>& policy_body_names) {
  std::ifstream file(plan_path);
  if (!file) throw std::runtime_error("cannot open Metatron recovery plan");
  const auto plan = nlohmann::json::parse(file);
  if (plan.at("schema") != "metatron.recovery.local_plan/v1" ||
      plan.at("anchor_body_index") != 16 ||
      plan.at("quaternion_order") != "xyzw" ||
      plan.at("body_names").get<std::vector<std::string>>() != policy_body_names ||
      policy_body_names.size() != 33) {
    throw std::runtime_error("recovery plan and scene-policy body layout differ");
  }
  frame_dt_seconds_ = plan.at("dt_seconds").get<float>();
  const std::size_t frames = plan.at("frames").get<std::size_t>();
  if (frames < 2 || !std::isfinite(frame_dt_seconds_) ||
      frame_dt_seconds_ <= 0.0F || plan.at("positions").size() != frames ||
      plan.at("rotations").size() != frames) {
    throw std::runtime_error("invalid Metatron recovery plan frame count");
  }
  positions_.resize(frames);
  rotations_.resize(frames);
  mirrored_positions_.resize(frames);
  mirrored_rotations_.resize(frames);
  std::array<std::size_t, 33> mirror_indices{};
  for (std::size_t body = 0; body < 33; ++body) {
    std::string opposite = policy_body_names[body];
    if (opposite.starts_with("left_")) opposite.replace(0, 5, "right_");
    else if (opposite.starts_with("right_")) opposite.replace(0, 6, "left_");
    const auto found = std::find(policy_body_names.begin(), policy_body_names.end(),
                                 opposite);
    if (found == policy_body_names.end())
      throw std::runtime_error("recovery plan body lacks reflection partner");
    mirror_indices[body] = static_cast<std::size_t>(found - policy_body_names.begin());
  }
  for (std::size_t frame = 0; frame < frames; ++frame) {
    const auto& positions = plan.at("positions")[frame];
    const auto& rotations = plan.at("rotations")[frame];
    if (positions.size() != 33 || rotations.size() != 33)
      throw std::runtime_error("invalid recovery plan body dimensions");
    for (std::size_t body = 0; body < 33; ++body) {
      if (positions[body].size() != 3 || rotations[body].size() != 4)
        throw std::runtime_error("invalid recovery plan pose dimensions");
      float squared = 0.0F;
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const float value = positions[body][axis].get<float>();
        if (!std::isfinite(value)) throw std::runtime_error("non-finite recovery position");
        positions_[frame][body * 3 + axis] = value;
      }
      for (std::size_t axis = 0; axis < 4; ++axis) {
        const float value = rotations[body][axis].get<float>();
        if (!std::isfinite(value)) throw std::runtime_error("non-finite recovery rotation");
        rotations_[frame][body * 4 + axis] = value;
        squared += value * value;
      }
      if (std::abs(squared - 1.0F) > 2.0e-4F)
        throw std::runtime_error("non-unit recovery quaternion");
    }
    for (std::size_t body = 0; body < 33; ++body) {
      const std::size_t source = mirror_indices[body];
      for (std::size_t axis = 0; axis < 3; ++axis) {
        mirrored_positions_[frame][body * 3 + axis] =
            positions_[frame][source * 3 + axis] * (axis == 1 ? -1.0F : 1.0F);
      }
      for (std::size_t axis = 0; axis < 4; ++axis) {
        mirrored_rotations_[frame][body * 4 + axis] =
            rotations_[frame][source * 4 + axis] *
            ((axis == 0 || axis == 2) ? -1.0F : 1.0F);
      }
    }
  }
  reset();
}

void StagedRecovery::reset() noexcept {
  phase_ = Phase::kInactive;
  elapsed_ = 0.0F;
  phase_start_ = 0.0F;
  ready_since_ = 0.0F;
  has_ready_since_ = false;
  mirror_prone_ = false;
  just_finished_ = false;
}

void StagedRecovery::transition(Phase phase) noexcept {
  phase_ = phase;
  phase_start_ = elapsed_;
}

bool StagedRecovery::intentional_low(const Inputs& inputs, float height,
                                     float up) const {
  if (up <= 0.3F) return false;
  float requested = -std::numeric_limits<float>::infinity();
  for (std::size_t slot = 0; slot < 5; ++slot) {
    if (inputs.booleans(13)[slot] && inputs.booleans(12)[slot * 11]) {
      requested = std::max(requested, inputs.floats(14)[slot * 5 * 9 + 2]);
    }
    for (const bool local : {false, true}) {
      const std::size_t mask_index = local ? 7 : 16;
      const std::size_t pose_index = local ? 8 : 17;
      const std::size_t position_index = local ? 5 : 10;
      const bool visible = inputs.booleans(mask_index)[slot * 10] &&
                           inputs.booleans(pose_index)[slot] &&
                           (local || inputs.booleans(3)[slot * 2 + 1]);
      if (visible) {
        requested = std::max(requested,
                             height_from_positions(inputs.floats(position_index), slot));
      }
    }
  }
  return requested < 0.5F && std::abs(requested - height) < 0.15F;
}

bool StagedRecovery::step(Inputs& inputs, const RecoveryPose& pose,
                          float dt_seconds) {
  if (!std::isfinite(dt_seconds) || dt_seconds <= 0.0F)
    throw std::invalid_argument("recovery step needs positive finite dt");
  float lowest = std::numeric_limits<float>::infinity();
  for (std::size_t body = 0; body < 33; ++body)
    lowest = std::min(lowest, pose.positions[body * 3 + 2]);
  const float height = pose.positions[16 * 3 + 2] - lowest;
  const float x = pose.rotations[16 * 4];
  const float y = pose.rotations[16 * 4 + 1];
  const float z = pose.rotations[16 * 4 + 2];
  const float w = pose.rotations[16 * 4 + 3];
  const float up = 1.0F - 2.0F * (x * x + y * y);
  const float forward = 2.0F * (x * z - w * y);
  const float knees = (pose.joints[3] + pose.joints[9]) * 0.5F;
  const bool fallen = height < 0.5F || up < 0.25F;
  just_finished_ = false;
  if (phase_ == Phase::kInactive &&
      (!fallen || intentional_low(inputs, height, up))) return false;
  if (phase_ == Phase::kInactive || (phase_ == Phase::kSettle && fallen)) {
    elapsed_ = 0.0F;
    has_ready_since_ = false;
    if (height < 0.35F && forward < -0.65F) transition(Phase::kPronePlan);
    else if (height < 0.35F && up > 0.3F && knees > 0.8F)
      transition(Phase::kSeatedPlan);
    else transition(Phase::kRamp);
    const float hand_dx = pose.positions[32 * 3] - pose.positions[16 * 3];
    const float hand_dy = pose.positions[32 * 3 + 1] - pose.positions[16 * 3 + 1];
    const float hand_dz = pose.positions[32 * 3 + 2] - pose.positions[16 * 3 + 2];
    const float lateral = 2.0F * (x * y - w * z) * hand_dx +
                          (1.0F - 2.0F * (x * x + z * z)) * hand_dy +
                          2.0F * (y * z + w * x) * hand_dz;
    mirror_prone_ = phase_ == Phase::kPronePlan && std::abs(lateral) < 0.12F;
  } else {
    elapsed_ += dt_seconds;
  }
  if (phase_ != Phase::kSettle) {
    if (height > 0.55F && up > 0.5F) {
      if (!has_ready_since_) ready_since_ = elapsed_;
      has_ready_since_ = true;
    } else {
      has_ready_since_ = false;
    }
    if (has_ready_since_ && elapsed_ - ready_since_ >= 0.2F - 1.0e-9F)
      transition(Phase::kSettle);
  }
  apply(inputs, height);
  if (phase_ == Phase::kSettle &&
      elapsed_ - phase_start_ >= 0.5F - 1.0e-9F &&
      height > 0.7F && up > 0.85F) {
    transition(Phase::kInactive);
    just_finished_ = true;
  }
  return true;
}

void StagedRecovery::apply(Inputs& inputs, float height) {
  for (std::size_t index : {16U, 17U, 3U, 2U, 12U, 13U, 19U, 7U, 8U, 37U}) {
    auto& values = inputs.booleans(index);
    std::fill(values.begin(), values.end(), 0U);
  }
  auto& masks = inputs.booleans(7);
  std::fill(inputs.booleans(8).begin(), inputs.booleans(8).end(), 1U);
  std::copy(kOffsets.begin(), kOffsets.end(), inputs.floats(9).begin());
  auto& positions = inputs.floats(5);
  auto& rotations = inputs.floats(6);
  if (phase_ == Phase::kPronePlan || phase_ == Phase::kSeatedPlan) {
    sample_plan(inputs, phase_ == Phase::kSeatedPlan ? 4.5F : 0.0F);
    for (std::size_t slot = 0; slot < 5; ++slot) {
      for (std::size_t body = 0; body < 5; ++body)
        masks[slot * 10 + body * 2] = 1;
      masks[slot * 10 + 1] = 1;
    }
  } else {
    std::fill(positions.begin(), positions.end(), 0.0F);
    std::fill(rotations.begin(), rotations.end(), 0.0F);
    for (std::size_t body = 0; body < 5 * 33; ++body)
      rotations[body * 4 + 3] = 1.0F;
    const bool settling = phase_ == Phase::kSettle;
    for (std::size_t slot = 0; slot < 5; ++slot) {
      const float offset = kOffsets[slot];
      positions[slot * 33 * 3 + 16 * 3 + 2] = settling
          ? 0.79F : height + std::clamp(0.79F - height, -0.8F * offset,
                                       0.8F * offset);
      masks[slot * 10] = 1;
      masks[slot * 10 + 1] = static_cast<std::uint8_t>(settling);
    }
    std::fill(inputs.floats(4).begin(), inputs.floats(4).end(), 0.0F);
    inputs.booleans(19)[0] = static_cast<std::uint8_t>(settling);
  }
}

void StagedRecovery::sample_plan(Inputs& inputs, float source_start_seconds) {
  const auto& source_pos = phase_ == Phase::kPronePlan && mirror_prone_
                               ? mirrored_positions_ : positions_;
  const auto& source_rot = phase_ == Phase::kPronePlan && mirror_prone_
                               ? mirrored_rotations_ : rotations_;
  auto& positions = inputs.floats(5);
  auto& rotations = inputs.floats(6);
  const float plan_seconds =
      source_start_seconds + (elapsed_ - phase_start_) / 3.0F;
  for (std::size_t slot = 0; slot < 5; ++slot) {
    const float frame = std::clamp((plan_seconds + kOffsets[slot]) /
                                        frame_dt_seconds_,
                                    0.0F, static_cast<float>(positions_.size() - 1));
    const auto low = static_cast<std::size_t>(frame);
    const auto high = std::min(low + 1, positions_.size() - 1);
    const float fraction = frame - static_cast<float>(low);
    for (std::size_t axis = 0; axis < 33 * 3; ++axis) {
      positions[slot * 33 * 3 + axis] =
          source_pos[low][axis] + fraction *
          (source_pos[high][axis] - source_pos[low][axis]);
    }
    for (std::size_t body = 0; body < 33; ++body) {
      const auto rotation = slerp(source_rot[low].data() + body * 4,
                                  source_rot[high].data() + body * 4, fraction);
      std::copy(rotation.begin(), rotation.end(),
                rotations.begin() + slot * 33 * 4 + body * 4);
    }
  }
}
}  // namespace protomotions::deployment::metatron_v1
