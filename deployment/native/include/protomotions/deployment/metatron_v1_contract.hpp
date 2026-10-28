// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace protomotions::deployment::metatron_v1 {
inline constexpr std::string_view kContract =
    "protomotions.masked_mimic.g1.v5_raw_context_scene";
inline constexpr std::size_t kInputCount = 39;
inline constexpr std::size_t kOutputCount = 4;
inline constexpr std::size_t kDofCount = 29;
inline constexpr std::size_t kHistoryCount = 8;
inline constexpr std::size_t kActionHistoryCount = 9;

struct InputSpec {
  const char* name;
  std::array<int, 5> dimensions;
  int rank;
  bool is_bool;
  constexpr std::size_t count() const noexcept {
    std::size_t result = 1;
    for (int i = 0; i < rank; ++i) {
      result *= static_cast<std::size_t>(dimensions[static_cast<std::size_t>(i)]);
    }
    return result;
  }
};

// Metatron v1 manifest order. Runtime validates name, rank, dims and dtype.
// The serialized contract identifier and tensor names are artifact ABI, not product names.
inline constexpr std::array<InputSpec, kInputCount> kInputs = {{
    {"current_root_pos", {1, 3}, 2, false},
    {"current_root_rot", {1, 4}, 2, false},
    {"masked_mimic_anchor_rotation_mask", {1, 5, 2}, 3, true},
    {"masked_mimic_anchor_translation_mask", {1, 5, 2}, 3, true},
    {"masked_mimic_average_velocity", {1, 2}, 2, false},
    {"masked_mimic_local_ref_pos", {1, 5, 33, 3}, 4, false},
    {"masked_mimic_local_ref_rot", {1, 5, 33, 4}, 4, false},
    {"masked_mimic_local_target_bodies_masks", {1, 50}, 2, true},
    {"masked_mimic_local_target_poses_masks", {1, 5}, 2, true},
    {"masked_mimic_local_time_offsets", {1, 5}, 2, false},
    {"masked_mimic_ref_pos", {1, 5, 33, 3}, 4, false},
    {"masked_mimic_ref_rot", {1, 5, 33, 4}, 4, false},
    {"masked_mimic_style_condition_masks", {1, 5, 11}, 3, true},
    {"masked_mimic_style_frame_masks", {1, 5}, 2, true},
    {"masked_mimic_style_pose_samples", {1, 225}, 2, false},
    {"masked_mimic_style_velocity_samples", {1, 150}, 2, false},
    {"masked_mimic_target_bodies_masks", {1, 50}, 2, true},
    {"masked_mimic_target_poses_masks", {1, 5}, 2, true},
    {"masked_mimic_time_offsets", {1, 5}, 2, false},
    {"masked_mimic_velocity_command_mask", {1, 1}, 2, true},
    {"noisy_anchor_pos", {1, 3}, 2, false},
    {"noisy_anchor_rot", {1, 4}, 2, false},
    {"noisy_dof_pos", {1, 29}, 2, false},
    {"noisy_dof_vel", {1, 29}, 2, false},
    {"noisy_root_local_ang_vel", {1, 3}, 2, false},
    {"noisy_historical_anchor_rot", {1, 8, 4}, 3, false},
    {"noisy_historical_dof_pos", {1, 8, 29}, 3, false},
    {"noisy_historical_dof_vel", {1, 8, 29}, 3, false},
    {"noisy_historical_raw_actions", {1, 8, 29}, 3, false},
    {"noisy_historical_root_local_ang_vel", {1, 8, 3}, 3, false},
    {"raw_action_history", {1, 9, 29}, 3, false},
    {"scene_boxes_centers", {1, 1, 2, 3}, 4, false},
    {"scene_boxes_extents", {1, 1, 2, 3}, 4, false},
    {"scene_boxes_object_pos", {1, 1, 3}, 3, false},
    {"scene_boxes_object_rot", {1, 1, 4}, 3, false},
    {"scene_boxes_position_offset", {1, 3}, 2, false},
    {"scene_boxes_valid", {1, 1, 2}, 3, true},
    {"scene_boxes_visible", {1, 1, 1}, 3, true},
    {"scene_boxes_yaw_offset", {1}, 1, false},
}};
inline constexpr std::array<const char*, kOutputCount> kOutputs = {
    "action", "stiffness", "damping", "raw_action"};

struct SensorReading {
  std::array<float, 29> dof_pos{};
  std::array<float, 29> dof_vel{};
  std::array<float, 3> root_local_ang_vel{};
  std::array<float, 4> anchor_rot{0, 0, 0, 1};
  std::array<float, 3> anchor_pos{};
  std::array<float, 3> root_pos{};
  std::array<float, 4> root_rot{0, 0, 0, 1};
};

// Fixed storage is allocated once. Inference does not advance history;
// only an accepted actuator command may call commit().
class Inputs {
 public:
  Inputs();
  void reset() noexcept;
  void prepare(const SensorReading& reading);
  void commit(const SensorReading& before_step,
              const std::array<float, kDofCount>& accepted_raw_action);
  std::vector<float>& floats(std::size_t index);
  const std::vector<float>& floats(std::size_t index) const;
  std::vector<std::uint8_t>& booleans(std::size_t index);
  const std::vector<std::uint8_t>& booleans(std::size_t index) const;

 private:
  std::array<std::vector<float>, kInputCount> float_values_{};
  std::array<std::vector<std::uint8_t>, kInputCount> bool_values_{};
  std::array<SensorReading, kHistoryCount> states_{};
  std::array<std::array<float, kDofCount>, kActionHistoryCount> actions_{};
  bool initialized_ = false;
  bool prepared_ = false;
};
}  // namespace protomotions::deployment::metatron_v1
