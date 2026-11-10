// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace protomotions::deployment {

// Reference motions for tracking, in the bundle the browser demo ships
// (web_demo/scripts/extract_tracking_motions.py): per motion a name, a frame
// period and body positions (xyz) and rotations (xyzw) for every policy body
// at every frame, plus the anchor body and the conditionable-body order the
// policy expects. Loaded once at start; no Torch, Python or simulator here.
class MotionLibrary final {
 public:
  MotionLibrary() = default;
  explicit MotionLibrary(const std::string& path) { load(path); }

  void load(const std::string& path);

  std::size_t motion_count() const noexcept { return motions_.size(); }
  std::size_t body_count() const noexcept { return body_count_; }
  int anchor_body_index() const noexcept { return anchor_body_index_; }
  const std::vector<int>& conditionable_body_ids() const noexcept { return conditionable_; }
  const std::vector<std::string>& names() const noexcept { return names_; }
  std::optional<std::size_t> index_of(std::string_view name) const noexcept;
  std::size_t frame_count(std::size_t motion_index) const;
  float motion_dt(std::size_t motion_index) const;
  float motion_length(std::size_t motion_index) const;

  // Linear positions and slerped rotations at a time clamped to the motion.
  // body_pos has body_count()*3 elements and body_rot body_count()*4.
  void sample(std::size_t motion_index, float time_seconds,
              std::span<float> body_pos,
              std::span<float> body_rot) const;

 private:
  struct MotionIndex {
    std::size_t start = 0;  // first frame in the shared position/rotation arrays
    std::size_t frames = 0;
    float dt = 0.0F;
  };

  std::size_t body_count_ = 0;
  int anchor_body_index_ = 0;
  std::vector<int> conditionable_;
  std::vector<std::string> names_;
  std::vector<MotionIndex> motions_;
  std::vector<float> positions_;
  std::vector<float> rotations_;
};

}  // namespace protomotions::deployment
