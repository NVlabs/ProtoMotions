// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace protomotions::deployment {

// Compact runtime-only motion library. The .pt-to-binary conversion happens
// offline; this class has no Torch, Python, or simulator dependency.
class MotionLibrary final {
 public:
  MotionLibrary() = default;
  explicit MotionLibrary(const std::string& path) { load(path); }

  void load(const std::string& path);

  std::size_t motion_count() const noexcept { return motions_.size(); }
  std::size_t body_count() const noexcept { return body_count_; }
  std::size_t dof_count() const noexcept { return dof_count_; }
  std::size_t frame_count(std::size_t motion_index) const;
  float motion_dt(std::size_t motion_index) const;
  float motion_length(std::size_t motion_index) const;

  // body_pos must have body_count()*3 elements and body_rot must have
  // body_count()*4 elements. Rotations are xyzw.
  void sample(std::size_t motion_index, float time_seconds,
              std::span<float> body_pos,
              std::span<float> body_rot) const;

 private:
  struct MotionIndex {
    std::uint64_t start = 0;
    std::uint32_t frames = 0;
    float dt = 0.0F;
  };

  std::size_t body_count_ = 0;
  std::size_t dof_count_ = 0;
  std::vector<MotionIndex> motions_;
  std::vector<float> positions_;
  std::vector<float> rotations_;
};

}  // namespace protomotions::deployment
