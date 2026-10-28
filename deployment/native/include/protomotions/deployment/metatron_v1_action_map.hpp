// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include "protomotions/deployment/metatron_v1_contract.hpp"

namespace protomotions::deployment::metatron_v1 {

// Invert the artifact's affine action map *after* the command boundary accepts
// a PD target. Inference itself never commits historical action state.
class ActionMap final {
 public:
  ActionMap(const std::array<float, kDofCount>& scale,
            const std::array<float, kDofCount>& offset)
      : scale_(scale), offset_(offset) {
    for (std::size_t joint = 0; joint < kDofCount; ++joint) {
      if (!std::isfinite(scale_[joint]) || scale_[joint] == 0.0F ||
          !std::isfinite(offset_[joint])) {
        throw std::invalid_argument("scene action map must be finite and invertible");
      }
    }
  }

  std::array<float, kDofCount> raw_from_applied(
      const std::array<float, kDofCount>& accepted_target) const {
    std::array<float, kDofCount> result{};
    for (std::size_t joint = 0; joint < kDofCount; ++joint) {
      if (!std::isfinite(accepted_target[joint])) {
        throw std::invalid_argument("non-finite accepted scene PD target");
      }
      result[joint] =
          (accepted_target[joint] - offset_[joint]) / scale_[joint];
    }
    return result;
  }

  bool matches_output(const std::array<float, kDofCount>& processed,
                      const std::array<float, kDofCount>& raw) const noexcept {
    for (std::size_t joint = 0; joint < kDofCount; ++joint) {
      const float expected = offset_[joint] + scale_[joint] * raw[joint];
      if (!std::isfinite(processed[joint]) || !std::isfinite(raw[joint]) ||
          std::abs(processed[joint] - expected) > 0.0001F) {
        return false;
      }
    }
    return true;
  }

 private:
  std::array<float, kDofCount> scale_;
  std::array<float, kDofCount> offset_;
};
}  // namespace protomotions::deployment::metatron_v1
