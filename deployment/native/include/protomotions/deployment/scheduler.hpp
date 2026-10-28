// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>
#include <cstdint>
#include <thread>

namespace protomotions::deployment {

// Scheduling is deliberately independent from policy and robot semantics.
// A caller can pin the owning thread and apply platform-specific RT policy;
// this class only keeps a monotonic fixed-rate deadline and counts overruns.
class FixedRateScheduler {
 public:
  explicit FixedRateScheduler(std::chrono::microseconds period)
      : period_(period), valid_(period.count() > 0) {}

  bool valid() const noexcept { return valid_; }

  void reset(std::chrono::steady_clock::time_point now) {
    if (!valid_) {
      initialized_ = false;
      return;
    }
    next_deadline_ = now + period_;
    overruns_ = 0;
    initialized_ = true;
  }

  bool wait() {
    if (!initialized_ || !valid_) return false;
    std::this_thread::sleep_until(next_deadline_);
    advance(std::chrono::steady_clock::now());
    return true;
  }

  void advance(std::chrono::steady_clock::time_point now) {
    if (!initialized_ || !valid_) return;
    if (now <= next_deadline_) {
      next_deadline_ += period_;
      return;
    }
    const auto elapsed = now - next_deadline_;
    const auto skipped = elapsed / period_;
    overruns_ += static_cast<std::uint64_t>(skipped) + 1;
    next_deadline_ += period_ * (skipped + 1);
  }

  std::uint64_t overruns() const noexcept { return overruns_; }

 private:
  std::chrono::microseconds period_;
  std::chrono::steady_clock::time_point next_deadline_{};
  std::uint64_t overruns_ = 0;
  bool initialized_ = false;
  bool valid_ = false;
};

}  // namespace protomotions::deployment
