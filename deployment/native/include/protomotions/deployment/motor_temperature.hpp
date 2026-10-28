// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace protomotions::deployment {

// Telemetry only: never changes commands or control state. Both hardware
// launchers use this class. Times are monotonic
// seconds; temperatures are Celsius, in hardware order: winding, driver.
class MotorTemperatureMonitor {
 public:
  bool any_hot() const {
    return std::any_of(hot_.begin(), hot_.end(), [](bool hot) { return hot; });
  }
  explicit MotorTemperatureMonitor(double enter = 90.0, double clear = 85.0,
                                   double interval = 1.0)
      : enter_(enter), clear_(clear), interval_(interval) {
    if (!std::isfinite(enter) || !std::isfinite(clear) ||
        !std::isfinite(interval) || clear >= enter || interval <= 0.0) {
      throw std::invalid_argument("invalid motor temperature warning thresholds");
    }
  }

  std::string update(const std::vector<std::array<double, 2>>& temperatures,
                     double now) {
    if (!std::isfinite(now)) return {};
    if (temperatures.empty()) {
      if (reported_ && now >= last_report_ && now - last_report_ < interval_) return {};
      reported_ = true;
      last_report_ = now;
      return "[MotorTemperature] telemetry unavailable; previous warning state retained";
    }
    if (hot_.size() != temperatures.size()) hot_.resize(temperatures.size(), false);
    bool invalid = false;
    bool changed = false;
    for (std::size_t i = 0; i < temperatures.size(); ++i) {
      const auto& t = temperatures[i];
      if (!std::isfinite(t[0]) || !std::isfinite(t[1])) {
        invalid = true;
        continue;  // Missing telemetry is not evidence of cooling.
      }
      const double maximum = std::max(t[0], t[1]);
      const bool hot = hot_[i] ? maximum >= clear_ : maximum >= enter_;
      changed |= hot != hot_[i];
      hot_[i] = hot;
    }
    const bool any_hot = std::any_of(hot_.begin(), hot_.end(), [](bool hot) { return hot; });
    if (!changed && !invalid && !any_hot) return {};
    if (!changed && reported_ && now >= last_report_ && now - last_report_ < interval_) return {};
    reported_ = true;
    last_report_ = now;
    std::ostringstream message;
    if (invalid) message << "[MotorTemperature] telemetry unavailable/invalid; ";
    if (any_hot) {
      message << "[HighTemp] ";
      for (std::size_t i = 0; i < hot_.size(); ++i) {
        if (!hot_[i]) continue;
        message << "motor[" << i << "]";
        if (temperatures.size() == names_.size()) message << " " << names_[i];
        message << " winding=" << temperatures[i][0]
                << "C driver=" << temperatures[i][1] << "C; ";
      }
      message << "warning only (control unchanged)";
    } else if (changed && !invalid) {
      message << "[MotorTemperature] high-temperature warning cleared";
    }
    return message.str();
  }

 private:
  double enter_, clear_, interval_, last_report_ = 0.0;
  bool reported_ = false;
  std::vector<bool> hot_;
  inline static constexpr std::array<const char*, 29> names_ = {
      "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
      "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch", "right_hip_roll",
      "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
      "waist_yaw", "waist_roll", "waist_pitch", "left_shoulder_pitch",
      "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll",
      "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch",
      "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
      "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"};
};
}  // namespace protomotions::deployment
