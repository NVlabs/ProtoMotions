// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

#include "protomotions/deployment/types.hpp"

namespace protomotions::deployment {

class SafetySupervisor {
 public:
  explicit SafetySupervisor(SafetyLimits limits = {}) : limits_(limits) {}

  bool validate_state(const RobotState& state,
                     const RobotCapabilities& capabilities,
                     std::chrono::steady_clock::time_point now,
                     std::string* reason = nullptr) const;
  // Validate backend-advertised default gains before arming. Applications
  // must provide finite caps whenever a backend can supply implicit gains;
  // failing at startup is clearer and safer than faulting on the first cycle.
  bool validate_capabilities(const RobotCapabilities& capabilities,
                             std::string* reason = nullptr) const;
  bool validate_command(const ActuatorCommand& command,
                        const RobotCapabilities& capabilities,
                        std::string* reason = nullptr) const;

  std::chrono::milliseconds max_cycle_interval() const noexcept {
    return limits_.max_cycle_interval;
  }
  std::chrono::milliseconds repeated_state_grace() const noexcept {
    return limits_.repeated_state_grace;
  }

 private:
  SafetyLimits limits_;
};

}  // namespace protomotions::deployment
