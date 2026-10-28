// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <string>
#include <vector>

#include "protomotions/deployment/metatron_v1_goals.hpp"
#include "protomotions/deployment/metatron_v1_recovery.hpp"
#include "protomotions/deployment/metatron_v1_rest.hpp"
#include "protomotions/deployment/types.hpp"

namespace protomotions::deployment::metatron_v1 {

// Stateless transport interface: the launcher writes a user command, and the
// same task object builds all 39 policy inputs from either backend's RobotState.
class MetatronTask final {
 public:
  enum class Mode { kIdle, kSteering };

  MetatronTask(FkEngine& fk, RestPose rest, const std::string& recovery_plan,
            const std::vector<std::string>& body_names);
  void reset() noexcept;
  void set_idle() noexcept;
  void set_steering(float target_heading,
                    std::array<float, 2> velocity_world);
  void build(Inputs& inputs, const RobotState& state);
  Mode mode() const noexcept { return mode_; }
  StagedRecovery::Phase recovery_phase() const noexcept { return recovery_.phase(); }

 private:
  void steering_goals(Inputs& inputs, const SensorReading& reading,
                      const RecoveryPose& measured, float heading);

  FkEngine& fk_;
  RestPose rest_;
  MetatronGoals goals_;
  StagedRecovery recovery_;
  Mode mode_ = Mode::kIdle;
  float target_heading_ = 0.0F;
  std::array<float, 2> velocity_world_{};
  float gait_phase_ = 0.0F;
  float hold_heading_ = 0.0F;
  bool heading_ready_ = false;
  bool support_ready_ = false;
  std::array<float, 2 * 3> support_feet_{};
  std::array<float, 3> support_anchor_{};
  float support_heading_ = 0.0F;
};

}  // namespace protomotions::deployment::metatron_v1
