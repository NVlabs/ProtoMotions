// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "protomotions/deployment/metatron_v1_contract.hpp"
#include "protomotions/deployment/metatron_v1_engine.hpp"
#include "protomotions/deployment/types.hpp"

namespace protomotions::deployment::metatron_v1 {

// Bounded arm-only differential IK on top of the policy's PD targets, the
// native counterpart of the browser demo's ArmIkCorrection. Each cycle it
// solves, per arm, a damped least-squares step from the measured hand pose
// error to a seven-joint PD-target offset, with the soft joint range, the
// offset budget and the rate limit folded into the box constraints before the
// solve. The offset is re-solved from the current error every cycle (the H*old
// term in the right-hand side), so it never integrates: when the policy
// reaches the goal on its own the correction returns to zero. It edits only the
// arm entries of the applied PD target and rewrites their raw actions through
// the manifest's affine map, so accepted-command history stays consistent.
struct ArmIkSettings {
  float fraction = 0.15F;          // share of the measured error corrected per solve
  float damping = 0.03F;           // DLS damping, metres per radian
  float max_offset = 0.025F;       // radians relative to the policy target
  float max_raw_offset = 0.05F;    // same budget in raw-action units (scaled per joint)
  float max_rate = 0.05F;          // radians per second the offset may change
  float soft_limit_factor = 0.9F;  // central share of the joint range the offset may use
  float orientation_length = 0.15F;  // metres per radian: weighs rotation against position
  float translation_clamp = 0.08F;   // metres: error seen by one solve
  float rotation_clamp = 0.4F;       // radians: error seen by one solve
};

struct HandGoal {
  bool enabled = false;
  bool translation = true;
  bool rotation = false;
  std::array<float, 3> position{};
  std::array<float, 4> rotation_xyzw{0.0F, 0.0F, 0.0F, 1.0F};
};

struct ArmIkDiagnostics {
  std::array<float, 2> hand_error_m{};      // left, right; measured before the solve
  std::array<float, 2> hand_rotation_rad{};  // left, right
  std::array<float, 2> offset_norm_rad{};    // left, right; applied offset magnitude
  float solve_milliseconds = 0.0F;
};

class ArmIkCorrection final {
 public:
  // `xml_path` is the robot MJCF used for kinematics only (never stepped);
  // `joint_names` is the policy's 29-joint order; scale/offset the manifest's
  // affine action map. Throws when an arm chain is not seven hinge joints.
  ArmIkCorrection(const std::string& xml_path,
                  const std::vector<std::string>& joint_names,
                  const std::array<float, kDofCount>& scale,
                  const std::array<float, kDofCount>& offset,
                  ArmIkSettings settings = {});
  ~ArmIkCorrection();
  ArmIkCorrection(const ArmIkCorrection&) = delete;
  ArmIkCorrection& operator=(const ArmIkCorrection&) = delete;

  void reset() noexcept;
  // side 0 = left, 1 = right; goals are world-frame, the frame of RobotState.
  void set_goal(int side, const HandGoal& goal);
  const HandGoal& goal(int side) const;
  ArmIkSettings& settings() noexcept;

  // Edits `outputs` in place using the measured `state`; `dt` is the control
  // period in seconds. Disabled goals leave their arm untouched and zero its
  // offset.
  void apply(Outputs& outputs, const RobotState& state, float dt);

  const std::array<float, kDofCount>& delta() const noexcept;
  const ArmIkDiagnostics& diagnostics() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Increment bounds in radians for one joint: how far the offset may move this
// cycle given the policy target, the soft range and the budget. Beyond the soft
// range the bounds pull back toward it at the rate limit instead of clipping
// the policy's command. Exposed for tests; mirrors the browser demo.
std::array<float, 2> correction_bounds(float target, float offset, float lower,
                                       float upper, float dt, float budget,
                                       float max_rate);

}  // namespace protomotions::deployment::metatron_v1
