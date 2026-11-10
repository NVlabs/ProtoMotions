// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "protomotions/deployment/metatron_v1_goals.hpp"
#include "protomotions/deployment/metatron_v1_recovery.hpp"
#include "protomotions/deployment/metatron_v1_rest.hpp"
#include "protomotions/deployment/metatron_v1_teleop.hpp"
#include "protomotions/deployment/metatron_v1_tracking.hpp"
#include <optional>
#include "protomotions/deployment/teleop_settings.hpp"
#include "protomotions/deployment/types.hpp"

namespace protomotions::deployment::metatron_v1 {

// Stateless transport interface: the launcher writes a user command, and the
// same task object builds all 39 policy inputs from either backend's RobotState.
class MetatronTask final {
 public:
  enum class Mode { kIdle, kSteering, kTeleop, kTracking };

  // Hand targets for teleoperation, world frame. They follow the current
  // feet frame while walking and the planted support frame while standing.
  struct HandTarget {
    bool enabled = false;
    bool rotation = false;
    std::array<float, 3> position{};
    std::array<float, 4> rotation_xyzw{0.0F, 0.0F, 0.0F, 1.0F};
  };

  // Where planted-foot goals are expressed while stationary.
  struct Support {
    std::array<float, 3> center{};
    float yaw = 0.0F;
    bool valid = false;
  };

  // Per-cycle task quantities for the metrics stream; the launcher decides
  // what to publish. Errors are between the command and the measurement.
  struct Metrics {
    float heading_error_rad = 0.0F;
    std::array<float, 2> velocity_error_mps{};   // requested minus measured anchor velocity, world xy
    float measured_speed_mps = 0.0F;
    float requested_speed_mps = 0.0F;
    std::array<float, 2> hand_error_m{};         // left, right; NaN when no goal
    float support_slip_m = 0.0F;                 // planted feet drift since planting
    bool stationary = true;
    std::uint32_t replants = 0;                  // stances re-planted after a step
    float torso_height_goal_m = 0.0F;            // commanded torso height above the support floor
    float torso_height_m = 0.0F;                 // measured torso height above the support floor
    std::array<float, 2> prompt_bias_m{};        // left, right: length of the pre-emphasis bias
    TrackingMetrics tracking{};                  // while a motion plays
    float tracking_elapsed_s = 0.0F;
    float tracking_duration_s = 0.0F;
    std::array<float, 15> tracking_reference{};  // conditionable bodies' reference xyz at the current time
  };

  // ``motions`` is optional: without a library the tracking mode is unavailable.
  MetatronTask(FkEngine& fk, RestPose rest, const std::string& recovery_plan,
            const std::vector<std::string>& body_names, const MotionLibrary* motions = nullptr);
  void reset() noexcept;
  void set_idle() noexcept;
  void set_steering(float target_heading,
                    std::array<float, 2> velocity_world);
  // Steering plus hand targets. Locomotion does not disable the hand goals.
  void set_teleop(float target_heading, std::array<float, 2> velocity_world,
                  const HandTarget& left, const HandTarget& right);
  // Operator-commanded torso height above the support floor (Pico headset
  // height, or a script). Applies on planted feet in teleop, clamped between
  // the crouch minimum and the upright height; the lower of this and the
  // hand-derived crouch wins. Without a command the torso follows the hands.
  void set_torso_height(bool enabled, float height_above_floor = 0.0F);
  // Play a reference motion from the library, aligned to the robot's heading
  // and floor at the next cycle and re-anchored to its torso xy every cycle.
  // Ends in idle when the motion finishes. Needs a library; see tracking_available().
  void start_tracking(std::size_t motion);
  bool tracking_available() const noexcept { return tracking_.has_value(); }
  const TrackingPlayer* tracking() const noexcept { return tracking_ ? &*tracking_ : nullptr; }
  void build(Inputs& inputs, const RobotState& state);
  Mode mode() const noexcept { return mode_; }
  // The task's own word for its mode, for operator displays; the enum value
  // means nothing to a viewer that serves more than one task.
  const char* mode_name() const noexcept {
    switch (mode_) {
      case Mode::kIdle: return "Idle";
      case Mode::kSteering: return "Steering";
      case Mode::kTeleop: return "Teleop";
      case Mode::kTracking: return "Tracking";
    }
    return "Unknown";
  }
  StagedRecovery::Phase recovery_phase() const noexcept { return recovery_.phase(); }
  Support support() const noexcept;
  // Hand frame uses planted support when stationary, otherwise the currently
  // measured feet midpoint and heading. Both backends use the same FK result.
  Support hand_frame() const noexcept;
  const Metrics& metrics() const noexcept { return metrics_; }

  // The operator tunables this task owns, with the values its qualification
  // covered. The speed default is what every qualification grid and tracking
  // run drove at; the operator may pull up to 5 m/s, which set_steering also
  // bounds, and the dashboard shows anything past the default in red. The
  // heading goal was never rate-limited before the yaw cap existed; a quarter
  // turn per second is the brisk, controllable default, and the hard bound is a
  // full turn in two seconds.
  static TeleopSettingsSchema settings_schema() {
    return {{"max_speed_mps", "Max speed", "m/s", 1.0F, 1.0F, 5.0F},
            {"max_yaw_rate_rps", "Max yaw rate", "rad/s", 1.5708F, 1.5708F, 3.1416F}};
  }
  // Measured hand positions (left, right) in the world frame from the last build;
  // the launcher anchors teleop goals to them when the operator takes over.
  bool measured_hands_ready() const noexcept { return measured_hands_ready_; }
  const std::array<std::array<float, 3>, 2>& measured_hands() const noexcept { return measured_hands_; }
  const std::array<std::array<float, 4>, 2>& measured_hand_rotations() const noexcept {
    return measured_hand_rotations_;
  }

 private:
  void steering_goals(Inputs& inputs, const SensorReading& reading,
                      const RecoveryPose& measured, float heading);
  void teleop_hand_goals(Inputs& inputs, const RecoveryPose& measured);

  FkEngine& fk_;
  RestPose rest_;
  MetatronGoals goals_;
  StagedRecovery recovery_;
  Mode mode_ = Mode::kIdle;
  float target_heading_ = 0.0F;
  std::array<float, 2> velocity_world_{};
  bool stationary_ = true;  // hysteresis around the stand/walk transition
  float gait_phase_ = 0.0F;
  float hold_heading_ = 0.0F;
  bool heading_ready_ = false;
  bool support_ready_ = false;
  std::array<float, 2 * 3> support_feet_{};
  std::array<float, 3> support_anchor_{};
  float support_heading_ = 0.0F;
  Support measured_hand_frame_{};
  std::array<HandTarget, 2> hands_{};
  float torso_height_sent_ = std::numeric_limits<float>::quiet_NaN();  // last torso height target, for the slew
  bool torso_goal_enabled_ = false;
  PromptEmphasis emphasis_{};
  std::array<std::array<float, 3>, 2> emphasis_bias_{};
  std::array<std::array<float, 3>, 2> emphasis_goal_{};      // goal the bias was built for
  std::array<std::array<float, 3>, 2> emphasis_measured_{};  // last measured hand, for velocity
  std::array<bool, 2> emphasis_ready_{};
  std::array<std::array<float, 3>, 2> measured_hands_{};
  std::array<std::array<float, 4>, 2> measured_hand_rotations_{};  // xyzw
  bool measured_hands_ready_ = false;
  float torso_goal_height_ = 0.0F;
  Metrics metrics_{};
  std::optional<TrackingPlayer> tracking_;
  std::optional<std::size_t> tracking_pending_;  // start at the next build(), once a pose is measured
  std::array<float, 3> previous_anchor_{};
  bool previous_anchor_ready_ = false;
  std::uint32_t replant_count_ = 0;
};

}  // namespace protomotions::deployment::metatron_v1
