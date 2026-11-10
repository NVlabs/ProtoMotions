// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_task.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>
#include <stdexcept>

#include "protomotions/deployment/metatron_v1_sensor.hpp"

namespace protomotions::deployment::metatron_v1 {
namespace {
constexpr float kPi = 3.14159265358979323846F;
constexpr std::array<float, 5> kHorizons = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F};
constexpr int kTorso = 16, kRightFoot = 13, kLeftFoot = 7;
constexpr float kHandOnFloor = 0.03F;  // rubber hand resting on the floor, centre height
// Torso target while the feet are planted and the hands are low: it follows the
// lowest hand goal from this far above it, never below the minimum (a deeper
// squat sinks past what the fall detector allows) and never above upright.
// Ramped, because a stepped torso target makes the policy step with it.
constexpr float kCrouchAboveHand = 0.45F, kCrouchMinHeight = 0.55F, kCrouchRateMps = 0.4F;
constexpr int kLeftHand = 24, kRightHand = 32;
// Planar drift of a planted foot that counts as a step rather than a lean.
constexpr float kReplantDistance = 0.12F;

float wrap(float angle) {
  return std::remainder(angle, 2.0F * kPi);
}
std::array<float, 4> yaw_quat(float yaw) {
  return {0.0F, 0.0F, std::sin(yaw * 0.5F), std::cos(yaw * 0.5F)};
}
float yaw_of(const std::array<float, 4>& q) {
  return std::atan2(2.0F * (q[3] * q[2] + q[0] * q[1]),
                    1.0F - 2.0F * (q[1] * q[1] + q[2] * q[2]));
}
std::array<float, 3> rotate_yaw(float yaw, const std::array<float, 3>& v) {
  const float c = std::cos(yaw), s = std::sin(yaw);
  return {c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]};
}
std::array<float, 4> quat_mul(const std::array<float, 4>& a,
                              const std::array<float, 4>& b) {
  return {a[3]*b[0]+a[0]*b[3]+a[1]*b[2]-a[2]*b[1],
          a[3]*b[1]-a[0]*b[2]+a[1]*b[3]+a[2]*b[0],
          a[3]*b[2]+a[0]*b[1]-a[1]*b[0]+a[2]*b[3],
          a[3]*b[3]-a[0]*b[0]-a[1]*b[1]-a[2]*b[2]};
}
std::array<float, 3> pos(const RecoveryPose& pose, int body) {
  const float* p = pose.positions.data() + body * 3;
  return {p[0], p[1], p[2]};
}
std::array<float, 3> pos(const RestPose& pose, int body) {
  const float* p = pose.positions.data() + body * 3;
  return {p[0], p[1], p[2]};
}
std::array<float, 4> rot(const RestPose& pose, int body) {
  const float* q = pose.rotations.data() + body * 4;
  return {q[0], q[1], q[2], q[3]};
}
void put_pos(Inputs& inputs, int slot, int body, const std::array<float, 3>& p) {
  std::copy(p.begin(), p.end(), inputs.floats(10).begin() + (slot * 33 + body) * 3);
}
void put_rot(Inputs& inputs, int slot, int body, const std::array<float, 4>& q) {
  std::copy(q.begin(), q.end(), inputs.floats(11).begin() + (slot * 33 + body) * 4);
}
void copy_measured(Inputs& inputs, const RecoveryPose& pose) {
  for (int slot = 0; slot < 5; ++slot) {
    std::copy(pose.positions.begin(), pose.positions.end(),
              inputs.floats(10).begin() + slot * 33 * 3);
    std::copy(pose.rotations.begin(), pose.rotations.end(),
              inputs.floats(11).begin() + slot * 33 * 4);
    inputs.floats(18)[slot] = kHorizons[slot];
    inputs.booleans(17)[slot] = 1;
  }
}
}  // namespace

MetatronTask::MetatronTask(FkEngine& fk, RestPose rest,
                     const std::string& recovery_plan,
                     const std::vector<std::string>& body_names,
                           const MotionLibrary* motions)
    : fk_(fk), rest_(std::move(rest)),
      goals_(rest_.positions, rest_.rotations, rest_.style),
      recovery_(recovery_plan, body_names) {
  if (motions) tracking_.emplace(*motions);
}

void MetatronTask::reset() noexcept {
  recovery_.reset();
  mode_ = Mode::kIdle;
  target_heading_ = 0.0F;
  velocity_world_ = {};
  stationary_ = true;
  gait_phase_ = 0.0F;
  hold_heading_ = 0.0F;
  heading_ready_ = false;
  support_ready_ = false;
  support_heading_ = 0.0F;
  measured_hand_frame_ = {};
  hands_ = {};
  metrics_ = {};
  previous_anchor_ready_ = false;
  replant_count_ = 0;
  if (tracking_) tracking_->stop();
  tracking_pending_.reset();
}

void MetatronTask::set_idle() noexcept {
  if (mode_ != Mode::kIdle) {
    mode_ = Mode::kIdle;
    support_ready_ = false;
    heading_ready_ = false;
  }
  velocity_world_ = {};
  stationary_ = true;
  hands_ = {};
  torso_height_sent_ = std::numeric_limits<float>::quiet_NaN();
  if (tracking_) tracking_->stop();
  tracking_pending_.reset();
}

void MetatronTask::set_steering(float target_heading,
                             std::array<float, 2> velocity_world) {
  if (!std::isfinite(target_heading) || !std::isfinite(velocity_world[0]) ||
      !std::isfinite(velocity_world[1]) ||
      std::hypot(velocity_world[0], velocity_world[1]) > 5.01F)  // the schema's hard bound on speed
    throw std::invalid_argument("invalid Metatron steering request");
  if (mode_ != Mode::kSteering) {
    // Teleop and steering share the planted-feet state; only idle drops it.
    if (mode_ != Mode::kTeleop) support_ready_ = false;
    if (mode_ == Mode::kTracking && tracking_) tracking_->stop();
    tracking_pending_.reset();
    mode_ = Mode::kSteering;
  }
  target_heading_ = wrap(target_heading);
  velocity_world_ = velocity_world;
  hands_ = {};
}

void MetatronTask::set_teleop(float target_heading, std::array<float, 2> velocity_world,
                              const HandTarget& left, const HandTarget& right) {
  for (const HandTarget* hand : {&left, &right}) {
    if (!hand->enabled) continue;
    for (float v : hand->position)
      if (!std::isfinite(v)) throw std::invalid_argument("non-finite teleop hand target");
    for (float v : hand->rotation_xyzw)
      if (!std::isfinite(v)) throw std::invalid_argument("non-finite teleop hand rotation");
  }
  const Mode previous = mode_;
  set_steering(target_heading, velocity_world);
  mode_ = Mode::kTeleop;
  if (previous != Mode::kTeleop && previous != Mode::kSteering) support_ready_ = false;
  hands_ = {left, right};
}

void MetatronTask::start_tracking(std::size_t motion) {
  if (!tracking_) throw std::logic_error("no tracking motion library was loaded");
  tracking_->stop();
  tracking_pending_ = motion;
  hands_ = {};
  support_ready_ = false;
  torso_height_sent_ = std::numeric_limits<float>::quiet_NaN();
  mode_ = Mode::kTracking;
}

void MetatronTask::set_torso_height(bool enabled, float height_above_floor) {
  if (enabled && !std::isfinite(height_above_floor))
    throw std::invalid_argument("non-finite torso height goal");
  torso_goal_enabled_ = enabled;
  torso_goal_height_ = height_above_floor;
}

MetatronTask::Support MetatronTask::support() const noexcept {
  Support result;
  result.valid = support_ready_;
  if (support_ready_) {
    result.center = {0.5F * (support_feet_[0] + support_feet_[3]),
                     0.5F * (support_feet_[1] + support_feet_[4]),
                     std::min(support_feet_[2], support_feet_[5])};
    result.yaw = support_heading_;
  }
  return result;
}

MetatronTask::Support MetatronTask::hand_frame() const noexcept {
  if (support_ready_ && stationary_)
    return support();
  return measured_hand_frame_;
}

void MetatronTask::build(Inputs& inputs, const RobotState& state) {
  const SensorReading reading = reading_from_robot_state(state);
  const RecoveryPose measured = fk_.run(reading);
  const float heading = yaw_of(reading.anchor_rot);
  copy_measured(inputs, measured);
  measured_hands_ = {pos(measured, kLeftHand), pos(measured, kRightHand)};
  for (int side = 0; side < 2; ++side) {
    const float* q = measured.rotations.data() + (side == 0 ? kLeftHand : kRightHand) * 4;
    measured_hand_rotations_[side] = {q[0], q[1], q[2], q[3]};
  }
  measured_hands_ready_ = true;
  const auto right_foot = pos(measured, kRightFoot);
  const auto left_foot = pos(measured, kLeftFoot);
  measured_hand_frame_.center = {
      0.5F * (right_foot[0] + left_foot[0]),
      0.5F * (right_foot[1] + left_foot[1]),
      std::min(right_foot[2], left_foot[2])};
  measured_hand_frame_.yaw = heading;
  measured_hand_frame_.valid = std::isfinite(heading) &&
      std::all_of(measured_hand_frame_.center.begin(), measured_hand_frame_.center.end(),
                  [](float value) { return std::isfinite(value); });
  const float speed = std::hypot(velocity_world_[0], velocity_world_[1]);
  stationary_ = stationary_with_hysteresis(stationary_, speed);
  // "standing" swaps the local targets for the hands-down idle pose. With
  // teleop hands active the local targets must carry the hand goals instead,
  // so a stationary teleop stance is not treated as standing.
  const bool hands_active = mode_ == Mode::kTeleop && (hands_[0].enabled || hands_[1].enabled);
  const bool standing = mode_ == Mode::kIdle ||
      (mode_ != Mode::kTracking && !hands_active && speed <= 0.03F &&
       std::abs(wrap(target_heading_ - heading)) <= kPi / 9.0F);
  if (mode_ == Mode::kIdle) {
    if (!heading_ready_) {
      hold_heading_ = heading;
      heading_ready_ = true;
    } else {
      const float alpha = 1.0F - std::exp(-0.02F / 0.5F);
      hold_heading_ = wrap(hold_heading_ + alpha * wrap(heading - hold_heading_));
    }
    put_rot(inputs, 4, kTorso, yaw_quat(hold_heading_));
    inputs.booleans(16)[4 * 10 + 1] = 1;
    inputs.booleans(2)[4 * 2 + 1] = 1;
  } else if (mode_ == Mode::kTracking) {
    if (tracking_pending_) {
      tracking_->start(*tracking_pending_, measured);
      tracking_pending_.reset();
    }
    tracking_->write_inputs(inputs, measured);
    metrics_.tracking = tracking_->metrics(measured);
    metrics_.tracking_elapsed_s = tracking_->elapsed_seconds();
    metrics_.tracking_duration_s = tracking_->duration_seconds();
    {
      static thread_local std::vector<float> ref_pos(33 * 3), ref_rot(33 * 4);
      tracking_->reference(tracking_->elapsed_seconds(), measured, ref_pos, ref_rot);
      const auto& bodies = tracking_->library().conditionable_body_ids();
      for (std::size_t c = 0; c < bodies.size() && c < 5; ++c)
        for (int k = 0; k < 3; ++k) metrics_.tracking_reference[c * 3 + k] = ref_pos[bodies[c] * 3 + k];
    }
    tracking_->advance(0.02F);
  } else {
    steering_goals(inputs, reading, measured, heading);
    if (mode_ == Mode::kTeleop) teleop_hand_goals(inputs, measured);
  }
  // Tracking writes the global and local tensors itself; the steering goal
  // post-processing would move and mask them again.
  goals_.apply(inputs, mode_ == Mode::kIdle ? "idle" : mode_ == Mode::kTracking ? "odometry" :
               mode_ == Mode::kTeleop ? "teleop" : "steering", standing);
  if (mode_ == Mode::kTracking && tracking_->complete()) {
    // The motion is over: stand where it ended, as after any other command.
    tracking_->stop();
    mode_ = Mode::kIdle;
    heading_ready_ = false;
    velocity_world_ = {};
  }
  if (recovery_.step(inputs, measured, 0.02F)) {
    support_ready_ = false;
  }

  // Task metrics: what was asked against what the body did this cycle.
  const bool stationary = stationary_;
  metrics_.stationary = stationary;
  metrics_.requested_speed_mps = mode_ == Mode::kIdle ? 0.0F : speed;
  metrics_.heading_error_rad = mode_ == Mode::kIdle ? wrap(hold_heading_ - heading)
                                                     : wrap(target_heading_ - heading);
  std::array<float, 2> measured_velocity{};
  if (previous_anchor_ready_) {
    measured_velocity = {(reading.anchor_pos[0] - previous_anchor_[0]) / 0.02F,
                         (reading.anchor_pos[1] - previous_anchor_[1]) / 0.02F};
  }
  previous_anchor_ = reading.anchor_pos;
  previous_anchor_ready_ = true;
  metrics_.measured_speed_mps = std::hypot(measured_velocity[0], measured_velocity[1]);
  const std::array<float, 2> requested = mode_ == Mode::kIdle ? std::array<float, 2>{} : velocity_world_;
  metrics_.velocity_error_mps = {requested[0] - measured_velocity[0], requested[1] - measured_velocity[1]};
  metrics_.support_slip_m = 0.0F;
  metrics_.replants = replant_count_;
  if (mode_ != Mode::kTracking) {
    metrics_.tracking = {};
    metrics_.tracking_elapsed_s = metrics_.tracking_duration_s = 0.0F;
  }
  if (support_ready_) {
    for (int side = 0; side < 2; ++side) {
      const auto foot = pos(measured, side == 0 ? kRightFoot : kLeftFoot);
      const float slip = std::hypot(foot[0] - support_feet_[side * 3], foot[1] - support_feet_[side * 3 + 1]);
      metrics_.support_slip_m = std::max(metrics_.support_slip_m, slip);
    }
  }
  for (int side = 0; side < 2; ++side) {
    const HandTarget& hand = hands_[side];
    if (mode_ != Mode::kTeleop || !hand.enabled) {
      metrics_.hand_error_m[side] = std::numeric_limits<float>::quiet_NaN();
      continue;
    }
    const auto measured_hand = pos(measured, side == 0 ? kLeftHand : kRightHand);
    metrics_.hand_error_m[side] = std::sqrt(
        (measured_hand[0] - hand.position[0]) * (measured_hand[0] - hand.position[0]) +
        (measured_hand[1] - hand.position[1]) * (measured_hand[1] - hand.position[1]) +
        (measured_hand[2] - hand.position[2]) * (measured_hand[2] - hand.position[2]));
  }
}

void MetatronTask::teleop_hand_goals(Inputs& inputs, const RecoveryPose& measured) {
  // Keep hand goals active while walking. The launcher updates their world
  // positions from the measured feet frame each cycle, so they travel with the
  // robot instead of snapping back to the steering rest pose.
  const Support frame = hand_frame();
  if (!frame.valid) {
    emphasis_ready_ = {};
    return;
  }
  for (int side = 0; side < 2; ++side) {
    const HandTarget& hand = hands_[side];
    metrics_.prompt_bias_m[side] = 0.0F;
    if (!hand.enabled) {
      emphasis_ready_[side] = false;
      continue;
    }
    const int body = side == 0 ? kLeftHand : kRightHand;
    const int mask_index = side == 0 ? 6 : 8;  // conditionable order: torso, right foot, left foot, left hand, right hand
    std::array<float, 3> prompt = hand.position;
    if (!stationary_) {
      // Gait motion makes measured hand velocity a poor feedback signal.
      // Keep the direct Pico goal and restart emphasis when the feet settle.
      emphasis_ready_[side] = false;
    } else {
      const auto now = pos(measured, body);
      // A goal that jumps is a new destination (a script step, an operator
      // edit): the bias built for the old one would push toward the wrong
      // place, so it restarts. A goal that drifts keeps it, as a headset does.
      const auto& previous_goal = emphasis_goal_[side];
      const bool jumped = emphasis_ready_[side] &&
          std::hypot(hand.position[0] - previous_goal[0], hand.position[1] - previous_goal[1],
                     hand.position[2] - previous_goal[2]) > 0.05F;
      if (!emphasis_ready_[side] || jumped) {
        emphasis_bias_[side] = {};
        emphasis_measured_[side] = now;
        emphasis_ready_[side] = true;
      } else {
        std::array<float, 3> velocity{};
        for (int i = 0; i < 3; ++i) velocity[i] = (now[i] - emphasis_measured_[side][i]) / 0.02F;
        emphasis_measured_[side] = now;
        emphasis_bias_[side] = prompt_emphasis_step(emphasis_bias_[side], hand.position, now, velocity, emphasis_, 0.02F);
      }
      emphasis_goal_[side] = hand.position;
      for (int i = 0; i < 3; ++i) prompt[i] += emphasis_bias_[side][i];
      metrics_.prompt_bias_m[side] = std::sqrt(emphasis_bias_[side][0] * emphasis_bias_[side][0] +
                                               emphasis_bias_[side][1] * emphasis_bias_[side][1] +
                                               emphasis_bias_[side][2] * emphasis_bias_[side][2]);
    }
    prompt = clamp_hand_prompt({frame.center, frame.yaw, frame.valid}, prompt, TeleopHandMapping{});
    // A prompt under the floor is unreachable, even while walking.
    prompt[2] = std::max(prompt[2], frame.center[2] + kHandOnFloor);
    for (int slot = 0; slot < 5; ++slot) {
      put_pos(inputs, slot, body, prompt);
      inputs.booleans(16)[slot * 10 + mask_index] = 1;
      if (hand.rotation) {
        put_rot(inputs, slot, body, hand.rotation_xyzw);
        inputs.booleans(16)[slot * 10 + mask_index + 1] = 1;
      }
    }
  }
}

void MetatronTask::steering_goals(Inputs& inputs, const SensorReading& reading,
                                const RecoveryPose& measured, float heading) {
  const float speed = std::hypot(velocity_world_[0], velocity_world_[1]);
  const bool stationary = stationary_;
  // A planted stance is a promise about where the feet are. When a push or a
  // reach makes the policy step, holding the old foot targets would pull it
  // back toward a stance it has already left, and teleop hand goals anchored
  // there would point at the wrong place. Re-plant where the feet now stand.
  if (stationary && support_ready_) {
    for (int side = 0; side < 2; ++side) {
      const auto foot = pos(measured, side == 0 ? kRightFoot : kLeftFoot);
      if (std::hypot(foot[0] - support_feet_[side * 3], foot[1] - support_feet_[side * 3 + 1]) > kReplantDistance) {
        support_ready_ = false;
        ++replant_count_;
        break;
      }
    }
  }
  if (stationary && !support_ready_) {
    for (int side = 0; side < 2; ++side) {
      const auto foot = pos(measured, side == 0 ? kRightFoot : kLeftFoot);
      std::copy(foot.begin(), foot.end(), support_feet_.begin() + side * 3);
    }
    support_anchor_ = pos(measured, kTorso);
    support_ready_ = true;
    support_heading_ = heading;
  }
  if (!stationary) {
    support_ready_ = false;
    torso_height_sent_ = std::numeric_limits<float>::quiet_NaN();
  }
  const auto default_torso = pos(rest_, kTorso);
  const auto right_default = pos(rest_, kRightFoot);
  const auto left_default = pos(rest_, kLeftFoot);
  std::array<float, 3> foot_center{};
  for (int axis = 0; axis < 3; ++axis)
    foot_center[axis] = 0.5F * (right_default[axis] + left_default[axis]) - default_torso[axis];
  const float default_foot_floor = std::min(right_default[2], left_default[2]);
  const float half_width = std::max(0.07F, 0.5F * std::abs(left_default[1] - right_default[1]));
  const float cadence = std::min(2.0F, 0.83F + 0.93F * speed);
  const float swing = stationary ? 0.0F : 0.13F + 0.065F * speed;
  const float move_heading = stationary ? heading : std::atan2(velocity_world_[1], velocity_world_[0]);
  float ground = measured.positions[2];
  for (int body = 1; body < 33; ++body)
    ground = std::min(ground, measured.positions[body * 3 + 2]);
  for (int slot = 0; slot < 5; ++slot) {
    const float time = kHorizons[slot];
    const float yaw = wrap(heading + std::clamp(wrap(target_heading_ - heading),
                                                -kPi * time, kPi * time));
    const auto yaw_q = yaw_quat(yaw);
    std::array<float, 3> torso{};
    if (stationary) {
      torso = support_anchor_;
      const float cx = 0.5F * (support_feet_[0] + support_feet_[3]);
      const float cy = 0.5F * (support_feet_[1] + support_feet_[4]);
      torso[0] = cx;
      torso[1] = cy;
      const float floor = std::min(support_feet_[2], support_feet_[5]);
      float torso_height = rest_.upright_height - default_foot_floor;
      if (mode_ == Mode::kTeleop) {
        // Reaching the floor from a full-height torso target asks the policy to
        // fold at the waist; dropping the target with the lowest hand lets it
        // squat, which keeps the feet planted and the reach inside the arm span.
        float lowest = std::numeric_limits<float>::infinity();
        for (const HandTarget& hand : hands_)
          if (hand.enabled) lowest = std::min(lowest, hand.position[2] - floor);
        if (std::isfinite(lowest))
          torso_height = std::clamp(lowest + kCrouchAboveHand, kCrouchMinHeight, torso_height);
        if (torso_goal_enabled_)
          torso_height = std::min(torso_height, std::max(kCrouchMinHeight, torso_goal_height_));
      }
      // A stepped torso target makes the policy step with it; ramp the target
      // so a crouch or a stand-up is a motion the feet can stay planted for.
      if (slot == 0) {
        if (std::isfinite(torso_height_sent_)) {
          const float budget = kCrouchRateMps * 0.02F;
          torso_height = std::clamp(torso_height, torso_height_sent_ - budget, torso_height_sent_ + budget);
        }
        torso_height_sent_ = torso_height;
      } else {
        torso_height = torso_height_sent_;
      }
      torso[2] = floor + torso_height;
      if (slot == 0) {
        metrics_.torso_height_goal_m = torso_height;
        metrics_.torso_height_m = pos(measured, kTorso)[2] - floor;
      }
    } else {
      torso = {reading.anchor_pos[0] + velocity_world_[0] * time,
               reading.anchor_pos[1] + velocity_world_[1] * time,
               ground + rest_.upright_height};
    }
    put_pos(inputs, slot, kTorso, torso);
    // In a planted teleop stance ask for the MJCF neutral torso roll/pitch
    // as well as the commanded heading. Walking keeps heading-only control so
    // natural gait lean remains available to the policy.
    const bool neutral_torso_rotation = stationary && mode_ == Mode::kTeleop;
    put_rot(inputs, slot, kTorso,
            neutral_torso_rotation ? quat_mul(yaw_q, rot(rest_, kTorso)) : yaw_q);
    const bool turn_feet = stationary &&
        std::abs(wrap(yaw - support_heading_)) > kPi / 6.0F;
    std::array<float, 2 * 3> feet{};
    for (int side = 0; side < 2; ++side) {
      const int body = side == 0 ? kRightFoot : kLeftFoot;
      const auto original = pos(rest_, body);
      std::array<float, 3> offset{};
      for (int axis = 0; axis < 3; ++axis)
        offset[axis] = original[axis] - default_torso[axis];
      const float sign = side == 0 ? -1.0F : 1.0F;
      std::array<float, 3> foot{};
      if (stationary && !turn_feet) {
        std::copy(support_feet_.begin() + side * 3,
                  support_feet_.begin() + side * 3 + 3, foot.begin());
      } else if (stationary) {
        const auto relative = rotate_yaw(yaw,
            {offset[0] - foot_center[0], offset[1] - foot_center[1],
             offset[2] - foot_center[2]});
        foot = {torso[0] + relative[0], torso[1] + relative[1],
                support_feet_[side * 3 + 2]};
      } else {
        const float rel = wrap(move_heading - yaw);
        offset[1] = sign * (half_width + 0.12F * std::abs(std::sin(rel)));
        const float phase = gait_phase_ + 2.0F * kPi * cadence * time +
                            (side == 1 ? kPi : 0.0F);
        const float lateral = std::min(1.0F, 0.12F / swing) * std::abs(std::cos(rel));
        offset[0] += swing * std::cos(phase) * 0.75F * std::cos(rel);
        offset[1] += swing * std::cos(phase) * lateral * std::sin(rel);
        offset[2] += 0.08F * std::max(0.0F, -std::sin(phase));
        const auto rotated = rotate_yaw(yaw, offset);
        for (int axis = 0; axis < 3; ++axis)
          foot[axis] = torso[axis] + rotated[axis];
      }
      std::copy(foot.begin(), foot.end(), feet.begin() + side * 3);
      put_pos(inputs, slot, body, foot);
      put_rot(inputs, slot, body, quat_mul(yaw_q, rot(rest_, body)));
    }
    const std::array<float, 3> support = {
        0.5F * (feet[0] + feet[3]), 0.5F * (feet[1] + feet[4]),
        std::min(feet[2], feet[5])};
    for (const int body : {kLeftHand, kRightHand}) {
      const auto original = pos(rest_, body);
      const auto relative = rotate_yaw(yaw,
          {original[0] - default_torso[0] - foot_center[0],
           original[1] - default_torso[1] - foot_center[1],
           original[2] - default_torso[2] - foot_center[2]});
      put_pos(inputs, slot, body,
              {support[0] + relative[0], support[1] + relative[1],
               support[2] + relative[2]});
    }
    auto& mask = inputs.booleans(16);
    mask[slot * 10] = 1;
    mask[slot * 10 + 1] = 1;
    mask[slot * 10 + 4] = static_cast<std::uint8_t>(!stationary || turn_feet || mode_ == Mode::kTeleop);
    mask[slot * 10 + 2] = mask[slot * 10 + 4];
    mask[slot * 10 + 6] = static_cast<std::uint8_t>(stationary || slot == 4);
    mask[slot * 10 + 8] = mask[slot * 10 + 6];
    inputs.booleans(3)[slot * 2] = static_cast<std::uint8_t>(!stationary);
    inputs.booleans(3)[slot * 2 + 1] = 1;
    inputs.booleans(2)[slot * 2] = static_cast<std::uint8_t>(neutral_torso_rotation);
    inputs.booleans(2)[slot * 2 + 1] = 1;
  }
  inputs.floats(4)[0] = velocity_world_[0];
  inputs.floats(4)[1] = velocity_world_[1];
  inputs.booleans(19)[0] = 1;
  gait_phase_ = stationary ? 0.0F :
      std::fmod(gait_phase_ + 2.0F * kPi * cadence * 0.02F, 2.0F * kPi);
}

}  // namespace protomotions::deployment::metatron_v1
