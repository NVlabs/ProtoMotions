// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_task.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "protomotions/deployment/metatron_v1_sensor.hpp"

namespace protomotions::deployment::metatron_v1 {
namespace {
constexpr float kPi = 3.14159265358979323846F;
constexpr std::array<float, 5> kHorizons = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F};
constexpr int kTorso = 16, kRightFoot = 13, kLeftFoot = 7;
constexpr int kLeftHand = 24, kRightHand = 32;

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
                     const std::vector<std::string>& body_names)
    : fk_(fk), rest_(std::move(rest)),
      goals_(rest_.positions, rest_.rotations, rest_.style),
      recovery_(recovery_plan, body_names) {}

void MetatronTask::reset() noexcept {
  recovery_.reset();
  mode_ = Mode::kIdle;
  target_heading_ = 0.0F;
  velocity_world_ = {};
  gait_phase_ = 0.0F;
  hold_heading_ = 0.0F;
  heading_ready_ = false;
  support_ready_ = false;
  support_heading_ = 0.0F;
}

void MetatronTask::set_idle() noexcept {
  if (mode_ != Mode::kIdle) {
    mode_ = Mode::kIdle;
    support_ready_ = false;
    heading_ready_ = false;
  }
  velocity_world_ = {};
}

void MetatronTask::set_steering(float target_heading,
                             std::array<float, 2> velocity_world) {
  if (!std::isfinite(target_heading) || !std::isfinite(velocity_world[0]) ||
      !std::isfinite(velocity_world[1]) ||
      std::hypot(velocity_world[0], velocity_world[1]) > 2.01F)
    throw std::invalid_argument("invalid Metatron steering request");
  if (mode_ != Mode::kSteering) {
    mode_ = Mode::kSteering;
    support_ready_ = false;
  }
  target_heading_ = wrap(target_heading);
  velocity_world_ = velocity_world;
}

void MetatronTask::build(Inputs& inputs, const RobotState& state) {
  const SensorReading reading = reading_from_robot_state(state);
  const RecoveryPose measured = fk_.run(reading);
  const float heading = yaw_of(reading.anchor_rot);
  copy_measured(inputs, measured);
  const float speed = std::hypot(velocity_world_[0], velocity_world_[1]);
  const bool standing = mode_ == Mode::kIdle ||
      (speed <= 0.03F && std::abs(wrap(target_heading_ - heading)) <= kPi / 9.0F);
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
  } else {
    steering_goals(inputs, reading, measured, heading);
  }
  goals_.apply(inputs, mode_ == Mode::kIdle ? "idle" : "steering", standing);
  if (recovery_.step(inputs, measured, 0.02F)) {
    support_ready_ = false;
  }
}

void MetatronTask::steering_goals(Inputs& inputs, const SensorReading& reading,
                                const RecoveryPose& measured, float heading) {
  const float speed = std::hypot(velocity_world_[0], velocity_world_[1]);
  const bool stationary = speed <= 0.05F;
  if (stationary && !support_ready_) {
    for (int side = 0; side < 2; ++side) {
      const auto foot = pos(measured, side == 0 ? kRightFoot : kLeftFoot);
      std::copy(foot.begin(), foot.end(), support_feet_.begin() + side * 3);
    }
    support_anchor_ = pos(measured, kTorso);
    support_ready_ = true;
    support_heading_ = heading;
  }
  if (!stationary) support_ready_ = false;
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
      torso[2] = std::min(support_feet_[2], support_feet_[5]) +
                 rest_.upright_height - default_foot_floor;
    } else {
      torso = {reading.anchor_pos[0] + velocity_world_[0] * time,
               reading.anchor_pos[1] + velocity_world_[1] * time,
               ground + rest_.upright_height};
    }
    put_pos(inputs, slot, kTorso, torso);
    put_rot(inputs, slot, kTorso, yaw_q);
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
    mask[slot * 10 + 4] = static_cast<std::uint8_t>(!stationary || turn_feet);
    mask[slot * 10 + 2] = mask[slot * 10 + 4];
    mask[slot * 10 + 6] = static_cast<std::uint8_t>(stationary || slot == 4);
    mask[slot * 10 + 8] = mask[slot * 10 + 6];
    inputs.booleans(3)[slot * 2] = static_cast<std::uint8_t>(!stationary);
    inputs.booleans(3)[slot * 2 + 1] = 1;
    inputs.booleans(2)[slot * 2 + 1] = 1;
  }
  inputs.floats(4)[0] = velocity_world_[0];
  inputs.floats(4)[1] = velocity_world_[1];
  inputs.booleans(19)[0] = 1;
  gait_phase_ = stationary ? 0.0F :
      std::fmod(gait_phase_ + 2.0F * kPi * cadence * 0.02F, 2.0F * kPi);
}

}  // namespace protomotions::deployment::metatron_v1
