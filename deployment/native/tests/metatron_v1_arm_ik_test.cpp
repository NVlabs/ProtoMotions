// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/metatron_v1_arm_ik.hpp"

#include <mujoco/mujoco.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace protomotions::deployment;
using namespace protomotions::deployment::metatron_v1;

namespace {

const std::vector<std::string> kJointNames = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"};

RobotState standing_state(const std::vector<double>& q) {
  RobotState state;
  state.dof_position = q;
  state.dof_velocity.assign(q.size(), 0.0);
  state.root_position = {0.0, 0.0, 0.793};
  state.root_orientation_xyzw = {0.0, 0.0, 0.0, 1.0};
  return state;
}

std::array<double, 3> hand_position(const char* xml, const RobotState& state, const char* body) {
  char error[512]{};
  mjModel* model = mj_loadXML(xml, nullptr, error, sizeof(error));
  assert(model);
  mjData* data = mj_makeData(model);
  for (std::size_t i = 0; i < kJointNames.size(); ++i) {
    const int joint = mj_name2id(model, mjOBJ_JOINT, kJointNames[i].c_str());
    data->qpos[model->jnt_qposadr[joint]] = state.dof_position[i];
  }
  data->qpos[2] = state.root_position[2];
  data->qpos[3] = 1.0;
  mj_kinematics(model, data);
  const int id = mj_name2id(model, mjOBJ_BODY, body);
  std::array<double, 3> out = {data->xpos[3 * id], data->xpos[3 * id + 1], data->xpos[3 * id + 2]};
  mj_deleteData(data);
  mj_deleteModel(model);
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc > 1);
  const char* xml = argv[1];
  std::array<float, kDofCount> scale{};
  std::array<float, kDofCount> offset{};
  scale.fill(0.25F);
  offset.fill(0.1F);
  ArmIkCorrection ik(xml, kJointNames, scale, offset);

  std::vector<double> q(kDofCount, 0.0);
  q[15] = q[22] = 0.2; q[16] = 0.2; q[23] = -0.2; q[18] = q[25] = 0.6;
  const RobotState state = standing_state(q);
  Outputs outputs;
  for (std::size_t j = 0; j < kDofCount; ++j) {
    outputs.raw_action[j] = static_cast<float>(q[j] - offset[j]) / scale[j];
    outputs.action[j] = offset[j] + scale[j] * outputs.raw_action[j];
  }
  const Outputs untouched = outputs;

  // 1. No goals: nothing changes.
  ik.apply(outputs, state, 0.02F);
  for (std::size_t j = 0; j < kDofCount; ++j) assert(outputs.action[j] == untouched.action[j]);

  // 2. A goal 5 cm in front of the left hand: only the left arm moves, within
  //    the per-step rate and total budget, and the affine map still holds.
  const auto left = hand_position(xml, state, "left_rubber_hand");
  HandGoal goal;
  goal.enabled = true;
  goal.position = {static_cast<float>(left[0] + 0.05), static_cast<float>(left[1]), static_cast<float>(left[2])};
  ik.set_goal(0, goal);
  Outputs stepped = untouched;
  ik.apply(stepped, state, 0.02F);
  const float rate = 0.05F * 0.02F;
  float moved = 0.0F;
  for (std::size_t j = 0; j < kDofCount; ++j) {
    const float delta = ik.delta()[j];
    const bool left_arm = j >= 15 && j <= 21;
    if (!left_arm) assert(delta == 0.0F && stepped.action[j] == untouched.action[j]);
    assert(std::abs(delta) <= rate + 1e-6F);
    moved += std::abs(delta);
    assert(std::abs(stepped.action[j] - (offset[j] + scale[j] * stepped.raw_action[j])) < 1e-5F);
  }
  assert(moved > 0.0F);
  assert(ik.diagnostics().hand_error_m[0] > 0.049F && ik.diagnostics().hand_error_m[0] < 0.051F);

  // 3. Repeated solves saturate at the offset budget, never beyond.
  for (int i = 0; i < 200; ++i) {
    stepped = untouched;
    ik.apply(stepped, state, 0.02F);
  }
  for (std::size_t j = 15; j <= 21; ++j) assert(std::abs(ik.delta()[j]) <= 0.025F + 1e-6F);

  // 4. Goal at the measured hand: the offset decays back toward zero.
  goal.position = {static_cast<float>(left[0]), static_cast<float>(left[1]), static_cast<float>(left[2])};
  ik.set_goal(0, goal);
  float before = 0.0F, after = 0.0F;
  for (std::size_t j = 15; j <= 21; ++j) before += std::abs(ik.delta()[j]);
  for (int i = 0; i < 200; ++i) {
    stepped = untouched;
    ik.apply(stepped, state, 0.02F);
  }
  for (std::size_t j = 15; j <= 21; ++j) after += std::abs(ik.delta()[j]);
  assert(after < 0.1F * before + 1e-4F);

  // 5. Disabling the goal clears the offset at once.
  goal.enabled = false;
  ik.set_goal(0, goal);
  stepped = untouched;
  ik.apply(stepped, state, 0.02F);
  for (std::size_t j = 0; j < kDofCount; ++j) assert(ik.delta()[j] == 0.0F);

  // 6. Bounds helper: beyond the soft range the increment points back inside.
  const auto bounds = correction_bounds(1.0F, 0.0F, -0.5F, 0.5F, 0.02F, 0.025F, 0.05F);
  assert(bounds[0] <= 0.0F && bounds[1] <= 0.0F);
  std::printf("arm IK: left-hand goal moved %.4f rad total in one step, budget and rate respected\n", moved);
  return 0;
}
