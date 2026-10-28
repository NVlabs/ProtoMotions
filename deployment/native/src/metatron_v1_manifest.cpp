// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_manifest.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace protomotions::deployment::metatron_v1 {
namespace {
constexpr std::array<std::string_view, kDofCount> kJointOrder = {
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint"};

void require(bool condition, const char* reason) {
  if (!condition) throw std::invalid_argument(reason);
}
}  // namespace

Manifest Manifest::load(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("cannot open scene manifest: " + path);
  const auto json = nlohmann::json::parse(file);
  require(json.at("contract").get<std::string>() == kContract,
          "wrong scene policy contract");
  require(json.at("previous_action_domain") == "raw_policy_action",
          "scene policy requires raw incoming-action history");
  require(json.at("pd_target_ema_alpha").get<double>() == 1.0,
          "native scene policy does not support a hidden PD action EMA");

  const auto& names = json.at("input_names");
  const auto& shapes = json.at("input_shapes");
  const auto& types = json.at("input_dtypes");
  const auto& keys = json.at("input_keys");
  require(names.size() == kInputCount && shapes.size() == kInputCount &&
              types.size() == kInputCount && keys.size() == kInputCount,
          "scene manifest must describe 39 inputs");
  for (std::size_t input = 0; input < kInputCount; ++input) {
    const auto& spec = kInputs[input];
    require(names[input].get<std::string>() == spec.name,
            "scene manifest input name/order mismatch");
    require(types[input].get<std::string>() ==
                (spec.is_bool ? "bool" : "float32"),
            "scene manifest input dtype mismatch");
    require(shapes[input].is_array() &&
                shapes[input].size() == static_cast<std::size_t>(spec.rank - 1),
            "scene manifest input rank mismatch");
    for (int axis = 1; axis < spec.rank; ++axis) {
      require(shapes[input][axis - 1].get<int>() ==
                  spec.dimensions[static_cast<std::size_t>(axis)],
              "scene manifest input shape mismatch");
    }
  }
  const auto& outputs = json.at("output_names");
  require(outputs.size() == kOutputCount, "scene manifest output count mismatch");
  for (std::size_t index = 0; index < kOutputCount; ++index) {
    require(outputs[index].get<std::string>() == kOutputs[index],
            "scene manifest output order mismatch");
  }

  const auto& robot = json.at("robot_contract");
  require(robot.at("quaternion_order") == "xyzw" &&
              robot.at("anchor_body_index") == 16 &&
              robot.at("conditionable_body_ids") ==
                  nlohmann::json::array({16, 13, 7, 24, 32}),
          "scene manifest G1 body layout mismatch");
  const auto& joints = robot.at("joint_names");
  const auto& bodies = robot.at("body_names");
  require(joints.size() == kDofCount && bodies.size() == 33,
          "scene manifest G1 joint/body count mismatch");
  for (std::size_t index = 0; index < kDofCount; ++index) {
    require(joints[index].get<std::string>() == kJointOrder[index],
            "scene manifest G1 motor order mismatch");
  }
  require(bodies[7] == "left_ankle_roll_link" &&
              bodies[13] == "right_ankle_roll_link" &&
              bodies[16] == "torso_link" &&
              bodies[24] == "left_rubber_hand" &&
              bodies[32] == "right_rubber_hand",
          "scene manifest G1 conditionable body order mismatch");

  const auto& action = json.at("action_map");
  require(action.at("kind") == "affine", "scene action map must be affine");
  const auto& scales = action.at("scale");
  const auto& offsets = action.at("offset");
  require(scales.size() == kDofCount && offsets.size() == kDofCount,
          "scene action map needs 29 scales and offsets");
  Manifest manifest;
  for (std::size_t joint = 0; joint < kDofCount; ++joint) {
    manifest.scale[joint] = scales[joint].get<float>();
    manifest.offset[joint] = offsets[joint].get<float>();
  }
  (void)manifest.action_map();  // Checks finiteness and invertibility.
  return manifest;
}

}  // namespace protomotions::deployment::metatron_v1
