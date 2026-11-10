// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_tracking.hpp"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace protomotions::deployment;
using namespace protomotions::deployment::metatron_v1;

namespace {

constexpr float kPi = 3.14159265F;

// A 33-body bundle: the reference walks along +x at 1 m/s for two seconds
// (61 frames at 30 Hz) with the torso at z = 0.8 and the feet at z = 0.05;
// the left hand (24) sits 0.3 m ahead of the torso and 0.2 m to its left.
nlohmann::json walking_bundle() {
  const int frames = 61;
  nlohmann::json positions = nlohmann::json::array(), rotations = nlohmann::json::array();
  for (int f = 0; f < frames; ++f) {
    const float x = f / 30.0F;
    for (int b = 0; b < 33; ++b) {
      float px = x, py = 0.0F, pz = 0.8F;
      if (b == 7 || b == 13) pz = 0.05F;
      if (b == 24) { px += 0.3F; py += 0.2F; pz = 1.0F; }
      if (b == 32) { px += 0.3F; py -= 0.2F; pz = 1.0F; }
      positions.insert(positions.end(), {px, py, pz});
      rotations.insert(rotations.end(), {0.0F, 0.0F, 0.0F, 1.0F});  // facing +x
    }
  }
  nlohmann::json motion = {{"frameCount", frames}, {"frameDtSeconds", 1.0F / 30.0F},
                           {"positions", positions}, {"rotations", rotations}};
  return {{"version", 1}, {"bodyCount", 33}, {"anchorBodyIndex", 16},
          {"conditionableBodyIds", {16, 13, 7, 24, 32}}, {"motions", {{"walk", motion}}}};
}

// The robot stands at (5, 5), 0.1 m higher floor, facing +y (yaw 90 degrees).
RecoveryPose robot_pose() {
  RecoveryPose pose;
  for (int b = 0; b < 33; ++b) {
    pose.positions[b * 3] = 5.0F;
    pose.positions[b * 3 + 1] = 5.0F;
    pose.positions[b * 3 + 2] = b == 7 || b == 13 ? 0.15F : 0.9F;
    pose.rotations[b * 4 + 2] = std::sin(0.25F * kPi);
    pose.rotations[b * 4 + 3] = std::cos(0.25F * kPi);
  }
  return pose;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path path(argv[1]);
  std::ofstream(path) << walking_bundle();
  MotionLibrary library(path.string());
  TrackingPlayer player(library);
  assert(!player.active());

  RecoveryPose robot = robot_pose();
  player.start(0, robot);
  assert(player.active() && std::abs(player.duration_seconds() - 2.0F) < 1e-4F);

  // 1. At the start the reference torso sits exactly on the robot's torso xy,
  //    the floor is lifted by the robot's floor offset, and the heading is the
  //    robot's: the left hand, 0.3 m ahead and 0.2 m left in the reference, is
  //    now 0.3 m along +y and 0.2 m along -x of the robot.
  std::vector<float> pos(33 * 3), rot(33 * 4);
  player.reference(0.0F, robot, pos, rot);
  assert(std::abs(pos[16 * 3] - 5.0F) < 1e-5F && std::abs(pos[16 * 3 + 1] - 5.0F) < 1e-5F);
  assert(std::abs(pos[16 * 3 + 2] - 0.9F) < 1e-5F);          // 0.8 + floor offset 0.1
  assert(std::abs(pos[24 * 3] - (5.0F - 0.2F)) < 1e-4F && std::abs(pos[24 * 3 + 1] - (5.0F + 0.3F)) < 1e-4F);
  assert(std::abs(rot[16 * 4 + 2] - std::sin(0.25F * kPi)) < 1e-4F);  // reference rotated to the robot's yaw

  // 2. A future slot keeps its offset from the current reference torso: half a
  //    second ahead the reference has walked 0.5 m, which in the robot's frame
  //    is 0.5 m along +y, wherever the robot actually is.
  player.reference(0.5F, robot, pos, rot);
  assert(std::abs(pos[16 * 3] - 5.0F) < 1e-4F && std::abs(pos[16 * 3 + 1] - 5.5F) < 1e-4F);

  // 3. Re-anchoring is per cycle: move the robot and advance time; the current
  //    reference torso follows the robot, the robot does not have to be where
  //    the reference was.
  player.advance(1.0F);
  for (int b = 0; b < 33; ++b) { robot.positions[b * 3] = 2.0F; robot.positions[b * 3 + 1] = -3.0F; }
  player.reference(player.elapsed_seconds(), robot, pos, rot);
  assert(std::abs(pos[16 * 3] - 2.0F) < 1e-4F && std::abs(pos[16 * 3 + 1] + 3.0F) < 1e-4F);
  assert(std::abs(pos[24 * 3] - (2.0F - 0.2F)) < 1e-4F && std::abs(pos[24 * 3 + 1] - (-3.0F + 0.3F)) < 1e-4F);

  // 4. Inputs: five slots, conditionable bodies masked in position and rotation,
  //    the velocity command is the reference's 1 m/s turned into the robot's +y.
  Inputs inputs;
  player.write_inputs(inputs, robot);
  for (int slot = 0; slot < 5; ++slot) {
    assert(inputs.booleans(17)[slot] == 1);
    assert(std::abs(inputs.floats(18)[slot] - TrackingPlayer::kHorizons[slot]) < 1e-6F);
    for (int c = 0; c < 5; ++c) assert(inputs.booleans(16)[slot * 10 + c * 2] == 1 && inputs.booleans(16)[slot * 10 + c * 2 + 1] == 1);
    assert(inputs.booleans(3)[slot * 2] == 1 && inputs.booleans(2)[slot * 2] == 1 && inputs.booleans(2)[slot * 2 + 1] == 0);
  }
  assert(inputs.booleans(19)[0] == 1);
  assert(std::abs(inputs.floats(4)[0]) < 1e-3F && std::abs(inputs.floats(4)[1] - 1.0F) < 1e-3F);
  assert(std::abs(inputs.floats(10)[(1 * 33 + 16) * 3 + 1] - (-3.0F + 0.04F)) < 1e-3F);  // slot 1: 40 ms ahead

  // 5. Metrics: the anchor xy error is zero by construction; the height and the
  //    hands are measured honestly. The robot's hand is at its torso, 0.36 m from
  //    the reference hand.
  const auto m = player.metrics(robot);
  assert(std::abs(m.anchor_height_error_m - 0.0F) < 1e-4F);
  assert(m.mean_body_error_m > 0.1F && m.mean_body_error_m < 0.3F);
  assert(m.mean_rotation_error_deg < 1e-2F);

  // 6. Completion and stop.
  player.advance(5.0F);
  assert(player.complete());
  player.stop();
  assert(!player.active() && player.duration_seconds() == 0.0F);
  std::filesystem::remove(path);
  return 0;
}
