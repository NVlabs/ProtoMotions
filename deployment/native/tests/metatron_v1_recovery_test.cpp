// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_recovery.hpp"

#include <nlohmann/json.hpp>

#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

using namespace protomotions::deployment::metatron_v1;

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::ifstream file(argv[1]);
  const auto plan = nlohmann::json::parse(file);
  StagedRecovery recovery(argv[1],
      plan.at("body_names").get<std::vector<std::string>>());
  RecoveryPose pose;
  for (int body = 0; body < 33; ++body) pose.rotations[body * 4 + 3] = 1.0F;
  pose.positions[16 * 3 + 2] = 0.2F;
  pose.rotations[16 * 4 + 1] = std::sin(3.14159265F / 4.0F);
  pose.rotations[16 * 4 + 3] = std::cos(3.14159265F / 4.0F);
  Inputs inputs;
  assert(recovery.step(inputs, pose, 0.02F));
  assert(recovery.phase() == StagedRecovery::Phase::kPronePlan);
  assert(inputs.booleans(7)[0] == 1);
  assert(inputs.booleans(7)[1] == 1);
  assert(inputs.booleans(16)[0] == 0);
  assert(inputs.booleans(37)[0] == 0);
  assert(inputs.floats(9)[0] == 0.2F);
  assert(std::isfinite(inputs.floats(5)[0]));

  recovery.reset();
  inputs.reset();
  pose.rotations[16 * 4 + 1] = 0.0F;
  pose.rotations[16 * 4 + 3] = 1.0F;
  pose.joints[3] = pose.joints[9] = 1.0F;
  assert(recovery.step(inputs, pose, 0.02F));
  assert(recovery.phase() == StagedRecovery::Phase::kSeatedPlan);

  recovery.reset();
  inputs.reset();
  pose.joints[3] = pose.joints[9] = 0.0F;
  pose.positions[16 * 3 + 2] = 0.4F;
  assert(recovery.step(inputs, pose, 0.02F));
  assert(recovery.phase() == StagedRecovery::Phase::kRamp);
  assert(inputs.booleans(7)[0] == 1);
  assert(inputs.booleans(7)[1] == 0);
  assert(inputs.floats(5)[16 * 3 + 2] > 0.4F);
  return 0;
}
