// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_task.hpp"
#include "protomotions/deployment/metatron_v1_sensor.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace protomotions::deployment;
using namespace protomotions::deployment::metatron_v1;

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: metatron_v1_task_dump FK_PLAN RECOVERY_PLAN MANIFEST MODE HEADING VX,VY\n";
    return 2;
  }
  try {
    const auto manifest = nlohmann::json::parse(std::ifstream(argv[3]));
    const auto body_names = manifest.at("robot_contract").at("body_names").get<std::vector<std::string>>();
    FkEngine fk(argv[1], true);
    RestPose rest = make_rest_pose(fk);
    MetatronTask task(fk, rest, argv[2], body_names);
    RobotState state;
    state.timestamp = std::chrono::steady_clock::now();
    state.sequence = 1;
    state.dof_position.assign(29, 0.0);
    state.dof_velocity.assign(29, 0.0);
    state.dof_position[0] = state.dof_position[6] = -0.312;
    state.dof_position[3] = state.dof_position[9] = 0.669;
    state.dof_position[4] = state.dof_position[10] = -0.363;
    state.dof_position[16] = 0.2;
    state.dof_position[23] = -0.2;
    state.dof_position[18] = state.dof_position[25] = 1.4;
    state.root_position = {0.0, 0.0, 0.75};
    state.base_position = state.root_position;
    // The standing torso over that pelvis (waist joints zero): the FK takes
    // the anchor pose, so it is set before anything runs it.
    for (int axis = 0; axis < 3; ++axis)
      state.anchor_position[axis] =
          state.root_position[axis] + static_cast<double>(kTorsoOffsetInPelvis[axis]);
    state.has_anchor_position = true;
    state.has_root_kinematics = true;
    // The measured pose the parity tool reads back, from the anchored FK.
    const RecoveryPose pose = fk.run(reading_from_robot_state(state));
    if (std::string(argv[4]) == "steering") {
      std::string velocity(argv[6]);
      const auto comma = velocity.find(',');
      if (comma == std::string::npos) throw std::invalid_argument("VX,VY required");
      task.set_steering(std::stof(argv[5]),
                        {std::stof(velocity.substr(0, comma)),
                         std::stof(velocity.substr(comma + 1))});
    } else if (std::string(argv[4]) != "idle") {
      throw std::invalid_argument("mode must be idle or steering");
    }
    Inputs inputs;
    inputs.prepare(reading_from_robot_state(state));
    task.build(inputs, state);
    nlohmann::json result;
    for (std::size_t i = 2; i < 20; ++i) {
      if (kInputs[i].is_bool) result[kInputs[i].name] = inputs.booleans(i);
      else result[kInputs[i].name] = inputs.floats(i);
    }
    result["measured_positions"] = pose.positions;
    result["measured_rotations"] = pose.rotations;
    result["anchor_position"] = state.anchor_position;
    result["recovery_phase"] = static_cast<int>(task.recovery_phase());
    std::cout << result.dump() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
