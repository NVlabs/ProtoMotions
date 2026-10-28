// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <mujoco/mujoco.h>

#include <iostream>
#include <string>

#include "protomotions/deployment/mujoco_model_loader.hpp"

int main(int argc, char** argv) {
  if (argc != 2) return 2;
  try {
    std::string note;
    mjModel* model =
        protomotions::deployment::load_kinematics_model(argv[1], &note);
    if (!model || model->nbody < 2 || note.empty()) return 3;
    mj_deleteModel(model);
    std::cout << note << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 4;
  }
}
