// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_rest.hpp"

#include <nlohmann/json.hpp>

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: metatron_v1_rest_dump FK_PLAN\n";
    return 2;
  }
  try {
    protomotions::deployment::metatron_v1::FkEngine fk(argv[1], true);
    const auto rest = protomotions::deployment::metatron_v1::make_rest_pose(fk);
    nlohmann::json output;
    output["positions"] = rest.positions;
    output["rotations"] = rest.rotations;
    output["style"] = rest.style;
    output["upright_height"] = rest.upright_height;
    std::cout << output.dump() << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
