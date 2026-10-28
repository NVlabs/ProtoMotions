// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/steering_style.hpp"
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  using namespace protomotions::deployment;
  if (argc != 1 && argc != 4) return 2;
  const float heading = argc == 4 ? std::stof(argv[1]) : 0.0F;
  const std::array<float, 2> world = {
      argc == 4 ? std::stof(argv[2]) : 0.0F,
      argc == 4 ? std::stof(argv[3]) : 0.0F};
  std::array<std::uint8_t, 50> bodies;
  std::array<std::uint8_t, 5> poses;
  std::array<std::uint8_t, 10> translation, rotation;
  std::array<float, 5> times = {0.1F, 0.2F, 0.3F, 0.4F, 0.5F};
  bodies.fill(1); poses.fill(1); translation.fill(1); rotation.fill(1);
  // Exercise both preserved XY and fully hidden future slots.
  translation[2] = 0;
  poses[3] = 0;
  crawling_pose_masks(bodies, poses, translation, rotation, times);
  const auto print = [](const auto& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i) std::cout << ',';
      std::cout << +values[i];
    }
    std::cout << ']';
  };
  std::cout << std::setprecision(9) << "{\"pose\":";
  print(crawling_style_preset().pose);
  std::cout << ",\"velocity\":"; print(crawling_style_velocity(heading, world, true));
  std::cout << ",\"bodies\":"; print(bodies);
  std::cout << ",\"poses\":"; print(poses);
  std::cout << ",\"translation\":"; print(translation);
  std::cout << ",\"rotation\":"; print(rotation);
  std::cout << ",\"times\":"; print(times);
  std::cout << "}\n";
}
