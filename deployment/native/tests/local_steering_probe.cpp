// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <iomanip>
#include <iostream>
#include <string>
#include "protomotions/deployment/local_steering_handrail.hpp"

int main(int argc, char** argv) {
  using namespace protomotions::deployment;
  std::cout << std::setprecision(9);
  if (argc != 2) return 2;
  if (std::string(argv[1]) == "hands") {
    LocalSteeringPicoHands hands;
    hands.latch({}, 0.7F);
    for (int i = 0; i < 80; ++i) {
      const float t = static_cast<float>(i) * 0.02F;
      LocalSteeringPicoHands::Hands world{{{t, -t, 2*t}, {-t, 0.5F*t, -t}}};
      const auto cmd = hands.update(world, {i < 30 || i >= 50 ? 1.0F : 0.0F,
                                            i >= 10 && i < 60 ? 1.0F : 0.0F}, 0.02F);
      for (float v : cmd) std::cout << v << ' ';
      std::cout << '\n';
    }
  } else if (std::string(argv[1]) == "targets") {
    const std::array<std::array<float, 3>, 5> relative{{{0,0,0}, {0,-.1F,-.7F},
        {0,.1F,-.7F}, {.05F,.2F,-.1F}, {.05F,-.2F,-.1F}}};
    const std::array<std::array<float, 2>, 4> velocities{{{0,0}, {1,0}, {0,1}, {-.7F,-.4F}}};
    for (int scenario = 0; scenario < 4; ++scenario) {
      for (int slot = 1; slot <= 5; ++slot) {
        const auto frame = local_steering_frame(relative, {.1F,-.2F,.73F}, 2.9F,
            -2.5F + static_cast<float>(scenario), velocities[scenario], {.1F,0,.2F,0,-.1F,-.05F},
            .4F, static_cast<float>(scenario)/3.0F, .1F*static_cast<float>(slot));
        for (const auto& position : frame.positions) for (float v : position) std::cout << v << ' ';
        std::cout << frame.yaw << '\n';
      }
    }
  } else return 2;
}
