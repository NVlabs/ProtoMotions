// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_fk_engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace protomotions::deployment::metatron_v1;

namespace {
template <class Container>
void read(std::ifstream& file, Container& values) {
  file.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(values[0])));
  if (!file) throw std::runtime_error("truncated FK fixture");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4)
      throw std::invalid_argument("usage: metatron_v1_fk_parity FK_PLAN FIXTURE [--cuda-graphs]");
    const bool graphs = argc == 4;
    if (graphs && std::string(argv[3]) != "--cuda-graphs")
      throw std::invalid_argument("unknown FK parity option");
    FkEngine engine(argv[1], graphs);
    if (graphs && !engine.cuda_graph_enabled())
      throw std::runtime_error("FK CUDA graph was requested but not captured");
    std::ifstream file(argv[2], std::ios::binary);
    if (!file) throw std::runtime_error("cannot open FK fixture");
    std::array<char, 12> magic{};
    read(file, magic);
    if (std::string(magic.data()) != "PM_FK_V2")
      throw std::runtime_error("invalid FK fixture magic (expected PM_FK_V2: joints, anchor pose)");
    std::uint32_t samples = 0;
    file.read(reinterpret_cast<char*>(&samples), sizeof(samples));
    if (!file || samples < 1 || samples > 1000)
      throw std::runtime_error("invalid FK fixture count");
    float max_pos = 0.0F, max_rot = 0.0F;
    double time_us = 0.0;
    for (std::uint32_t sample = 0; sample < samples; ++sample) {
      SensorReading reading;
      RecoveryPose reference;
      read(file, reading.dof_pos);
      read(file, reading.anchor_pos);
      read(file, reading.anchor_rot);
      read(file, reference.positions);
      read(file, reference.rotations);
      const auto before = std::chrono::steady_clock::now();
      const RecoveryPose actual = engine.run(reading);
      time_us += std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - before).count();
      for (std::size_t axis = 0; axis < actual.positions.size(); ++axis)
        max_pos = std::max(max_pos, std::abs(actual.positions[axis] - reference.positions[axis]));
      for (std::size_t body = 0; body < 33; ++body) {
        float same = 0.0F, flipped = 0.0F;
        for (std::size_t axis = 0; axis < 4; ++axis) {
          const std::size_t index = body * 4 + axis;
          same = std::max(same, std::abs(actual.rotations[index] - reference.rotations[index]));
          flipped = std::max(flipped, std::abs(actual.rotations[index] + reference.rotations[index]));
        }
        max_rot = std::max(max_rot, std::min(same, flipped));
      }
    }
    std::cout << "FK parity samples=" << samples << " graph=" << engine.cuda_graph_enabled()
              << " max_pos_m=" << max_pos << " max_quat_component=" << max_rot
              << " mean_us=" << time_us / samples << '\n';
    return max_pos <= 1e-4F && max_rot <= 1e-4F ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "FK parity failed: " << error.what() << '\n';
    return 1;
  }
}
