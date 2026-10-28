// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_engine.hpp"
#include "protomotions/deployment/metatron_v1_manifest.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace protomotions::deployment::metatron_v1;

namespace {
void read_exact(std::ifstream& input, void* pointer, std::size_t bytes) {
  input.read(static_cast<char*>(pointer), static_cast<std::streamsize>(bytes));
  if (!input) throw std::runtime_error("truncated native parity fixture");
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 5)
      throw std::invalid_argument("usage: metatron_v1_parity PLAN FIXTURE MANIFEST [--cuda-graphs]");
    const bool graphs = argc == 5;
    if (graphs && std::string(argv[4]) != "--cuda-graphs") throw std::invalid_argument("unknown parity option");
    const Manifest manifest = Manifest::load(argv[3]);
    Engine engine(argv[1], graphs);
    std::ifstream file(argv[2], std::ios::binary);
    if (!file) throw std::runtime_error("cannot open native parity fixture");
    std::array<char, 12> magic{};
    read_exact(file, magic.data(), magic.size());
    if (std::string(magic.data()) != "PM_SCENE_V1")
      throw std::runtime_error("invalid native parity fixture header");
    std::uint32_t samples = 0;
    read_exact(file, &samples, sizeof(samples));
    if (samples < 1 || samples > 1000) throw std::runtime_error("invalid fixture sample count");
    Inputs inputs;
    std::array<float, kOutputCount> maximum{};
    float affine_maximum = 0.0F;
    for (std::uint32_t sample = 0; sample < samples; ++sample) {
      for (std::size_t index = 0; index < kInputCount; ++index) {
        const auto& spec = kInputs[index];
        if (spec.is_bool) {
          auto& values = inputs.booleans(index);
          read_exact(file, values.data(), values.size());
        } else {
          auto& values = inputs.floats(index);
          read_exact(file, values.data(), values.size() * sizeof(float));
        }
      }
      std::array<std::array<float, kDofCount>, kOutputCount> reference{};
      for (auto& values : reference)
        read_exact(file, values.data(), values.size() * sizeof(float));
      const Outputs result = engine.run(inputs);
      for (std::size_t joint = 0; joint < kDofCount; ++joint) {
        affine_maximum = std::max(
            affine_maximum,
            std::abs(result.action[joint] - (manifest.offset[joint] +
                 manifest.scale[joint] * result.raw_action[joint])));
      }
      const std::array<const std::array<float, kDofCount>*, kOutputCount> actual = {
          &result.action, &result.stiffness, &result.damping, &result.raw_action};
      for (std::size_t output = 0; output < kOutputCount; ++output) {
        for (std::size_t joint = 0; joint < kDofCount; ++joint) {
          const float error = std::abs(reference[output][joint] - (*actual[output])[joint]);
          if (!std::isfinite(error)) throw std::runtime_error("non-finite parity error");
          maximum[output] = std::max(maximum[output], error);
        }
      }
    }
    std::cout << "native_parity samples=" << samples
              << " graph=" << (engine.cuda_graph_enabled() ? "true" : "false")
              << " action_max_abs=" << maximum[0]
              << " stiffness_max_abs=" << maximum[1]
              << " damping_max_abs=" << maximum[2]
              << " raw_action_max_abs=" << maximum[3]
              << " action_affine_max_abs=" << affine_maximum << '\n';
    // On the 18 recorded Python feeds, the graph-safe TRT plan's worst raw
    // difference is 0.00500152; processed motor-target error is 0.00175381 rad.
    // Give only the normalized raw output the measured FP32 allowance.
    if (maximum[0] > 0.002F || maximum[1] > 0.00001F ||
        maximum[2] > 0.00001F || maximum[3] > 0.006F ||
        affine_maximum > 0.0001F) return 2;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native parity failed: " << error.what() << '\n';
    return 1;
  }
}
