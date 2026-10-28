// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using protomotions::deployment::metatron_v1::Engine;
using protomotions::deployment::metatron_v1::Inputs;
using protomotions::deployment::metatron_v1::SensorReading;

int main(int argc, char** argv) {
  try {
    std::string plan;
    int iterations = 100;
    bool graph = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--plan" && i + 1 < argc) plan = argv[++i];
      else if (arg == "--iterations" && i + 1 < argc) iterations = std::stoi(argv[++i]);
      else if (arg == "--cuda-graphs") graph = true;
      else throw std::invalid_argument("usage: metatron_v1_inference_benchmark --plan FILE [--iterations N] [--cuda-graphs]");
    }
    if (plan.empty() || iterations < 1) throw std::invalid_argument("plan and positive iterations required");
    Inputs inputs;
    SensorReading reading;
    reading.root_pos[2] = 0.79F;
    reading.anchor_pos[2] = 0.79F;
    inputs.prepare(reading);
    for (int frame = 0; frame < 5 * 33; ++frame) {
      inputs.floats(6)[static_cast<std::size_t>(frame) * 4 + 3] = 1.0F;
      inputs.floats(11)[static_cast<std::size_t>(frame) * 4 + 3] = 1.0F;
    }
    inputs.booleans(19)[0] = 1;
    Engine engine(plan, graph);
    engine.warmup(10);
    std::vector<double> milliseconds;
    milliseconds.reserve(static_cast<std::size_t>(iterations));
    for (int i = 0; i < iterations; ++i) {
      const auto begin = std::chrono::steady_clock::now();
      const auto output = engine.run(inputs);
      const auto end = std::chrono::steady_clock::now();
      if (!std::isfinite(output.action[0])) throw std::runtime_error("non-finite action");
      milliseconds.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    const auto percentile = [&](double fraction) {
      const auto index = static_cast<std::size_t>(fraction * (milliseconds.size() - 1));
      return milliseconds[index];
    };
    std::cout << "native_metatron_inference iterations=" << iterations
              << " cuda_graph=" << (engine.cuda_graph_enabled() ? "true" : "false")
              << " p50_ms=" << percentile(0.50) << " p95_ms=" << percentile(0.95)
              << " max_ms=" << milliseconds.back() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "native Metatron benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
