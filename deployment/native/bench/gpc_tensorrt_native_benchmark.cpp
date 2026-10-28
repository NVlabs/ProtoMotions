// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Native fixed-buffer TensorRT benchmark for the unconditional GPC prior.
// Input construction and external sampling-noise generation are intentionally
// outside the timed policy boundary.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "protomotions/deployment/gpc_policy.hpp"

namespace {

using protomotions::deployment::GpcSamplingMode;
using protomotions::deployment::NativeGpcPolicy;
using protomotions::deployment::kNativeGpcNoiseDim;

struct Statistics {
  std::vector<double> values_ms;

  void print(const char* label) {
    if (values_ms.empty()) throw std::runtime_error("no timing samples");
    std::sort(values_ms.begin(), values_ms.end());
    const auto percentile = [&](double p) {
      const auto index = static_cast<std::size_t>(p * (values_ms.size() - 1));
      return values_ms[index];
    };
    const double mean = std::accumulate(values_ms.begin(), values_ms.end(), 0.0) /
                        static_cast<double>(values_ms.size());
    std::cout << label << ": samples=" << values_ms.size()
              << " mean=" << std::fixed << std::setprecision(6) << mean << "ms"
              << " p50=" << percentile(0.50) << "ms"
              << " p99=" << percentile(0.99) << "ms"
              << " min=" << percentile(0.00) << "ms"
              << " max=" << percentile(1.00) << "ms\n";
  }
};

GpcSamplingMode parse_mode(const std::string& value) {
  if (value == "argmax") return GpcSamplingMode::kArgmax;
  if (value == "gumbel") return GpcSamplingMode::kGumbel;
  if (value == "topp") return GpcSamplingMode::kTopP;
  throw std::runtime_error("--mode must be argmax, gumbel, or topp");
}

const char* mode_name(GpcSamplingMode mode) {
  switch (mode) {
    case GpcSamplingMode::kArgmax: return "argmax";
    case GpcSamplingMode::kGumbel: return "gumbel";
    case GpcSamplingMode::kTopP: return "topp";
  }
  return "unknown";
}

void print_usage() {
  std::cerr << "usage: gpc_tensorrt_native_benchmark ENGINE.plan [samples=1000] "
               "--mode argmax|gumbel|topp [--cuda-graphs|--no-cuda-graphs] "
               "[--plugin-library FILE] [--warmup N]\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage();
    return 2;
  }
  try {
    const std::string engine_path = argv[1];
    int samples = 1000;
    int warmup = 20;
    std::string mode_value;
    std::string plugin_library;
    bool use_cuda_graph = true;
    for (int i = 2; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto next = [&](const char* option) {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + option);
        return std::string(argv[++i]);
      };
      if (argument == "--mode") mode_value = next("--mode");
      else if (argument == "--plugin-library") plugin_library = next("--plugin-library");
      else if (argument == "--cuda-graphs") use_cuda_graph = true;
      else if (argument == "--no-cuda-graphs") use_cuda_graph = false;
      else if (argument == "--warmup") warmup = std::stoi(next("--warmup"));
      else if (!argument.empty() && argument[0] != '-') samples = std::stoi(argument);
      else throw std::runtime_error("unknown option: " + argument);
    }
    if (mode_value.empty()) throw std::runtime_error("--mode is required");
    if (samples < 1 || warmup < 1) throw std::runtime_error("samples and warmup must be positive");
    const auto mode = parse_mode(mode_value);

    NativeGpcPolicy policy(engine_path, mode, use_cuda_graph, plugin_library);
    policy.warmup(warmup);

    std::array<float, 64> observation{};
    std::array<float, 29> previous_action{};
    std::vector<float> noise(kNativeGpcNoiseDim, 0.0F);
    std::mt19937 rng(184250U);
    std::uniform_real_distribution<float> uniform(1.0e-5F, 1.0F - 1.0e-5F);
    if (mode != GpcSamplingMode::kArgmax) {
      for (float& value : noise) value = -std::log(-std::log(uniform(rng)));
    }
    std::array<float, 29> action{};
    Statistics statistics;
    statistics.values_ms.reserve(static_cast<std::size_t>(samples));
    bool finite = true;
    for (int sample = 0; sample < samples; ++sample) {
      // Keep host-side input construction and noise generation out of the
      // policy timing. This models the fixed buffers supplied by the sensor
      // contract and the external sampler.
      policy.run(observation, previous_action,
                 mode == GpcSamplingMode::kArgmax ? nullptr : noise.data(), action);
      statistics.values_ms.push_back(static_cast<double>(policy.last_step_ns()) / 1.0e6);
      for (float value : action) finite = finite && std::isfinite(value);
    }

    std::cout << "engine=" << engine_path
              << " mode=" << mode_name(mode)
              << " cuda_graphs=" << (use_cuda_graph ? "true" : "false")
              << " finite=" << (finite ? "true" : "false") << "\n";
    statistics.print("native_policy_boundary");
    return finite ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
