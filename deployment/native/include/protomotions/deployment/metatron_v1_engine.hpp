// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <memory>
#include <string>

#include "protomotions/deployment/metatron_v1_contract.hpp"

namespace protomotions::deployment::metatron_v1 {

struct Outputs {
  std::array<float, kDofCount> action{};
  std::array<float, kDofCount> stiffness{};
  std::array<float, kDofCount> damping{};
  std::array<float, kDofCount> raw_action{};
};

// A single inference implementation for both MuJoCo and SDK2 adapters.
// This class knows nothing about robot transport, control modes, or MuJoCo.
class Engine final {
 public:
  Engine(const std::string& plan_path, bool cuda_graph);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;
  Outputs run(const Inputs& inputs);
  void warmup(int iterations);
  bool cuda_graph_enabled() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace protomotions::deployment::metatron_v1
