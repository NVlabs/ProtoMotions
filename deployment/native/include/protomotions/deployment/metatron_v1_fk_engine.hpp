// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>

#include "protomotions/deployment/metatron_v1_contract.hpp"
#include "protomotions/deployment/metatron_v1_recovery.hpp"

namespace protomotions::deployment::metatron_v1 {

// Compiled ProtoMotions FK; identical for MuJoCo and Unitree SensorReading.
// This small graph is used to construct recovery/goal geometry, never as a
// simulator-only privileged state source.
class FkEngine final {
 public:
  FkEngine(const std::string& plan_path, bool cuda_graph);
  ~FkEngine();
  FkEngine(const FkEngine&) = delete;
  FkEngine& operator=(const FkEngine&) = delete;

  RecoveryPose run(const SensorReading& reading);
  bool cuda_graph_enabled() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace protomotions::deployment::metatron_v1
