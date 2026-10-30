// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>

#include "protomotions/deployment/metatron_v1_contract.hpp"
#include "protomotions/deployment/metatron_v1_recovery.hpp"

namespace protomotions::deployment::metatron_v1 {

// The policy's own FK kernel, exported on its own as fk.onnx
// (deployment/export_fk_to_onnx.py) and compiled to a TensorRT plan: joint
// angles plus the torso anchor pose in, world-frame body positions and xyzw
// rotations out.
// Identical for MuJoCo and Unitree SensorReading. It constructs recovery/goal
// geometry, never a simulator-only privileged state source, and it is the very
// FK the policy runs, so the host's body poses and the policy's agree.
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
