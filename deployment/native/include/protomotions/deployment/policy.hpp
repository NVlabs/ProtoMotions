// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "protomotions/deployment/controller.hpp"

namespace protomotions::deployment {

// CUDA graph capture is an execution option on a CUDA provider, not a second
// provider. Keeping it in PolicyArtifactConfig avoids two names for one path.
enum class PolicyBackend { kCallback, kCpu, kOnnxRuntime, kTensorRt };

inline std::optional<PolicyBackend> policy_backend_from_name(
    std::string_view name) {
  if (name == "callback") return PolicyBackend::kCallback;
  if (name == "cpu") return PolicyBackend::kCpu;
  if (name == "onnxruntime") return PolicyBackend::kOnnxRuntime;
  if (name == "tensorrt") return PolicyBackend::kTensorRt;
  return std::nullopt;
}

struct PolicyArtifactConfig {
  std::string artifact_path;
  std::string input_contract;
  PolicyBackend backend = PolicyBackend::kCallback;
  bool capture_cuda_graph = false;
  bool requires_odometry = false;
  bool requires_projected_gravity = false;
  bool requires_vision = false;
  bool requires_lidar = false;
};

// A backend adapter owns artifact loading and execution. The deployment core
// only calls this contract; it never imports ONNX Runtime, TensorRT, or CUDA.
class PolicyRuntime {
 public:
  virtual ~PolicyRuntime() = default;
  virtual bool load(const PolicyArtifactConfig& config) = 0;
  virtual bool loaded() const noexcept = 0;
  virtual ActuatorCommand step(const RobotState& state) = 0;
  virtual void stop() noexcept = 0;
};

using PolicyRuntimePtr = std::unique_ptr<PolicyRuntime>;

// Generated ProtoMotions adapters can use this controller for any backend.
// The adapter remains responsible for FK, obs processing, normalization,
// inference, and action postprocessing.
class ArtifactController final : public Controller {
 public:
  ArtifactController(std::string name, PolicyArtifactConfig config,
                     PolicyRuntimePtr runtime)
      : name_(std::move(name)),
        config_(std::move(config)),
        runtime_(std::move(runtime)) {}

  std::string_view name() const noexcept override { return name_; }
  bool start(const RobotState& state) override {
    if (!runtime_ || (config_.requires_odometry && !state.has_odometry) ||
        (config_.requires_projected_gravity && !state.has_projected_gravity) ||
        (config_.requires_vision && !state.has_vision) ||
        (config_.requires_lidar && !state.has_lidar)) {
      return false;
    }
    // Startup/provider errors stay visible. Runtime safety handles failures
    // during an active cycle; swallowing a load exception here would hide a
    // bad artifact or provider configuration.
    if (runtime_->load(config_) && runtime_->loaded()) return true;
    runtime_->stop();
    return false;
  }
  ActuatorCommand step(const RobotState& state) override {
    return runtime_->step(state);
  }
  void stop() noexcept override {
    if (runtime_) runtime_->stop();
  }

 private:
  std::string name_;
  PolicyArtifactConfig config_;
  PolicyRuntimePtr runtime_;
};

}  // namespace protomotions::deployment
