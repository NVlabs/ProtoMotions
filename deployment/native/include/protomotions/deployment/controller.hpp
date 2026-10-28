// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "protomotions/deployment/types.hpp"

namespace protomotions::deployment {

// A controller owns intent, not robot I/O. A generated ProtoMotions policy
// adapter may implement this interface and keep FK, obs processing,
// normalization, inference, and action postprocessing inside the artifact.
class Controller {
 public:
  virtual ~Controller() = default;
  virtual std::string_view name() const noexcept = 0;
  virtual bool start(const RobotState& state) = 0;
  virtual ActuatorCommand step(const RobotState& state) = 0;
  // Only a newly accepted backend command advances a controller's action
  // history. Bounded repeated-state holds never call this hook.
  virtual void on_command_accepted(const ActuatorCommand&) {}
  virtual void stop() noexcept = 0;
};

using ControllerPtr = std::unique_ptr<Controller>;
using PolicyStep = std::function<ActuatorCommand(const RobotState&)>;

// Thin adapter for generated policy artifacts. The callback is where
// ProtoMotions-owned preprocessing and postprocessing are executed.
class CallbackController final : public Controller {
 public:
  CallbackController(std::string name, PolicyStep step)
      : name_(std::move(name)), step_(std::move(step)) {}

  std::string_view name() const noexcept override { return name_; }
  bool start(const RobotState&) override { return static_cast<bool>(step_); }
  ActuatorCommand step(const RobotState& state) override {
    return step_(state);
  }
  void stop() noexcept override {}

 private:
  std::string name_;
  PolicyStep step_;
};

}  // namespace protomotions::deployment
