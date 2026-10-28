// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>

#include "protomotions/deployment/controller.hpp"

namespace protomotions::deployment {

// ControllerManager performs only lifecycle and selection. It does not know
// whether a controller is Metatron v1, GPC, or a human hold mode.
class ControllerManager {
 public:
  bool add(ControllerPtr controller);
  bool request_switch(const std::string& name);
  bool activate_requested(const RobotState& state);
  std::optional<ActuatorCommand> step(const RobotState& state);
  void on_command_accepted(const ActuatorCommand& command);
  void stop() noexcept;

  std::string active_name() const;
  std::string requested_name() const;

  // Requests and stepping must be serialized by the application. Keeping this
  // boundary single-threaded avoids a mutex in the real-time loop; a UI/input
  // thread should queue its request to the control thread.

 private:
  std::map<std::string, ControllerPtr, std::less<>> controllers_;
  Controller* active_ = nullptr;
  std::string requested_;
};

}  // namespace protomotions::deployment
