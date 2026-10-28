// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <iostream>
#include <memory>
#include <string>

#include "protomotions/deployment/input.hpp"
#include "protomotions/deployment/mujoco_backend.hpp"
#include "protomotions/deployment/runtime.hpp"

using namespace protomotions::deployment;

namespace {

ActuatorCommand idle(const RobotState&) {
  ActuatorCommand command;
  command.mode = ControlMode::kPosition;
  command.values = {0.0, 0.0};
  return command;
}

ActuatorCommand steering(const RobotState&) {
  ActuatorCommand command;
  command.mode = ControlMode::kPosition;
  command.values = {0.25, -0.25};
  return command;
}

ActuatorCommand tracking(const RobotState&) {
  ActuatorCommand command;
  command.mode = ControlMode::kPosition;
  command.values = {-0.25, 0.25};
  return command;
}

ActuatorCommand handshake_idle(const RobotState&) {
  ActuatorCommand command;
  command.mode = ControlMode::kPosition;
  command.values = {0.1, -0.1};
  return command;
}

ActuatorCommand explicit_gains(const RobotState&) {
  ActuatorCommand command;
  command.mode = ControlMode::kPosition;
  command.values = {0.0, 0.0};
  command.stiffness = {0.25, 0.25};
  command.damping = {0.05, 0.05};
  return command;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string xml = argc > 1 ? argv[1] : "tests/assets/two_joint.xml";
  MuJoCoBackendConfig config;
  config.xml_path = xml;
  MuJoCoBackend backend(config);
  ControllerManager controllers;
  controllers.add(std::make_unique<CallbackController>("idle", idle));
  controllers.add(std::make_unique<CallbackController>("steering", steering));
  controllers.add(std::make_unique<CallbackController>("tracking", tracking));
  controllers.add(std::make_unique<CallbackController>(
      "handshake_idle", handshake_idle));
  controllers.add(
      std::make_unique<CallbackController>("explicit_gains", explicit_gains));

  auto input = std::make_unique<BufferedInputSource>("xbox", InputDevice::kXbox);
  auto* input_writer = input.get();
  InputRouter router;
  router.add_source(std::move(input));
  router.bind("xbox", xbox_button::kA, "steering");
  router.bind("xbox", xbox_button::kB, "handshake_idle");
  router.bind("xbox", xbox_button::kLeftBumper, "idle");
  router.bind("xbox", xbox_button::kLeftTrigger, "tracking");
  router.bind_emergency_stop("xbox", xbox_button::kY);
  router.bind_revive("xbox", xbox_button::kX);

  SafetyLimits limits;
  limits.max_abs_command = 1.0;
  limits.max_abs_stiffness = 1.0;
  limits.max_abs_damping = 1.0;
  DeploymentRuntime runtime(backend, controllers, SafetySupervisor(limits));
  if (!runtime.start() || !runtime.reset_anchor_frame()) return 1;

  // Exercise the MuJoCo gain override/restore contract: an explicit policy
  // gain is followed by a fault-safe damping command, then a gain-omitting
  // policy command must restore the XML baseline before stepping again.
  if (!runtime.request_controller("explicit_gains") ||
      runtime.run_cycle(std::chrono::steady_clock::now()) != CycleResult::kSent) {
    return 1;
  }
  runtime.emergency_stop();
  if (!runtime.revive() || !runtime.request_controller("idle") ||
      runtime.run_cycle(std::chrono::steady_clock::now()) != CycleResult::kSent) {
    return 1;
  }

  for (const std::uint32_t buttons : {xbox_button::kA, xbox_button::kB,
                                      xbox_button::kLeftBumper,
                                      xbox_button::kLeftTrigger}) {
    InputState input_state;
    input_state.connected = true;
    input_state.buttons = buttons;
    input_writer->publish(input_state);
    const auto target = router.poll();
    if (!target || target->type != InputEvent::Type::kControllerSwitch ||
        !runtime.request_controller(target->controller)) {
      return 1;
    }
    for (int step = 0; step < 50; ++step) {
      if (runtime.run_cycle(std::chrono::steady_clock::now()) !=
          CycleResult::kSent) {
        runtime.emergency_stop();
        return 1;
      }
    }
    std::cout << controllers.active_name() << " cycles=50\n";
  }
  InputState disconnected;
  disconnected.connected = false;
  input_writer->publish(disconnected);
  const auto disconnect_event = router.poll();
  if (!disconnect_event ||
      disconnect_event->type != InputEvent::Type::kEmergencyStop) {
    return 1;
  }
  runtime.emergency_stop();
  runtime.stop();
  return 0;
}
