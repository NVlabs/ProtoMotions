// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "protomotions/deployment/controller.hpp"
#include "protomotions/deployment/metatron_v1_action_map.hpp"
#include "protomotions/deployment/metatron_v1_command.hpp"
#include "protomotions/deployment/metatron_v1_sensor.hpp"

namespace protomotions::deployment::metatron_v1 {

// One policy/history boundary for both MuJoCo and Unitree RobotState sources.
// TaskGoalBuilder is shared by both launchers; it must fill all task inputs
// every cycle. The runner is the native TensorRT Engine::run in production.
class MetatronController final : public Controller {
 public:
  using TaskGoalBuilder = std::function<void(Inputs&, const RobotState&)>;
  using Runner = std::function<Outputs(const Inputs&)>;

  MetatronController(ActionMap action_map, TaskGoalBuilder goals, Runner runner)
      : action_map_(std::move(action_map)),
        goals_(std::move(goals)),
        runner_(std::move(runner)) {
    if (!goals_ || !runner_) {
      throw std::invalid_argument("scene controller requires goals and TensorRT runner");
    }
  }

  std::string_view name() const noexcept override { return "metatron_v1"; }

  bool start(const RobotState& state) override {
    (void)reading_from_robot_state(state);
    inputs_.reset();
    pending_ = false;
    active_ = true;
    return true;
  }

  ActuatorCommand step(const RobotState& state) override {
    if (!active_ || pending_) {
      throw std::logic_error("scene step requires start and acknowledged prior command");
    }
    const SensorReading reading = reading_from_robot_state(state);
    inputs_.prepare(reading);
    clear_task_inputs();
    goals_(inputs_, state);
    const Outputs output = runner_(inputs_);
    if (!action_map_.matches_output(output.action, output.raw_action)) {
      throw std::runtime_error("scene plan action offset/scale disagrees with manifest");
    }
    ActuatorCommand command = command_from_outputs(output);
    pending_reading_ = reading;
    pending_ = true;
    return command;
  }

  void on_command_accepted(const ActuatorCommand& command) override {
    if (!active_ || !pending_ || command.mode != ControlMode::kPosition ||
        command.values.size() != kDofCount) {
      throw std::logic_error("scene command acknowledgement is missing or invalid");
    }
    std::array<float, kDofCount> accepted{};
    for (std::size_t joint = 0; joint < kDofCount; ++joint) {
      accepted[joint] = static_cast<float>(command.values[joint]);
    }
    inputs_.commit(pending_reading_, action_map_.raw_from_applied(accepted));
    pending_ = false;
  }

  void stop() noexcept override {
    active_ = false;
    pending_ = false;
    inputs_.reset();
  }

 private:
  // Python constructs fresh task tensors on every call. Clearing only task
  // inputs prevents a previous mode's masks/style/scene pose leaking into the
  // next call, without erasing sensor or accepted-action history.
  void clear_task_inputs() {
    for (std::size_t index = 2; index < 20; ++index) {
      if (kInputs[index].is_bool) {
        std::fill(inputs_.booleans(index).begin(), inputs_.booleans(index).end(), 0U);
      } else {
        std::fill(inputs_.floats(index).begin(), inputs_.floats(index).end(), 0.0F);
      }
    }
    for (std::size_t index = 31; index < kInputCount; ++index) {
      if (kInputs[index].is_bool) {
        std::fill(inputs_.booleans(index).begin(), inputs_.booleans(index).end(), 0U);
      } else {
        std::fill(inputs_.floats(index).begin(), inputs_.floats(index).end(), 0.0F);
      }
    }
    for (std::size_t frame = 0; frame < 5 * 33; ++frame) {
      inputs_.floats(6)[frame * 4 + 3] = 1.0F;
      inputs_.floats(11)[frame * 4 + 3] = 1.0F;
    }
    inputs_.floats(34)[3] = 1.0F;
  }

  ActionMap action_map_;
  TaskGoalBuilder goals_;
  Runner runner_;
  Inputs inputs_;
  SensorReading pending_reading_{};
  bool active_ = false;
  bool pending_ = false;
};
}  // namespace protomotions::deployment::metatron_v1
