// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stdexcept>

#include "protomotions/deployment/metatron_v1_engine.hpp"
#include "protomotions/deployment/types.hpp"

namespace protomotions::deployment::metatron_v1 {

// The exported ONNX graph already applies the learned action transform and
// per-joint affine offset/scale. Never send raw_action or apply the map twice.
inline ActuatorCommand command_from_outputs(const Outputs& outputs) {
  ActuatorCommand command;
  command.mode = ControlMode::kPosition;
  command.values.assign(outputs.action.begin(), outputs.action.end());
  command.stiffness.assign(outputs.stiffness.begin(), outputs.stiffness.end());
  command.damping.assign(outputs.damping.begin(), outputs.damping.end());
  if (!command.finite())
    throw std::invalid_argument("non-finite scene policy actuator command");
  return command;
}
}  // namespace protomotions::deployment::metatron_v1
