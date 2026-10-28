// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <memory>
#include <vector>
#include <string>

#include "protomotions/deployment/backend.hpp"

namespace protomotions::deployment {

struct MuJoCoBackendConfig {
  std::string xml_path;
  ControlMode command_mode = ControlMode::kPosition;
  bool step_simulation_on_send = true;
  // Opt-in scene-policy sensor: publish torso pose alongside free-root pose.
  // Empty retains the generic backend's previous behavior.
  std::string policy_anchor_body_name;
  // One remains the generic default; scene launchers set physics decimation.
  int substeps_per_command = 1;
  // Safe-stop damping is a backend safety action, separate from policy gain
  // limits. It is applied for the safe step and XML gains are restored before
  // a later gain-omitting command.
  double emergency_damping = 8.0;
  // Optional startup joint pose in actuator order. Empty preserves the
  // source XML qpos0; scene-policy simulation supplies its calibrated stand.
  // This is simulation initialization, never a policy observation input.
  std::vector<double> initial_dof_position;
};

// Optional headless MuJoCo backend used to validate the complete deployment
// loop before connecting a physical robot. It maps one scalar hinge/slide
// actuator per control DOF and keeps MuJoCo types out of the core library.
class MuJoCoBackend final : public RobotBackend {
 public:
  explicit MuJoCoBackend(MuJoCoBackendConfig config);
  ~MuJoCoBackend() override;

  MuJoCoBackend(const MuJoCoBackend&) = delete;
  MuJoCoBackend& operator=(const MuJoCoBackend&) = delete;

  const RobotCapabilities& capabilities() const noexcept override;
  bool read_state(RobotState& state) override;
  bool send_command(const ActuatorCommand& command) override;
  bool reset_anchor_frame() override;
  bool emergency_stop() noexcept override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace protomotions::deployment
