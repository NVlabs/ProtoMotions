// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "protomotions/deployment/types.hpp"

namespace protomotions::deployment {

// Backend adapters are the only layer that knows a robot SDK or transport.
// They must be non-blocking for the configured control period. `read_state`
// returns the newest complete sample available at the call boundary; adapters
// must not queue stale samples for the controller to drain. This latest-only
// rule is the default transport contract, not an optional optimization.
class RobotBackend {
 public:
  virtual ~RobotBackend() = default;

  virtual const RobotCapabilities& capabilities() const noexcept = 0;
  virtual bool read_state(RobotState& state) = 0;
  virtual bool send_command(const ActuatorCommand& command) = 0;
  virtual bool reset_anchor_frame() = 0;
  // Returns true only when a safe command was accepted by the transport.
  // Fault paths remain noexcept, while startup/re-arm can refuse to proceed
  // when the hard stop cannot be confirmed.
  virtual bool emergency_stop() noexcept = 0;
};

}  // namespace protomotions::deployment
