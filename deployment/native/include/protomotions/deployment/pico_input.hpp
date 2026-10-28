// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <string_view>

#include "protomotions/deployment/input.hpp"

namespace protomotions::deployment {

// Native XRoboToolkit/Pico source. PXREA invokes a callback on its service
// thread; poll() copies the newest parsed snapshot and checks its age.
class PicoInputSource final : public InputSource {
 public:
  explicit PicoInputSource(
      std::chrono::milliseconds stale_timeout = std::chrono::milliseconds(250));
  ~PicoInputSource() override;

  PicoInputSource(const PicoInputSource&) = delete;
  PicoInputSource& operator=(const PicoInputSource&) = delete;

  std::string_view name() const noexcept override { return "pico"; }
  InputDevice device() const noexcept override { return InputDevice::kPico; }
  bool poll(InputState& state) override;
  std::string diagnostics() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace protomotions::deployment
