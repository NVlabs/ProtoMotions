// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>

#include "protomotions/deployment/input.hpp"

namespace protomotions::deployment {

// Linux evdev adapter for an Xbox-compatible controller. It has no third-
// party dependency and expects an explicit /dev/input/event* path selected by
// the application (udev rules can provide a stable path).
class EvdevXboxInputSource final : public InputSource {
 public:
  explicit EvdevXboxInputSource(std::string device_path,
                                std::string name = "xbox");
  ~EvdevXboxInputSource() override;

  EvdevXboxInputSource(const EvdevXboxInputSource&) = delete;
  EvdevXboxInputSource& operator=(const EvdevXboxInputSource&) = delete;

  std::string_view name() const noexcept override;
  InputDevice device() const noexcept override { return InputDevice::kXbox; }
  bool poll(InputState& state) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace protomotions::deployment
