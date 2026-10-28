// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <string>

#include "protomotions/deployment/input.hpp"

namespace protomotions::deployment {

// Optional SDL2 adapter for a laptop Xbox-compatible controller. The core
// library does not depend on SDL2; this class is built only by the optional
// protomotions_sdl2_input target.
class Sdl2XboxInputSource final : public InputSource {
 public:
  explicit Sdl2XboxInputSource(std::string name = "xbox", int device_index = 0);
  ~Sdl2XboxInputSource() override;

  Sdl2XboxInputSource(const Sdl2XboxInputSource&) = delete;
  Sdl2XboxInputSource& operator=(const Sdl2XboxInputSource&) = delete;

  std::string_view name() const noexcept override;
  InputDevice device() const noexcept override { return InputDevice::kXbox; }
  bool poll(InputState& state) override;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace protomotions::deployment
