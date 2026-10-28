// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include "protomotions/deployment/mode_voice.hpp"

namespace protomotions::deployment {

// Construct only after the backend has initialized the DDS channel factory.
// All AudioClient RPCs, including initialization, execute on the worker.
class UnitreeModeVoice {
 public:
  UnitreeModeVoice();
  ~UnitreeModeVoice();
  void update(VoiceMode mode) noexcept;
  void temperature(bool hot) noexcept;
  void close();
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace protomotions::deployment
