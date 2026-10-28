// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/unitree_mode_voice.hpp"

#include <stdexcept>
#include <string>
#include <unitree/robot/g1/audio/g1_audio_client.hpp>

namespace protomotions::deployment {
class UnitreeModeVoice::Impl {
 public:
  Impl() : worker([this](const char* text) {
    if (!initialized) {
      client.SetTimeout(1.0F);
      client.Init();
      initialized = true;
    }
    const auto result = client.TtsMaker(text, 1);
    if (result != 0) throw std::runtime_error("English TTS RPC failed: " + std::to_string(result));
  }) {}
  unitree::robot::g1::AudioClient client;
  bool initialized = false;
  ModeVoiceWorker worker;  // Join before destroying the client.
};
UnitreeModeVoice::UnitreeModeVoice() : impl_(std::make_unique<Impl>()) {}
UnitreeModeVoice::~UnitreeModeVoice() = default;
void UnitreeModeVoice::update(VoiceMode mode) noexcept { impl_->worker.update(mode); }
void UnitreeModeVoice::temperature(bool hot) noexcept { impl_->worker.temperature(hot); }
void UnitreeModeVoice::close() { impl_->worker.close(); }
}  // namespace protomotions::deployment
