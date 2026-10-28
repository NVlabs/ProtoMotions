// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/mode_voice.hpp"
#include "protomotions/deployment/motor_temperature.hpp"
#include <future>
#include <stdexcept>
#include <string>

using namespace protomotions::deployment;
void require(bool ok) { if (!ok) throw std::runtime_error("voice regression failed"); }
int main() {
  ModeVoiceGate gate;
  auto poll = [&](VoiceMode m, bool hot, double t) { return std::string(gate.poll(m, hot, t)); };
  require(poll(VoiceMode::kSilent, false, 0).empty());
  require(poll(VoiceMode::kIdle, false, 1).empty());
  require(poll(VoiceMode::kIdle, false, 1.3) == "Idle");
  require(poll(VoiceMode::kIdle, false, 5).empty());
  require(poll(VoiceMode::kTracking, false, 6).empty());
  require(poll(VoiceMode::kSteering, false, 6.1).empty());
  require(poll(VoiceMode::kSteering, false, 6.4) == "Steering");
  require(poll(VoiceMode::kTeleop, true, 6.5) == "Motor temperature high");
  require(poll(VoiceMode::kTracking, true, 7).empty());
  require(poll(VoiceMode::kSilent, true, 21.5) == "Motor temperature high");
  require(poll(VoiceMode::kSilent, false, 22).empty());
  require(poll(VoiceMode::kSilent, true, 22.1) == "Motor temperature high");
  require(voice_mode_from_name("pico_teleop") == VoiceMode::kTeleop);
  require(voice_mode_from_name("recovery") == VoiceMode::kSilent);

  MotorTemperatureMonitor thermal;
  std::vector<std::array<double, 2>> t(29, {30., 30.});
  t[3][1] = 90.; thermal.update(t, 0);
  require(thermal.any_hot());
  thermal.update({}, 1); require(thermal.any_hot());
  t[3][1] = 85.; thermal.update(t, 2); require(thermal.any_hot());
  t[3][1] = 84.; thermal.update(t, 3); require(!thermal.any_hot());

  // A blocked TTS RPC cannot block the producer's mode updates.
  std::promise<void> entered, release;
  auto ready = entered.get_future();
  auto wait = release.get_future();
  ModeVoiceWorker worker([&](const char*) { entered.set_value(); wait.wait(); });
  worker.temperature(true);
  const bool started = ready.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
  if (!started) { release.set_value(); worker.close(); require(false); }
  for (int i = 0; i < 10000; ++i) worker.update(VoiceMode::kSteering);
  worker.update(VoiceMode::kSilent);
  release.set_value();
  worker.close();
  worker.close();
}
