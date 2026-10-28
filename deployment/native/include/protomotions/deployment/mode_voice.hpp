// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <string_view>
#include <thread>

namespace protomotions::deployment {

enum class VoiceMode { kSilent, kIdle, kSteering, kTracking, kTeleop,
                       kNavigation, kHandshake, kFootRaise, kUnconditional };

inline const char* voice_mode_text(VoiceMode mode) {
  switch (mode) {
    case VoiceMode::kIdle: return "Idle";
    case VoiceMode::kSteering: return "Steering";
    case VoiceMode::kTracking: return "Tracking";
    case VoiceMode::kTeleop: return "Teleoperation";
    case VoiceMode::kNavigation: return "Navigation";
    case VoiceMode::kHandshake: return "Handshake";
    case VoiceMode::kFootRaise: return "Foot raise";
    case VoiceMode::kUnconditional: return "Unconditional";
    default: return "";
  }
}

inline VoiceMode voice_mode_from_name(std::string_view mode) {
  if (mode == "idle") return VoiceMode::kIdle;
  if (mode == "steering") return VoiceMode::kSteering;
  if (mode == "tracking") return VoiceMode::kTracking;
  if (mode == "teleop" || mode == "pico_teleop") return VoiceMode::kTeleop;
  if (mode == "odometry") return VoiceMode::kNavigation;
  if (mode == "handshake_idle") return VoiceMode::kHandshake;
  if (mode == "right_foot_raise") return VoiceMode::kFootRaise;
  if (mode == "unconditional") return VoiceMode::kUnconditional;
  return VoiceMode::kSilent;
}

// Pure debounce/rate-limit state, shared by native and Python launchers.
// Only the latest stable mode is spoken; no queue of stale transitions.
class ModeVoiceGate {
 public:
  const char* poll(VoiceMode mode, bool hot, double now) {
    if (mode != pending_) { pending_ = mode; changed_at_ = now; }
    if (hot) {
      spoken_ = mode;
      if (!was_hot_ || now >= next_temperature_) {
        was_hot_ = true;
        next_temperature_ = now + 15.0;
        return "Motor temperature high";
      }
      return "";
    }
    was_hot_ = false;
    if (mode == VoiceMode::kSilent) { spoken_ = mode; return ""; }
    if (mode == spoken_ || now - changed_at_ < 0.25 || now < next_allowed_) return "";
    spoken_ = mode;
    next_allowed_ = now + 1.5;
    return voice_mode_text(mode);
  }
 private:
  VoiceMode pending_ = VoiceMode::kSilent, spoken_ = VoiceMode::kSilent;
  double changed_at_ = 0.0, next_allowed_ = 0.0;
  bool was_hot_ = false;
  double next_temperature_ = 0.0;
};

class ModeVoiceWorker {
 public:
  explicit ModeVoiceWorker(std::function<void(const char*)> speak)
      : thread_([this, speak = std::move(speak)] {
          ModeVoiceGate gate;
          try {
            while (!stopping_.load()) {
              const auto now = std::chrono::steady_clock::now().time_since_epoch();
              const char* text = gate.poll(mode_.load(), hot_.load(), std::chrono::duration<double>(now).count());
              if (*text && !stopping_.load()) speak(text);
              std::unique_lock<std::mutex> lock(mutex_);
              cv_.wait_for(lock, std::chrono::milliseconds(50), [this] { return stopping_.load(); });
            }
          } catch (const std::exception& error) {
            std::cerr << "mode voice disabled: " << error.what() << '\n';
          }
        }) {}
  ~ModeVoiceWorker() { close(); }
  void update(VoiceMode mode) noexcept { mode_.store(mode); }
  void temperature(bool hot) noexcept { hot_.store(hot); }
  void close() {
    stopping_.store(true);
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }
 private:
  std::atomic<VoiceMode> mode_{VoiceMode::kSilent};
  std::atomic<bool> stopping_{false};
  std::atomic<bool> hot_{false};
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread thread_;
};
}  // namespace protomotions::deployment
