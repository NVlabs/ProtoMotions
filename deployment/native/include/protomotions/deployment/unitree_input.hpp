// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <stdexcept>
#include <utility>

#include "protomotions/deployment/input.hpp"

namespace protomotions::deployment {

// Unitree REMOTE_DATA_RX uses two button bytes followed by five little-endian
// floats (lx, rx, ry, L2, ly). This decoder deliberately accepts raw bytes so
// the core library does not include Unitree SDK headers. UnitreeSdk2Backend
// exposes the same 40-byte payload in RobotState::operator_input_data.
inline InputState decode_unitree_remote(std::span<const std::uint8_t> data,
                                        std::uint64_t sequence = 0,
                                        std::chrono::steady_clock::time_point
                                            timestamp = {}) {
  if (data.size() < 24) {
    throw std::invalid_argument("Unitree remote payload is shorter than 24 bytes");
  }
  InputState state;
  state.timestamp = timestamp.time_since_epoch().count() == 0
                        ? std::chrono::steady_clock::now()
                        : timestamp;
  state.sequence = sequence;
  state.connected = true;
  const auto raw_buttons = static_cast<std::uint16_t>(data[2]) |
                           (static_cast<std::uint16_t>(data[3]) << 8U);
  const auto set_raw_bit = [&state, raw_buttons](unsigned bit,
                                                 std::uint32_t output) {
    if ((raw_buttons & (1U << bit)) != 0) state.buttons |= output;
  };
  // Unitree's 16-bit map follows the SDK remote ordering: R1, L1, Start,
  // Select, R2, L2, F1, F2, A, B, X, Y, Up, Right, Down, Left.
  set_raw_bit(0, xbox_button::kRightBumper);
  set_raw_bit(1, xbox_button::kLeftBumper);
  set_raw_bit(2, xbox_button::kMenu);
  set_raw_bit(3, xbox_button::kView);
  set_raw_bit(4, xbox_button::kRightTrigger);
  set_raw_bit(5, xbox_button::kLeftTrigger);
  set_raw_bit(6, xbox_button::kF1);
  set_raw_bit(7, xbox_button::kF2);
  set_raw_bit(8, xbox_button::kA);
  set_raw_bit(9, xbox_button::kB);
  set_raw_bit(10, xbox_button::kX);
  set_raw_bit(11, xbox_button::kY);
  set_raw_bit(12, xbox_button::kDpadUp);
  set_raw_bit(13, xbox_button::kDpadRight);
  set_raw_bit(14, xbox_button::kDpadDown);
  set_raw_bit(15, xbox_button::kDpadLeft);

  float values[5]{};
  std::memcpy(values, data.data() + 4, sizeof(values));
  for (const float value : values) {
    if (!std::isfinite(value) || std::abs(value) > 1.0F) {
      throw std::invalid_argument("Unitree remote axis is outside [-1, 1]");
    }
  }
  state.axes[0] = values[0];  // lx
  state.axes[1] = values[4];  // ly
  state.axes[2] = values[1];  // rx
  state.axes[3] = values[2];  // ry
  state.axes[4] = values[3];  // L2
  return state;
}

// Converts a backend's raw Unitree payload into the same latest-only input
// source used by keyboard and Xbox adapters. Call update once per fresh
// RobotState on the control thread, then route with InputRouter::poll().
class UnitreeControllerInputSource final : public InputSource {
 public:
  explicit UnitreeControllerInputSource(std::string name = "unitree")
      : name_(std::move(name)) {}

  std::string_view name() const noexcept override { return name_; }
  InputDevice device() const noexcept override {
    return InputDevice::kUnitreeController;
  }

  bool update(std::span<const std::uint8_t> remote,
              std::uint64_t sequence = 0,
              std::chrono::steady_clock::time_point timestamp = {}) noexcept {
    try {
      auto state = decode_unitree_remote(remote, sequence, timestamp);
      if (state.sequence == 0) {
        state.sequence = sequence_ + 1;
      } else if (state.sequence <= sequence_) {
        return false;
      }
      sequence_ = state.sequence;
      return mailbox_.publish(std::make_shared<const InputState>(std::move(state)),
                              sequence_);
    } catch (...) {
      return false;
    }
  }

  bool poll(InputState& state) override {
    const auto snapshot = mailbox_.snapshot();
    if (!snapshot.value) return false;
    state = *snapshot.value;
    return state.connected;
  }

 private:
  std::string name_;
  std::uint64_t sequence_ = 0;
  LatestMailbox<InputState> mailbox_;
};

}  // namespace protomotions::deployment
