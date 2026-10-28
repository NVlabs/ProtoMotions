// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "protomotions/deployment/latest_mailbox.hpp"

namespace protomotions::deployment {

// The four face buttons and the common Xbox navigation controls share these
// values across SDL2, evdev, Unitree's remote packet, and injected keyboard
// adapters. Applications may bind different controls, but adapters should
// normalize into this vocabulary.
namespace xbox_button {
inline constexpr std::uint32_t kA = 1U << 0;
inline constexpr std::uint32_t kB = 1U << 1;
inline constexpr std::uint32_t kX = 1U << 2;
inline constexpr std::uint32_t kY = 1U << 3;
inline constexpr std::uint32_t kLeftBumper = 1U << 4;
inline constexpr std::uint32_t kRightBumper = 1U << 5;
inline constexpr std::uint32_t kView = 1U << 6;
inline constexpr std::uint32_t kMenu = 1U << 7;
inline constexpr std::uint32_t kDpadUp = 1U << 8;
inline constexpr std::uint32_t kDpadRight = 1U << 9;
inline constexpr std::uint32_t kDpadDown = 1U << 10;
inline constexpr std::uint32_t kDpadLeft = 1U << 11;
// Unitree's remote has four additional digital buttons that do not have a
// direct SDL/Xbox face-button equivalent. Keep them in the normalized surface
// so an application can bind them without decoding vendor bits itself.
inline constexpr std::uint32_t kRightTrigger = 1U << 12;
inline constexpr std::uint32_t kLeftTrigger = 1U << 13;
inline constexpr std::uint32_t kF1 = 1U << 14;
inline constexpr std::uint32_t kF2 = 1U << 15;
}  // namespace xbox_button

// Device adapters normalize their native events into this small, transport-
// independent surface. Button bits are deliberately opaque to the runtime;
// an application chooses a binding map for each input source.
enum class InputDevice { kKeyboard, kUnitreeController, kXbox, kPico };

// Transport-neutral Pico/XRoboToolkit snapshot. The optional vendor adapter
// parses JSON off the control thread and publishes only fixed-size values here.
// Positions are robot/floor-frame world coordinates; local positions are
// headset-yaw-relative; quaternions use xyzw ordering.
struct PicoInputState {
  std::uint64_t device_timestamp_ns = 0;
  std::array<float, 3> left_hand_position_world{};
  std::array<float, 3> right_hand_position_world{};
  std::array<float, 3> left_hand_position_local{};
  std::array<float, 3> right_hand_position_local{};
  std::array<float, 4> left_hand_rotation_local{0.0F, 0.0F, 0.0F, 1.0F};
  std::array<float, 4> right_hand_rotation_local{0.0F, 0.0F, 0.0F, 1.0F};
  float headset_heading = 0.0F;
  float left_trigger = 0.0F;
  float right_trigger = 0.0F;
  float left_grip = 0.0F;
  float right_grip = 0.0F;
  bool pose_valid = false;
  bool heading_valid = false;
  bool hand_rotations_valid = false;
};

struct InputState {
  std::chrono::steady_clock::time_point timestamp{};
  std::uint64_t sequence = 0;
  std::uint32_t buttons = 0;
  std::array<float, 8> axes{};
  PicoInputState pico{};
  bool has_pico = false;
  bool connected = false;
};

struct InputEvent {
  enum class Type {
    kControllerSwitch,
    kEmergencyStop,
    kRevive,
    kSpeedLimit,
  };

  Type type = Type::kControllerSwitch;
  std::string controller;
  std::size_t speed_level = 0;
};

class InputSource {
 public:
  virtual ~InputSource() = default;
  virtual std::string_view name() const noexcept = 0;
  virtual InputDevice device() const noexcept = 0;
  // Returns the newest complete sample. It never drains an event backlog.
  virtual bool poll(InputState& state) = 0;
};

using InputSourcePtr = std::unique_ptr<InputSource>;

// A dependency-free source used by keyboard, Unitree-controller, and adapter
// integrations. The device-specific callback translates native events and
// publishes a normalized snapshot; the control thread only calls poll().
class BufferedInputSource final : public InputSource {
 public:
  BufferedInputSource(std::string name, InputDevice device)
      : name_(std::move(name)), device_(device) {}

  std::string_view name() const noexcept override { return name_; }
  InputDevice device() const noexcept override { return device_; }

  bool publish(InputState state) noexcept {
    if (state.sequence == 0) {
      state.sequence = sequence_ + 1;
    } else if (state.sequence <= sequence_) {
      return false;
    }
    sequence_ = state.sequence;
    if (state.timestamp.time_since_epoch().count() == 0) {
      state.timestamp = std::chrono::steady_clock::now();
    }
    const auto sequence = state.sequence;
    return mailbox_.publish(std::make_shared<const InputState>(std::move(state)),
                            sequence);
  }

  bool poll(InputState& state) override {
    const auto snapshot = mailbox_.snapshot();
    if (!snapshot.value) return false;
    state = *snapshot.value;
    return state.connected;
  }

 private:
  std::string name_;
  InputDevice device_;
  std::uint64_t sequence_ = 0;
  LatestMailbox<InputState> mailbox_;
};

struct InputBinding {
  std::uint32_t button_mask = 0;
  InputEvent event;
};

// Maps normalized button chords to named controllers. It performs only edge
// detection and routing; lifecycle, safety, and controller startup stay in
// ControllerManager/DeploymentRuntime.
class InputRouter final {
 public:
  explicit InputRouter(
      std::chrono::milliseconds source_timeout = std::chrono::milliseconds(250))
      : source_timeout_(source_timeout) {
    if (source_timeout_.count() <= 0) {
      throw std::invalid_argument("input source timeout must be positive");
    }
  }

  bool add_source(InputSourcePtr source) {
    if (!source || source->name().empty()) return false;
    const std::string key(source->name());
    if (sources_.contains(key)) return false;
    sources_.emplace(key, std::move(source));
    return true;
  }

  bool bind(std::string_view source_name, std::uint32_t button_mask,
            std::string controller) {
    if (button_mask == 0 || controller.empty()) return false;
    auto source = sources_.find(source_name);
    if (source == sources_.end()) return false;
    auto& bindings = bindings_[std::string(source_name)];
    if (bindings.contains(button_mask)) return false;
    bindings.emplace(button_mask,
                     InputBinding{button_mask,
                                  InputEvent{InputEvent::Type::kControllerSwitch,
                                             std::move(controller), 0}});
    return true;
  }

  bool bind_emergency_stop(std::string_view source_name,
                           std::uint32_t button_mask) {
    return bind_safety(source_name, button_mask, InputEvent::Type::kEmergencyStop);
  }

  bool bind_revive(std::string_view source_name, std::uint32_t button_mask) {
    return bind_safety(source_name, button_mask, InputEvent::Type::kRevive);
  }

  bool bind_speed_limit(std::string_view source_name,
                        std::uint32_t button_mask, std::size_t level) {
    if (button_mask == 0) return false;
    if (sources_.find(source_name) == sources_.end()) return false;
    auto& bindings = bindings_[std::string(source_name)];
    if (bindings.contains(button_mask) || level >= kMaxSpeedLevels) return false;
    bindings.emplace(button_mask,
                     InputBinding{button_mask,
                                  InputEvent{InputEvent::Type::kSpeedLimit, {},
                                             level}});
    return true;
  }

  // Returns the first edge-triggered event in deterministic source/mask order.
  // Safety events have priority over controller and speed requests. A held
  // button does not repeatedly request an action.
  std::optional<InputEvent> poll() {
    std::optional<InputEvent> safety_event;
    std::optional<InputEvent> normal_event;
    for (auto& [name, source] : sources_) {
      InputState state;
      if (!source->poll(state)) {
        if (active_[name]) {
          active_[name] = false;
          safety_event = InputEvent{InputEvent::Type::kEmergencyStop, {}, 0};
        }
        previous_.erase(name);
        latest_.erase(name);
        continue;
      }
      // Direct hardware sources timestamp the completed poll. Capture the
      // comparison clock after polling so a healthy sample is not rejected
      // merely because it was stamped a few microseconds later.
      const auto now = std::chrono::steady_clock::now();
      if (state.timestamp.time_since_epoch().count() == 0 ||
          now < state.timestamp || now - state.timestamp > source_timeout_) {
        if (active_[name]) {
          active_[name] = false;
          safety_event = InputEvent{InputEvent::Type::kEmergencyStop, {}, 0};
        }
        previous_.erase(name);
        latest_.erase(name);
        continue;
      }
      active_[name] = true;
      auto& previous = previous_[name];
      if (state.sequence == previous.sequence) continue;
      latest_[name] = state;
      const std::uint32_t pressed = state.buttons;
      const std::uint32_t previous_buttons = previous.buttons;
      previous = Previous{state.sequence, state.buttons};
      if (pressed == previous_buttons) continue;
      const auto binding_it = bindings_.find(name);
      if (binding_it == bindings_.end()) continue;
      const auto edge_event = [&](bool safety) -> std::optional<InputEvent> {
        for (const auto& [mask, binding] : binding_it->second) {
          const bool is_safety =
              binding.event.type == InputEvent::Type::kEmergencyStop ||
              binding.event.type == InputEvent::Type::kRevive;
          if (is_safety != safety) continue;
          const bool now_pressed = (pressed & mask) == mask;
          const bool was_pressed = (previous_buttons & mask) == mask;
          if (now_pressed && !was_pressed) return binding.event;
        }
        return std::nullopt;
      };
      if (!safety_event) safety_event = edge_event(true);
      if (!normal_event) normal_event = edge_event(false);
    }
    return safety_event ? safety_event : normal_event;
  }

  bool latest(std::string_view source_name, InputState& state) const {
    const auto it = latest_.find(source_name);
    if (it == latest_.end()) return false;
    state = it->second;
    return state.connected;
  }

  static constexpr std::size_t kMaxSpeedLevels = 3;

 private:
  bool bind_safety(std::string_view source_name, std::uint32_t button_mask,
                   InputEvent::Type type) {
    if (button_mask == 0 || sources_.find(source_name) == sources_.end()) {
      return false;
    }
    auto& bindings = bindings_[std::string(source_name)];
    if (bindings.contains(button_mask)) return false;
    bindings.emplace(button_mask, InputBinding{button_mask, {type, {}, 0}});
    return true;
  }

  struct Previous {
    std::uint64_t sequence = 0;
    std::uint32_t buttons = 0;
  };

  std::map<std::string, InputSourcePtr, std::less<>> sources_;
  std::map<std::string, std::map<std::uint32_t, InputBinding>, std::less<>>
      bindings_;
  std::map<std::string, Previous, std::less<>> previous_;
  std::map<std::string, InputState, std::less<>> latest_;
  std::map<std::string, bool, std::less<>> active_;
  std::chrono::milliseconds source_timeout_;
};

}  // namespace protomotions::deployment
