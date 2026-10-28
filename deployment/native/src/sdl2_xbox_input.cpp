// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/sdl2_xbox_input.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <SDL2/SDL.h>

namespace protomotions::deployment {

class Sdl2XboxInputSource::Impl {
 public:
  Impl(std::string name, int device_index)
      : name_(std::move(name)), device_index_(device_index) {
    if (name_.empty() || device_index_ < 0) {
      throw std::invalid_argument("invalid Xbox input source configuration");
    }
    if (SDL_InitSubSystem(SDL_INIT_GAMECONTROLLER | SDL_INIT_EVENTS) != 0) {
      throw std::runtime_error(SDL_GetError());
    }
    controller_ = SDL_GameControllerOpen(device_index_);
    if (!controller_) {
      SDL_QuitSubSystem(SDL_INIT_GAMECONTROLLER | SDL_INIT_EVENTS);
      throw std::runtime_error(SDL_GetError());
    }
  }

  ~Impl() {
    if (controller_) SDL_GameControllerClose(controller_);
    SDL_QuitSubSystem(SDL_INIT_GAMECONTROLLER | SDL_INIT_EVENTS);
  }

  std::string_view name() const noexcept { return name_; }

  bool poll(InputState& state) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_CONTROLLERDEVICEADDED && !controller_) {
        controller_ = SDL_GameControllerOpen(event.cdevice.which);
      } else if (event.type == SDL_CONTROLLERDEVICEREMOVED && controller_ &&
                 event.cdevice.which == SDL_JoystickInstanceID(
                     SDL_GameControllerGetJoystick(controller_))) {
        SDL_GameControllerClose(controller_);
        controller_ = nullptr;
      }
    }
    if (!controller_) return false;
    SDL_GameControllerUpdate();
    std::uint32_t buttons = 0;
    constexpr std::array<std::pair<SDL_GameControllerButton, std::uint32_t>, 12>
        kButtons{{{SDL_CONTROLLER_BUTTON_A, 1U << 0},
                  {SDL_CONTROLLER_BUTTON_B, 1U << 1},
                  {SDL_CONTROLLER_BUTTON_X, 1U << 2},
                  {SDL_CONTROLLER_BUTTON_Y, 1U << 3},
                  {SDL_CONTROLLER_BUTTON_LEFTSHOULDER,
                   xbox_button::kLeftBumper},
                  {SDL_CONTROLLER_BUTTON_RIGHTSHOULDER,
                   xbox_button::kRightBumper},
                  {SDL_CONTROLLER_BUTTON_BACK, xbox_button::kView},
                  {SDL_CONTROLLER_BUTTON_START, xbox_button::kMenu},
                  {SDL_CONTROLLER_BUTTON_DPAD_UP, xbox_button::kDpadUp},
                  {SDL_CONTROLLER_BUTTON_DPAD_RIGHT, xbox_button::kDpadRight},
                  {SDL_CONTROLLER_BUTTON_DPAD_DOWN, xbox_button::kDpadDown},
                  {SDL_CONTROLLER_BUTTON_DPAD_LEFT, xbox_button::kDpadLeft}}};
    for (const auto& [button, mask] : kButtons) {
      if (SDL_GameControllerGetButton(controller_, button)) buttons |= mask;
    }
    constexpr std::array<SDL_GameControllerAxis, 6> kAxes{
        SDL_CONTROLLER_AXIS_LEFTX, SDL_CONTROLLER_AXIS_LEFTY,
        SDL_CONTROLLER_AXIS_RIGHTX, SDL_CONTROLLER_AXIS_RIGHTY,
        SDL_CONTROLLER_AXIS_TRIGGERLEFT, SDL_CONTROLLER_AXIS_TRIGGERRIGHT};
    std::array<float, 8> axes{};
    for (std::size_t i = 0; i < kAxes.size(); ++i) {
      const auto raw = SDL_GameControllerGetAxis(controller_, kAxes[i]);
      const float denominator = i >= 4 ? 32767.0F : 32768.0F;
      axes[i] = std::clamp(static_cast<float>(raw) / denominator, -1.0F, 1.0F);
    }
    // SDL reports stick-up as -Y. Normalize the external Xbox source
    // to the shared deployment contract (+Y=forward).
    axes[1] = -axes[1];
    axes[3] = -axes[3];
    // SDL exposes triggers as axes (rest=-1, pressed=+1), while the
    // transport-independent input surface also offers digital trigger bits.
    // Use 25% travel to press and 10% travel to release because a normal
    // press may not reach the endpoint and analog trigger noise should not
    // generate repeated button edges.
    constexpr float kTriggerPressThreshold = -0.5F;
    constexpr float kTriggerReleaseThreshold = -0.8F;
    const bool left_trigger_pressed =
        (last_buttons_ & xbox_button::kLeftTrigger) != 0;
    if ((!left_trigger_pressed && axes[4] > kTriggerPressThreshold) ||
        (left_trigger_pressed && axes[4] >= kTriggerReleaseThreshold)) {
      buttons |= xbox_button::kLeftTrigger;
    }
    const bool right_trigger_pressed =
        (last_buttons_ & xbox_button::kRightTrigger) != 0;
    if ((!right_trigger_pressed && axes[5] > kTriggerPressThreshold) ||
        (right_trigger_pressed && axes[5] >= kTriggerReleaseThreshold)) {
      buttons |= xbox_button::kRightTrigger;
    }
    const bool changed =
        !has_sample_ || buttons != last_buttons_ || axes != last_axes_;
    if (changed) {
      last_buttons_ = buttons;
      last_axes_ = axes;
      sequence_ += 1;
      has_sample_ = true;
    }
    // A successful poll is a liveness heartbeat even when the operator holds
    // the same controls. Sequence numbers describe state changes; the
    // timestamp describes source health and must advance on every poll.
    last_timestamp_ = std::chrono::steady_clock::now();
    state.timestamp = last_timestamp_;
    state.sequence = sequence_;
    state.connected = true;
    state.buttons = last_buttons_;
    state.axes = last_axes_;
    return true;
  }

 private:
  std::string name_;
  int device_index_;
  SDL_GameController* controller_ = nullptr;
  std::uint64_t sequence_ = 0;
  std::uint32_t last_buttons_ = 0;
  std::array<float, 8> last_axes_{};
  std::chrono::steady_clock::time_point last_timestamp_{};
  bool has_sample_ = false;
};

Sdl2XboxInputSource::Sdl2XboxInputSource(std::string name, int device_index)
    : impl_(std::make_unique<Impl>(std::move(name), device_index)) {}

Sdl2XboxInputSource::~Sdl2XboxInputSource() = default;

std::string_view Sdl2XboxInputSource::name() const noexcept {
  return impl_->name();
}

bool Sdl2XboxInputSource::poll(InputState& state) { return impl_->poll(state); }

}  // namespace protomotions::deployment
