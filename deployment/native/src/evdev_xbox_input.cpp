// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/evdev_xbox_input.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <linux/input.h>
#include <optional>
#include <stdexcept>
#include <sys/ioctl.h>
#include <unistd.h>
#include <utility>

namespace protomotions::deployment {

class EvdevXboxInputSource::Impl {
 public:
  Impl(std::string device_path, std::string name)
      : device_path_(std::move(device_path)), name_(std::move(name)) {
    if (device_path_.empty() || name_.empty()) {
      throw std::invalid_argument("evdev Xbox path and name are required");
    }
    fd_ = ::open(device_path_.c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("could not open evdev Xbox device");
    input_id id{};
    if (::ioctl(fd_, EVIOCGID, &id) == 0) {
      // This PowerA pad reports physical X/Y opposite to standard Xbox evdev.
      swap_xy_ = id.vendor == 0x20d6 && id.product == 0x207b;
      if (swap_xy_)
        std::cout << "Xbox evdev: PowerA X/Y button layout corrected\n";
    }
    for (const auto& [code, slot] : kAxisCodes) {
      input_absinfo info{};
      if (::ioctl(fd_, EVIOCGABS(code), &info) == 0) {
        axes_[slot] = Axis{info.minimum, info.maximum, info.value};
      }
    }
  }

  ~Impl() {
    if (fd_ >= 0) ::close(fd_);
  }

  std::string_view name() const noexcept { return name_; }

  bool poll(InputState& state) {
    if (fd_ < 0) return false;
    input_event event{};
    bool disconnected = false;
    bool dropped = awaiting_resync_;
    while (true) {
      const auto bytes = ::read(fd_, &event, sizeof(event));
      if (bytes == static_cast<ssize_t>(sizeof(event))) {
        if (dropped && event.type == EV_SYN && event.code == SYN_REPORT) {
          awaiting_resync_ = false;
          dropped = false;
          if (!resynchronize()) {
            disconnected = true;
            break;
          }
          continue;
        }
        apply(event, dropped);
        if (dropped) {
          // After SYN_DROPPED, discard the remainder of the stale kernel
          // queue through SYN_REPORT, then rebuild the current state from
          // evdev ioctls. A transient queue overrun is not a disconnect.
          bool recovered = false;
          while (true) {
            const auto sync_bytes = ::read(fd_, &event, sizeof(event));
            if (sync_bytes == static_cast<ssize_t>(sizeof(event))) {
              if (event.type == EV_SYN && event.code == SYN_REPORT) {
                recovered = true;
                break;
              }
              continue;
            }
            if (sync_bytes < 0 &&
                (errno == EAGAIN || errno == EWOULDBLOCK)) {
              // Nonblocking evdev may deliver the overrun marker and the
              // matching SYN_REPORT in separate polls. Keep the fd open and
              // resume draining on the next poll instead of treating this
              // transient gap as device loss.
              awaiting_resync_ = true;
              break;
            }
            if (sync_bytes == 0 ||
                (sync_bytes < 0 && (errno == ENODEV || errno == EIO))) {
              disconnected = true;
            }
            break;
          }
          if (!recovered) {
            if (disconnected) break;
            return false;
          }
          awaiting_resync_ = false;
          dropped = false;
          if (!resynchronize()) {
            disconnected = true;
            break;
          }
        }
        continue;
      }
      if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK) && dropped) {
        // Still waiting for SYN_REPORT from a split overrun; do not publish
        // the stale snapshot as a fresh controller state.
        return false;
      }
      if (bytes < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) break;
      if (bytes == 0 || (bytes < 0 && (errno == ENODEV || errno == EIO))) {
        disconnected = true;
      }
      break;
    }
    if (disconnected) {
      ::close(fd_);
      fd_ = -1;
      return false;
    }
    std::array<float, 8> axes{};
    for (const auto& [code, slot] : kAxisCodes) {
      const auto& axis = axes_[slot];
      axes[slot] = axis.normalized();
    }
    // Linux evdev reports stick-up as -Y. Normalize the external
    // Xbox source to the shared deployment contract (+Y=forward).
    axes[1] = -axes[1];
    axes[3] = -axes[3];
    // Rebuild the published button snapshot from EV_KEY state. Hat-axis D-pad
    // values are layered below, while trigger bits are derived from the
    // analog axes.
    buttons_ = key_buttons_;
    // Some Xbox-compatible Linux drivers expose the D-pad as a hat axis
    // rather than BTN_DPAD_* key events. Normalize both representations into
    // the same edge-triggered button bits. Linux reports hat-up as -1 and
    // hat-down as +1.
    if (axes[6] < -0.5F) buttons_ |= xbox_button::kDpadLeft;
    if (axes[6] > 0.5F) buttons_ |= xbox_button::kDpadRight;
    if (axes[7] < -0.5F) buttons_ |= xbox_button::kDpadUp;
    if (axes[7] > 0.5F) buttons_ |= xbox_button::kDpadDown;
    // Most Xbox evdev mappings expose L2/R2 as ABS_Z/ABS_RZ with the same
    // rest=-1, pressed=+1 convention as SDL. A normal trigger press does not
    // necessarily reach the physical endpoint, so treat 25% travel as the
    // digital edge and preserve the analog values for future consumers.
    constexpr float kTriggerPressThreshold = -0.5F;
    constexpr float kTriggerReleaseThreshold = -0.8F;
    const bool left_trigger_pressed =
        (buttons_ & xbox_button::kLeftTrigger) != 0;
    if ((!left_trigger_pressed && axes[4] > kTriggerPressThreshold) ||
        (left_trigger_pressed && axes[4] >= kTriggerReleaseThreshold)) {
      buttons_ |= xbox_button::kLeftTrigger;
    } else {
      buttons_ &= ~xbox_button::kLeftTrigger;
    }
    const bool right_trigger_pressed =
        (buttons_ & xbox_button::kRightTrigger) != 0;
    if ((!right_trigger_pressed && axes[5] > kTriggerPressThreshold) ||
        (right_trigger_pressed && axes[5] >= kTriggerReleaseThreshold)) {
      buttons_ |= xbox_button::kRightTrigger;
    } else {
      buttons_ &= ~xbox_button::kRightTrigger;
    }
    const bool changed =
        !has_sample_ || buttons_ != last_buttons_ || axes != last_axes_;
    if (changed) {
      last_buttons_ = buttons_;
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
    state.buttons = last_buttons_;
    state.axes = last_axes_;
    state.connected = true;
    return true;
  }

 private:
  struct Axis {
    int minimum = -32768;
    int maximum = 32767;
    int value = 0;

    float normalized() const noexcept {
      const double span = static_cast<double>(maximum - minimum);
      if (span <= 0.0) return 0.0F;
      return static_cast<float>(std::clamp(
          (static_cast<double>(value - minimum) / span) * 2.0 - 1.0, -1.0,
          1.0));
    }
  };

  static constexpr std::array<std::pair<unsigned short, std::size_t>, 8>
      kAxisCodes{{{ABS_X, 0}, {ABS_Y, 1}, {ABS_RX, 2}, {ABS_RY, 3},
                  {ABS_Z, 4}, {ABS_RZ, 5}, {ABS_HAT0X, 6}, {ABS_HAT0Y, 7}}};

  std::optional<std::uint32_t> button_mask(
      unsigned short code) const noexcept {
    switch (code) {
      case BTN_SOUTH: return xbox_button::kA;
      case BTN_EAST: return xbox_button::kB;
      case BTN_WEST: return swap_xy_ ? xbox_button::kY : xbox_button::kX;
      case BTN_NORTH: return swap_xy_ ? xbox_button::kX : xbox_button::kY;
      case BTN_TL: return xbox_button::kLeftBumper;
      case BTN_TR: return xbox_button::kRightBumper;
      case BTN_SELECT: return xbox_button::kView;
      case BTN_START: return xbox_button::kMenu;
      case BTN_DPAD_UP: return xbox_button::kDpadUp;
      case BTN_DPAD_RIGHT: return xbox_button::kDpadRight;
      case BTN_DPAD_DOWN: return xbox_button::kDpadDown;
      case BTN_DPAD_LEFT: return xbox_button::kDpadLeft;
      default: return std::nullopt;
    }
  }

  bool resynchronize() {
    for (const auto& [code, slot] : kAxisCodes) {
      input_absinfo info{};
      if (::ioctl(fd_, EVIOCGABS(code), &info) == 0) {
        axes_[slot] = Axis{info.minimum, info.maximum, info.value};
      }
    }
    std::array<unsigned char, (KEY_MAX + 8) / 8> keys{};
    if (::ioctl(fd_, EVIOCGKEY(keys.size()), keys.data()) < 0) return false;
    key_buttons_ = 0;
    for (unsigned short code = 0; code <= KEY_MAX; ++code) {
      const auto mask = button_mask(code);
      if (mask && (keys[code / 8] & (1U << (code % 8)))) {
        key_buttons_ |= *mask;
      }
    }
    return true;
  }

  void apply(const input_event& event, bool& dropped) {
    if (event.type == EV_KEY) {
      const auto mask = button_mask(event.code);
      if (mask) {
        if (event.value == 0) key_buttons_ &= ~*mask;
        else if (event.value == 1 || event.value == 2) key_buttons_ |= *mask;
      }
    } else if (event.type == EV_ABS) {
      for (const auto& [code, slot] : kAxisCodes) {
        if (event.code == code) axes_[slot].value = event.value;
      }
    } else if (event.type == EV_SYN && event.code == SYN_DROPPED) {
      dropped = true;
    }
  }

  std::string device_path_;
  std::string name_;
  int fd_ = -1;
  bool swap_xy_ = false;
  std::array<Axis, 8> axes_{};
  std::uint32_t key_buttons_ = 0;
  std::uint32_t buttons_ = 0;
  std::uint64_t sequence_ = 0;
  std::uint32_t last_buttons_ = 0;
  std::array<float, 8> last_axes_{};
  std::chrono::steady_clock::time_point last_timestamp_{};
  bool has_sample_ = false;
  bool awaiting_resync_ = false;
};

EvdevXboxInputSource::EvdevXboxInputSource(std::string device_path,
                                           std::string name)
    : impl_(std::make_unique<Impl>(std::move(device_path), std::move(name))) {}

EvdevXboxInputSource::~EvdevXboxInputSource() = default;

std::string_view EvdevXboxInputSource::name() const noexcept {
  return impl_->name();
}

bool EvdevXboxInputSource::poll(InputState& state) { return impl_->poll(state); }

}  // namespace protomotions::deployment
