// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/pico_input.hpp"

#include <PXREARobotSDK.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/file.h>
#include <unistd.h>

namespace protomotions::deployment {
namespace {

using Json = nlohmann::json;
using Vec3 = std::array<float, 3>;
using Quat = std::array<float, 4>;  // xyzw
using Matrix3 = std::array<float, 9>;  // row-major

bool finite(const Vec3& value) {
  return std::all_of(value.begin(), value.end(),
                     [](float item) { return std::isfinite(item); });
}

bool finite(const Quat& value) {
  return std::all_of(value.begin(), value.end(),
                     [](float item) { return std::isfinite(item); });
}

float norm(const Quat& value) {
  float result = 0.0F;
  for (const float item : value) result += item * item;
  return std::sqrt(result);
}

Quat normalize(Quat value) {
  const float length = norm(value);
  if (!std::isfinite(length) || length <= 1.0e-8F) {
    return {0.0F, 0.0F, 0.0F, 1.0F};
  }
  for (float& item : value) item /= length;
  return value;
}

Quat multiply(const Quat& a, const Quat& b) {
  return {
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  };
}

Matrix3 quaternion_matrix(const Quat& input) {
  const Quat q = normalize(input);
  const float x = q[0];
  const float y = q[1];
  const float z = q[2];
  const float w = q[3];
  return {
      1.0F - 2.0F * (y * y + z * z), 2.0F * (x * y - z * w),
      2.0F * (x * z + y * w), 2.0F * (x * y + z * w),
      1.0F - 2.0F * (x * x + z * z), 2.0F * (y * z - x * w),
      2.0F * (x * z - y * w), 2.0F * (y * z + x * w),
      1.0F - 2.0F * (x * x + y * y),
  };
}

Matrix3 multiply(const Matrix3& a, const Matrix3& b) {
  Matrix3 result{};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      for (int inner = 0; inner < 3; ++inner) {
        result[3 * row + column] +=
            a[3 * row + inner] * b[3 * inner + column];
      }
    }
  }
  return result;
}

Vec3 rotate(const Matrix3& matrix, const Vec3& value) {
  return {
      matrix[0] * value[0] + matrix[1] * value[1] + matrix[2] * value[2],
      matrix[3] * value[0] + matrix[4] * value[1] + matrix[5] * value[2],
      matrix[6] * value[0] + matrix[7] * value[1] + matrix[8] * value[2],
  };
}

Quat matrix_quaternion(const Matrix3& matrix) {
  Quat q{};
  const float trace = matrix[0] + matrix[4] + matrix[8];
  if (trace > 0.0F) {
    const float s = 2.0F * std::sqrt(std::max(0.0F, trace + 1.0F));
    q[3] = 0.25F * s;
    q[0] = (matrix[7] - matrix[5]) / s;
    q[1] = (matrix[2] - matrix[6]) / s;
    q[2] = (matrix[3] - matrix[1]) / s;
  } else if (matrix[0] > matrix[4] && matrix[0] > matrix[8]) {
    const float s = 2.0F * std::sqrt(
        std::max(0.0F, 1.0F + matrix[0] - matrix[4] - matrix[8]));
    q[3] = (matrix[7] - matrix[5]) / s;
    q[0] = 0.25F * s;
    q[1] = (matrix[1] + matrix[3]) / s;
    q[2] = (matrix[2] + matrix[6]) / s;
  } else if (matrix[4] > matrix[8]) {
    const float s = 2.0F * std::sqrt(
        std::max(0.0F, 1.0F + matrix[4] - matrix[0] - matrix[8]));
    q[3] = (matrix[2] - matrix[6]) / s;
    q[0] = (matrix[1] + matrix[3]) / s;
    q[1] = 0.25F * s;
    q[2] = (matrix[5] + matrix[7]) / s;
  } else {
    const float s = 2.0F * std::sqrt(
        std::max(0.0F, 1.0F + matrix[8] - matrix[0] - matrix[4]));
    q[3] = (matrix[3] - matrix[1]) / s;
    q[0] = (matrix[2] + matrix[6]) / s;
    q[1] = (matrix[5] + matrix[7]) / s;
    q[2] = 0.25F * s;
  }
  return normalize(q);
}

constexpr Matrix3 unity_to_robot() {
  // Unity/XRoboToolkit: x right, y up, z forward. Deployment: x forward,
  // y left, z up. This matches examples/pico_input.py.
  return {0.0F, 0.0F, -1.0F, -1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
}

Vec3 unity_position_to_robot(const Vec3& value) {
  return rotate(unity_to_robot(), value);
}

Quat unity_rotation_to_robot(const Quat& value) {
  constexpr Matrix3 conversion = unity_to_robot();
  constexpr Matrix3 transpose = {
      conversion[0], conversion[3], conversion[6], conversion[1],
      conversion[4], conversion[7], conversion[2], conversion[5],
      conversion[8]};
  return matrix_quaternion(multiply(
      multiply(conversion, quaternion_matrix(value)), transpose));
}

Quat yaw_quaternion(float value) {
  const float half = value * 0.5F;
  return {0.0F, 0.0F, std::sin(half), std::cos(half)};
}

float yaw(const Quat& value) {
  const Quat q = normalize(value);
  return std::atan2(2.0F * (q[3] * q[2] + q[0] * q[1]),
                    1.0F - 2.0F * (q[1] * q[1] + q[2] * q[2]));
}

bool parse_csv(const Json& value, std::size_t count, float* output) {
  if (value.is_array()) {
    if (value.size() < count) return false;
    for (std::size_t i = 0; i < count; ++i) {
      if (!value[i].is_number()) return false;
      output[i] = value[i].get<float>();
    }
    return true;
  }
  if (!value.is_string()) return false;
  std::stringstream stream(value.get<std::string>());
  std::string item;
  std::size_t index = 0;
  while (std::getline(stream, item, ',') && index < count) {
    try {
      output[index++] = std::stof(item);
    } catch (const std::exception&) {
      return false;
    }
  }
  return index == count;
}

bool number(const Json& object, const char* key, float& output) {
  if (!object.contains(key) || !object.at(key).is_number()) return false;
  output = object.at(key).get<float>();
  return std::isfinite(output);
}

bool button(const Json& object, const char* key) {
  return object.contains(key) && object.at(key).is_boolean() &&
         object.at(key).get<bool>();
}

}  // namespace

class PicoInputSource::Impl {
 public:
  explicit Impl(std::chrono::milliseconds stale_timeout)
      : stale_timeout_(stale_timeout) {
    if (stale_timeout_.count() <= 0) {
      throw std::invalid_argument("Pico stale timeout must be positive");
    }
    const std::string lock_path =
        "/tmp/protomotions_pico_xrt_" + std::to_string(::getuid()) + ".lock";
    lock_fd_ = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (lock_fd_ < 0) {
      throw std::runtime_error(
          "cannot open Pico client lock " + lock_path + ": " +
          std::strerror(errno));
    }
    if (::flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
      std::array<char, 64> owner{};
      const auto size = ::pread(lock_fd_, owner.data(), owner.size() - 1, 0);
      const std::string owner_text =
          size > 0 ? std::string(owner.data(), static_cast<std::size_t>(size))
                   : std::string{};
      ::close(lock_fd_);
      lock_fd_ = -1;
      throw std::runtime_error(
          "another ProtoMotions Pico client is already running" +
          (owner_text.empty() ? std::string{}
                              : " (owner pid " + owner_text + ")") +
          "; stop it before starting a second deployment");
    }
    const std::string owner = std::to_string(::getpid());
    if (::ftruncate(lock_fd_, 0) != 0 ||
        ::pwrite(lock_fd_, owner.data(), owner.size(), 0) !=
            static_cast<ssize_t>(owner.size())) {
      const std::string detail = std::strerror(errno);
      release_lock();
      throw std::runtime_error("cannot write Pico client lock: " + detail);
    }
    if (PXREAInit(this, &Impl::callback, PXREAFullMask) != 0) {
      release_lock();
      throw std::runtime_error("PXREAInit failed");
    }
    initialized_ = true;
  }

  ~Impl() {
    if (initialized_) PXREADeinit();
    release_lock();
  }

  bool poll(InputState& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!received_) return false;
    state = latest_;
    if (std::chrono::steady_clock::now() - latest_.timestamp > stale_timeout_) {
      state.connected = false;
      return false;
    }
    state.connected = true;
    return true;
  }

  std::string diagnostics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::ostringstream stream;
    stream << "local_service="
           << (server_connected_ ? "connected" : "waiting")
           << " device="
           << (device_id_.empty() ? "not-found" : device_id_)
           << " packets=" << sequence_;
    if (received_) {
      const auto age = std::chrono::duration<double>(
          std::chrono::steady_clock::now() - latest_.timestamp).count();
      stream << " last_packet_age=" << age << "s pose="
             << (latest_.pico.pose_valid && latest_.pico.heading_valid
                     ? "valid"
                     : "unavailable");
    } else {
      stream << " last_packet=never";
    }
    if (!last_error_.empty()) stream << " error=" << last_error_;
    return stream.str();
  }

 private:
  void release_lock() noexcept {
    if (lock_fd_ < 0) return;
    static_cast<void>(::flock(lock_fd_, LOCK_UN));
    ::close(lock_fd_);
    lock_fd_ = -1;
  }

  static void callback(void* context, PXREAClientCallbackType type, int status,
                       void* user_data) {
    auto* self = static_cast<Impl*>(context);
    if (self != nullptr) self->handle(type, status, user_data);
  }

  void handle(PXREAClientCallbackType type, int status, void* user_data) {
    if (type == PXREAServerConnect) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        server_connected_ = true;
      }
      std::cout << "pico: XRoboToolkit server connected\n";
      return;
    }
    if (type == PXREAServerDisconnect) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        server_connected_ = false;
      }
      std::cerr << "pico: XRoboToolkit server disconnected\n";
      return;
    }
    if (type == PXREADeviceFind && user_data != nullptr) {
      const std::string device = static_cast<const char*>(user_data);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        device_id_ = device;
      }
      std::cout << "pico: device found " << device << "\n";
      return;
    }
    if (type == PXREADeviceMissing && user_data != nullptr) {
      const std::string device = static_cast<const char*>(user_data);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (device_id_ == device) device_id_.clear();
      }
      std::cerr << "pico: device missing " << device << "\n";
      return;
    }
    if (type == PXREADeviceConnect && user_data != nullptr) {
      const std::string device = static_cast<const char*>(user_data);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        device_id_ = device;
      }
      std::cout << "pico: device connected " << device
                << " status=" << status << "\n";
      return;
    }
    if (type != PXREADeviceStateJson || user_data == nullptr) return;

    const auto& state = *static_cast<const PXREADevStateJson*>(user_data);
    try {
      const Json outer = Json::parse(state.stateJson);
      if (!outer.contains("value")) return;
      const auto& raw_value = outer.at("value");
      const Json value = raw_value.is_string()
                             ? Json::parse(raw_value.get<std::string>())
                             : raw_value;
      parse_packet(value);
    } catch (const std::exception& error) {
      bool report = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        last_error_ = error.what();
        report = !parse_error_reported_;
        parse_error_reported_ = true;
      }
      if (report) {
        std::cerr << "pico: state JSON parse failed: " << error.what() << "\n";
      }
    }
  }

  void parse_packet(const Json& value) {
    if (!value.is_object()) return;
    InputState next{};
    next.timestamp = std::chrono::steady_clock::now();
    next.sequence = sequence_ + 1;
    next.has_pico = true;
    bool left_pose_valid = false;
    bool right_pose_valid = false;

    if (value.contains("timeStampNs") && value.at("timeStampNs").is_number()) {
      next.pico.device_timestamp_ns = value.at("timeStampNs").get<std::uint64_t>();
    }
    const auto controller = value.find("Controller");
    if (controller != value.end() && controller->is_object()) {
      const auto left = controller->find("left");
      const auto right = controller->find("right");
      if (left != controller->end() && left->is_object()) {
        float item = 0.0F;
        if (number(*left, "axisX", item)) next.axes[0] = std::clamp(item, -1.0F, 1.0F);
        if (number(*left, "axisY", item)) next.axes[1] = std::clamp(item, -1.0F, 1.0F);
        if (number(*left, "trigger", item)) next.pico.left_trigger = std::clamp(item, 0.0F, 1.0F);
        if (number(*left, "grip", item)) next.pico.left_grip = std::clamp(item, 0.0F, 1.0F);
        if (button(*left, "primaryButton")) next.buttons |= xbox_button::kX;
        if (button(*left, "secondaryButton")) next.buttons |= xbox_button::kY;
        if (left->contains("pose")) {
          std::array<float, 7> pose{};
          if (parse_csv(left->at("pose"), pose.size(), pose.data())) {
            const Quat raw_rotation{pose[3], pose[4], pose[5], pose[6]};
            left_pose_valid = std::all_of(pose.begin(), pose.end(),
                                          [](float value) { return std::isfinite(value); }) &&
                              norm(raw_rotation) > 1.0e-8F;
            if (left_pose_valid) {
              next.pico.left_hand_position_world = unity_position_to_robot(
                  {pose[0], pose[1], pose[2]});
              next.pico.left_hand_rotation_local = unity_rotation_to_robot(
                  raw_rotation);
            }
          }
        }
      }
      if (right != controller->end() && right->is_object()) {
        float item = 0.0F;
        if (number(*right, "axisX", item)) next.axes[2] = std::clamp(item, -1.0F, 1.0F);
        if (number(*right, "axisY", item)) next.axes[3] = std::clamp(item, -1.0F, 1.0F);
        if (number(*right, "trigger", item)) next.pico.right_trigger = std::clamp(item, 0.0F, 1.0F);
        if (number(*right, "grip", item)) next.pico.right_grip = std::clamp(item, 0.0F, 1.0F);
        if (button(*right, "primaryButton")) next.buttons |= xbox_button::kA;
        if (button(*right, "secondaryButton")) next.buttons |= xbox_button::kB;
        if (right->contains("pose")) {
          std::array<float, 7> pose{};
          if (parse_csv(right->at("pose"), pose.size(), pose.data())) {
            const Quat raw_rotation{pose[3], pose[4], pose[5], pose[6]};
            right_pose_valid = std::all_of(pose.begin(), pose.end(),
                                           [](float value) { return std::isfinite(value); }) &&
                               norm(raw_rotation) > 1.0e-8F;
            if (right_pose_valid) {
              next.pico.right_hand_position_world = unity_position_to_robot(
                  {pose[0], pose[1], pose[2]});
              next.pico.right_hand_rotation_local = unity_rotation_to_robot(
                  raw_rotation);
            }
          }
        }
      }
    }

    std::array<float, 7> headset_pose{};
    bool headset_valid = false;
    if (value.contains("Head") && value.at("Head").is_object() &&
        value.at("Head").contains("pose")) {
      headset_valid = parse_csv(value.at("Head").at("pose"), headset_pose.size(),
                                headset_pose.data());
    }
    if (headset_valid) {
      const Quat raw_headset_rotation{headset_pose[3], headset_pose[4],
                                      headset_pose[5], headset_pose[6]};
      headset_valid = std::all_of(headset_pose.begin(), headset_pose.end(),
                                  [](float value) { return std::isfinite(value); }) &&
                      norm(raw_headset_rotation) > 1.0e-8F;
    }
    if (headset_valid) {
      const Vec3 headset_world = unity_position_to_robot(
          {headset_pose[0], headset_pose[1], headset_pose[2]});
      const Quat headset_rotation = unity_rotation_to_robot(
          {headset_pose[3], headset_pose[4], headset_pose[5], headset_pose[6]});
      next.pico.headset_heading = yaw(headset_rotation);
      next.pico.heading_valid = finite(headset_world) && finite(headset_rotation) &&
                                norm(headset_rotation) > 1.0e-8F;
      if (next.pico.heading_valid) {
        const Quat inverse_headset_yaw = yaw_quaternion(-next.pico.headset_heading);
        const Vec3 left_delta = {
            next.pico.left_hand_position_world[0] - headset_world[0],
            next.pico.left_hand_position_world[1] - headset_world[1],
            next.pico.left_hand_position_world[2] - headset_world[2]};
        const Vec3 right_delta = {
            next.pico.right_hand_position_world[0] - headset_world[0],
            next.pico.right_hand_position_world[1] - headset_world[1],
            next.pico.right_hand_position_world[2] - headset_world[2]};
        next.pico.left_hand_position_local = rotate(
            quaternion_matrix(inverse_headset_yaw), left_delta);
        next.pico.right_hand_position_local = rotate(
            quaternion_matrix(inverse_headset_yaw), right_delta);
        next.pico.left_hand_rotation_local = normalize(multiply(
            inverse_headset_yaw, next.pico.left_hand_rotation_local));
        next.pico.right_hand_rotation_local = normalize(multiply(
            inverse_headset_yaw, next.pico.right_hand_rotation_local));
        next.pico.pose_valid = left_pose_valid && right_pose_valid &&
                               finite(next.pico.left_hand_position_world) &&
                               finite(next.pico.right_hand_position_world) &&
                               finite(next.pico.left_hand_position_local) &&
                               finite(next.pico.right_hand_position_local);
        next.pico.hand_rotations_valid =
            next.pico.pose_valid && norm(next.pico.left_hand_rotation_local) > 1.0e-8F &&
            norm(next.pico.right_hand_rotation_local) > 1.0e-8F;
      }
    }

    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = next;
    latest_.sequence = ++sequence_;
    received_ = true;
    parse_error_reported_ = false;
    last_error_.clear();
  }

  std::chrono::milliseconds stale_timeout_;
  mutable std::mutex mutex_;
  InputState latest_{};
  std::uint64_t sequence_ = 0;
  std::string device_id_;
  std::string last_error_;
  int lock_fd_ = -1;
  bool received_ = false;
  bool initialized_ = false;
  bool server_connected_ = false;
  bool parse_error_reported_ = false;
};

PicoInputSource::PicoInputSource(std::chrono::milliseconds stale_timeout)
    : impl_(std::make_unique<Impl>(stale_timeout)) {}

PicoInputSource::~PicoInputSource() = default;

bool PicoInputSource::poll(InputState& state) { return impl_->poll(state); }

std::string PicoInputSource::diagnostics() const {
  return impl_->diagnostics();
}

}  // namespace protomotions::deployment
