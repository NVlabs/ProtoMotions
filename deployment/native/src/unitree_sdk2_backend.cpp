// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/unitree_sdk2_backend.hpp"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <utility>

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/HandCmd_.hpp>
#include <unitree/idl/hg/HandState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>

namespace protomotions::deployment {
namespace {

constexpr std::size_t kG1DofCount = 29;

using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;  // xyzw

Quat quat_mul(const Quat& a, const Quat& b) {
  return {
      a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
      a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
      a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
      a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
  };
}

Quat quat_conjugate(const Quat& q) { return {-q[0], -q[1], -q[2], q[3]}; }

Quat quat_normalize(Quat q) {
  double norm = 0.0;
  for (const double value : q) norm += value * value;
  norm = std::sqrt(norm);
  if (!std::isfinite(norm) || norm <= 1.0e-12) return {0.0, 0.0, 0.0, 1.0};
  for (double& value : q) value /= norm;
  return q;
}

Vec3 quat_rotate(const Quat& q, const Vec3& value) {
  const Quat vector{value[0], value[1], value[2], 0.0};
  const Quat rotated = quat_mul(quat_mul(q, vector), quat_conjugate(q));
  return {rotated[0], rotated[1], rotated[2]};
}

Vec3 quat_to_rpy(const Quat& q) {
  const double sinr_cosp = 2.0 * (q[3] * q[0] + q[1] * q[2]);
  const double cosr_cosp = 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1]);
  const double roll = std::atan2(sinr_cosp, cosr_cosp);
  const double sinp =
      std::clamp(2.0 * (q[3] * q[1] - q[2] * q[0]), -1.0, 1.0);
  const double pitch = std::asin(sinp);
  const double siny_cosp = 2.0 * (q[3] * q[2] + q[0] * q[1]);
  const double cosy_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
  return {roll, pitch, std::atan2(siny_cosp, cosy_cosp)};
}

Quat axis_angle(const Vec3& axis, double angle) {
  const double half = angle * 0.5;
  const double s = std::sin(half);
  return {axis[0] * s, axis[1] * s, axis[2] * s, std::cos(half)};
}

struct WaistTransform {
  Quat rotation{0.0, 0.0, 0.0, 1.0};
  Vec3 angular_velocity{0.0, 0.0, 0.0};
};

WaistTransform waist_torso_transform(
    const std::array<float, kG1DofCount>& q,
    const std::array<float, kG1DofCount>& dq) {
  const Quat yaw = axis_angle({0.0, 0.0, 1.0}, q[12]);
  const Quat roll = axis_angle({1.0, 0.0, 0.0}, q[13]);
  const Quat pitch = axis_angle({0.0, 1.0, 0.0}, q[14]);
  const Quat rotation = quat_normalize(quat_mul(quat_mul(yaw, roll), pitch));
  Vec3 rate{0.0, 0.0, dq[12]};
  const Vec3 roll_component = quat_rotate(yaw, {dq[13], 0.0, 0.0});
  const Vec3 pitch_component =
      quat_rotate(quat_mul(yaw, roll), {0.0, dq[14], 0.0});
  for (int axis = 0; axis < 3; ++axis) {
    rate[axis] += roll_component[axis] + pitch_component[axis];
  }
  // The accumulated joint rate above is pelvis-local. The policy torso IMU
  // contract, and the Python reference implementation, use torso-local rates.
  return {rotation, quat_rotate(quat_conjugate(rotation), rate)};
}

Vec3 torso_anchor_position(const std::array<float, kG1DofCount>& q,
                           const Vec3& root_position,
                           const Quat& root_rotation) {
  // Exact fixed geometry used by the Python adapter:
  // root_position + root_rotation * Rz(waist_yaw) * [-.0039635, 0, .044].
  const Quat yaw = axis_angle({0.0, 0.0, 1.0}, q[12]);
  const Vec3 offset_in_root =
      quat_rotate(yaw, {-0.0039635, 0.0, 0.044});
  return {
      root_position[0] + quat_rotate(root_rotation, offset_in_root)[0],
      root_position[1] + quat_rotate(root_rotation, offset_in_root)[1],
      root_position[2] + quat_rotate(root_rotation, offset_in_root)[2],
  };
}

std::uint32_t crc32(const std::uint32_t* words, std::uint32_t count) {
  std::uint32_t crc = 0xFFFFFFFFU;
  for (std::uint32_t i = 0; i < count; ++i) {
    std::uint32_t bit = 1U << 31U;
    const std::uint32_t data = words[i];
    for (std::uint32_t j = 0; j < 32; ++j) {
      const bool carry = (crc & 0x80000000U) != 0;
      crc <<= 1U;
      if (carry) crc ^= 0x04C11DB7U;
      if ((data & bit) != 0) crc ^= 0x04C11DB7U;
      bit >>= 1U;
    }
  }
  return crc;
}

}  // namespace

G1ImuMeasurement g1_torso_sensor_from_pelvis(
    const std::array<float, kG1SdkDofCount>& dof_position,
    const std::array<float, kG1SdkDofCount>& dof_velocity,
    const std::array<double, 4>& pelvis_rotation_xyzw,
    const std::array<double, 3>& pelvis_angular_velocity) {
  const auto waist = waist_torso_transform(dof_position, dof_velocity);
  const auto pelvis_rate_in_torso = quat_rotate(
      quat_conjugate(waist.rotation), pelvis_angular_velocity);
  Vec3 torso_rate{};
  for (std::size_t axis = 0; axis < torso_rate.size(); ++axis) {
    torso_rate[axis] =
        pelvis_rate_in_torso[axis] + waist.angular_velocity[axis];
  }
  return {
      quat_normalize(quat_mul(quat_normalize(pelvis_rotation_xyzw),
                              waist.rotation)),
      torso_rate};
}

G1ImuMeasurement g1_pelvis_sensor_from_torso(
    const std::array<float, kG1SdkDofCount>& dof_position,
    const std::array<float, kG1SdkDofCount>& dof_velocity,
    const std::array<double, 4>& torso_rotation_xyzw,
    const std::array<double, 3>& torso_angular_velocity) {
  const auto waist = waist_torso_transform(dof_position, dof_velocity);
  Vec3 relative_torso_rate{};
  for (std::size_t axis = 0; axis < relative_torso_rate.size(); ++axis) {
    relative_torso_rate[axis] =
        torso_angular_velocity[axis] - waist.angular_velocity[axis];
  }
  return {
      quat_normalize(quat_mul(quat_normalize(torso_rotation_xyzw),
                              quat_conjugate(waist.rotation))),
      quat_rotate(waist.rotation, relative_torso_rate)};
}

class UnitreeSdk2Backend::Impl {
 public:
  using LowCmd = unitree_hg::msg::dds_::LowCmd_;
  using LowState = unitree_hg::msg::dds_::LowState_;
  using ImuState = unitree_hg::msg::dds_::IMUState_;
  using SportState = unitree_go::msg::dds_::SportModeState_;
  using HandCmd = unitree_hg::msg::dds_::HandCmd_;
  using HandState = unitree_hg::msg::dds_::HandState_;

  struct IntervalWindow {
    static constexpr std::size_t kCapacity = 2048;
    std::array<double, kCapacity> samples{};
    std::size_t count = 0, cursor = 0;
    void observe(std::chrono::steady_clock::time_point now,
                 std::chrono::steady_clock::time_point previous) {
      if (previous.time_since_epoch().count() == 0 || now < previous) return;
      samples[cursor] = std::chrono::duration<double, std::milli>(now - previous).count();
      cursor = (cursor + 1) % kCapacity;
      count = std::min(count + 1, kCapacity);
    }
    std::array<double, 3> percentiles() const {
      if (count == 0) return {-1.0, -1.0, -1.0};
      auto sorted = samples;
      std::sort(sorted.begin(), sorted.begin() + count);
      return {sorted[(count - 1) / 2], sorted[(count - 1) * 99 / 100],
              sorted[count - 1]};
    }
  };

  explicit Impl(UnitreeSdk2BackendConfig config) : config_(std::move(config)) {
    if (config_.network_interface.empty()) {
      throw std::invalid_argument("Unitree SDK2 network_interface is required");
    }
    if ((!config_.use_lowstate_imu || config_.imu_is_torso) &&
        config_.imu_topic.empty()) {
      throw std::invalid_argument(
          "Unitree SDK2 imu_topic is required for secondary IMU mode");
    }
    if (config_.enable_odometry && config_.sport_state_topic.empty()) {
      throw std::invalid_argument(
          "Unitree SDK2 sport_state_topic is required for odometry");
    }
    if ((config_.enable_dex3_hands || config_.damp_unused_dex3_hands) &&
        (config_.left_hand_state_topic.empty() ||
         config_.right_hand_state_topic.empty() ||
         config_.left_hand_command_topic.empty() ||
         config_.right_hand_command_topic.empty())) {
      throw std::invalid_argument(
          "Unitree SDK2 Dex-3 hand topics are required when enabled");
    }
    const auto gain_is_valid = [](double gain) {
      return std::isfinite(gain) && gain >= 0.0 &&
             gain <= std::numeric_limits<float>::max();
    };
    if (!std::isfinite(config_.default_stiffness) ||
        !std::isfinite(config_.default_damping) ||
        !std::isfinite(config_.emergency_damping) ||
        !std::all_of(config_.dex3_stiffness.begin(),
                     config_.dex3_stiffness.end(), gain_is_valid) ||
        !std::all_of(config_.dex3_damping.begin(),
                     config_.dex3_damping.end(), gain_is_valid) ||
        !std::isfinite(config_.dex3_emergency_damping) ||
        config_.default_stiffness < 0.0 || config_.default_damping < 0.0 ||
        config_.emergency_damping < 0.0 || config_.dex3_emergency_damping < 0.0 ||
        config_.default_stiffness > std::numeric_limits<float>::max() ||
        config_.default_damping > std::numeric_limits<float>::max() ||
        config_.emergency_damping > std::numeric_limits<float>::max() ||
        config_.dex3_emergency_damping > std::numeric_limits<float>::max()) {
      throw std::invalid_argument(
          "Unitree SDK2 gains must be finite, non-negative, and float-safe");
    }
    if (!std::isfinite(config_.command_period_seconds) ||
        config_.command_period_seconds < 0.0005 ||
        config_.command_period_seconds > 0.1) {
      throw std::invalid_argument(
          "Unitree SDK2 command_period_seconds must be in [0.0005,0.1]");
    }
    if (config_.max_odometry_age.count() < 0)
      throw std::invalid_argument("max_odometry_age must be non-negative");
    const bool use_lowstate_imu =
        config_.use_lowstate_imu && !config_.imu_is_torso;
    capabilities_.has_default_position_gains = true;
    capabilities_.has_projected_gravity = true;
    capabilities_.has_odometry = config_.enable_odometry;
    capabilities_.default_stiffness = config_.default_stiffness;
    capabilities_.default_damping = config_.default_damping;
    capabilities_.dof_count = kG1DofCount;
    capabilities_.supports_position = true;
    unitree::robot::ChannelFactory::Instance()->Init(
        0, config_.network_interface);
    if (!config_.sensor_only) {
      // Release high-level control only when an actuator backend is requested.
      motion_switcher_ =
          std::make_shared<unitree::robot::b2::MotionSwitcherClient>();
      motion_switcher_->SetTimeout(5.0F);
      motion_switcher_->Init();
      for (int attempt = 0; attempt < 20; ++attempt) {
        std::string form;
        std::string name;
        motion_switcher_->CheckMode(form, name);
        if (name.empty()) break;
        if (motion_switcher_->ReleaseMode() != 0) {
          std::cerr << "warning: SDK2 ReleaseMode failed; retrying\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
      publisher_ = std::make_shared<unitree::robot::ChannelPublisher<LowCmd>>(
          config_.command_topic);
      publisher_->InitChannel();
    }
    subscriber_ = std::make_shared<unitree::robot::ChannelSubscriber<LowState>>(
        config_.state_topic);
    // Queue length one plus overwrite-on-callback gives the default
    // latest-only contract: a slow controller never drains stale LowState
    // samples.
    subscriber_->InitChannel(
        [this](const void* message) { handle_state(message); }, 1);
    if (!use_lowstate_imu) {
      imu_subscriber_ =
          std::make_shared<unitree::robot::ChannelSubscriber<ImuState>>(
              config_.imu_topic);
      imu_subscriber_->InitChannel(
          [this](const void* message) { handle_torso_imu(message); }, 1);
    }
    if (config_.enable_odometry) {
      sport_subscriber_ =
          std::make_shared<unitree::robot::ChannelSubscriber<SportState>>(
              config_.sport_state_topic);
      sport_subscriber_->InitChannel(
          [this](const void* message) { handle_sport_state(message); }, 1);
    }
    if (!config_.sensor_only &&
        (config_.enable_dex3_hands || config_.damp_unused_dex3_hands)) {
      left_hand_subscriber_ =
          std::make_shared<unitree::robot::ChannelSubscriber<HandState>>(
              config_.left_hand_state_topic);
      left_hand_subscriber_->InitChannel(
          [this](const void* message) { handle_left_hand_state(message); }, 1);
      right_hand_subscriber_ =
          std::make_shared<unitree::robot::ChannelSubscriber<HandState>>(
              config_.right_hand_state_topic);
      right_hand_subscriber_->InitChannel(
          [this](const void* message) { handle_right_hand_state(message); }, 1);
      left_hand_publisher_ =
          std::make_shared<unitree::robot::ChannelPublisher<HandCmd>>(
              config_.left_hand_command_topic);
      left_hand_publisher_->InitChannel();
      right_hand_publisher_ =
          std::make_shared<unitree::robot::ChannelPublisher<HandCmd>>(
              config_.right_hand_command_topic);
      right_hand_publisher_->InitChannel();
      std::cout << "Dex-3 hand transport enabled: subscribe "
                << config_.left_hand_state_topic << ", "
                << config_.right_hand_state_topic << "; publish "
                << config_.left_hand_command_topic << ", "
                << config_.right_hand_command_topic << "\n";
    } else {
      std::cout << "Dex-3 hand transport disabled; no hand DDS endpoints opened\n";
    }
    if (!config_.sensor_only)
      command_writer_ = std::thread([this] { low_command_writer_loop(); });
    if (!config_.sensor_only && config_.damp_unused_dex3_hands &&
        !config_.enable_dex3_hands) {
      std::cout << "Unused Dex-3 hands: damping-only; waiting for fresh validated states\n";
      unused_hand_writer_ = std::thread([this] { unused_hand_writer_loop(); });
    }
  }

  ~Impl() {
    command_writer_stop_.store(true, std::memory_order_release);
    if (unused_hand_writer_.joinable()) unused_hand_writer_.join();
    if (command_writer_.joinable()) command_writer_.join();
    // Stop callbacks before releasing the mutex and state storage.
    if (right_hand_subscriber_) right_hand_subscriber_->CloseChannel();
    if (left_hand_subscriber_) left_hand_subscriber_->CloseChannel();
    if (sport_subscriber_) sport_subscriber_->CloseChannel();
    if (imu_subscriber_) imu_subscriber_->CloseChannel();
    if (subscriber_) subscriber_->CloseChannel();
    if (right_hand_publisher_) right_hand_publisher_->CloseChannel();
    if (left_hand_publisher_) left_hand_publisher_->CloseChannel();
    if (publisher_) publisher_->CloseChannel();
    right_hand_subscriber_.reset();
    left_hand_subscriber_.reset();
    sport_subscriber_.reset();
    imu_subscriber_.reset();
    subscriber_.reset();
    right_hand_publisher_.reset();
    left_hand_publisher_.reset();
    publisher_.reset();
    motion_switcher_.reset();
  }

  const RobotCapabilities& capabilities() const noexcept { return capabilities_; }
  UnitreeSensorStatus sensor_status() const {
    UnitreeSensorStatus status;
    IntervalWindow low_intervals, odometry_intervals;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      status = sensor_status_;
      low_intervals = low_intervals_;
      odometry_intervals = odometry_intervals_;
      const auto now = std::chrono::steady_clock::now();
      if (has_state_) status.lowstate_age_ms =
          std::chrono::duration<double, std::milli>(now - timestamp_).count();
      if (has_sport_state_) status.odometry_age_ms =
          std::chrono::duration<double, std::milli>(now - sport_timestamp_).count();
    }
    const auto low = low_intervals.percentiles();
    const auto odometry = odometry_intervals.percentiles();
    status.lowstate_p50_interval_ms = low[0];
    status.lowstate_p99_interval_ms = low[1];
    status.lowstate_max_interval_ms = low[2];
    status.odometry_p50_interval_ms = odometry[0];
    status.odometry_p99_interval_ms = odometry[1];
    status.odometry_max_interval_ms = odometry[2];
    return status;
  }


  bool read_state(RobotState& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool use_lowstate_imu =
        config_.use_lowstate_imu && !config_.imu_is_torso;
    if (!has_state_ || (!use_lowstate_imu && !has_torso_imu_) ||
        (config_.enable_odometry && !has_sport_state_)) {
      return false;
    }
    if (config_.enable_odometry && config_.max_odometry_age.count() > 0 &&
        std::chrono::steady_clock::now() - sport_timestamp_ >
            config_.max_odometry_age) return false;

    std::array<float, kG1DofCount> q{};
    std::array<float, kG1DofCount> dq{};
    for (std::size_t i = 0; i < kG1DofCount; ++i) {
      q[i] = latest_.motor_state()[i].q();
      dq[i] = latest_.motor_state()[i].dq();
    }

    const auto pelvis_quaternion = latest_.imu_state().quaternion();
    const auto pelvis_gyro = latest_.imu_state().gyroscope();
    const Quat root_rotation = quat_normalize(
        {pelvis_quaternion[1], pelvis_quaternion[2],
         pelvis_quaternion[3], pelvis_quaternion[0]});
    const Vec3 root_angular_velocity{
        pelvis_gyro[0], pelvis_gyro[1], pelvis_gyro[2]};

    Quat anchor_rotation = root_rotation;
    Vec3 anchor_angular_velocity = root_angular_velocity;
    Quat carrier_rotation = root_rotation;
    Vec3 carrier_angular_velocity = root_angular_velocity;
    if (use_lowstate_imu) {
      // This synchronized LowState path converts the pelvis IMU into the
      // policy torso anchor exactly once.
      const auto torso = g1_torso_sensor_from_pelvis(
          q, dq, root_rotation, root_angular_velocity);
      anchor_rotation = torso.rotation_xyzw;
      anchor_angular_velocity = torso.angular_velocity;
    } else {
      const auto torso_quaternion = latest_torso_imu_.quaternion();
      const auto torso_gyro = latest_torso_imu_.gyroscope();
      anchor_rotation = quat_normalize(
          {torso_quaternion[1], torso_quaternion[2],
           torso_quaternion[3], torso_quaternion[0]});
      anchor_angular_velocity = {
          torso_gyro[0], torso_gyro[1], torso_gyro[2]};
      const auto pelvis = g1_pelvis_sensor_from_torso(
          q, dq, anchor_rotation, anchor_angular_velocity);
      carrier_rotation = pelvis.rotation_xyzw;
      carrier_angular_velocity = pelvis.angular_velocity;
    }

    Vec3 root_position = config_.root_position_without_odometry;
    Vec3 odometry_velocity{0.0, 0.0, 0.0};
    if (config_.enable_odometry) {
      const auto position = latest_sport_state_.position();
      const auto velocity = latest_sport_state_.velocity();
      root_position = {position[0], position[1], position[2]};
      odometry_velocity = {velocity[0], velocity[1], velocity[2]};
      for (const double value : root_position) {
        if (!std::isfinite(value)) return false;
      }
      for (const double value : odometry_velocity) {
        if (!std::isfinite(value)) return false;
      }
    }

    if (config_.align_born_place) {
      if (!born_place_aligned_) {
        const double yaw = quat_to_rpy(anchor_rotation)[2];
        born_place_rotation_ = axis_angle({0.0, 0.0, 1.0}, -yaw);
        born_place_position_ = config_.enable_odometry
                                   ? root_position
                                   : Vec3{0.0, 0.0, 0.0};
        born_place_position_[2] = 0.0;
        born_place_aligned_ = true;
      }
      if (config_.enable_odometry) {
        const Vec3 relative_position{
            root_position[0] - born_place_position_[0],
            root_position[1] - born_place_position_[1],
            root_position[2] - born_place_position_[2]};
        root_position = quat_rotate(born_place_rotation_, relative_position);
      }
      anchor_rotation = quat_normalize(
          quat_mul(born_place_rotation_, anchor_rotation));
      carrier_rotation = quat_normalize(
          quat_mul(born_place_rotation_, carrier_rotation));
    }
    const Vec3 anchor_position =
        torso_anchor_position(q, root_position, carrier_rotation);

    out.timestamp = timestamp_;
    out.sequence = sequence_;
    if (out.dof_position.size() != kG1DofCount) {
      out.dof_position.resize(kG1DofCount);
    }
    if (out.dof_velocity.size() != kG1DofCount) {
      out.dof_velocity.resize(kG1DofCount);
    }
    if (out.dof_torque.size() != kG1DofCount) {
      out.dof_torque.resize(kG1DofCount);
    }
    for (std::size_t i = 0; i < kG1DofCount; ++i) {
      out.dof_position[i] = q[i];
      out.dof_velocity[i] = dq[i];
      out.dof_torque[i] = latest_.motor_state()[i].tau_est();
    }
    out.motor_temperature.resize(kG1DofCount);
    for (std::size_t i = 0; i < kG1DofCount; ++i) {
      const auto& temperature = latest_.motor_state()[i].temperature();
      out.motor_temperature[i] = {static_cast<double>(temperature[0]),
                                  static_cast<double>(temperature[1])};
    }
    out.base_position = root_position;
    out.root_position = root_position;
    out.anchor_position = anchor_position;
    out.base_orientation_xyzw = anchor_rotation;
    out.root_orientation_xyzw = carrier_rotation;
    out.base_angular_velocity = anchor_angular_velocity;
    out.root_angular_velocity = carrier_angular_velocity;
    out.base_rpy = quat_to_rpy(anchor_rotation);
    out.projected_gravity =
        quat_rotate(quat_conjugate(anchor_rotation), {0.0, 0.0, -1.0});
    out.odometry_velocity = odometry_velocity;
    out.has_odometry = config_.enable_odometry;
    out.has_anchor_position = true;
    out.has_root_kinematics = true;
    out.has_projected_gravity = true;
    const auto& remote = latest_.wireless_remote();
    std::copy_n(remote.begin(),
                std::min(remote.size(), out.operator_input_data.size()),
                out.operator_input_data.begin());
    out.has_operator_input = true;
    return out.finite();
  }

  bool send_command(const ActuatorCommand& command) {
    if (config_.sensor_only) return false;
    const auto sized_or_empty = [](const auto& values) {
      return values.empty() || values.size() == kG1DofCount;
    };
    const auto float_safe = [](const auto& values) {
      for (const double value : values) {
        if (!std::isfinite(value) ||
            value < std::numeric_limits<float>::lowest() ||
            value > std::numeric_limits<float>::max()) {
          return false;
        }
      }
      return true;
    };
    if (command.mode != ControlMode::kPosition ||
        command.values.size() != kG1DofCount ||
        !sized_or_empty(command.stiffness) ||
        !sized_or_empty(command.damping) ||
        !sized_or_empty(command.armature) || !command.finite() ||
        !float_safe(command.values) || !float_safe(command.stiffness) ||
        !float_safe(command.damping) || !float_safe(command.armature)) {
      return false;
    }
    if (command.stiffness.empty() &&
        (!std::isfinite(config_.default_stiffness) ||
         config_.default_stiffness < 0.0)) {
      return false;
    }
    if (command.damping.empty() &&
        (!std::isfinite(config_.default_damping) ||
         config_.default_damping < 0.0)) {
      return false;
    }
    std::uint8_t current_mode_machine = 0;
    {
      std::lock_guard<std::mutex> state_lock(mutex_);
      current_mode_machine = mode_machine_;
    }
    std::lock_guard<std::mutex> command_lock(command_mutex_);
    command_message_.mode_machine() = current_mode_machine;
    command_message_.mode_pr() = 0;
    for (std::size_t i = 0; i < kG1DofCount; ++i) {
      auto& motor = command_message_.motor_cmd().at(i);
      const float stiffness = static_cast<float>(command.stiffness.empty()
          ? config_.default_stiffness : command.stiffness.at(i));
      const float damping = static_cast<float>(command.damping.empty()
          ? config_.default_damping : command.damping.at(i));
      // The SDK packet must carry kp/kd on every cycle, but avoid repeating
      // gain-setting work when the policy emitted the same vectors. A changed
      // policy gain is applied on the next packet; the serialized packet
      // retains the last applied values otherwise.
      if (!gain_cache_valid_ || cached_stiffness_[i] != stiffness) {
        cached_stiffness_[i] = stiffness;
        motor.kp() = stiffness;
      }
      if (!gain_cache_valid_ || cached_damping_[i] != damping) {
        cached_damping_[i] = damping;
        motor.kd() = damping;
      }
      motor.mode() = 1;
      motor.q() = static_cast<float>(command.values[i]);
      motor.dq() = 0.0F;
      motor.tau() = 0.0F;
    }
    gain_cache_valid_ = true;
    command_message_.crc() = crc32(
        reinterpret_cast<const std::uint32_t*>(&command_message_),
        (sizeof(LowCmd) >> 2U) - 1U);
    command_ready_ = true;
    // Publish the new target immediately, then let
    // the recurrent writer hold that exact packet at the low-level cadence.
    return publisher_->Write(command_message_, 0);
  }

  bool emergency_stop() noexcept {
    if (config_.sensor_only) return false;
    // The backend contract is noexcept because this path is called while
    // faulting. Report the transport result so startup/re-arm can refuse to
    // proceed when the hard-stop packet was not accepted.
    try {
      ActuatorCommand stop;
      stop.mode = ControlMode::kPosition;
      stop.values.assign(kG1DofCount, 0.0);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (has_state_) {
          for (std::size_t i = 0; i < kG1DofCount; ++i) {
            const double position = latest_.motor_state()[i].q();
            // Damping is the safety action; a corrupt position sample must
            // not prevent the finite command packet from being sent.
            stop.values[i] = std::isfinite(position) ? position : 0.0;
          }
        }
      }
      stop.stiffness.assign(kG1DofCount, 0.0);
      stop.damping.assign(kG1DofCount, config_.emergency_damping);
      return send_command(stop);
    } catch (...) {
      return false;
    }
  }

  bool dex3_hands_enabled() const noexcept {
    return config_.enable_dex3_hands;
  }

  bool dex3_hands_ready() const noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      return hand_state_ready_locked();
    } catch (...) {
      return false;
    }
  }

  std::string dex3_hand_status() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!config_.enable_dex3_hands && !config_.damp_unused_dex3_hands) return "disabled";
    const auto now = std::chrono::steady_clock::now();
    const auto describe = [now](const auto& state,
                                std::chrono::steady_clock::time_point stamp,
                                const char* name) {
      std::ostringstream result;
      if (!state) {
        result << name << "=missing";
      } else {
        const double age_ms =
            std::chrono::duration<double, std::milli>(now - stamp).count();
        result << name << "="
               << (age_ms <= 250.0 ? "ready" : "stale")
               << "(age_ms=" << static_cast<int>(std::max(0.0, age_ms)) << ")";
      }
      return result.str();
    };
    return describe(latest_left_hand_, left_hand_timestamp_, "left") + "," +
           describe(latest_right_hand_, right_hand_timestamp_, "right") +
           (hand_state_ready_locked() ? ",transport=ready"
                                       : ",transport=stale");
  }

  bool send_dex3_position_command(
      const std::array<float, UnitreeSdk2Backend::kDex3DofCount>& left,
      const std::array<float, UnitreeSdk2Backend::kDex3DofCount>& right) noexcept {
    try {
      if (config_.sensor_only || !config_.enable_dex3_hands ||
          !std::all_of(left.begin(), left.end(),
                       [](float value) { return std::isfinite(value); }) ||
          !std::all_of(right.begin(), right.end(),
                       [](float value) { return std::isfinite(value); })) {
        return false;
      }
      std::lock_guard<std::mutex> lock(mutex_);
      std::array<float, UnitreeSdk2Backend::kDex3DofCount> stiffness{};
      std::array<float, UnitreeSdk2Backend::kDex3DofCount> damping{};
      std::transform(config_.dex3_stiffness.begin(),
                     config_.dex3_stiffness.end(), stiffness.begin(),
                     [](double value) { return static_cast<float>(value); });
      std::transform(config_.dex3_damping.begin(), config_.dex3_damping.end(),
                     damping.begin(),
                     [](double value) { return static_cast<float>(value); });
      return write_dex3_locked(left, right, stiffness, damping);
    } catch (...) {
      return false;
    }
  }

  bool send_dex3_damping_hold() noexcept {
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      if (config_.sensor_only ||
          (!config_.enable_dex3_hands && !config_.damp_unused_dex3_hands) ||
          !hand_state_ready_locked()) return false;
      std::array<float, UnitreeSdk2Backend::kDex3DofCount> left{};
      std::array<float, UnitreeSdk2Backend::kDex3DofCount> right{};
      for (std::size_t i = 0; i < UnitreeSdk2Backend::kDex3DofCount; ++i) {
        left[i] = latest_left_hand_->motor_state()[i].q();
        right[i] = latest_right_hand_->motor_state()[i].q();
      }
      std::array<float, UnitreeSdk2Backend::kDex3DofCount> stiffness{};
      std::array<float, UnitreeSdk2Backend::kDex3DofCount> damping{};
      damping.fill(static_cast<float>(config_.dex3_emergency_damping));
      return write_dex3_locked(left, right, stiffness, damping);
    } catch (...) {
      return false;
    }
  }

 private:
  void unused_hand_writer_loop() noexcept {
    bool was_sent = false;
    while (!command_writer_stop_.load(std::memory_order_acquire)) {
      const bool sent = send_dex3_damping_hold();
      if (sent != was_sent) {
        std::cout << "Unused Dex-3 damping: "
                  << (sent ? "fresh states verified; active" : "state stale or write failed; suspended")
                  << "\n";
        was_sent = sent;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  void low_command_writer_loop() noexcept {
    const auto period =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(config_.command_period_seconds));
    auto next = std::chrono::steady_clock::now() + period;
    while (!command_writer_stop_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_until(next);
      next += period;
      if (std::chrono::steady_clock::now() > next + period) {
        next = std::chrono::steady_clock::now() + period;
      }
      try {
        std::lock_guard<std::mutex> lock(command_mutex_);
        if (command_ready_ && publisher_) {
          (void)publisher_->Write(command_message_, 0);
        }
      } catch (...) {
        // send_command() and the runtime safety watchdog remain responsible
        // for latching transport failures; the hold thread must not terminate
        // the process or throw during backend destruction.
      }
    }
  }

  void handle_state(const void* message) {
    if (message == nullptr) return;
    const auto& state = *static_cast<const LowState*>(message);
    const auto expected_crc = crc32(
        reinterpret_cast<const std::uint32_t*>(&state),
        (sizeof(LowState) >> 2U) - 1U);
    if (state.crc() != expected_crc) {
      std::lock_guard<std::mutex> lock(mutex_);
      ++sensor_status_.lowstate_crc_errors;
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = state;
    mode_machine_ = state.mode_machine();
    const auto now = std::chrono::steady_clock::now();
    low_intervals_.observe(now, timestamp_);
    timestamp_ = now;
    ++sensor_status_.lowstate_packets;
    // LowState.tick identifies the SDK sample. Extend it across the 32-bit
    // wrap so repeated reads of one queued sample do not look like new data,
    // while the runtime can still reject stale/replayed samples by sequence.
    const auto tick = state.tick();
    if (has_tick_ && tick == last_tick_) ++sensor_status_.repeated_ticks;
    if (has_tick_ && tick < last_tick_ &&
        (last_tick_ - tick) < (std::numeric_limits<std::uint32_t>::max() / 2U))
      ++sensor_status_.backward_ticks;
    sensor_status_.last_tick = tick;
    if (has_tick_ && tick < last_tick_ &&
        (last_tick_ - tick) > (std::numeric_limits<std::uint32_t>::max() / 2U)) {
      ++tick_epoch_;
    }
    last_tick_ = tick;
    has_tick_ = true;
    // Reserve sequence zero for an uninitialized sample while preserving
    // monotonic ordering across the 32-bit SDK tick wrap.
    sequence_ = ((tick_epoch_ << 32U) | static_cast<std::uint64_t>(tick)) + 1U;
    has_state_ = true;
  }

  void handle_torso_imu(const void* message) {
    if (message == nullptr) return;
    const auto& imu = *static_cast<const ImuState*>(message);
    std::lock_guard<std::mutex> lock(mutex_);
    latest_torso_imu_ = imu;
    has_torso_imu_ = true;
  }

  void handle_sport_state(const void* message) {
    if (message == nullptr) return;
    const auto& state = *static_cast<const SportState*>(message);
    std::lock_guard<std::mutex> lock(mutex_);
    latest_sport_state_ = state;
    const auto now = std::chrono::steady_clock::now();
    odometry_intervals_.observe(now, sport_timestamp_);
    sport_timestamp_ = now;
    ++sensor_status_.odometry_packets;
    has_sport_state_ = true;
  }

  static bool valid_hand_state(const HandState& state) {
    if (state.motor_state().size() != UnitreeSdk2Backend::kDex3DofCount ||
        state.error().size() < 2 || state.error()[0] != 0 ||
        state.error()[1] != 0) {
      return false;
    }
    for (const auto& motor : state.motor_state()) {
      if (!std::isfinite(motor.q()) || !std::isfinite(motor.dq()) ||
          !std::isfinite(motor.tau_est())) {
        return false;
      }
    }
    return true;
  }

  bool hand_state_ready_locked() const {
    if (!latest_left_hand_ || !latest_right_hand_ ||
        !valid_hand_state(*latest_left_hand_) ||
        !valid_hand_state(*latest_right_hand_)) {
      return false;
    }
    const auto now = std::chrono::steady_clock::now();
    constexpr auto max_age = std::chrono::milliseconds(250);
    return now - left_hand_timestamp_ <= max_age &&
           now - right_hand_timestamp_ <= max_age;
  }

  void handle_left_hand_state(const void* message) {
    if (message == nullptr) return;
    const auto& state = *static_cast<const HandState*>(message);
    if (!valid_hand_state(state)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    latest_left_hand_ = std::make_shared<HandState>(state);
    left_hand_timestamp_ = std::chrono::steady_clock::now();
  }

  void handle_right_hand_state(const void* message) {
    if (message == nullptr) return;
    const auto& state = *static_cast<const HandState*>(message);
    if (!valid_hand_state(state)) return;
    std::lock_guard<std::mutex> lock(mutex_);
    latest_right_hand_ = std::make_shared<HandState>(state);
    right_hand_timestamp_ = std::chrono::steady_clock::now();
  }

  bool write_dex3_locked(
      const std::array<float, UnitreeSdk2Backend::kDex3DofCount>& left,
      const std::array<float, UnitreeSdk2Backend::kDex3DofCount>& right,
      const std::array<float, UnitreeSdk2Backend::kDex3DofCount>& stiffness,
      const std::array<float, UnitreeSdk2Backend::kDex3DofCount>& damping) {
    if (!hand_state_ready_locked() || !left_hand_publisher_ ||
        !right_hand_publisher_) {
      return false;
    }
    HandCmd left_command;
    HandCmd right_command;
    left_command.motor_cmd().resize(UnitreeSdk2Backend::kDex3DofCount);
    right_command.motor_cmd().resize(UnitreeSdk2Backend::kDex3DofCount);
    for (std::size_t i = 0; i < UnitreeSdk2Backend::kDex3DofCount; ++i) {
      auto& left_motor = left_command.motor_cmd().at(i);
      left_motor.mode() = dex3_position_motor_mode();
      left_motor.q() = left[i];
      left_motor.dq() = 0.0F;
      left_motor.tau() = 0.0F;
      left_motor.kp() = stiffness[i];
      left_motor.kd() = damping[i];
      auto& right_motor = right_command.motor_cmd().at(i);
      right_motor.mode() = dex3_position_motor_mode();
      right_motor.q() = right[i];
      right_motor.dq() = 0.0F;
      right_motor.tau() = 0.0F;
      right_motor.kp() = stiffness[i];
      right_motor.kd() = damping[i];
    }
    const bool left_written = left_hand_publisher_->Write(left_command, 0);
    const bool right_written = right_hand_publisher_->Write(right_command, 0);
    return left_written && right_written;
  }

  UnitreeSdk2BackendConfig config_;
  RobotCapabilities capabilities_{};
  mutable std::mutex mutex_;
  std::mutex command_mutex_;
  LowState latest_{};
  ImuState latest_torso_imu_{};
  SportState latest_sport_state_{};
  std::shared_ptr<HandState> latest_left_hand_;
  std::shared_ptr<HandState> latest_right_hand_;
  std::chrono::steady_clock::time_point left_hand_timestamp_{};
  std::chrono::steady_clock::time_point right_hand_timestamp_{};
  LowCmd command_message_{};
  std::chrono::steady_clock::time_point timestamp_{};
  std::chrono::steady_clock::time_point sport_timestamp_{};
  UnitreeSensorStatus sensor_status_{};
  IntervalWindow low_intervals_{};
  IntervalWindow odometry_intervals_{};
  std::uint64_t sequence_ = 0;
  std::uint64_t tick_epoch_ = 0;
  std::uint32_t last_tick_ = 0;
  std::uint8_t mode_machine_ = 0;
  bool has_state_ = false;
  bool has_torso_imu_ = false;
  bool has_sport_state_ = false;
  bool has_tick_ = false;
  bool born_place_aligned_ = false;
  Quat born_place_rotation_{0.0, 0.0, 0.0, 1.0};
  Vec3 born_place_position_{};
  std::array<float, kG1DofCount> cached_stiffness_{};
  std::array<float, kG1DofCount> cached_damping_{};
  bool gain_cache_valid_ = false;
  bool command_ready_ = false;
  std::atomic<bool> command_writer_stop_{false};
  std::thread command_writer_;
  std::thread unused_hand_writer_;
  // Keep SDK channels last so they are destroyed first. The subscriber can
  // still deliver a final callback while its destructor runs; all callback
  // state and its mutex must remain alive until that teardown completes.
  std::shared_ptr<unitree::robot::b2::MotionSwitcherClient> motion_switcher_;
  std::shared_ptr<unitree::robot::ChannelPublisher<LowCmd>> publisher_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<LowState>> subscriber_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<ImuState>> imu_subscriber_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<SportState>> sport_subscriber_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<HandState>> left_hand_subscriber_;
  std::shared_ptr<unitree::robot::ChannelSubscriber<HandState>> right_hand_subscriber_;
  std::shared_ptr<unitree::robot::ChannelPublisher<HandCmd>> left_hand_publisher_;
  std::shared_ptr<unitree::robot::ChannelPublisher<HandCmd>> right_hand_publisher_;
};

UnitreeSdk2Backend::UnitreeSdk2Backend(UnitreeSdk2BackendConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

UnitreeSdk2Backend::~UnitreeSdk2Backend() = default;

const RobotCapabilities& UnitreeSdk2Backend::capabilities() const noexcept {
  return impl_->capabilities();
}

bool UnitreeSdk2Backend::read_state(RobotState& state) {
  return impl_->read_state(state);
}

UnitreeSensorStatus UnitreeSdk2Backend::sensor_status() const {
  return impl_->sensor_status();
}

bool UnitreeSdk2Backend::send_command(const ActuatorCommand& command) {
  return impl_->send_command(command);
}

bool UnitreeSdk2Backend::reset_anchor_frame() {
  // SDK2 owns transport state, not the policy's world-frame anchor. Claiming
  // success here would re-arm a policy with an unchanged anchor.
  return false;
}

bool UnitreeSdk2Backend::emergency_stop() noexcept {
  return impl_->emergency_stop();
}

bool UnitreeSdk2Backend::dex3_hands_enabled() const noexcept {
  return impl_->dex3_hands_enabled();
}

bool UnitreeSdk2Backend::dex3_hands_ready() const noexcept {
  return impl_->dex3_hands_ready();
}

std::string UnitreeSdk2Backend::dex3_hand_status() const {
  return impl_->dex3_hand_status();
}

bool UnitreeSdk2Backend::send_dex3_position_command(
    const std::array<float, kDex3DofCount>& left,
    const std::array<float, kDex3DofCount>& right) noexcept {
  return impl_->send_dex3_position_command(left, right);
}

bool UnitreeSdk2Backend::send_dex3_damping_hold() noexcept {
  return impl_->send_dex3_damping_hold();
}

}  // namespace protomotions::deployment
