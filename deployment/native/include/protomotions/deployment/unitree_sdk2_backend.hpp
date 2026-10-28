// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

#include "protomotions/deployment/backend.hpp"

namespace protomotions::deployment {

inline constexpr std::size_t kDex3DofCount = 7;
inline constexpr std::size_t kG1SdkDofCount = 29;

// Pure, SDK-independent G1 IMU conversion used by the native backend and its
// Python-parity regression test. Quaternions are deployment-order xyzw and
// angular velocities are expressed in the corresponding local body frame.
struct G1ImuMeasurement {
  std::array<double, 4> rotation_xyzw{0.0, 0.0, 0.0, 1.0};
  std::array<double, 3> angular_velocity{};
};

G1ImuMeasurement g1_torso_sensor_from_pelvis(
    const std::array<float, kG1SdkDofCount>& dof_position,
    const std::array<float, kG1SdkDofCount>& dof_velocity,
    const std::array<double, 4>& pelvis_rotation_xyzw,
    const std::array<double, 3>& pelvis_angular_velocity);

G1ImuMeasurement g1_pelvis_sensor_from_torso(
    const std::array<float, kG1SdkDofCount>& dof_position,
    const std::array<float, kG1SdkDofCount>& dof_velocity,
    const std::array<double, 4>& torso_rotation_xyzw,
    const std::array<double, 3>& torso_angular_velocity);

// Dex-3 HandCmd uses mode=1 to enable position control. The motor index is the
// element's position in motor_cmd; it must not be encoded into the mode byte.
inline constexpr std::uint8_t dex3_position_motor_mode() noexcept { return 1U; }

struct UnitreeSdk2BackendConfig {
  std::string network_interface;
  // Diagnostics only: subscribe without MotionSwitcher, publishers, or writers.
  bool sensor_only = false;
  // Zero disables this optional cutoff; Metatron v1 opts into 100 ms.
  std::chrono::milliseconds max_odometry_age{0};
  std::string state_topic = "rt/lowstate";
  std::string command_topic = "rt/lowcmd";
  std::string imu_topic = "rt/secondary_imu";
  std::string sport_state_topic = "rt/odommodestate";
  double default_stiffness = 40.0;
  double default_damping = 1.0;
  // LowState's synchronized IMU is the default, matching the qualified
  // Python path. It is a pelvis/root sample and is converted once
  // into the torso-anchor contract below. Set false only when intentionally
  // qualifying the asynchronous secondary_imu stream.
  bool imu_is_torso = false;
  bool enable_odometry = false;
  bool use_lowstate_imu = true;
  // Match the Python real-robot adapter: remove the initial torso yaw and,
  // when odometry is active, the initial planar position. Local angular
  // velocities are unchanged by this world-frame alignment.
  bool align_born_place = true;
  // Without SportModeState the robot has no measured root translation. The
  // launcher supplies the MJCF qpos0 carrier so native and Python paths feed
  // the same anchor height instead of silently using a zero-height root.
  std::array<double, 3> root_position_without_odometry{};
  // Policy targets arrive at 50 Hz, while the SDK transport repeats the most
  // recent valid LowCmd at the G1's 500 Hz low-level cadence. This matches the
  // deployment-local Python binding and avoids a backend-dependent hold rate.
  double command_period_seconds = 0.002;
  // Safe-stop damping is a backend safety action and must be qualified
  // separately from normal policy gain caps.
  double emergency_damping = 8.0;
  // Interactive control and unused-hand damping require separate permission.
  bool enable_dex3_hands = false;
  bool damp_unused_dex3_hands = false;
  std::string left_hand_state_topic = "rt/dex3/left/state";
  std::string right_hand_state_topic = "rt/dex3/right/state";
  std::string left_hand_command_topic = "rt/dex3/left/cmd";
  std::string right_hand_command_topic = "rt/dex3/right/cmd";
  // Sonic WBC interactive gains. Safe damping is configured separately.
  std::array<double, kDex3DofCount> dex3_stiffness = {
      2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  std::array<double, kDex3DofCount> dex3_damping = {
      0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
  double dex3_emergency_damping = 0.1;
};
struct UnitreeSensorStatus {
  std::uint64_t lowstate_packets = 0;
  std::uint64_t lowstate_crc_errors = 0;
  std::uint64_t odometry_packets = 0;
  std::uint64_t repeated_ticks = 0;
  std::uint64_t backward_ticks = 0;
  std::uint64_t last_tick = 0;
  double lowstate_age_ms = -1.0;  // negative means never received
  double odometry_age_ms = -1.0;
  double lowstate_p50_interval_ms = -1.0;
  double lowstate_p99_interval_ms = -1.0;
  double lowstate_max_interval_ms = -1.0;
  double odometry_p50_interval_ms = -1.0;
  double odometry_p99_interval_ms = -1.0;
  double odometry_max_interval_ms = -1.0;
};


// Optional Unitree SDK2 adapter. The SDK headers and library stay out of the
// core runtime; enable PROTOMOTIONS_BUILD_UNITREE_SDK2 to build this target.
// The adapter owns only DDS state/command transport and maps SDK data into the
// deployment RobotState/ActuatorCommand contract.
class UnitreeSdk2Backend final : public RobotBackend {
 public:
  explicit UnitreeSdk2Backend(UnitreeSdk2BackendConfig config);
  ~UnitreeSdk2Backend() override;

  UnitreeSdk2Backend(const UnitreeSdk2Backend&) = delete;
  UnitreeSdk2Backend& operator=(const UnitreeSdk2Backend&) = delete;

  const RobotCapabilities& capabilities() const noexcept override;
  bool read_state(RobotState& state) override;
  bool send_command(const ActuatorCommand& command) override;
  bool reset_anchor_frame() override;
  bool emergency_stop() noexcept override;

  static constexpr std::size_t kDex3DofCount =
      protomotions::deployment::kDex3DofCount;
  bool dex3_hands_enabled() const noexcept;
  bool dex3_hands_ready() const noexcept;
  std::string dex3_hand_status() const;
  bool send_dex3_position_command(
      const std::array<float, kDex3DofCount>& left,
      const std::array<float, kDex3DofCount>& right) noexcept;
  bool send_dex3_damping_hold() noexcept;

  UnitreeSensorStatus sensor_status() const;
 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace protomotions::deployment
