// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>

// Subscribe-only SDK2 diagnostic. Never releases MotionSwitcher or opens a writer.
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

#include "protomotions/deployment/unitree_dds_config.hpp"
#include "protomotions/deployment/unitree_sdk2_backend.hpp"

int main(int argc, char** argv) {
  try {
    std::string net_if;
    int participant_index = 0;
    int seconds = 15;
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + flag);
      const std::string value = argv[++i];
      if (flag == "--net-if") net_if = value;
      else if (flag == "--participant-index") participant_index = std::stoi(value);
      else if (flag == "--seconds") seconds = std::stoi(value);
      else throw std::invalid_argument("unknown option " + flag);
    }
    if (net_if.empty() || seconds < 1 || seconds > 120)
      throw std::invalid_argument("usage: unitree_sensor_probe --net-if IFACE [--participant-index 0] [--seconds 15]");
    protomotions::deployment::configure_unitree_dds(net_if, participant_index);
    protomotions::deployment::UnitreeSdk2BackendConfig config;
    config.network_interface = net_if;
    config.sensor_only = true;
    config.enable_odometry = true;
    config.max_odometry_age = std::chrono::milliseconds(100);
    config.use_lowstate_imu = true;
    config.imu_is_torso = false;
    protomotions::deployment::UnitreeSdk2Backend backend(config);
    std::cout << "sensor-only probe: no MotionSwitcher release, LowCmd/Dex-3 publisher, or motor command\n";
    auto previous = backend.sensor_status();
    bool sustained = true;
    double worst_low_gap_ms = 0.0, worst_odometry_gap_ms = 0.0;
    for (int i = 0; i < seconds; ++i) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      const auto s = backend.sensor_status();
      if (i >= 2 && (s.lowstate_packets - previous.lowstate_packets < 100 ||
                     s.odometry_packets - previous.odometry_packets < 30))
        sustained = false;
      worst_low_gap_ms = std::max(worst_low_gap_ms, s.lowstate_max_interval_ms);
      worst_odometry_gap_ms = std::max(worst_odometry_gap_ms,
                                       s.odometry_max_interval_ms);
      std::cout << "second=" << i + 1
                << " lowstate_hz=" << s.lowstate_packets - previous.lowstate_packets
                << " lowstate_age_ms=" << s.lowstate_age_ms
                << " lowstate_interval_ms(p50/p99/max)="
                << s.lowstate_p50_interval_ms << '/' << s.lowstate_p99_interval_ms
                << '/' << s.lowstate_max_interval_ms
                << " odometry_hz=" << s.odometry_packets - previous.odometry_packets
                << " odometry_age_ms=" << s.odometry_age_ms
                << " odometry_interval_ms(p50/p99/max)="
                << s.odometry_p50_interval_ms << '/' << s.odometry_p99_interval_ms
                << '/' << s.odometry_max_interval_ms
                << " tick=" << s.last_tick
                << " repeated_ticks=" << s.repeated_ticks
                << " backward_ticks=" << s.backward_ticks
                << " crc_errors=" << s.lowstate_crc_errors << '\n';
      previous = s;
    }
    const auto s = backend.sensor_status();
    const bool ready = sustained &&
        s.lowstate_packets >= static_cast<std::uint64_t>(seconds * 100) &&
        s.odometry_packets >= static_cast<std::uint64_t>(seconds * 30) &&
        s.lowstate_age_ms >= 0 && s.lowstate_age_ms < 100 &&
        s.odometry_age_ms >= 0 && s.odometry_age_ms < 100 &&
        s.lowstate_max_interval_ms >= 0 && worst_low_gap_ms < 60 &&
        s.odometry_max_interval_ms >= 0 && worst_odometry_gap_ms < 100 &&
        s.lowstate_crc_errors == 0 && s.backward_ticks == 0;
    std::cout << "sensor-only result=" << (ready ? "fresh" : "NOT READY") << '\n';
    return ready ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "sensor-only probe failed: " << error.what() << '\n';
    return 1;
  }
}
