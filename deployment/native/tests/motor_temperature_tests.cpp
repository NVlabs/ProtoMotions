// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/motor_temperature.hpp"
#include "protomotions/deployment/native_time_series_recorder.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <chrono>

using namespace protomotions::deployment;
void require(bool ok) { if (!ok) throw std::runtime_error("temperature regression failed"); }

int main() {
  MotorTemperatureMonitor monitor;
  std::vector<std::array<double, 2>> t(29, {30.0, 31.0});
  require(monitor.update(t, 0.0).empty());
  t[3][1] = 90.0;  // Driver alone triggers; names use hardware order.
  auto warning = monitor.update(t, 0.1);
  require(warning.find("left_knee") != std::string::npos);
  require(warning.find("driver=90C") != std::string::npos);
  require(monitor.update(t, 0.5).empty());
  require(!monitor.update(t, 1.1).empty());
  t[3][1] = 85.0;
  require(monitor.update(t, 2.2).find("HighTemp") != std::string::npos);
  require(monitor.update({}, 3.3).find("unavailable") != std::string::npos);
  require(monitor.update(t, 4.4).find("HighTemp") != std::string::npos);
  t[3][1] = std::numeric_limits<double>::quiet_NaN();
  require(monitor.update(t, 5.5).find("invalid") != std::string::npos);
  t[3][1] = 84.9;
  require(monitor.update(t, 5.6).find("cleared") != std::string::npos);
  require(monitor.update(t, 6.7).empty());
  t[9][0] = 91.0;
  require(monitor.update(t, 6.8).find("right_knee") != std::string::npos);
  MotorTemperatureMonitor custom(70.0, 60.0);
  require(custom.update(t, 0.0).find("HighTemp") != std::string::npos);
  bool rejected = false;
  try { MotorTemperatureMonitor bad(80.0, 90.0); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected);

  // Real recorder round trip: schema version, raw temperatures, validity.
  const auto path = std::filesystem::temp_directory_path() /
      ("pm-temperature-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  NativeTimeSeriesRecord record;
  record.motor_temperature[3] = {81.0F, 92.0F};
  record.motor_temperature_valid = 1;
  { NativeTimeSeriesRecorder recorder(path.string()); require(recorder.append(record)); recorder.close(); }
  std::ifstream stream(path, std::ios::binary);
  std::array<char, 8> magic;
  std::array<std::uint32_t, 4> header;
  stream.read(magic.data(), magic.size());
  stream.read(reinterpret_cast<char*>(header.data()), sizeof(header));
  require(header[0] == 2 && header[1] == sizeof(NativeTimeSeriesRecord));
  NativeTimeSeriesRecord loaded;
  stream.read(reinterpret_cast<char*>(&loaded), sizeof(loaded));
  require(stream.good());
  require(loaded.motor_temperature_valid == 1 && loaded.motor_temperature[3][1] == 92.0F);
  stream.close();
  std::filesystem::remove(path);
}
