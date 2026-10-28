// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace protomotions::deployment {

constexpr std::size_t kNativeRecordDofs = 29;

// Fixed-size record written by the native control loop. The file header stores
// this structure size and version, so a replay tool can reject incompatible
// logs instead of guessing field offsets.
struct NativeTimeSeriesRecord {
  std::uint64_t timestamp_ns = 0;
  std::uint64_t sequence = 0;
  std::uint32_t buttons = 0;
  std::array<float, 8> axes{};
  std::uint8_t mode = 0;
  std::uint8_t armed = 0;
  std::uint8_t sent = 0;
  std::uint8_t reserved = 0;
  std::array<float, kNativeRecordDofs> q{};
  std::array<float, kNativeRecordDofs> dq{};
  std::array<float, kNativeRecordDofs> action{};
  std::array<float, kNativeRecordDofs> raw_action{};
  std::array<float, kNativeRecordDofs> stiffness{};
  std::array<float, kNativeRecordDofs> damping{};
  std::array<float, 3> root_position{};
  std::array<float, 3> anchor_position{};
  std::array<float, 4> root_orientation_xyzw{0.0F, 0.0F, 0.0F, 1.0F};
  std::array<float, 4> anchor_orientation_xyzw{0.0F, 0.0F, 0.0F, 1.0F};
  std::array<float, 3> projected_gravity{0.0F, 0.0F, -1.0F};
  std::array<float, 3> odometry_goal{};
  float odometry_distance = 0.0F;
  // Version 2 extension. Validity distinguishes missing data from 0 Celsius.
  std::array<std::array<float, 2>, kNativeRecordDofs> motor_temperature{};
  std::uint8_t motor_temperature_valid = 0;
};

class NativeTimeSeriesRecorder final {
 public:
  explicit NativeTimeSeriesRecorder(const std::string& path,
                                    std::size_t capacity = 4096);
  ~NativeTimeSeriesRecorder();

  NativeTimeSeriesRecorder(const NativeTimeSeriesRecorder&) = delete;
  NativeTimeSeriesRecorder& operator=(const NativeTimeSeriesRecorder&) = delete;

  // Single-producer, single-consumer bounded queue. A full queue drops the
  // record and increments dropped_count; it never blocks the policy thread.
  bool append(const NativeTimeSeriesRecord& record) noexcept;
  void close() noexcept;
  std::uint64_t dropped_count() const noexcept {
    return dropped_count_.load(std::memory_order_relaxed);
  }

 private:
  void writer_loop() noexcept;

  std::vector<NativeTimeSeriesRecord> ring_;
  const std::size_t capacity_;
  std::atomic<std::uint64_t> write_index_{0};
  std::atomic<std::uint64_t> read_index_{0};
  std::atomic<std::uint64_t> dropped_count_{0};
  std::mutex wake_mutex_;
  std::condition_variable wake_condition_;
  bool wake_requested_ = false;
  bool stopping_ = false;
  std::ofstream stream_;
  std::thread writer_;
};

}  // namespace protomotions::deployment
