// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/native_time_series_recorder.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace protomotions::deployment {
namespace {

constexpr std::array<char, 8> kMagic = {'P', 'M', 'R', 'E', 'C', '1', '\0', '\0'};
constexpr std::uint32_t kVersion = 2;

struct FileHeader {
  std::array<char, 8> magic = kMagic;
  std::uint32_t version = kVersion;
  std::uint32_t record_size = sizeof(NativeTimeSeriesRecord);
  std::uint32_t dof_count = kNativeRecordDofs;
  std::uint32_t reserved = 0;
};

}  // namespace

NativeTimeSeriesRecorder::NativeTimeSeriesRecorder(const std::string& path,
                                                   std::size_t capacity)
    : ring_(std::max<std::size_t>(capacity, 2)), capacity_(ring_.size()) {
  stream_.open(path, std::ios::binary | std::ios::trunc);
  if (!stream_) throw std::runtime_error("cannot open native recorder: " + path);
  const FileHeader header;
  stream_.write(reinterpret_cast<const char*>(&header), sizeof(header));
  if (!stream_) throw std::runtime_error("cannot write native recorder header");
  writer_ = std::thread(&NativeTimeSeriesRecorder::writer_loop, this);
}

NativeTimeSeriesRecorder::~NativeTimeSeriesRecorder() { close(); }

bool NativeTimeSeriesRecorder::append(const NativeTimeSeriesRecord& record) noexcept {
  const auto write = write_index_.load(std::memory_order_relaxed);
  const auto read = read_index_.load(std::memory_order_acquire);
  if (write - read >= capacity_) {
    dropped_count_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  ring_[write % capacity_] = record;
  write_index_.store(write + 1, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(wake_mutex_);
    wake_requested_ = true;
  }
  wake_condition_.notify_one();
  return true;
}

void NativeTimeSeriesRecorder::close() noexcept {
  {
    std::lock_guard<std::mutex> lock(wake_mutex_);
    if (stopping_) return;
    stopping_ = true;
    wake_requested_ = true;
  }
  wake_condition_.notify_one();
  if (writer_.joinable()) writer_.join();
  if (stream_.is_open()) stream_.close();
}

void NativeTimeSeriesRecorder::writer_loop() noexcept {
  while (true) {
    std::unique_lock<std::mutex> lock(wake_mutex_);
    wake_condition_.wait(lock, [this] {
      return wake_requested_ || stopping_ ||
             read_index_.load(std::memory_order_acquire) <
                 write_index_.load(std::memory_order_acquire);
    });
    wake_requested_ = false;
    const bool stopping = stopping_;
    lock.unlock();

    while (true) {
      const auto read = read_index_.load(std::memory_order_relaxed);
      const auto write = write_index_.load(std::memory_order_acquire);
      if (read >= write) break;
      const auto& record = ring_[read % capacity_];
      stream_.write(reinterpret_cast<const char*>(&record), sizeof(record));
      read_index_.store(read + 1, std::memory_order_release);
      if (!stream_) {
        // Disk errors must never reach the policy thread. Drain the ring and
        // leave the file in a detectable truncated state.
        read_index_.store(write, std::memory_order_release);
        break;
      }
    }
    stream_.flush();
    if (stopping && read_index_.load(std::memory_order_acquire) >=
                        write_index_.load(std::memory_order_acquire)) {
      break;
    }
  }
}

}  // namespace protomotions::deployment
