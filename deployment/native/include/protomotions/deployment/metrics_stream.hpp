// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace protomotions::deployment {

// One control cycle's named scalars and UTF-8 text metadata. Producers (the launcher, a task, a
// backend) append what they know; names are "group/name" so a viewer can lay
// them out without a schema. Vectors are appended as name/0, name/1, ...
class MetricsFrame {
 public:
  void reset(std::uint64_t timestamp_ns, std::uint64_t cycle) {
    values_.clear();
    text_values_.clear();
    timestamp_ns_ = timestamp_ns;
    cycle_ = cycle;
  }
  void set(std::string name, float value) { values_.emplace_back(std::move(name), value); }
  void set_text(std::string name, std::string value) {
    text_values_.emplace_back(std::move(name), std::move(value));
  }
  template <class Range>
  void set_vector(const std::string& prefix, const Range& values) {
    std::size_t index = 0;
    for (const auto& value : values) {
      values_.emplace_back(prefix + "/" + std::to_string(index++), static_cast<float>(value));
    }
  }
  std::uint64_t timestamp_ns() const noexcept { return timestamp_ns_; }
  std::uint64_t cycle() const noexcept { return cycle_; }
  const std::vector<std::pair<std::string, float>>& values() const noexcept { return values_; }

  // One JSON object per frame; NaN becomes null so the line stays valid JSON.
  std::string to_json() const {
    std::string out;
    out.reserve(64 + (values_.size() + text_values_.size()) * 24);
    out += "{\"t_ns\":" + std::to_string(timestamp_ns_) + ",\"cycle\":" + std::to_string(cycle_);
    char buffer[32];
    for (const auto& [name, value] : values_) {
      out += ',';
      append_json_string(out, name);
      out += ':';
      if (std::isfinite(value)) {
        std::snprintf(buffer, sizeof(buffer), "%.6g", static_cast<double>(value));
        out += buffer;
      } else {
        out += "null";
      }
    }
    for (const auto& [name, value] : text_values_) {
      out += ',';
      append_json_string(out, name);
      out += ':';
      append_json_string(out, value);
    }
    out += "}";
    return out;
  }

 private:
  static void append_json_string(std::string& out, const std::string& value) {
    constexpr char hex[] = "0123456789abcdef";
    out += '"';
    for (const unsigned char ch : value) {
      switch (ch) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        default:
          if (ch < 0x20) {
            // Includes embedded NUL, newlines and every other JSON control byte.
            out += "\\u00";
            out += hex[ch >> 4];
            out += hex[ch & 0x0f];
          } else {
            out += static_cast<char>(ch);  // Preserve UTF-8 bytes unchanged.
          }
      }
    }
    out += '"';
  }

  std::vector<std::pair<std::string, float>> values_;
  std::vector<std::pair<std::string, std::string>> text_values_;
  std::uint64_t timestamp_ns_ = 0;
  std::uint64_t cycle_ = 0;
};

// Publishes frames as JSON lines over UDP (for the live viewer) and/or to a
// file (for the report). Nothing here may stall the 50 Hz control thread:
// the UDP send is non-blocking and a full socket buffer drops the datagram,
// and the file is written by a worker thread fed through a bounded queue that
// drops the oldest frame when the disk falls behind. A viewer that hangs, a
// network that disappears or a disk that stalls therefore costs frames of
// telemetry, never a control cycle.
class MetricsPublisher {
 public:
  MetricsPublisher() = default;
  MetricsPublisher(const MetricsPublisher&) = delete;
  MetricsPublisher& operator=(const MetricsPublisher&) = delete;
  ~MetricsPublisher() { close(); }

  void open_udp(const std::string& host, std::uint16_t port) {
    socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_ < 0) throw std::runtime_error("metrics: cannot open UDP socket");
    address_ = {};
    address_.sin_family = AF_INET;
    address_.sin_port = htons(port);
    if (::inet_pton(AF_INET, host.c_str(), &address_.sin_addr) != 1)
      throw std::invalid_argument("metrics: UDP host must be a dotted IPv4 address");
  }

  void open_file(const std::string& path, std::size_t queue_capacity = 2048) {
    file_.open(path, std::ios::out | std::ios::trunc);
    if (!file_) throw std::runtime_error("metrics: cannot open " + path);
    queue_capacity_ = queue_capacity;
    stop_ = false;
    writer_ = std::thread([this] { write_loop(); });
  }

  bool enabled() const noexcept { return socket_ >= 0 || file_.is_open(); }

  // Called on the control thread. Cost: one string build, one non-blocking
  // sendto, one bounded queue push under a mutex the writer holds only briefly.
  void publish(const MetricsFrame& frame) {
    if (!enabled()) return;
    std::string line = frame.to_json();
    if (socket_ >= 0) {
      const auto sent = ::sendto(socket_, line.data(), line.size(), MSG_DONTWAIT,
                                 reinterpret_cast<const sockaddr*>(&address_), sizeof(address_));
      if (sent < 0) ++udp_dropped_;
    }
    if (file_.is_open()) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= queue_capacity_) {
          queue_.pop_front();
          ++file_dropped_;
        }
        queue_.push_back(std::move(line));
      }
      ready_.notify_one();
    }
  }

  std::uint64_t dropped() const noexcept { return udp_dropped_ + file_dropped_; }
  std::uint64_t udp_dropped() const noexcept { return udp_dropped_; }
  std::uint64_t file_dropped() const noexcept { return file_dropped_; }

  // Flushes the queue to disk and closes everything; safe to call twice.
  void close() {
    if (writer_.joinable()) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
      }
      ready_.notify_all();
      writer_.join();
    }
    if (file_.is_open()) file_.close();
    if (socket_ >= 0) {
      ::close(socket_);
      socket_ = -1;
    }
  }

 private:
  void write_loop() {
    std::deque<std::string> batch;
    for (;;) {
      {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        batch.swap(queue_);
        if (batch.empty() && stop_) return;
      }
      for (const auto& line : batch) file_ << line << '\n';
      file_.flush();
      batch.clear();
    }
  }

  int socket_ = -1;
  sockaddr_in address_{};
  std::ofstream file_;
  std::thread writer_;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<std::string> queue_;
  std::size_t queue_capacity_ = 2048;
  bool stop_ = false;
  std::atomic<std::uint64_t> udp_dropped_{0};
  std::uint64_t file_dropped_ = 0;
};

}  // namespace protomotions::deployment
