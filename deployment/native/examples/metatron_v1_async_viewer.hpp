// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "metatron_v1_mujoco_viewer.hpp"

namespace protomotions::deployment::metatron_v1 {

// Rendering owns its own MuJoCo data and GL context. The control thread only
// publishes a small measured-state snapshot, never waits for a GPU draw.
class AsyncMujocoViewer final {
 public:
  AsyncMujocoViewer(std::string xml, std::string glfw_library)
      : thread_([this, xml = std::move(xml), glfw_library = std::move(glfw_library)] {
          try {
            MujocoViewer viewer(xml, glfw_library);
            while (!stop_.load(std::memory_order_relaxed)) {
              RobotState state;
              bool available = false;
              {
                std::lock_guard<std::mutex> lock(mutex_);
                if (has_state_) {
                  state = latest_;
                  available = true;
                }
              }
              if (available && !viewer.render(state)) {
                closed_.store(true, std::memory_order_relaxed);
                break;
              }
              std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
          } catch (const std::exception& error) {
            std::cerr << "MuJoCo viewer failed: " << error.what() << '\n';
            closed_.store(true, std::memory_order_relaxed);
          }
        }) {}

  ~AsyncMujocoViewer() {
    stop_.store(true, std::memory_order_relaxed);
    if (thread_.joinable()) thread_.join();
  }
  AsyncMujocoViewer(const AsyncMujocoViewer&) = delete;
  AsyncMujocoViewer& operator=(const AsyncMujocoViewer&) = delete;

  void publish(const RobotState& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = state;
    has_state_ = true;
  }
  bool closed() const noexcept { return closed_.load(std::memory_order_relaxed); }

 private:
  std::mutex mutex_;
  RobotState latest_;
  bool has_state_ = false;
  std::atomic<bool> stop_{false};
  std::atomic<bool> closed_{false};
  std::thread thread_;
};

}  // namespace protomotions::deployment::metatron_v1
