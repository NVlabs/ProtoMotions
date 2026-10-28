// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "protomotions/deployment/input_c_api.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <new>
#include <string>

#include "protomotions/deployment/evdev_xbox_input.hpp"

struct pm_xbox_input {
  std::unique_ptr<protomotions::deployment::EvdevXboxInputSource> source;
};

extern "C" pm_xbox_input_t* pm_xbox_evdev_create(const char* device_path,
                                                  const char* name) {
  if (device_path == nullptr || device_path[0] == '\0') return nullptr;
  try {
    auto input = std::make_unique<pm_xbox_input>();
    input->source = std::make_unique<protomotions::deployment::EvdevXboxInputSource>(
        device_path, name == nullptr || name[0] == '\0' ? "xbox" : name);
    return input.release();
  } catch (...) {
    return nullptr;
  }
}

extern "C" void pm_xbox_input_destroy(pm_xbox_input_t* input) {
  delete input;
}

extern "C" int pm_xbox_input_poll(pm_xbox_input_t* input,
                                   pm_xbox_sample_t* sample) {
  if (input == nullptr || input->source == nullptr || sample == nullptr) {
    return -1;
  }
  protomotions::deployment::InputState state;
  try {
    if (!input->source->poll(state) || !state.connected) return -1;
  } catch (...) {
    return -1;
  }
  sample->sequence = state.sequence;
  sample->timestamp_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          state.timestamp.time_since_epoch())
          .count());
  sample->buttons = state.buttons;
  std::memcpy(sample->axes, state.axes.data(), sizeof(sample->axes));
  sample->connected = state.connected ? 1 : 0;
  return 1;
}
