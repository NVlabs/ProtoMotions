// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/pico_input.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

namespace {

using namespace protomotions::deployment;

void print_vec(const std::array<float, 3>& value) {
  std::cout << '[' << value[0] << ',' << value[1] << ',' << value[2] << ']';
}

}  // namespace

int main() {
  try {
    PicoInputSource source;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(30);
    InputState state{};
    std::uint64_t last_sequence = 0;
    std::cout << "waiting for native Pico/XRoboToolkit packets...\n";
    while (std::chrono::steady_clock::now() < deadline) {
      if (source.poll(state) && state.sequence != last_sequence) {
        last_sequence = state.sequence;
        std::cout << std::fixed << std::setprecision(3)
                  << "seq=" << state.sequence << " buttons=0x" << std::hex
                  << state.buttons << std::dec << " axes=[" << state.axes[0]
                  << ',' << state.axes[1] << ',' << state.axes[2] << ','
                  << state.axes[3] << "] trigger=["
                  << state.pico.left_trigger << ','
                  << state.pico.right_trigger << "] grip=["
                  << state.pico.left_grip << ',' << state.pico.right_grip
                  << "] heading_valid="
                  << state.pico.heading_valid << " pose_valid="
                  << state.pico.pose_valid << " heading="
                  << state.pico.headset_heading << " left=";
        print_vec(state.pico.left_hand_position_world);
        std::cout << " right=";
        print_vec(state.pico.right_hand_position_world);
        std::cout << '\n';
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return last_sequence == 0 ? 2 : 0;
  } catch (const std::exception& error) {
    std::cerr << "pico_input_demo: " << error.what() << '\n';
    return 1;
  }
}
