// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstring>

#include "protomotions/deployment/c_api.h"

namespace {

std::uint64_t source_sequence = 0;

int read_state(void*, pm_state_output_t* state, std::uint64_t* sequence) {
  if (state == nullptr || sequence == nullptr || state->dof_position == nullptr ||
      state->dof_velocity == nullptr || state->base_orientation_xyzw == nullptr ||
      state->projected_gravity == nullptr || state->operator_input_data == nullptr ||
      state->operator_input_data_size != 40) {
    return 0;
  }
  for (std::size_t i = 0; i < state->dof_count; ++i) {
    state->dof_position[i] = 0.0;
    state->dof_velocity[i] = 0.0;
  }
  state->base_orientation_xyzw[3] = 1.0;
  state->projected_gravity[2] = -1.0;
  state->has_projected_gravity = 1;
  state->timestamp_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
  static const float vision[] = {1.0F, 2.0F};
  state->vision_data = vision;
  state->vision_size = 2;
  state->has_vision = 1;
  *sequence = ++source_sequence;
  return 1;
}

int send_command(void*, const double* values, const double* stiffness,
                 const double* damping, const double* armature,
                 std::size_t count) {
  if (!values || !stiffness || !damping || armature || count != 2) return 0;
  return stiffness[0] == 40.0 && damping[0] == 1.0;
}
int reset_anchor(void*) {
  source_sequence = 0;
  return 1;
}
int emergency_stop(void*) { return 1; }
int step(void*, const pm_state_view_t* state, pm_command_output_t* output) {
  if (state == nullptr || output == nullptr || output->values == nullptr ||
      output->stiffness == nullptr || output->damping == nullptr ||
      output->armature == nullptr || state->dof_count != output->dof_count ||
      state->has_projected_gravity != 1 || state->projected_gravity == nullptr ||
      state->projected_gravity[2] != -1.0 || state->has_vision != 1 ||
      state->vision_data == nullptr || state->vision_size != 2 ||
      state->vision_data[1] != 2.0F) {
    return 0;
  }
  for (std::size_t i = 0; i < output->dof_count; ++i) {
    output->values[i] = 0.0;
    output->stiffness[i] = 40.0;
    output->damping[i] = 1.0;
    output->armature[i] = 0.0;
  }
  output->has_stiffness = 1;
  output->has_damping = 1;
  return 1;
}

}  // namespace

int main() {
  const auto require = [](bool condition, const char* message) {
    if (condition) return true;
    std::fprintf(stderr, "c_api_smoke: %s\n", message);
    return false;
  };
  if (!require(pm_runtime_abi_version() == PROTOMOTIONS_DEPLOYMENT_C_ABI_VERSION,
               "ABI version mismatch")) {
    return 1;
  }
  pm_runtime_callbacks_t callbacks{read_state, send_command, reset_anchor,
                                   emergency_stop, nullptr};
  pm_runtime_t* runtime = pm_runtime_create(2, callbacks, 1.0, 100, 100.0, 20.0);
  if (!require(runtime != nullptr, "create failed")) return 1;
  if (!require(pm_runtime_add_controller(runtime, "idle", step, nullptr),
               "add controller failed")) {
    pm_runtime_destroy(runtime);
    return 1;
  }
  if (!require(pm_runtime_start(runtime), "start failed") ||
      !require(pm_runtime_reset_anchor_frame(runtime), "reset failed") ||
      !require(pm_runtime_request_controller(runtime, "idle"),
               "request failed") ||
      !require(pm_runtime_run_cycle(runtime), "cycle failed") ||
      !require(pm_runtime_last_state_to_command_ns(runtime) > 0,
               "timing was not recorded")) {
    pm_runtime_destroy(runtime);
    return 1;
  }
  if (!require(pm_runtime_reset_anchor_frame(runtime), "second reset failed") ||
      !require(!pm_runtime_run_cycle(runtime),
               "reset should clear the active controller") ||
      !require(pm_runtime_request_controller(runtime, "idle"),
               "re-request after reset failed") ||
      !require(pm_runtime_run_cycle(runtime), "cycle after re-request failed")) {
    pm_runtime_destroy(runtime);
    return 1;
  }
  pm_runtime_emergency_stop(runtime);
  if (!require(std::strcmp(pm_runtime_last_fault_reason(runtime),
                           "operator emergency stop") == 0,
               "fault reason missing from C ABI")) {
    pm_runtime_destroy(runtime);
    return 1;
  }
  pm_runtime_stop(runtime);
  pm_runtime_destroy(runtime);
  return 0;
}
