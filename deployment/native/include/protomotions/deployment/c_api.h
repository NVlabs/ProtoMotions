// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Small C ABI for applications that keep policy semantics in Python or
// another language while using the deployment runtime's C++ lifecycle and
// safety path. The callbacks are synchronous and run on the caller's control
// thread; they must not retain the supplied buffers.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pm_runtime pm_runtime_t;

// Bump when callback struct layouts or required runtime semantics change.
#define PROTOMOTIONS_DEPLOYMENT_C_ABI_VERSION 2

typedef struct pm_state_output {
  double* dof_position;
  double* dof_velocity;
  double* base_position;          // length 3
  // xyzw policy-anchor orientation; for G1 this is the direct torso IMU.
  double* base_orientation_xyzw; // length 4
  double* projected_gravity;     // length 3
  uint8_t* operator_input_data;
  size_t operator_input_data_size;  // writable capacity in bytes
  const float* vision_data;      // callback-owned optional payload
  size_t vision_size;
  const float* lidar_data;       // callback-owned optional payload
  size_t lidar_size;
  size_t dof_count;
  int has_odometry;
  int has_projected_gravity;
  int has_vision;
  int has_lidar;
  // Monotonic steady-clock timestamp in nanoseconds. The callback must fill
  // this from the sensor sample; zero is rejected so completion time cannot
  // masquerade as sensor freshness.
  uint64_t timestamp_ns;
} pm_state_output_t;

typedef struct pm_state_view {
  const double* dof_position;
  const double* dof_velocity;
  const double* base_position;
  // xyzw policy-anchor orientation; for G1 this is the direct torso IMU.
  const double* base_orientation_xyzw;
  const double* projected_gravity;
  const uint8_t* operator_input_data;
  size_t operator_input_data_size;
  const float* vision_data;
  size_t vision_size;
  const float* lidar_data;
  size_t lidar_size;
  size_t dof_count;
  uint64_t sequence;
  int has_odometry;
  int has_projected_gravity;
  int has_vision;
  int has_lidar;
} pm_state_view_t;

typedef struct pm_command_output {
  double* values;
  double* stiffness;
  double* damping;
  double* armature;
  size_t dof_count;
  int has_stiffness;
  int has_damping;
  int has_armature;
} pm_command_output_t;

typedef int (*pm_read_state_fn)(void* user, pm_state_output_t* state,
                                uint64_t* sequence);
typedef int (*pm_step_controller_fn)(void* user, const pm_state_view_t* state,
                                     pm_command_output_t* output);
typedef int (*pm_send_command_fn)(void* user, const double* values,
                                  const double* stiffness,
                                  const double* damping,
                                  const double* armature,
                                  size_t dof_count);
typedef int (*pm_reset_fn)(void* user);
typedef int (*pm_stop_fn)(void* user);

typedef struct pm_runtime_callbacks {
  pm_read_state_fn read_state;
  pm_send_command_fn send_command;
  pm_reset_fn reset_anchor_frame;
  pm_stop_fn emergency_stop;
  void* user;
} pm_runtime_callbacks_t;

int pm_runtime_abi_version(void);

// Creates a position-command runtime. The C++ safety core validates finite
// state/commands, dimensions, freshness, cycle timing, and failures before
// invoking send_command. `max_abs_command` and `max_cycle_interval_ms` are
// required safety limits and must be positive finite values. Gain limits are
// also required when a controller supplies stiffness or damping arrays. The
// callbacks own no buffers: they fill the writable state/command views for one
// cycle.
pm_runtime_t* pm_runtime_create(size_t dof_count,
                                pm_runtime_callbacks_t callbacks,
                                double max_abs_command,
                                int max_cycle_interval_ms,
                                double max_abs_stiffness,
                                double max_abs_damping);
void pm_runtime_destroy(pm_runtime_t* runtime);

// Registers one named controller. A controller callback may call into the
// host policy, must fill exactly dof_count finite position targets, and may
// set the optional gain flags after filling the corresponding per-DOF arrays.
int pm_runtime_add_controller(pm_runtime_t* runtime, const char* name,
                              pm_step_controller_fn step, void* user);
int pm_runtime_start(pm_runtime_t* runtime);
// Returns zero if the backend rejects the safe stop command; such a failure
// leaves the runtime faulted rather than reporting a clean halt.
int pm_runtime_stop(pm_runtime_t* runtime);
int pm_runtime_reset_anchor_frame(pm_runtime_t* runtime);
int pm_runtime_request_controller(pm_runtime_t* runtime, const char* name);
// Returns non-zero when either a controller command or a safe damping/hold
// command was accepted. A faulted runtime keeps sending the safe command until
// revive; callers can inspect pm_runtime_lifecycle() to distinguish the two.
int pm_runtime_run_cycle(pm_runtime_t* runtime);
// Borrowed string, valid until the next runtime fault or destruction.
const char* pm_runtime_last_fault_reason(const pm_runtime_t* runtime);
void pm_runtime_emergency_stop(pm_runtime_t* runtime);
int pm_runtime_revive(pm_runtime_t* runtime);

// Returns the last state-ready -> command-ready duration in nanoseconds.
uint64_t pm_runtime_last_state_to_command_ns(const pm_runtime_t* runtime);
int pm_runtime_lifecycle(const pm_runtime_t* runtime);

#ifdef __cplusplus
}
#endif
