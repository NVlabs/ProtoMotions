// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pm_xbox_input pm_xbox_input_t;

typedef struct pm_xbox_sample {
  uint64_t sequence;
  uint64_t timestamp_ns;
  uint32_t buttons;
  float axes[8];
  int connected;
} pm_xbox_sample_t;

// The returned object owns the device file and is safe to poll from one
// control thread. A null return means the device could not be opened.
pm_xbox_input_t* pm_xbox_evdev_create(const char* device_path,
                                       const char* name);
void pm_xbox_input_destroy(pm_xbox_input_t* input);

// Returns 1 for a current sample, 0 when the adapter has no sample yet, and
// -1 after a disconnect or malformed source. Callers should apply their own
// bounded source timeout before continuing actuation.
int pm_xbox_input_poll(pm_xbox_input_t* input, pm_xbox_sample_t* sample);

#ifdef __cplusplus
}
#endif
