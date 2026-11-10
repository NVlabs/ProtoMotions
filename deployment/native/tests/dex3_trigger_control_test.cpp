// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <limits>

#include "protomotions/deployment/dex3_trigger_control.hpp"

using protomotions::deployment::Dex3TriggerControl;

int main() {
  Dex3TriggerControl fingers;
  assert(fingers.target(0) == Dex3TriggerControl::kOpen);
  assert(fingers.target(1) == Dex3TriggerControl::kOpenRight);
  fingers.update(1.0F, 0.0F);
  assert(fingers.target(0) == Dex3TriggerControl::kCloseLeft);
  assert(fingers.target(1) == Dex3TriggerControl::kOpenRight);
  fingers.update(0.0F, 1.0F);
  assert(fingers.target(0) == Dex3TriggerControl::kOpen);
  assert(fingers.target(1) == Dex3TriggerControl::kCloseRight);
  fingers.update(0.0F, 0.0F);
  assert(fingers.target(0) == Dex3TriggerControl::kOpen);
  assert(fingers.target(1) == Dex3TriggerControl::kOpenRight);
  fingers.update(std::numeric_limits<float>::quiet_NaN(), 1.0F);
  assert(fingers.target(0) == Dex3TriggerControl::kOpen);
  assert(fingers.target(1) == Dex3TriggerControl::kCloseRight);
  fingers.reset();
  assert(fingers.target(0) == Dex3TriggerControl::kOpen);
  assert(fingers.target(1) == Dex3TriggerControl::kOpenRight);
}
