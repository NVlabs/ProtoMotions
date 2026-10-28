// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_contract.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace protomotions::deployment::metatron_v1 {
namespace {
template <std::size_t N>
void copy_to(std::vector<float>& output, const std::array<float, N>& input,
             std::size_t offset = 0) {
  std::copy(input.begin(), input.end(), output.begin() + offset);
}

bool finite(const SensorReading& state) {
  const auto all_finite = [](const auto& values) {
    return std::all_of(values.begin(), values.end(), [](float value) {
      return std::isfinite(value);
    });
  };
  return all_finite(state.dof_pos) && all_finite(state.dof_vel) &&
         all_finite(state.root_local_ang_vel) && all_finite(state.anchor_rot) &&
         all_finite(state.anchor_pos) && all_finite(state.root_pos) &&
         all_finite(state.root_rot);
}
}  // namespace

Inputs::Inputs() {
  for (std::size_t index = 0; index < kInputCount; ++index) {
    if (kInputs[index].is_bool) {
      bool_values_[index].resize(kInputs[index].count());
    } else {
      float_values_[index].resize(kInputs[index].count());
    }
  }
  reset();
}

void Inputs::reset() noexcept {
  for (auto& state : states_) state = {};
  for (auto& action : actions_) action.fill(0.0F);
  for (auto& values : float_values_) std::fill(values.begin(), values.end(), 0.0F);
  for (auto& values : bool_values_) std::fill(values.begin(), values.end(), 0U);
  for (std::size_t frame = 0; frame < 5 * 33; ++frame) {
    float_values_[6][frame * 4 + 3] = 1.0F;
    float_values_[11][frame * 4 + 3] = 1.0F;
  }
  // Hidden scene has a finite identity placeholder, but no visible object.
  float_values_[34][3] = 1.0F;
  initialized_ = false;
  prepared_ = false;
}

void Inputs::prepare(const SensorReading& reading) {
  if (!finite(reading)) throw std::invalid_argument("non-finite scene sensor reading");
  if (!initialized_) {
    states_.fill(reading);
    initialized_ = true;
  }
  copy_to(float_values_[0], reading.root_pos);
  copy_to(float_values_[1], reading.root_rot);
  copy_to(float_values_[20], reading.anchor_pos);
  copy_to(float_values_[21], reading.anchor_rot);
  copy_to(float_values_[22], reading.dof_pos);
  copy_to(float_values_[23], reading.dof_vel);
  copy_to(float_values_[24], reading.root_local_ang_vel);
  for (std::size_t row = 0; row < kHistoryCount; ++row) {
    const auto& previous = states_[row];
    copy_to(float_values_[25], previous.anchor_rot, row * 4);
    copy_to(float_values_[26], previous.dof_pos, row * kDofCount);
    copy_to(float_values_[27], previous.dof_vel, row * kDofCount);
    copy_to(float_values_[29], previous.root_local_ang_vel, row * 3);
    copy_to(float_values_[28], actions_[row + 1], row * kDofCount);
  }
  for (std::size_t row = 0; row < kActionHistoryCount; ++row) {
    copy_to(float_values_[30], actions_[row], row * kDofCount);
  }
  prepared_ = true;
}

void Inputs::commit(const SensorReading& before_step,
                    const std::array<float, kDofCount>& accepted_raw_action) {
  if (!prepared_) throw std::logic_error("cannot commit before preparing inputs");
  if (!finite(before_step) ||
      !std::all_of(accepted_raw_action.begin(), accepted_raw_action.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("non-finite accepted scene command");
  }
  for (std::size_t row = kHistoryCount - 1; row > 0; --row) {
    states_[row] = states_[row - 1];
  }
  states_[0] = before_step;
  for (std::size_t row = kActionHistoryCount - 1; row > 0; --row) {
    actions_[row] = actions_[row - 1];
  }
  actions_[0] = accepted_raw_action;
  prepared_ = false;
}

std::vector<float>& Inputs::floats(std::size_t index) {
  if (index >= kInputCount || kInputs[index].is_bool)
    throw std::out_of_range("not a scene float input");
  return float_values_[index];
}
const std::vector<float>& Inputs::floats(std::size_t index) const {
  if (index >= kInputCount || kInputs[index].is_bool)
    throw std::out_of_range("not a scene float input");
  return float_values_[index];
}
std::vector<std::uint8_t>& Inputs::booleans(std::size_t index) {
  if (index >= kInputCount || !kInputs[index].is_bool)
    throw std::out_of_range("not a scene boolean input");
  return bool_values_[index];
}
const std::vector<std::uint8_t>& Inputs::booleans(std::size_t index) const {
  if (index >= kInputCount || !kInputs[index].is_bool)
    throw std::out_of_range("not a scene boolean input");
  return bool_values_[index];
}
}  // namespace protomotions::deployment::metatron_v1
