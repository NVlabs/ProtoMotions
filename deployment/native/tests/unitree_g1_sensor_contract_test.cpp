// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <array>
#include <cmath>
#include <iostream>

#include "protomotions/deployment/unitree_sdk2_backend.hpp"

using protomotions::deployment::g1_pelvis_sensor_from_torso;
using protomotions::deployment::g1_torso_sensor_from_pelvis;
using protomotions::deployment::kG1SdkDofCount;

namespace {

bool near(double actual, double expected, double tolerance = 1.0e-6) {
  return std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
}

template <std::size_t N>
bool near_array(const std::array<double, N>& actual,
                const std::array<double, N>& expected) {
  for (std::size_t index = 0; index < N; ++index) {
    if (!near(actual[index], expected[index])) {
      std::cerr << "mismatch at " << index << ": actual=" << actual[index]
                << " expected=" << expected[index] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  std::array<float, kG1SdkDofCount> position{};
  std::array<float, kG1SdkDofCount> velocity{};
  position[12] = 0.3F;
  position[13] = -0.2F;
  position[14] = 0.4F;
  velocity[12] = 0.7F;
  velocity[13] = -0.5F;
  velocity[14] = 0.25F;
  const std::array<double, 4> pelvis_rotation{
      0.10259783520851541, -0.20519567041703082,
      0.3077935056255462, 0.9233805168766387};
  const std::array<double, 3> pelvis_rate{0.4, -0.3, 0.2};

  // Golden values are from the Python deployment's scipy Rotation reference:
  // pelvis * Rz(waist_yaw) * Rx(waist_roll) * Ry(waist_pitch), with every
  // angular velocity represented in its corresponding local sensor frame.
  const auto torso = g1_torso_sensor_from_pelvis(
      position, velocity, pelvis_rotation, pelvis_rate);
  if (!near_array(torso.rotation_xyzw,
                  std::array<double, 4>{
                      -0.0989168658853, -0.0832910239697,
                      0.406786799431, 0.904324412346}) ||
      !near_array(torso.angular_velocity,
                  std::array<double, 3>{
                      -0.502390921116, -0.325542181730,
                      0.657933115959})) {
    return 1;
  }

  const auto recovered = g1_pelvis_sensor_from_torso(
      position, velocity, torso.rotation_xyzw, torso.angular_velocity);
  if (!near_array(recovered.rotation_xyzw, pelvis_rotation) ||
      !near_array(recovered.angular_velocity, pelvis_rate)) {
    return 1;
  }
  return 0;
}
