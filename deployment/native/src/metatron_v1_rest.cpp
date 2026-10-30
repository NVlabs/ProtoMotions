// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_rest.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace protomotions::deployment::metatron_v1 {
namespace {
constexpr std::array<int, 5> kConditionable = {16, 13, 7, 24, 32};

std::array<float, 3> rotate(const float* q, const std::array<float, 3>& v) {
  const std::array<float, 3> u = {q[0], q[1], q[2]};
  const float dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
  const float uu = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
  const std::array<float, 3> cross = {
      u[1] * v[2] - u[2] * v[1],
      u[2] * v[0] - u[0] * v[2],
      u[0] * v[1] - u[1] * v[0]};
  return {(q[3] * q[3] - uu) * v[0] + 2 * dot * u[0] + 2 * q[3] * cross[0],
          (q[3] * q[3] - uu) * v[1] + 2 * dot * u[1] + 2 * q[3] * cross[1],
          (q[3] * q[3] - uu) * v[2] + 2 * dot * u[2] + 2 * q[3] * cross[2]};
}
}  // namespace

RestPose make_rest_pose(FkEngine& fk) {
  SensorReading state;
  // MJCF-calibrated standing pose, in the pinned manifest's actuator order.
  state.dof_pos[0] = state.dof_pos[6] = -0.312F;
  state.dof_pos[3] = state.dof_pos[9] = 0.669F;
  state.dof_pos[4] = state.dof_pos[10] = -0.363F;
  state.dof_pos[15] = state.dof_pos[22] = 0.0F;
  state.dof_pos[16] = 0.2F;
  state.dof_pos[23] = -0.2F;
  state.dof_pos[18] = state.dof_pos[25] = 1.4F;
  // The FK is anchored on the torso: give it the standing torso pose over a
  // pelvis at the origin (waist joints are zero in this pose).
  state.anchor_pos = kTorsoOffsetInPelvis;
  state.anchor_rot = {0, 0, 0, 1};
  const RecoveryPose pose = fk.run(state);
  RestPose rest;
  rest.positions = pose.positions;
  rest.rotations = pose.rotations;
  float floor = rest.positions[2];
  for (int body = 1; body < 33; ++body)
    floor = std::min(floor, rest.positions[body * 3 + 2]);
  for (int body = 0; body < 33; ++body)
    rest.positions[body * 3 + 2] -= floor;
  rest.upright_height = rest.positions[16 * 3 + 2];
  if (!std::isfinite(rest.upright_height) || rest.upright_height <= 0.0F)
    throw std::runtime_error("invalid Metatron calibrated upright height");

  // ProtoMotions v5 style features: per frame, five selected bodies each
  // contribute local XY, floor-relative Z, tangent and normal (9 values).
  // The static template has zero torso yaw, so its heading inverse is identity.
  const float* torso = rest.positions.data() + 16 * 3;
  for (int frame = 0; frame < 5; ++frame) {
    for (int selected = 0; selected < 5; ++selected) {
      const int body = kConditionable[selected];
      const float* p = rest.positions.data() + body * 3;
      const float* q = rest.rotations.data() + body * 4;
      const std::size_t base = static_cast<std::size_t>(frame * 5 + selected) * 9;
      rest.style[base] = p[0] - torso[0];
      rest.style[base + 1] = p[1] - torso[1];
      rest.style[base + 2] = p[2];
      const auto tangent = rotate(q, {1.0F, 0.0F, 0.0F});
      const auto normal = rotate(q, {0.0F, 0.0F, 1.0F});
      for (int axis = 0; axis < 3; ++axis) {
        rest.style[base + 3 + axis] = tangent[axis];
        rest.style[base + 6 + axis] = normal[axis];
      }
    }
  }
  return rest;
}

}  // namespace protomotions::deployment::metatron_v1
