// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>

#include "protomotions/deployment/metatron_v1_fk_engine.hpp"

namespace protomotions::deployment::metatron_v1 {

// The torso_link origin in the pelvis frame with every waist joint at zero
// (g1 MJCF: waist_roll_link pos="-0.0039635 0 0.044", torso_link no offset;
// the torso's rotation then equals the pelvis's). The FK takes the anchor
// pose, so the rest pose and the task dump place the standing anchor here,
// which keeps the pelvis at the origin as before.
inline constexpr std::array<float, 3> kTorsoOffsetInPelvis = {-0.0039635F, 0.0F, 0.044F};

// One startup-only calibrated pose, shared by the MuJoCo and SDK launchers.
// The default joint values match ModelBridge::_standing_pose; the relaxed arms
// match the Metatron v1 offline reference in tools/metatron_v1_reference.py.
struct RestPose {
  std::array<float, 33 * 3> positions{};
  std::array<float, 33 * 4> rotations{};
  std::array<float, 225> style{};
  float upright_height = 0.0F;
};

RestPose make_rest_pose(FkEngine& fk);

}  // namespace protomotions::deployment::metatron_v1
