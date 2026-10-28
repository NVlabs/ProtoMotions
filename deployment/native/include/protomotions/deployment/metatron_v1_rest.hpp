// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>

#include "protomotions/deployment/metatron_v1_fk_engine.hpp"

namespace protomotions::deployment::metatron_v1 {

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
