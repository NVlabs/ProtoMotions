// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

struct mjModel_;

namespace protomotions::deployment {

// Load a robot-only MJCF for kinematic use. If the source references scene
// contacts (for example a floor geom supplied by a training environment), the
// fallback removes only the contact section and resolves meshdir relative to
// the source file. The source XML is never modified.
mjModel_* load_kinematics_model(const std::string& xml_path,
                                std::string* load_note = nullptr);

}  // namespace protomotions::deployment
