// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/motion_library.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace {

// A one-body bundle in the browser demo's layout: two frames 20 ms apart, the
// body moving from x = 0 to x = 2 and turning 90 degrees about z.
nlohmann::json bundle(int frames = 2, float dt = 0.02F, int bodies = 1) {
  nlohmann::json motion;
  motion["frameCount"] = frames;
  motion["frameDtSeconds"] = dt;
  nlohmann::json positions = nlohmann::json::array(), rotations = nlohmann::json::array();
  for (int f = 0; f < frames; ++f) {
    for (int b = 0; b < bodies; ++b) {
      positions.insert(positions.end(), {2.0F * f, 0.0F, 0.0F});
      const float half = f == 0 ? 0.0F : 0.25F * 3.14159265F;
      rotations.insert(rotations.end(), {0.0F, 0.0F, std::sin(half), std::cos(half)});
    }
  }
  motion["positions"] = positions;
  motion["rotations"] = rotations;
  return {{"version", 1}, {"bodyCount", bodies}, {"anchorBodyIndex", 0},
          {"conditionableBodyIds", {0}}, {"motions", {{"slide", motion}}}};
}

void write(const std::filesystem::path& path, const nlohmann::json& json) {
  std::ofstream(path) << json;
}

void expect_rejected(const std::filesystem::path& path, const nlohmann::json& json) {
  write(path, json);
  bool rejected = false;
  try {
    protomotions::deployment::MotionLibrary library(path.string());
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path path(argv[1]);
  write(path, bundle());
  protomotions::deployment::MotionLibrary library(path.string());
  assert(library.motion_count() == 1 && library.names()[0] == "slide");
  assert(library.index_of("slide") == 0 && !library.index_of("missing"));
  assert(library.frame_count(0) == 2 && library.body_count() == 1 && library.anchor_body_index() == 0);
  assert(std::abs(library.motion_length(0) - 0.02F) < 1e-6F);
  std::array<float, 3> position{};
  std::array<float, 4> rotation{};
  library.sample(0, 0.01F, position, rotation);
  assert(std::abs(position[0] - 1.0F) < 1e-5F);          // linear in position
  assert(std::abs(rotation[2] - std::sin(0.125F * 3.14159265F)) < 1e-4F);  // slerp halfway: 45 degrees
  library.sample(0, 5.0F, position, rotation);
  assert(std::abs(position[0] - 2.0F) < 1e-5F);          // clamped to the last frame
  library.sample(0, -1.0F, position, rotation);
  assert(position[0] == 0.0F);

  auto bad = bundle();
  bad["version"] = 2;
  expect_rejected(path, bad);
  bad = bundle(1);
  expect_rejected(path, bad);  // one frame is not a motion
  bad = bundle();
  bad["motions"]["slide"]["positions"].push_back(1.0F);
  expect_rejected(path, bad);  // wrong length
  bad = bundle();
  bad["motions"]["slide"]["frameDtSeconds"] = 0.0F;
  expect_rejected(path, bad);
  bad = bundle();
  bad["conditionableBodyIds"] = {3};
  expect_rejected(path, bad);  // body id outside the bundle
  std::filesystem::remove(path);
  return 0;
}
