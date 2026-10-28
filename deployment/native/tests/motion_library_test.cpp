// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/motion_library.hpp"

#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

template <typename T>
void write_value(std::ofstream& stream, T value) {
  stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_motion(const std::filesystem::path& path, std::uint64_t start,
                  std::uint32_t frames) {
  std::ofstream stream(path, std::ios::binary);
  const std::array<char, 8> magic = {'P', 'M', 'M', 'O', 'T', '4', '\0', '\0'};
  stream.write(magic.data(), magic.size());
  write_value<std::uint32_t>(stream, 1);  // Version.
  write_value<std::uint32_t>(stream, 1);  // Bodies.
  write_value<std::uint32_t>(stream, 0);  // DOFs.
  write_value<std::uint32_t>(stream, 1);  // Motions.
  write_value<std::uint64_t>(stream, 2);  // Total frames.
  write_value(stream, start);
  write_value(stream, frames);
  write_value<float>(stream, 0.02F);
  for (float value : {0.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F}) {
    write_value(stream, value);
  }
  for (int frame = 0; frame < 2; ++frame) {
    for (float value : {0.0F, 0.0F, 0.0F, 1.0F}) write_value(stream, value);
  }
  assert(stream.good());
}

void expect_invalid(const std::filesystem::path& path, std::uint64_t start,
                    std::uint32_t frames) {
  write_motion(path, start, frames);
  protomotions::deployment::MotionLibrary library;
  bool rejected = false;
  try {
    library.load(path.string());
  } catch (const std::runtime_error& error) {
    rejected = std::string(error.what()) == "invalid motion index entry";
  }
  assert(rejected);
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::filesystem::path path(argv[1]);
  write_motion(path, 0, 2);
  protomotions::deployment::MotionLibrary library;
  library.load(path.string());
  assert(library.frame_count(0) == 2);
  std::array<float, 3> position{};
  std::array<float, 4> rotation{};
  library.sample(0, 0.01F, position, rotation);
  assert(position[0] == 1.0F);
  assert(rotation[3] == 1.0F);
  library.sample(0, 0.02F, position, rotation);
  assert(position[0] == 2.0F);

  expect_invalid(path, 0, 0);
  expect_invalid(path, 0, 3);
  expect_invalid(path, 2, 1);
  expect_invalid(path, 3, 1);
  // UINT64_MAX + 3 wraps to 2, matching total_frames in the old check.
  expect_invalid(path, std::numeric_limits<std::uint64_t>::max(), 3);
  std::filesystem::remove(path);
}
