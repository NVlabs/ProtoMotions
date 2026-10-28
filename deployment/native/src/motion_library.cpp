// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/motion_library.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace protomotions::deployment {
namespace {

constexpr std::array<char, 8> kMagic = {'P', 'M', 'M', 'O', 'T', '4', '\0', '\0'};
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kMaxMotions = 100000;
constexpr std::size_t kMaxBodies = 4096;
constexpr std::size_t kMaxFrames = 100000000;

template <typename T>
T read_value(std::ifstream& stream, const char* field) {
  T value{};
  stream.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!stream) throw std::runtime_error(std::string("motion binary truncated at ") + field);
  return value;
}

void read_floats(std::ifstream& stream, std::vector<float>& values,
                 const char* field) {
  if (values.empty()) return;
  stream.read(reinterpret_cast<char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
  if (!stream) throw std::runtime_error(std::string("motion binary truncated at ") + field);
}

struct Quat {
  float x;
  float y;
  float z;
  float w;
};

Quat normalize(Quat q) {
  const float norm = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (!std::isfinite(norm) || norm <= 1.0e-8F) return {0.0F, 0.0F, 0.0F, 1.0F};
  q.x /= norm;
  q.y /= norm;
  q.z /= norm;
  q.w /= norm;
  return q;
}

Quat slerp(Quat a, Quat b, float alpha) {
  a = normalize(a);
  b = normalize(b);
  float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  if (dot < 0.0F) {
    dot = -dot;
    b = {-b.x, -b.y, -b.z, -b.w};
  }
  if (dot > 0.9995F) {
    return normalize({a.x + alpha * (b.x - a.x),
                      a.y + alpha * (b.y - a.y),
                      a.z + alpha * (b.z - a.z),
                      a.w + alpha * (b.w - a.w)});
  }
  const float theta = std::acos(std::clamp(dot, -1.0F, 1.0F));
  const float sin_theta = std::sin(theta);
  if (std::abs(sin_theta) <= 1.0e-8F) return a;
  const float aw = std::sin((1.0F - alpha) * theta) / sin_theta;
  const float bw = std::sin(alpha * theta) / sin_theta;
  return normalize({aw * a.x + bw * b.x, aw * a.y + bw * b.y,
                    aw * a.z + bw * b.z, aw * a.w + bw * b.w});
}

}  // namespace

void MotionLibrary::load(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open motion binary: " + path);
  std::array<char, 8> magic{};
  stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!stream || magic != kMagic) throw std::runtime_error("invalid motion binary magic");
  if (read_value<std::uint32_t>(stream, "version") != kVersion) {
    throw std::runtime_error("unsupported motion binary version");
  }
  const auto bodies = read_value<std::uint32_t>(stream, "body_count");
  const auto dofs = read_value<std::uint32_t>(stream, "dof_count");
  const auto motion_count = read_value<std::uint32_t>(stream, "motion_count");
  const auto total_frames = read_value<std::uint64_t>(stream, "total_frames");
  if (bodies == 0 || bodies > kMaxBodies || dofs > 4096 || motion_count == 0 ||
      motion_count > kMaxMotions || total_frames == 0 || total_frames > kMaxFrames) {
    throw std::runtime_error("invalid motion binary dimensions");
  }
  body_count_ = bodies;
  dof_count_ = dofs;
  motions_.assign(motion_count, {});
  std::uint64_t previous_end = 0;
  for (auto& motion : motions_) {
    motion.start = read_value<std::uint64_t>(stream, "motion start");
    motion.frames = read_value<std::uint32_t>(stream, "motion frames");
    motion.dt = read_value<float>(stream, "motion dt");
    if (motion.frames == 0 || !std::isfinite(motion.dt) || motion.dt <= 0.0F ||
        motion.start < previous_end || motion.start > total_frames ||
        motion.frames > total_frames - motion.start) {
      throw std::runtime_error("invalid motion index entry");
    }
    previous_end = motion.start + motion.frames;
  }
  if (previous_end != total_frames) throw std::runtime_error("motion index coverage mismatch");
  positions_.assign(static_cast<std::size_t>(total_frames) * body_count_ * 3, 0.0F);
  rotations_.assign(static_cast<std::size_t>(total_frames) * body_count_ * 4, 0.0F);
  read_floats(stream, positions_, "positions");
  read_floats(stream, rotations_, "rotations");
}

std::size_t MotionLibrary::frame_count(std::size_t motion_index) const {
  return motions_.at(motion_index).frames;
}

float MotionLibrary::motion_dt(std::size_t motion_index) const {
  return motions_.at(motion_index).dt;
}

float MotionLibrary::motion_length(std::size_t motion_index) const {
  const auto& motion = motions_.at(motion_index);
  return motion.dt * static_cast<float>(motion.frames - 1);
}

void MotionLibrary::sample(std::size_t motion_index, float time_seconds,
                           std::span<float> body_pos,
                           std::span<float> body_rot) const {
  if (motion_index >= motions_.size()) throw std::out_of_range("motion index");
  if (body_pos.size() != body_count_ * 3 || body_rot.size() != body_count_ * 4) {
    throw std::invalid_argument("motion sample output span has the wrong size");
  }
  const auto& motion = motions_[motion_index];
  const float clamped = std::clamp(time_seconds, 0.0F,
                                   motion.dt * (motion.frames - 1));
  const float frame = clamped / motion.dt;
  const auto frame0 = static_cast<std::size_t>(std::floor(frame));
  const auto frame1 = std::min<std::size_t>(frame0 + 1, motion.frames - 1);
  const float alpha = frame - static_cast<float>(frame0);
  const auto global0 = static_cast<std::size_t>(motion.start) + frame0;
  const auto global1 = static_cast<std::size_t>(motion.start) + frame1;
  const auto pos0 = global0 * body_count_ * 3;
  const auto pos1 = global1 * body_count_ * 3;
  for (std::size_t i = 0; i < body_count_ * 3; ++i) {
    body_pos[i] = positions_[pos0 + i] +
                  alpha * (positions_[pos1 + i] - positions_[pos0 + i]);
  }
  const auto rot0 = global0 * body_count_ * 4;
  const auto rot1 = global1 * body_count_ * 4;
  for (std::size_t body = 0; body < body_count_; ++body) {
    const auto a = rot0 + body * 4;
    const auto b = rot1 + body * 4;
    const Quat q = slerp({rotations_[a], rotations_[a + 1],
                          rotations_[a + 2], rotations_[a + 3]},
                         {rotations_[b], rotations_[b + 1],
                          rotations_[b + 2], rotations_[b + 3]},
                         alpha);
    body_rot[body * 4] = q.x;
    body_rot[body * 4 + 1] = q.y;
    body_rot[body * 4 + 2] = q.z;
    body_rot[body * 4 + 3] = q.w;
  }
}

}  // namespace protomotions::deployment
