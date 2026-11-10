// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/motion_library.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace protomotions::deployment {
namespace {

constexpr int kBundleVersion = 1;
constexpr std::size_t kMaxBodies = 4096;
constexpr std::size_t kMaxFrames = 100000000;

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

void append_finite(const nlohmann::json& values, std::size_t expected, std::vector<float>& out,
                   const std::string& what) {
  if (!values.is_array() || values.size() != expected)
    throw std::runtime_error("tracking bundle: " + what + " must hold " + std::to_string(expected) + " values");
  for (const auto& value : values) {
    if (!value.is_number()) throw std::runtime_error("tracking bundle: " + what + " has a non-numeric value");
    const float v = value.get<float>();
    if (!std::isfinite(v)) throw std::runtime_error("tracking bundle: " + what + " has a non-finite value");
    out.push_back(v);
  }
}

}  // namespace

void MotionLibrary::load(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("cannot open tracking motion bundle: " + path);
  nlohmann::json bundle;
  try {
    stream >> bundle;
  } catch (const nlohmann::json::exception& error) {
    throw std::runtime_error(std::string("tracking bundle is not valid JSON: ") + error.what());
  }
  if (!bundle.is_object() || bundle.value("version", 0) != kBundleVersion)
    throw std::runtime_error("tracking bundle: unsupported version");
  const std::size_t bodies = bundle.value("bodyCount", std::size_t{0});
  if (bodies == 0 || bodies > kMaxBodies) throw std::runtime_error("tracking bundle: invalid bodyCount");
  const auto motions = bundle.find("motions");
  if (motions == bundle.end() || !motions->is_object() || motions->empty())
    throw std::runtime_error("tracking bundle: no motions");

  std::vector<int> conditionable;
  if (const auto ids = bundle.find("conditionableBodyIds"); ids != bundle.end()) {
    for (const auto& id : *ids) {
      if (!id.is_number_integer() || id.get<int>() < 0 || static_cast<std::size_t>(id.get<int>()) >= bodies)
        throw std::runtime_error("tracking bundle: conditionableBodyIds out of range");
      conditionable.push_back(id.get<int>());
    }
  }
  const int anchor = bundle.value("anchorBodyIndex", 0);
  if (anchor < 0 || static_cast<std::size_t>(anchor) >= bodies)
    throw std::runtime_error("tracking bundle: anchorBodyIndex out of range");

  std::vector<std::string> names;
  std::vector<MotionIndex> index;
  std::vector<float> positions, rotations;
  std::size_t total_frames = 0;
  for (const auto& [name, motion] : motions->items()) {
    if (name.empty() || !motion.is_object()) throw std::runtime_error("tracking bundle: malformed motion entry");
    const std::size_t frames = motion.value("frameCount", std::size_t{0});
    const float dt = motion.value("frameDtSeconds", 0.0F);
    if (frames < 2 || frames > kMaxFrames || total_frames > kMaxFrames - frames)
      throw std::runtime_error("tracking bundle: " + name + " needs at least two frames");
    if (!std::isfinite(dt) || dt <= 0.0F) throw std::runtime_error("tracking bundle: " + name + " has an invalid frame period");
    append_finite(motion.at("positions"), frames * bodies * 3, positions, name + ".positions");
    append_finite(motion.at("rotations"), frames * bodies * 4, rotations, name + ".rotations");
    index.push_back({total_frames, frames, dt});
    names.push_back(name);
    total_frames += frames;
  }
  body_count_ = bodies;
  anchor_body_index_ = anchor;
  conditionable_ = std::move(conditionable);
  names_ = std::move(names);
  motions_ = std::move(index);
  positions_ = std::move(positions);
  rotations_ = std::move(rotations);
}

std::optional<std::size_t> MotionLibrary::index_of(std::string_view name) const noexcept {
  for (std::size_t i = 0; i < names_.size(); ++i)
    if (names_[i] == name) return i;
  return std::nullopt;
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
  const float clamped = std::clamp(std::isfinite(time_seconds) ? time_seconds : 0.0F, 0.0F,
                                   motion.dt * static_cast<float>(motion.frames - 1));
  const float frame = clamped / motion.dt;
  const std::size_t lower = std::min(motion.frames - 1, static_cast<std::size_t>(std::floor(frame)));
  const std::size_t upper = std::min(motion.frames - 1, lower + 1);
  const float alpha = std::clamp(frame - static_cast<float>(lower), 0.0F, 1.0F);
  const float* p0 = positions_.data() + (motion.start + lower) * body_count_ * 3;
  const float* p1 = positions_.data() + (motion.start + upper) * body_count_ * 3;
  const float* q0 = rotations_.data() + (motion.start + lower) * body_count_ * 4;
  const float* q1 = rotations_.data() + (motion.start + upper) * body_count_ * 4;
  for (std::size_t i = 0; i < body_count_ * 3; ++i) body_pos[i] = p0[i] + (p1[i] - p0[i]) * alpha;
  for (std::size_t b = 0; b < body_count_; ++b) {
    const Quat q = slerp({q0[b * 4], q0[b * 4 + 1], q0[b * 4 + 2], q0[b * 4 + 3]},
                         {q1[b * 4], q1[b * 4 + 1], q1[b * 4 + 2], q1[b * 4 + 3]}, alpha);
    body_rot[b * 4] = q.x;
    body_rot[b * 4 + 1] = q.y;
    body_rot[b * 4 + 2] = q.z;
    body_rot[b * 4 + 3] = q.w;
  }
}

}  // namespace protomotions::deployment
