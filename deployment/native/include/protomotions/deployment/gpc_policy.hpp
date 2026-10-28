// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace protomotions::deployment {

inline constexpr std::size_t kNativeGpcObservationDim = 64;
inline constexpr std::size_t kNativeGpcPreviousActionDim = 29;
inline constexpr std::size_t kNativeGpcTokens = 8;
inline constexpr std::size_t kNativeGpcVocabulary = 6561;
inline constexpr std::size_t kNativeGpcNoiseDim =
    kNativeGpcTokens * kNativeGpcVocabulary;
inline constexpr std::size_t kNativeGpcActionDim = 29;

enum class GpcSamplingMode {
  kArgmax,
  kGumbel,
  kTopP,
};

struct GpcInferenceTiming {
  float h2d_ms = 0.0F;
  float tensorrt_ms = 0.0F;
  float d2h_ms = 0.0F;
  float total_gpu_ms = 0.0F;
  bool valid = false;
};

// Fixed-shape native TensorRT runner for the G1 unconditional GPC prior.
// The caller owns observation/action construction; this class owns only the
// inference contract, persistent pinned/device buffers, CUDA graph state, and
// optional top-p plugin loading.
class NativeGpcPolicy final {
 public:
  NativeGpcPolicy(const std::string& engine_path, GpcSamplingMode mode,
                  bool use_cuda_graph,
                  const std::string& plugin_library = {});
  ~NativeGpcPolicy();

  NativeGpcPolicy(const NativeGpcPolicy&) = delete;
  NativeGpcPolicy& operator=(const NativeGpcPolicy&) = delete;

  void warmup(int count = 10);
  void run(const std::array<float, kNativeGpcObservationDim>& observation,
           const std::array<float, kNativeGpcPreviousActionDim>& previous_action,
           const float* gumbel_noise,
           std::array<float, kNativeGpcActionDim>& action);

  GpcSamplingMode sampling_mode() const noexcept;
  bool using_cuda_graph() const noexcept;
  std::int64_t last_step_ns() const noexcept;
  const GpcInferenceTiming& last_inference_timing() const noexcept;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace protomotions::deployment
