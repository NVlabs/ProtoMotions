// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/gpc_policy.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace protomotions::deployment {
namespace {

using Clock = std::chrono::steady_clock;

void cuda_check(cudaError_t status, std::string_view operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      std::fprintf(stderr, "[TensorRT] %s\n", message);
    }
  }
};

std::vector<char> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot open TensorRT engine: " + path);
  const auto size = file.tellg();
  if (size <= 0) throw std::runtime_error("TensorRT engine is empty: " + path);
  std::vector<char> data(static_cast<std::size_t>(size));
  file.seekg(0);
  file.read(data.data(), static_cast<std::streamsize>(data.size()));
  if (!file) throw std::runtime_error("could not read TensorRT engine: " + path);
  return data;
}

std::size_t volume(nvinfer1::Dims dims) {
  std::size_t result = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] <= 0) throw std::runtime_error("GPC TensorRT engine has dynamic/invalid shape");
    result *= static_cast<std::size_t>(dims.d[i]);
  }
  return result;
}

const char* mode_name(GpcSamplingMode mode) {
  switch (mode) {
    case GpcSamplingMode::kArgmax: return "argmax";
    case GpcSamplingMode::kGumbel: return "gumbel";
    case GpcSamplingMode::kTopP: return "topp";
  }
  return "unknown";
}

}  // namespace

class NativeGpcPolicy::Impl {
 public:
  Impl(const std::string& engine_path, GpcSamplingMode mode,
       bool use_cuda_graph, const std::string& plugin_library)
      : mode_(mode), use_cuda_graph_(use_cuda_graph) {
    if (mode_ == GpcSamplingMode::kTopP && plugin_library.empty()) {
      throw std::runtime_error("GPC top-p requires --plugin-library");
    }
    if (!plugin_library.empty()) {
      plugin_handle_ = dlopen(plugin_library.c_str(), RTLD_NOW | RTLD_GLOBAL);
      if (plugin_handle_ == nullptr) {
        throw std::runtime_error(std::string("could not load TensorRT plugin: ") + dlerror());
      }
    }
    const auto serialized = read_file(engine_path);
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (runtime_ == nullptr) throw std::runtime_error("TensorRT runtime creation failed");
    engine_ = runtime_->deserializeCudaEngine(serialized.data(), serialized.size());
    if (engine_ == nullptr) throw std::runtime_error("TensorRT engine deserialization failed");
    context_ = engine_->createExecutionContext();
    if (context_ == nullptr) throw std::runtime_error("TensorRT execution context creation failed");
    cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");
    bind_and_allocate();
  }

  ~Impl() {
    if (graph_exec_ != nullptr) cudaGraphExecDestroy(graph_exec_);
    if (graph_ != nullptr) cudaGraphDestroy(graph_);
    for (auto& buffer : inputs_) {
      if (buffer.device != nullptr) cudaFree(buffer.device);
      if (buffer.host != nullptr) cudaFreeHost(buffer.host);
    }
    if (output_device_ != nullptr) cudaFree(output_device_);
    if (output_host_ != nullptr) cudaFreeHost(output_host_);
    if (stream_ != nullptr) cudaStreamDestroy(stream_);
    delete context_;
    delete engine_;
    delete runtime_;
    if (plugin_handle_ != nullptr) dlclose(plugin_handle_);
  }

  void warmup(int count) {
    if (count < 1) throw std::runtime_error("GPC warmup count must be positive");
    std::array<float, kNativeGpcObservationDim> observation{};
    std::array<float, kNativeGpcPreviousActionDim> previous_action{};
    std::vector<float> noise(kNativeGpcNoiseDim, 0.0F);
    for (int i = 0; i < count; ++i) {
      prepare_inputs(observation, previous_action, mode_ == GpcSamplingMode::kArgmax ? nullptr : noise.data());
      enqueue_direct();
      cuda_check(cudaStreamSynchronize(stream_), "warmup synchronize");
    }
    if (use_cuda_graph_ && graph_exec_ == nullptr) capture_graph();
  }

  void run(const std::array<float, kNativeGpcObservationDim>& observation,
           const std::array<float, kNativeGpcPreviousActionDim>& previous_action,
           const float* gumbel_noise,
           std::array<float, kNativeGpcActionDim>& action) {
    if (mode_ != GpcSamplingMode::kArgmax && gumbel_noise == nullptr) {
      throw std::runtime_error(std::string(mode_name(mode_)) + " requires external gumbel noise");
    }
    prepare_inputs(observation, previous_action, gumbel_noise);
    const auto start = Clock::now();
    if (use_cuda_graph_) {
      if (graph_exec_ == nullptr) capture_graph();
      cuda_check(cudaGraphLaunch(graph_exec_, stream_), "cudaGraphLaunch");
    } else {
      enqueue_direct();
    }
    cuda_check(cudaStreamSynchronize(stream_), "policy synchronize");
    last_step_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
    std::memcpy(action.data(), output_host_, sizeof(float) * kNativeGpcActionDim);
    for (float value : action) {
      if (!std::isfinite(value)) throw std::runtime_error("GPC TensorRT action is non-finite");
    }
  }

  GpcSamplingMode mode() const noexcept { return mode_; }
  bool use_cuda_graph() const noexcept { return use_cuda_graph_; }
  std::int64_t last_step_ns() const noexcept { return last_step_ns_; }
  const GpcInferenceTiming& timing() const noexcept { return timing_; }

 private:
  struct Buffer {
    std::string name;
    std::size_t elements = 0;
    void* host = nullptr;
    void* device = nullptr;
  };

  void bind_and_allocate() {
    bool saw_observation = false;
    bool saw_previous_action = false;
    bool saw_noise = false;
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char* name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
        if (std::string(name) != "action" || engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT
            || volume(engine_->getTensorShape(name)) != kNativeGpcActionDim) {
          throw std::runtime_error("unexpected GPC TensorRT output contract");
        }
        cuda_check(cudaMallocHost(&output_host_, sizeof(float) * kNativeGpcActionDim), "cudaMallocHost output");
        cuda_check(cudaMalloc(&output_device_, sizeof(float) * kNativeGpcActionDim), "cudaMalloc output");
        context_->setTensorAddress(name, output_device_);
        continue;
      }
      if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("GPC TensorRT inputs must be float32");
      }
      Buffer buffer;
      buffer.name = name;
      buffer.elements = volume(engine_->getTensorShape(name));
      if (buffer.name == "reduced_coords_obs") {
        saw_observation = buffer.elements == kNativeGpcObservationDim;
      } else if (buffer.name == "previous_processed_action") {
        saw_previous_action = buffer.elements == kNativeGpcPreviousActionDim;
      } else if (buffer.name == "gumbel_noise") {
        saw_noise = buffer.elements == kNativeGpcNoiseDim;
      } else {
        throw std::runtime_error("unexpected GPC TensorRT input: " + buffer.name);
      }
      const auto bytes = buffer.elements * sizeof(float);
      cuda_check(cudaMallocHost(&buffer.host, bytes), "cudaMallocHost input");
      cuda_check(cudaMalloc(&buffer.device, bytes), "cudaMalloc input");
      context_->setTensorAddress(buffer.name.c_str(), buffer.device);
      inputs_.push_back(buffer);
    }
    if (!saw_observation || !saw_previous_action
        || (mode_ != GpcSamplingMode::kArgmax && !saw_noise)
        || (mode_ == GpcSamplingMode::kArgmax && saw_noise)) {
      throw std::runtime_error("GPC engine inputs do not match selected sampling mode");
    }
  }

  Buffer& input(std::string_view name) {
    const auto it = std::find_if(inputs_.begin(), inputs_.end(), [&](const Buffer& buffer) {
      return buffer.name == name;
    });
    if (it == inputs_.end()) throw std::runtime_error("missing GPC input: " + std::string(name));
    return *it;
  }

  void prepare_inputs(const std::array<float, kNativeGpcObservationDim>& observation,
                      const std::array<float, kNativeGpcPreviousActionDim>& previous_action,
                      const float* noise) {
    auto& observation_buffer = input("reduced_coords_obs");
    auto& action_buffer = input("previous_processed_action");
    std::memcpy(observation_buffer.host, observation.data(), sizeof(float) * observation.size());
    std::memcpy(action_buffer.host, previous_action.data(), sizeof(float) * previous_action.size());
    if (mode_ != GpcSamplingMode::kArgmax) {
      auto& noise_buffer = input("gumbel_noise");
      std::memcpy(noise_buffer.host, noise, sizeof(float) * kNativeGpcNoiseDim);
    }
  }

  void enqueue_direct() {
    const auto start = Clock::now();
    for (const auto& buffer : inputs_) {
      cuda_check(cudaMemcpyAsync(buffer.device, buffer.host,
                                 buffer.elements * sizeof(float),
                                 cudaMemcpyHostToDevice, stream_), "GPC H2D");
    }
    const auto after_h2d = Clock::now();
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("GPC TensorRT enqueueV3 failed");
    const auto after_trt = Clock::now();
    cuda_check(cudaMemcpyAsync(output_host_, output_device_,
                               sizeof(float) * kNativeGpcActionDim,
                               cudaMemcpyDeviceToHost, stream_), "GPC D2H");
    const auto after_d2h = Clock::now();
    timing_.h2d_ms = std::chrono::duration<float, std::milli>(after_h2d - start).count();
    timing_.tensorrt_ms = std::chrono::duration<float, std::milli>(after_trt - after_h2d).count();
    timing_.d2h_ms = std::chrono::duration<float, std::milli>(after_d2h - after_trt).count();
    timing_.total_gpu_ms = std::chrono::duration<float, std::milli>(after_d2h - start).count();
    timing_.valid = true;
  }

  void capture_graph() {
    for (int i = 0; i < 3; ++i) {
      enqueue_direct();
      cuda_check(cudaStreamSynchronize(stream_), "GPC graph warmup synchronize");
    }
    cuda_check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal), "GPC begin graph capture");
    enqueue_direct();
    cuda_check(cudaStreamEndCapture(stream_, &graph_), "GPC end graph capture");
    cuda_check(cudaGraphInstantiate(&graph_exec_, graph_, 0), "GPC graph instantiate");
  }

  GpcSamplingMode mode_;
  bool use_cuda_graph_;
  void* plugin_handle_ = nullptr;
  Logger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  std::vector<Buffer> inputs_;
  void* output_host_ = nullptr;
  void* output_device_ = nullptr;
  std::int64_t last_step_ns_ = 0;
  GpcInferenceTiming timing_{};
};

NativeGpcPolicy::NativeGpcPolicy(const std::string& engine_path,
                                 GpcSamplingMode mode, bool use_cuda_graph,
                                 const std::string& plugin_library)
    : impl_(std::make_unique<Impl>(engine_path, mode, use_cuda_graph, plugin_library)) {}
NativeGpcPolicy::~NativeGpcPolicy() = default;
void NativeGpcPolicy::warmup(int count) { impl_->warmup(count); }
void NativeGpcPolicy::run(const std::array<float, kNativeGpcObservationDim>& observation,
                          const std::array<float, kNativeGpcPreviousActionDim>& previous_action,
                          const float* gumbel_noise,
                          std::array<float, kNativeGpcActionDim>& action) {
  impl_->run(observation, previous_action, gumbel_noise, action);
}
GpcSamplingMode NativeGpcPolicy::sampling_mode() const noexcept { return impl_->mode(); }
bool NativeGpcPolicy::using_cuda_graph() const noexcept { return impl_->use_cuda_graph(); }
std::int64_t NativeGpcPolicy::last_step_ns() const noexcept { return impl_->last_step_ns(); }
const GpcInferenceTiming& NativeGpcPolicy::last_inference_timing() const noexcept { return impl_->timing(); }

}  // namespace protomotions::deployment
