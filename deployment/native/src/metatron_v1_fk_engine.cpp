// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_fk_engine.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace protomotions::deployment::metatron_v1 {
namespace {
static_assert(NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR == 9,
              "Metatron FK plans require TensorRT 10.9 headers");
void check_tensorrt_version() {
  if (getInferLibVersion() != NV_TENSORRT_VERSION)
    throw std::runtime_error("TensorRT headers and linked runtime differ");
}
void check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity level, const char* message) noexcept override {
    if (level <= Severity::kERROR) std::fprintf(stderr, "TensorRT FK: %s\n", message);
  }
};

struct Buffer {
  void* host = nullptr;
  void* device = nullptr;
  std::size_t bytes = 0;
  Buffer() = default;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  ~Buffer() {
    if (device) cudaFree(device);
    if (host) cudaFreeHost(host);
  }
  void allocate(std::size_t size) {
    bytes = size;
    check(cudaHostAlloc(&host, bytes, cudaHostAllocDefault), "FK cudaHostAlloc");
    check(cudaMalloc(&device, bytes), "FK cudaMalloc");
  }
};

struct TensorSpec {
  const char* name;
  std::array<int, 3> dims;
  int rank;
  std::size_t count;
};
constexpr std::array<TensorSpec, 3> kFkInputs = {{
    {"root_pos", {1, 3}, 2, 3},
    {"root_rot", {1, 4}, 2, 4},
    {"dof_pos", {1, 29}, 2, 29},
}};
constexpr std::array<TensorSpec, 2> kFkOutputs = {{
    {"body_pos", {1, 33, 3}, 3, 33 * 3},
    {"body_rot", {1, 33, 4}, 3, 33 * 4},
}};
}  // namespace

class FkEngine::Impl {
 public:
  Impl(const std::string& plan_path, bool use_graph) {
    try {
      check_tensorrt_version();
      std::ifstream file(plan_path, std::ios::binary | std::ios::ate);
      if (!file) throw std::runtime_error("cannot open TensorRT FK plan: " + plan_path);
      const auto size = file.tellg();
      if (size <= 0) throw std::runtime_error("empty TensorRT FK plan");
      std::vector<char> plan(static_cast<std::size_t>(size));
      file.seekg(0);
      file.read(plan.data(), size);
      if (!file) throw std::runtime_error("cannot read TensorRT FK plan");
      runtime_ = nvinfer1::createInferRuntime(logger_);
      if (!runtime_) throw std::runtime_error("FK TensorRT runtime creation failed");
      engine_ = runtime_->deserializeCudaEngine(plan.data(), plan.size());
      if (!engine_) throw std::runtime_error("FK TensorRT deserialization failed");
      context_ = engine_->createExecutionContext();
      if (!context_) throw std::runtime_error("FK TensorRT context creation failed");
      validate();
      check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "FK cudaStreamCreate");
      for (std::size_t index = 0; index < kFkInputs.size(); ++index) {
        input_[index].allocate(kFkInputs[index].count * sizeof(float));
        if (!context_->setTensorAddress(kFkInputs[index].name, input_[index].device))
          throw std::runtime_error("FK input binding rejected");
      }
      for (std::size_t index = 0; index < kFkOutputs.size(); ++index) {
        output_[index].allocate(kFkOutputs[index].count * sizeof(float));
        if (!context_->setTensorAddress(kFkOutputs[index].name, output_[index].device))
          throw std::runtime_error("FK output binding rejected");
      }
      if (use_graph) capture();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~Impl() { cleanup(); }

  RecoveryPose run(const SensorReading& reading) {
    std::memcpy(input_[0].host, reading.root_pos.data(), input_[0].bytes);
    std::memcpy(input_[1].host, reading.root_rot.data(), input_[1].bytes);
    std::memcpy(input_[2].host, reading.dof_pos.data(), input_[2].bytes);
    for (const auto& buffer : input_)
      check(cudaMemcpyAsync(buffer.device, buffer.host, buffer.bytes,
                            cudaMemcpyHostToDevice, stream_), "FK input upload");
    if (graph_exec_) {
      check(cudaGraphLaunch(graph_exec_, stream_), "FK cudaGraphLaunch");
    } else if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error("FK TensorRT enqueueV3 failed");
    }
    for (const auto& buffer : output_)
      check(cudaMemcpyAsync(buffer.host, buffer.device, buffer.bytes,
                            cudaMemcpyDeviceToHost, stream_), "FK output download");
    check(cudaStreamSynchronize(stream_), "FK stream synchronize");
    RecoveryPose result;
    std::memcpy(result.positions.data(), output_[0].host, output_[0].bytes);
    std::memcpy(result.rotations.data(), output_[1].host, output_[1].bytes);
    result.joints = reading.dof_pos;
    const auto finite = [](const auto& values) {
      return std::all_of(values.begin(), values.end(),
                         [](float value) { return std::isfinite(value); });
    };
    if (!finite(result.positions) || !finite(result.rotations))
      throw std::runtime_error("non-finite TensorRT FK output");
    return result;
  }

  bool graph_enabled() const noexcept { return graph_exec_ != nullptr; }

 private:
  void validate() const {
    if (engine_->getNbIOTensors() != 5)
      throw std::runtime_error("FK plan must have exactly three inputs and two outputs");
    const auto validate_spec = [this](const TensorSpec& spec,
                                      nvinfer1::TensorIOMode mode) {
      if (engine_->getTensorIOMode(spec.name) != mode ||
          engine_->getTensorDataType(spec.name) != nvinfer1::DataType::kFLOAT)
        throw std::runtime_error(std::string("FK tensor name/dtype mismatch: ") + spec.name);
      const auto dims = engine_->getTensorShape(spec.name);
      if (dims.nbDims != spec.rank)
        throw std::runtime_error(std::string("FK tensor rank mismatch: ") + spec.name);
      for (int axis = 0; axis < spec.rank; ++axis)
        if (dims.d[axis] != spec.dims[static_cast<std::size_t>(axis)])
          throw std::runtime_error(std::string("FK tensor shape mismatch: ") + spec.name);
    };
    for (const auto& spec : kFkInputs) validate_spec(spec, nvinfer1::TensorIOMode::kINPUT);
    for (const auto& spec : kFkOutputs) validate_spec(spec, nvinfer1::TensorIOMode::kOUTPUT);
  }

  void capture() {
    SensorReading reading;
    reading.root_pos[2] = 0.75F;
    reading.root_rot = {0, 0, 0, 1};
    for (int pass = 0; pass < 10; ++pass) (void)run(reading);
    check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
          "FK graph capture begin");
    if (!context_->enqueueV3(stream_)) {
      cudaGraph_t abandoned = nullptr;
      cudaStreamEndCapture(stream_, &abandoned);
      if (abandoned) cudaGraphDestroy(abandoned);
      throw std::runtime_error("FK TensorRT graph enqueueV3 failed");
    }
    check(cudaStreamEndCapture(stream_, &graph_), "FK graph capture end");
    check(cudaGraphInstantiate(&graph_exec_, graph_, 0), "FK graph instantiate");
  }

  void cleanup() noexcept {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (stream_) cudaStreamDestroy(stream_);
    delete context_;
    delete engine_;
    delete runtime_;
    graph_exec_ = nullptr;
    graph_ = nullptr;
    stream_ = nullptr;
    context_ = nullptr;
    engine_ = nullptr;
    runtime_ = nullptr;
  }

  Logger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  std::array<Buffer, 3> input_{};
  std::array<Buffer, 2> output_{};
};

FkEngine::FkEngine(const std::string& plan_path, bool cuda_graph)
    : impl_(std::make_unique<Impl>(plan_path, cuda_graph)) {}
FkEngine::~FkEngine() = default;
RecoveryPose FkEngine::run(const SensorReading& reading) { return impl_->run(reading); }
bool FkEngine::cuda_graph_enabled() const noexcept { return impl_->graph_enabled(); }

}  // namespace protomotions::deployment::metatron_v1
