// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "protomotions/deployment/metatron_v1_engine.hpp"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace protomotions::deployment::metatron_v1 {
namespace {
static_assert(NV_TENSORRT_MAJOR == 10 && NV_TENSORRT_MINOR == 9,
              "Metatron plans require TensorRT 10.9 headers");
void check_tensorrt_version() {
  if (getInferLibVersion() != NV_TENSORRT_VERSION)
    throw std::runtime_error("TensorRT headers and linked runtime differ");
}
void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kERROR) {
      std::fprintf(stderr, "TensorRT scene engine: %s\n", message);
    }
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
    cuda_check(cudaHostAlloc(&host, bytes, cudaHostAllocDefault),
               "cudaHostAlloc");
    cuda_check(cudaMalloc(&device, bytes), "cudaMalloc");
  }
};
}  // namespace

class Engine::Impl {
 public:
  explicit Impl(const std::string& plan_path, bool use_graph) {
    check_tensorrt_version();
    std::ifstream file(plan_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open scene TensorRT plan: " + plan_path);
    const auto length = file.tellg();
    if (length <= 0) throw std::runtime_error("scene TensorRT plan is empty");
    std::vector<char> bytes(static_cast<std::size_t>(length));
    file.seekg(0);
    file.read(bytes.data(), length);
    if (!file) throw std::runtime_error("cannot read scene TensorRT plan");
    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) throw std::runtime_error("TensorRT runtime creation failed");
    engine_ = runtime_->deserializeCudaEngine(bytes.data(), bytes.size());
    if (!engine_) throw std::runtime_error("scene TensorRT plan deserialization failed");
    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("scene TensorRT context creation failed");
    validate_engine();
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
    for (std::size_t index = 0; index < kInputCount; ++index) {
      const auto& spec = kInputs[index];
      input_[index].allocate(spec.count() * (spec.is_bool ? 1 : sizeof(float)));
      if (!context_->setTensorAddress(spec.name, input_[index].device)) {
        throw std::runtime_error("TensorRT input binding rejected: " +
                                 std::string(spec.name));
      }
    }
    for (std::size_t index = 0; index < kOutputCount; ++index) {
      output_[index].allocate(kDofCount * sizeof(float));
      if (!context_->setTensorAddress(kOutputs[index], output_[index].device)) {
        throw std::runtime_error("TensorRT output binding rejected: " +
                                 std::string(kOutputs[index]));
      }
    }
    if (use_graph) capture_graph();
  }

  ~Impl() {
    if (graph_exec_) cudaGraphExecDestroy(graph_exec_);
    if (graph_) cudaGraphDestroy(graph_);
    if (stream_) cudaStreamDestroy(stream_);
    delete context_;
    delete engine_;
    delete runtime_;
  }

  Outputs run(const Inputs& inputs) {
    copy_inputs(inputs);
    if (graph_exec_) {
      cuda_check(cudaGraphLaunch(graph_exec_, stream_), "cudaGraphLaunch");
    } else {
      enqueue();
    }
    cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    Outputs result;
    std::array<std::array<float, kDofCount>*, 4> destinations = {
        &result.action, &result.stiffness, &result.damping, &result.raw_action};
    for (std::size_t index = 0; index < kOutputCount; ++index) {
      std::memcpy(destinations[index]->data(), output_[index].host,
                  output_[index].bytes);
      if (!std::all_of(destinations[index]->begin(), destinations[index]->end(),
                       [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("non-finite scene TensorRT output: " +
                                 std::string(kOutputs[index]));
      }
    }
    return result;
  }

  bool cuda_graph_enabled() const noexcept { return graph_exec_ != nullptr; }

 private:
  void validate_engine() const {
    if (engine_->getNbIOTensors() != static_cast<int>(kInputCount + kOutputCount)) {
      throw std::runtime_error("scene plan must have exactly 39 inputs and 4 outputs");
    }
    for (const auto& spec : kInputs) {
      if (engine_->getTensorIOMode(spec.name) != nvinfer1::TensorIOMode::kINPUT ||
          engine_->getTensorDataType(spec.name) !=
              (spec.is_bool ? nvinfer1::DataType::kBOOL
                            : nvinfer1::DataType::kFLOAT)) {
        throw std::runtime_error("scene plan input name or dtype mismatch: " +
                                 std::string(spec.name));
      }
      const auto shape = engine_->getTensorShape(spec.name);
      if (shape.nbDims != spec.rank) {
        throw std::runtime_error("scene plan input rank mismatch: " +
                                 std::string(spec.name));
      }
      for (int axis = 0; axis < spec.rank; ++axis) {
        if (shape.d[axis] != spec.dimensions[static_cast<std::size_t>(axis)]) {
          throw std::runtime_error("scene plan input shape mismatch: " +
                                   std::string(spec.name));
        }
      }
    }
    for (const char* name : kOutputs) {
      if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kOUTPUT ||
          engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("scene plan output name or dtype mismatch: " +
                                 std::string(name));
      }
      const auto shape = engine_->getTensorShape(name);
      if (shape.nbDims != 2 || shape.d[0] != 1 || shape.d[1] != 29) {
        throw std::runtime_error("scene plan output shape mismatch: " +
                                 std::string(name));
      }
    }
    for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
      const std::string_view name = engine_->getIOTensorName(index);
      const bool input = std::any_of(kInputs.begin(), kInputs.end(),
                                     [name](const InputSpec& spec) {
                                       return name == spec.name;
                                     });
      const bool output = std::any_of(kOutputs.begin(), kOutputs.end(),
                                      [name](const char* expected) {
                                        return name == expected;
                                      });
      if (!input && !output)
        throw std::runtime_error("unsupported scene plan tensor: " +
                                 std::string(name));
    }
  }

  void copy_inputs(const Inputs& inputs) {
    for (std::size_t index = 0; index < kInputCount; ++index) {
      const auto& spec = kInputs[index];
      if (spec.is_bool) {
        const auto& data = inputs.booleans(index);
        if (data.size() != spec.count())
          throw std::runtime_error("scene boolean input length mismatch");
        std::memcpy(input_[index].host, data.data(), input_[index].bytes);
      } else {
        const auto& data = inputs.floats(index);
        if (data.size() != spec.count())
          throw std::runtime_error("scene float input length mismatch");
        std::memcpy(input_[index].host, data.data(), input_[index].bytes);
      }
    }
  }

  void enqueue() {
    for (const auto& buffer : input_) {
      cuda_check(cudaMemcpyAsync(buffer.device, buffer.host, buffer.bytes,
                                 cudaMemcpyHostToDevice, stream_),
                 "cudaMemcpyAsync(scene input)");
    }
    if (!context_->enqueueV3(stream_))
      throw std::runtime_error("scene TensorRT enqueueV3 failed");
    for (const auto& buffer : output_) {
      cuda_check(cudaMemcpyAsync(buffer.host, buffer.device, buffer.bytes,
                                 cudaMemcpyDeviceToHost, stream_),
                 "cudaMemcpyAsync(scene output)");
    }
  }

  void capture_graph() {
    Inputs initial;
    SensorReading reading;
    initial.prepare(reading);
    copy_inputs(initial);
    for (int index = 0; index < 20; ++index) {
      enqueue();
      cuda_check(cudaStreamSynchronize(stream_), "scene graph warmup");
    }
    cuda_check(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
               "cudaStreamBeginCapture");
    // Capture transfers as well as inference; pinned host/device addresses stay
    // fixed, and copy_inputs refreshes the host buffers before each replay.
    enqueue();
    cuda_check(cudaStreamEndCapture(stream_, &graph_), "cudaStreamEndCapture");
    cuda_check(cudaGraphInstantiate(&graph_exec_, graph_, 0),
               "cudaGraphInstantiate");
  }

  Logger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  std::array<Buffer, kInputCount> input_{};
  std::array<Buffer, kOutputCount> output_{};
};

Engine::Engine(const std::string& plan_path, bool cuda_graph)
    : impl_(std::make_unique<Impl>(plan_path, cuda_graph)) {}
Engine::~Engine() = default;
Outputs Engine::run(const Inputs& inputs) { return impl_->run(inputs); }
void Engine::warmup(int iterations) {
  if (iterations < 0) throw std::invalid_argument("negative warmup iterations");
  Inputs inputs;
  SensorReading reading;
  inputs.prepare(reading);
  for (int index = 0; index < iterations; ++index) impl_->run(inputs);
}
bool Engine::cuda_graph_enabled() const noexcept {
  return impl_->cuda_graph_enabled();
}
}  // namespace protomotions::deployment::metatron_v1
