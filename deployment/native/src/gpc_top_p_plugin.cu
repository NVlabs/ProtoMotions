// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

constexpr int kVocabulary = 6561;
constexpr int kPaddedVocabulary = 8192;
constexpr int kThreads = 1024;

__device__ __forceinline__ bool comes_before(float lhs, int lhs_index, float rhs, int rhs_index) {
    const bool lhs_nan = isnan(lhs);
    const bool rhs_nan = isnan(rhs);
    if (lhs_nan != rhs_nan) {
        return !lhs_nan;
    }
    if (lhs > rhs) {
        return true;
    }
    if (lhs < rhs) {
        return false;
    }
    return lhs_index < rhs_index;
}

__global__ void top_p_filter_kernel(
    const float* input,
    float* output,
    int batch,
    float top_p) {
    extern __shared__ unsigned char shared[];
    auto* values = reinterpret_cast<float*>(shared);
    auto* indices = reinterpret_cast<int*>(values + kPaddedVocabulary);
    const int row = static_cast<int>(blockIdx.x);
    if (row >= batch) {
        return;
    }

    const int tid = static_cast<int>(threadIdx.x);
    const float* row_input = input + row * kVocabulary;
    float* row_output = output + row * kVocabulary;
    for (int i = tid; i < kPaddedVocabulary; i += kThreads) {
        if (i < kVocabulary) {
            values[i] = row_input[i];
            indices[i] = i;
        } else {
            values[i] = -INFINITY;
            indices[i] = -1;
        }
    }
    __syncthreads();

    // In-place bitonic sort. The final stage is descending, with original
    // index as a deterministic tie-breaker matching a stable descending sort.
    for (int k = 2; k <= kPaddedVocabulary; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < kPaddedVocabulary; i += kThreads) {
                const int partner = i ^ j;
                if (partner <= i || partner >= kPaddedVocabulary) {
                    continue;
                }
                const bool descending_half = (i & k) == 0;
                const float lhs = values[i];
                const float rhs = values[partner];
                const int lhs_index = indices[i];
                const int rhs_index = indices[partner];
                const bool swap = descending_half
                    ? comes_before(rhs, rhs_index, lhs, lhs_index)
                    : comes_before(lhs, lhs_index, rhs, rhs_index);
                if (swap) {
                    values[i] = rhs;
                    values[partner] = lhs;
                    indices[i] = rhs_index;
                    indices[partner] = lhs_index;
                }
            }
            __syncthreads();
        }
    }

    if (tid == 0) {
        float max_logit = values[0];
        if (!isfinite(max_logit)) {
            max_logit = 0.0F;
        }
        float denominator = 0.0F;
        for (int i = 0; i < kVocabulary; ++i) {
            const float value = values[i];
            if (isfinite(value)) {
                denominator += expf(value - max_logit);
            }
        }
        const float cutoff = top_p * denominator;
        float cumulative = 0.0F;
        for (int rank = 0; rank < kVocabulary; ++rank) {
            const int original_index = indices[rank];
            if (original_index < 0) {
                continue;
            }
            const float value = values[rank];
            const float probability = isfinite(value) ? expf(value - max_logit) : 0.0F;
            const bool keep = rank == 0 || cumulative <= cutoff;
            row_output[original_index] = keep ? row_input[original_index] : -INFINITY;
            cumulative += probability;
        }
    }
}

class TopPFilter final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit TopPFilter(float top_p) : top_p_(top_p) {}

    TopPFilter(const void* serial_data, size_t serial_length) {
        if (serial_length != sizeof(float)) {
            top_p_ = 0.9F;
            return;
        }
        std::memcpy(&top_p_, serial_data, sizeof(float));
    }

    const char* getPluginType() const noexcept override { return "TopPFilter"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t output_index,
        nvinfer1::DimsExprs const* inputs,
        int32_t nb_inputs,
        nvinfer1::IExprBuilder&) noexcept override {
        if (output_index != 0 || nb_inputs != 1) {
            return nvinfer1::DimsExprs{-1, {}};
        }
        return inputs[0];
    }

    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::PluginTensorDesc const* in_out,
        int32_t nb_inputs,
        int32_t nb_outputs) noexcept override {
        if (nb_inputs != 1 || nb_outputs != 1) {
            return false;
        }
        return in_out[pos].format == nvinfer1::TensorFormat::kLINEAR
            && in_out[pos].type == nvinfer1::DataType::kFLOAT;
    }

    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* inputs,
        int32_t nb_inputs,
        nvinfer1::DynamicPluginTensorDesc const*,
        int32_t nb_outputs) noexcept override {
        if (nb_inputs == 1 && nb_outputs == 1 && inputs[0].desc.dims.nbDims == 2
            && inputs[0].desc.dims.d[1] != kVocabulary) {
            valid_shape_ = false;
        }
    }

    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const*,
        int32_t,
        nvinfer1::PluginTensorDesc const*,
        int32_t) const noexcept override {
        return 0;
    }

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* input_desc,
        nvinfer1::PluginTensorDesc const* output_desc,
        void const* const* inputs,
        void* const* outputs,
        void*,
        cudaStream_t stream) noexcept override {
        if (!valid_shape_ || input_desc[0].dims.nbDims != 2
            || output_desc[0].dims.nbDims != 2
            || input_desc[0].dims.d[1] != kVocabulary
            || output_desc[0].dims.d[1] != kVocabulary) {
            std::fprintf(stderr, "TopPFilter invalid shape: valid=%d in=%d,%d out=%d,%d\n",
                valid_shape_ ? 1 : 0,
                input_desc[0].dims.nbDims,
                input_desc[0].dims.nbDims > 1 ? input_desc[0].dims.d[1] : -1,
                output_desc[0].dims.nbDims,
                output_desc[0].dims.nbDims > 1 ? output_desc[0].dims.d[1] : -1);
            return 1;
        }
        const int batch = input_desc[0].dims.d[0];
        constexpr size_t shared_bytes =
            kPaddedVocabulary * (sizeof(float) + sizeof(int));
        const cudaError_t attribute_status = cudaFuncSetAttribute(
            top_p_filter_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes));
        if (attribute_status != cudaSuccess) {
            std::fprintf(stderr, "TopPFilter cudaFuncSetAttribute failed: %s\n",
                cudaGetErrorString(attribute_status));
            return 1;
        }
        top_p_filter_kernel<<<batch, kThreads, shared_bytes, stream>>>(
            static_cast<const float*>(inputs[0]),
            static_cast<float*>(outputs[0]),
            batch,
            top_p_);
        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            std::fprintf(stderr, "TopPFilter kernel launch failed: %s\n",
                cudaGetErrorString(launch_status));
            return 1;
        }
        return 0;
    }

    nvinfer1::DataType getOutputDataType(
        int32_t index,
        nvinfer1::DataType const* input_types,
        int32_t nb_inputs) const noexcept override {
        return index == 0 && nb_inputs == 1 ? input_types[0] : nvinfer1::DataType::kFLOAT;
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* result = new TopPFilter(top_p_);
        result->namespace_ = namespace_;
        result->valid_shape_ = valid_shape_;
        return result;
    }

    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }

    size_t getSerializationSize() const noexcept override { return sizeof(float); }
    void serialize(void* buffer) const noexcept override {
        std::memcpy(buffer, &top_p_, sizeof(float));
    }

    void setPluginNamespace(const char* plugin_namespace) noexcept override {
        namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
    }
    const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

private:
    float top_p_ = 0.9F;
    std::string namespace_;
    bool valid_shape_ = true;
};

class TopPFilterCreator final : public nvinfer1::IPluginCreator {
public:
    TopPFilterCreator() {
        field_.name = "top_p";
        field_.type = nvinfer1::PluginFieldType::kFLOAT32;
        field_.data = &default_top_p_;
        field_.length = 1;
        fields_.nbFields = 1;
        fields_.fields = &field_;
    }

    const char* getPluginName() const noexcept override { return "TopPFilter"; }
    const char* getPluginVersion() const noexcept override { return "1"; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &fields_; }

    nvinfer1::IPluginV2* createPlugin(
        const char*,
        const nvinfer1::PluginFieldCollection* fields) noexcept override {
        float top_p = default_top_p_;
        if (fields != nullptr) {
            for (int32_t i = 0; i < fields->nbFields; ++i) {
                const auto& field = fields->fields[i];
                if (field.name != nullptr && std::strcmp(field.name, "top_p") == 0
                    && field.data != nullptr && field.length >= 1) {
                    if (field.type == nvinfer1::PluginFieldType::kFLOAT32) {
                        top_p = *static_cast<const float*>(field.data);
                    } else if (field.type == nvinfer1::PluginFieldType::kFLOAT64) {
                        top_p = static_cast<float>(*static_cast<const double*>(field.data));
                    }
                }
            }
        }
        return new TopPFilter(top_p);
    }

    nvinfer1::IPluginV2* deserializePlugin(
        const char*,
        const void* serial_data,
        size_t serial_length) noexcept override {
        return new TopPFilter(serial_data, serial_length);
    }

    void setPluginNamespace(const char* plugin_namespace) noexcept override {
        namespace_ = plugin_namespace == nullptr ? "" : plugin_namespace;
    }
    const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

private:
    float default_top_p_ = 0.9F;
    nvinfer1::PluginField field_{};
    nvinfer1::PluginFieldCollection fields_{};
    std::string namespace_;
};

} // namespace

struct TopPFilterRegistrar {
    TopPFilterCreator creator;
    TopPFilterRegistrar() {
        ::getPluginRegistry()->registerCreator(creator, "");
    }
};

static TopPFilterRegistrar top_p_filter_registrar;
