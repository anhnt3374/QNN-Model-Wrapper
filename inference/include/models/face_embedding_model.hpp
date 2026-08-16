#pragma once

#include "inference/qnn_backend.hpp"
#include "inference/qnn_model.hpp"
#include "inference/qnn_tensor_buffer.hpp"

#include <opencv2/core/mat.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace models {

struct FaceEmbeddingPreprocessInfo {
    int originalWidth = 0;
    int originalHeight = 0;

    int resizedWidth = 112;
    int resizedHeight = 112;
};

struct FaceEmbeddingResult {
    static constexpr std::size_t DIMENSION = 512;

    std::array<float, DIMENSION> values{};

    // Norm before normalization.
    float rawL2Norm = 0.0F;
};

class FaceEmbeddingModel {
public:
    static constexpr std::size_t EMBEDDING_DIM = 512;

    explicit FaceEmbeddingModel(
        inference::QnnBackend& backend
    ) noexcept;

    ~FaceEmbeddingModel();

    FaceEmbeddingModel(
        const FaceEmbeddingModel&
    ) = delete;

    FaceEmbeddingModel& operator=(
        const FaceEmbeddingModel&
    ) = delete;

    // =====================================================
    // Step 1
    //
    // Load model
    // compose graph
    // finalize graph
    // validate tensor contract
    // allocate persistent buffers
    // =====================================================

    bool initialize(
        const std::string& modelPath
    );

    void shutdown();

    // =====================================================
    // Step 2
    //
    // Input face:
    //
    // BGR
    // -> resize 112x112
    // -> RGB
    // -> pixel / 127.5 - 1.0
    // -> NHWC
    // -> runtime tensor
    //
    // IMPORTANT:
    // Input should already be cropped/aligned face.
    // =====================================================

    bool preprocess(
        const cv::Mat& bgrFace
    );

    // =====================================================
    // Step 3
    //
    // graphExecute()
    // =====================================================

    bool infer();

    // =====================================================
    // Step 4
    //
    // output [1,512]
    // -> FLOAT32 directly
    // or
    // -> UFIXED16 dequantization
    // -> L2 normalization
    // =====================================================

    bool postprocess(
        FaceEmbeddingResult& result
    );

    // =====================================================
    // Step 5
    //
    // Complete one-image API:
    //
    // preprocess
    // -> infer
    // -> postprocess
    // =====================================================

    bool extract(
        const cv::Mat& bgrFace,
        FaceEmbeddingResult& result
    );

    bool ready() const noexcept;

    const FaceEmbeddingPreprocessInfo&
    preprocessInfo() const noexcept;

    const inference::QnnTensorBuffer*
    inputBuffer() const noexcept;

    const inference::QnnTensorBuffer*
    outputBuffer() const noexcept;

    const std::string&
    lastError() const noexcept;

private:
    bool allocateRuntimeBuffers();

    bool buildExecutionTensorArrays();

    bool validateModelContract();

    static bool readTensorValue(
        const inference::QnnTensorBuffer& buffer,
        uint64_t index,
        float& value
    ) noexcept;

    static Qnn_DataType_t dataType(
        const Qnn_Tensor_t& tensor
    ) noexcept;

    static uint32_t rank(
        const Qnn_Tensor_t& tensor
    ) noexcept;

    static const uint32_t* dimensions(
        const Qnn_Tensor_t& tensor
    ) noexcept;

    static const Qnn_QuantizeParams_t*
    quantization(
        const Qnn_Tensor_t& tensor
    ) noexcept;

private:
    static constexpr int INPUT_WIDTH = 112;
    static constexpr int INPUT_HEIGHT = 112;
    static constexpr int INPUT_CHANNELS = 3;

    static constexpr float L2_EPSILON =
        1e-12F;

    inference::QnnModel model_;

    std::vector<
        std::unique_ptr<
            inference::QnnTensorBuffer
        >
    > inputBuffers_;

    std::vector<
        std::unique_ptr<
            inference::QnnTensorBuffer
        >
    > outputBuffers_;

    // Persistent descriptors for graphExecute().
    std::vector<Qnn_Tensor_t>
        executionInputTensors_;

    std::vector<Qnn_Tensor_t>
        executionOutputTensors_;

    FaceEmbeddingPreprocessInfo preprocessInfo_;

    std::string lastError_;
};

} // namespace models