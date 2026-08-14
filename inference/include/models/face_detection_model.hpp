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

struct FaceDetectionPreprocessInfo {
    int originalWidth = 0;
    int originalHeight = 0;

    int resizedWidth = 0;
    int resizedHeight = 0;

    float detScale = 0.0F;
};

struct FaceDetectionProposal {
    float score = 0.0F;

    // Model-input coordinate system:
    //
    // [x1, y1, x2, y2]
    std::array<float, 4> bbox{};

    // 5 keypoints:
    //
    // [x0, y0,
    //  x1, y1,
    //  ...
    //  x4, y4]
    std::array<float, 10> landmarks{};
};

class FaceDetectionModel {
public:
    explicit FaceDetectionModel(
        inference::QnnBackend& backend
    ) noexcept;

    ~FaceDetectionModel();

    FaceDetectionModel(
        const FaceDetectionModel&
    ) = delete;

    FaceDetectionModel& operator=(
        const FaceDetectionModel&
    ) = delete;

    bool initialize(
        const std::string& modelPath
    );

    void shutdown();

    bool preprocess(
        const cv::Mat& bgrImage
    );

    bool infer();

    // Kept for the 5H.3 stride-8 checkpoint.
    bool decodeStride8(
        std::vector<FaceDetectionProposal>& proposals,
        float scoreThreshold = 0.5F
    );

    // 5H.4:
    //
    // decode stride 8 + 16 + 32,
    // merge proposals,
    // sort by confidence descending.
    bool decodeAll(
        std::vector<FaceDetectionProposal>& proposals,
        float scoreThreshold = 0.5F
    );

    bool ready() const noexcept;

    const FaceDetectionPreprocessInfo&
    preprocessInfo() const noexcept;

    const inference::QnnTensorBuffer*
    inputBuffer() const noexcept;

    std::size_t outputCount() const noexcept;

    const inference::QnnTensorBuffer*
    outputBuffer(
        std::size_t index
    ) const noexcept;

    const std::string&
    lastError() const noexcept;

private:
    bool allocateRuntimeBuffers();

    bool buildExecutionTensorArrays();

    bool validateScrfdInput();

    bool decodeLevel(
        int stride,
        const char* scoreTensorName,
        const char* bboxTensorName,
        const char* kpsTensorName,
        float scoreThreshold,
        std::vector<FaceDetectionProposal>& proposals
    );

    const inference::QnnTensorBuffer*
    outputBufferByName(
        const char* name
    ) const noexcept;

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
    static constexpr int INPUT_WIDTH = 640;
    static constexpr int INPUT_HEIGHT = 640;
    static constexpr int INPUT_CHANNELS = 3;

    static constexpr int NUM_ANCHORS = 2;

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

    std::vector<Qnn_Tensor_t>
        executionInputTensors_;

    std::vector<Qnn_Tensor_t>
        executionOutputTensors_;

    FaceDetectionPreprocessInfo preprocessInfo_;

    std::string lastError_;
};

} // namespace models