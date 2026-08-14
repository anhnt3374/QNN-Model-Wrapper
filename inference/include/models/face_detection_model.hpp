#pragma once

#include "inference/qnn_backend.hpp"
#include "inference/qnn_model.hpp"
#include "inference/qnn_tensor_buffer.hpp"

#include <opencv2/core/mat.hpp>

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