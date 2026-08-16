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

struct PersonDetectionPreprocessInfo {
    int originalWidth = 0;
    int originalHeight = 0;

    int resizedWidth = 0;
    int resizedHeight = 0;

    float scale = 0.0F;

    int padLeft = 0;
    int padRight = 0;
    int padTop = 0;
    int padBottom = 0;
};

struct PersonDetectionProposal {
    float score = 0.0F;

    // Original image coordinates:
    //
    // [x1, y1, x2, y2]
    std::array<float, 4> bbox{};
};

struct PersonDetectionResult {
    float score = 0.0F;

    // Original image coordinates:
    //
    // [x1, y1, x2, y2]
    std::array<float, 4> bbox{};
};

class PersonDetectionModel {
public:
    explicit PersonDetectionModel(
        inference::QnnBackend& backend
    ) noexcept;

    ~PersonDetectionModel();

    PersonDetectionModel(
        const PersonDetectionModel&
    ) = delete;

    PersonDetectionModel& operator=(
        const PersonDetectionModel&
    ) = delete;

    bool initialize(
        const std::string& modelPath
    );

    void shutdown();

    bool preprocess(
        const cv::Mat& bgrImage
    );

    bool infer();

    bool decode(
        std::vector<PersonDetectionProposal>& proposals,
        float confidenceThreshold = 0.15F
    );

    bool postprocess(
        std::vector<PersonDetectionResult>& results,
        float confidenceThreshold = 0.15F,
        float nmsThreshold = 0.60F
    );

    // Complete one-image API:
    //
    // preprocess
    // -> inference
    // -> decode
    // -> NMS
    bool detect(
        const cv::Mat& bgrImage,
        std::vector<PersonDetectionResult>& results,
        float confidenceThreshold = 0.15F,
        float nmsThreshold = 0.60F
    );

    cv::Mat renderDetections(
        const cv::Mat& bgrImage,
        const std::vector<PersonDetectionResult>& results
    ) const;

    bool ready() const noexcept;

    const PersonDetectionPreprocessInfo&
    preprocessInfo() const noexcept;

    const inference::QnnTensorBuffer*
    inputBuffer() const noexcept;

    std::size_t outputCount() const noexcept;

    const inference::QnnTensorBuffer*
    outputBuffer(
        std::size_t index
    ) const noexcept;

    const inference::QnnTensorBuffer*
    outputBufferByName(
        const std::string& name
    ) const noexcept;

    const std::string&
    lastError() const noexcept;

private:
    bool allocateRuntimeBuffers();

    bool buildExecutionTensorArrays();

    bool validateModelContract();

    bool applyNms(
        const std::vector<PersonDetectionProposal>& proposals,
        std::vector<PersonDetectionProposal>& kept,
        float nmsThreshold
    );

    static float calculateIoU(
        const std::array<float, 4>& lhs,
        const std::array<float, 4>& rhs
    ) noexcept;

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

    static int pythonRound(
        double value
    ) noexcept;

private:
    static constexpr int INPUT_HEIGHT = 640;
    static constexpr int INPUT_WIDTH = 640;
    static constexpr int INPUT_CHANNELS = 3;

    static constexpr int PAD_VALUE = 114;

    static constexpr uint32_t NUM_PREDICTIONS = 8400;

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

    PersonDetectionPreprocessInfo preprocessInfo_;

    std::string lastError_;
};

} // namespace models