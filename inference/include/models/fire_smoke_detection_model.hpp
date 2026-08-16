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

struct FireSmokePreprocessInfo {
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

struct FireSmokeAnchor {
    float width = 0.0F;
    float height = 0.0F;
};

struct FireSmokeAnchors {
    std::array<FireSmokeAnchor, 3> s4{};
    std::array<FireSmokeAnchor, 3> s8{};
    std::array<FireSmokeAnchor, 3> s16{};
};

struct FireSmokeDetectionProposal {
    // Current model/script mapping:
    //
    // 0 = smoke
    // 1 = fire
    int classId = -1;

    float score = 0.0F;

    // Original image coordinates:
    //
    // [x1, y1, x2, y2]
    std::array<float, 4> bbox{};
};

struct FireSmokeDetectionResult {
    // Current model/script mapping:
    //
    // 0 = smoke
    // 1 = fire
    int classId = -1;

    float score = 0.0F;

    // Original image coordinates:
    //
    // [x1, y1, x2, y2]
    std::array<float, 4> bbox{};
};

class FireSmokeDetectionModel {
public:
    explicit FireSmokeDetectionModel(
        inference::QnnBackend& backend
    ) noexcept;

    ~FireSmokeDetectionModel();

    FireSmokeDetectionModel(
        const FireSmokeDetectionModel&
    ) = delete;

    FireSmokeDetectionModel& operator=(
        const FireSmokeDetectionModel&
    ) = delete;

    bool initialize(
        const std::string& modelPath
    );

    void shutdown();

    bool setAnchors(
        const FireSmokeAnchors& anchors
    );

    bool preprocess(
        const cv::Mat& bgrImage
    );

    bool infer();

    // Decode 3 raw YOLO heads.
    //
    // No NMS here.
    bool decode(
        std::vector<FireSmokeDetectionProposal>& proposals,
        float confidenceThreshold = 0.25F
    );

    // Decode + class-aware NMS.
    bool postprocess(
        std::vector<FireSmokeDetectionResult>& results,
        float confidenceThreshold = 0.25F,
        float nmsThreshold = 0.45F
    );

    // Complete one-image API:
    //
    // preprocess
    // -> infer
    // -> decode
    // -> class-aware NMS
    bool detect(
        const cv::Mat& bgrImage,
        std::vector<FireSmokeDetectionResult>& results,
        float confidenceThreshold = 0.25F,
        float nmsThreshold = 0.45F
    );

    cv::Mat renderDetections(
        const cv::Mat& bgrImage,
        const std::vector<FireSmokeDetectionResult>& results
    ) const;

    bool ready() const noexcept;

    bool anchorsConfigured() const noexcept;

    const FireSmokeAnchors&
    anchors() const noexcept;

    const FireSmokePreprocessInfo&
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

    bool validateOutput(
        const char* name,
        uint32_t expectedHeight,
        uint32_t expectedWidth
    );

    bool decodeHead(
        const inference::QnnTensorBuffer& output,
        uint32_t height,
        uint32_t width,
        uint32_t stride,
        const std::array<FireSmokeAnchor, 3>& anchors,
        float confidenceThreshold,
        std::vector<FireSmokeDetectionProposal>& proposals
    );

    bool applyClassAwareNms(
        const std::vector<FireSmokeDetectionProposal>& proposals,
        std::vector<FireSmokeDetectionProposal>& kept,
        float nmsThreshold
    );

    static float calculateIoU(
        const std::array<float, 4>& lhs,
        const std::array<float, 4>& rhs
    ) noexcept;

    static bool validateAnchorSet(
        const std::array<FireSmokeAnchor, 3>& anchors
    ) noexcept;

    static bool readTensorValue(
        const inference::QnnTensorBuffer& buffer,
        uint64_t index,
        float& value
    ) noexcept;

    static float sigmoid(
        float value
    ) noexcept;

    static const char* className(
        int classId
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
    static constexpr int INPUT_HEIGHT = 320;
    static constexpr int INPUT_WIDTH = 320;
    static constexpr int INPUT_CHANNELS = 3;

    static constexpr int PAD_VALUE = 114;

    static constexpr uint32_t NUM_CLASSES = 2;
    static constexpr uint32_t NUM_ANCHORS = 3;
    static constexpr uint32_t VALUES_PER_ANCHOR = 7;

    static constexpr uint32_t OUTPUT_CHANNELS =
        NUM_ANCHORS * VALUES_PER_ANCHOR;

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

    FireSmokeAnchors anchors_{};

    bool anchorsConfigured_ = false;

    FireSmokePreprocessInfo preprocessInfo_;

    std::string lastError_;
};

} // namespace models