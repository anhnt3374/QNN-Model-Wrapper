#include "models/fire_smoke_detection_model.hpp"

#include "inference/qnn_quantization.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <sstream>

namespace models {

FireSmokeDetectionModel::FireSmokeDetectionModel(
    inference::QnnBackend& backend
) noexcept
    : model_(backend)
{
}

FireSmokeDetectionModel::~FireSmokeDetectionModel()
{
    shutdown();
}

bool FireSmokeDetectionModel::initialize(
    const std::string& modelPath
)
{
    shutdown();

    lastError_.clear();

    if (!model_.load(modelPath)) {

        lastError_ =
            "Failed to load FireSmoke model: "
            + model_.lastError();

        return false;
    }

    if (!model_.composeGraphs()) {

        lastError_ =
            "Failed to compose FireSmoke graph: "
            + model_.lastError();

        shutdown();

        return false;
    }

    if (model_.graphCount() != 1) {

        std::ostringstream oss;

        oss
            << "FireSmoke expected exactly 1 graph, got "
            << model_.graphCount();

        lastError_ =
            oss.str();

        shutdown();

        return false;
    }

    if (!model_.finalizeGraphs()) {

        lastError_ =
            "Failed to finalize FireSmoke graph: "
            + model_.lastError();

        shutdown();

        return false;
    }

    const auto* graph =
        model_.graphInfo(0);

    if (graph == nullptr) {

        lastError_ =
            "FireSmoke graph metadata is null";

        shutdown();

        return false;
    }

    if (graph->numInputTensors != 1) {

        std::ostringstream oss;

        oss
            << "FireSmoke expected 1 input tensor, got "
            << graph->numInputTensors;

        lastError_ =
            oss.str();

        shutdown();

        return false;
    }

    if (graph->numOutputTensors != 3) {

        std::ostringstream oss;

        oss
            << "FireSmoke expected 3 output tensors, got "
            << graph->numOutputTensors;

        lastError_ =
            oss.str();

        shutdown();

        return false;
    }

    if (!allocateRuntimeBuffers()) {

        shutdown();

        return false;
    }

    if (!validateModelContract()) {

        shutdown();

        return false;
    }

    if (!buildExecutionTensorArrays()) {

        shutdown();

        return false;
    }

    lastError_.clear();

    return true;
}

bool FireSmokeDetectionModel::allocateRuntimeBuffers()
{
    const auto* graph =
        model_.graphInfo(0);

    if (graph == nullptr) {

        lastError_ =
            "Cannot allocate FireSmoke buffers: "
            "graph is null";

        return false;
    }

    inputBuffers_.clear();
    outputBuffers_.clear();

    inputBuffers_.reserve(
        graph->numInputTensors
    );

    outputBuffers_.reserve(
        graph->numOutputTensors
    );

    for (uint32_t i = 0;
         i < graph->numInputTensors;
         ++i) {

        auto buffer =
            std::make_unique<
                inference::QnnTensorBuffer
            >();

        if (!buffer->initialize(
                graph->inputTensors[i]
            )) {

            std::ostringstream oss;

            oss
                << "Failed to allocate FireSmoke input["
                << i
                << "]: "
                << buffer->lastError();

            lastError_ =
                oss.str();

            return false;
        }

        inputBuffers_.push_back(
            std::move(buffer)
        );
    }

    for (uint32_t i = 0;
         i < graph->numOutputTensors;
         ++i) {

        auto buffer =
            std::make_unique<
                inference::QnnTensorBuffer
            >();

        if (!buffer->initialize(
                graph->outputTensors[i]
            )) {

            std::ostringstream oss;

            oss
                << "Failed to allocate FireSmoke output["
                << i
                << "]: "
                << buffer->lastError();

            lastError_ =
                oss.str();

            return false;
        }

        outputBuffers_.push_back(
            std::move(buffer)
        );
    }

    return true;
}

bool FireSmokeDetectionModel::buildExecutionTensorArrays()
{
    executionInputTensors_.clear();
    executionOutputTensors_.clear();

    executionInputTensors_.reserve(
        inputBuffers_.size()
    );

    executionOutputTensors_.reserve(
        outputBuffers_.size()
    );

    for (const auto& buffer :
         inputBuffers_) {

        if (buffer == nullptr ||
            !buffer->ready()) {

            lastError_ =
                "FireSmoke execution input buffer "
                "is not ready";

            return false;
        }

        executionInputTensors_.push_back(
            buffer->tensor()
        );
    }

    for (const auto& buffer :
         outputBuffers_) {

        if (buffer == nullptr ||
            !buffer->ready()) {

            lastError_ =
                "FireSmoke execution output buffer "
                "is not ready";

            return false;
        }

        executionOutputTensors_.push_back(
            buffer->tensor()
        );
    }

    if (executionInputTensors_.size() != 1) {

        lastError_ =
            "FireSmoke execution input count must be 1";

        return false;
    }

    if (executionOutputTensors_.size() != 3) {

        lastError_ =
            "FireSmoke execution output count must be 3";

        return false;
    }

    return true;
}

bool FireSmokeDetectionModel::validateModelContract()
{
    if (inputBuffers_.size() != 1 ||
        outputBuffers_.size() != 3) {

        lastError_ =
            "Unexpected FireSmoke runtime tensor count";

        return false;
    }

    const auto& input =
        *inputBuffers_[0];

    if (input.name() !=
        "images") {

        lastError_ =
            "Expected FireSmoke input tensor 'images'";

        return false;
    }

    const Qnn_Tensor_t& tensor =
        input.tensor();

    if (rank(tensor) != 4) {

        lastError_ =
            "FireSmoke input rank must be 4";

        return false;
    }

    const uint32_t* dims =
        dimensions(tensor);

    if (dims == nullptr) {

        lastError_ =
            "FireSmoke input dimensions are null";

        return false;
    }

    if (dims[0] != 1 ||
        dims[1] != INPUT_HEIGHT ||
        dims[2] != INPUT_WIDTH ||
        dims[3] != INPUT_CHANNELS) {

        lastError_ =
            "Unexpected FireSmoke input shape";

        return false;
    }

    const Qnn_DataType_t inputType =
        dataType(tensor);

    if (inputType !=
            QNN_DATATYPE_FLOAT_32 &&
        inputType !=
            QNN_DATATYPE_UFIXED_POINT_16) {

        lastError_ =
            "Unsupported FireSmoke input datatype";

        return false;
    }

    if (!validateOutput(
            "output_s4",
            80,
            80
        )) {

        return false;
    }

    if (!validateOutput(
            "output_s8",
            40,
            40
        )) {

        return false;
    }

    if (!validateOutput(
            "output_s16",
            20,
            20
        )) {

        return false;
    }

    lastError_.clear();

    return true;
}

bool FireSmokeDetectionModel::validateOutput(
    const char* name,
    uint32_t expectedHeight,
    uint32_t expectedWidth
)
{
    const auto* output =
        outputBufferByName(name);

    if (output == nullptr) {

        lastError_ =
            std::string(
                "Missing FireSmoke output: "
            )
            +
            name;

        return false;
    }

    const Qnn_Tensor_t& tensor =
        output->tensor();

    if (rank(tensor) != 4) {

        lastError_ =
            std::string(name)
            +
            " rank must be 4";

        return false;
    }

    const uint32_t* dims =
        dimensions(tensor);

    if (dims == nullptr) {

        lastError_ =
            std::string(name)
            +
            " dimensions are null";

        return false;
    }

    if (dims[0] != 1 ||
        dims[1] != expectedHeight ||
        dims[2] != expectedWidth ||
        dims[3] != OUTPUT_CHANNELS) {

        lastError_ =
            std::string(
                "Unexpected FireSmoke output shape: "
            )
            +
            name;

        return false;
    }

    const uint64_t expectedElements =
        static_cast<uint64_t>(
            expectedHeight
        )
        *
        static_cast<uint64_t>(
            expectedWidth
        )
        *
        OUTPUT_CHANNELS;

    if (output->elementCount() !=
        expectedElements) {

        lastError_ =
            std::string(
                "Unexpected FireSmoke element count: "
            )
            +
            name;

        return false;
    }

    const Qnn_DataType_t type =
        dataType(tensor);

    if (type !=
            QNN_DATATYPE_FLOAT_32 &&
        type !=
            QNN_DATATYPE_UFIXED_POINT_16) {

        lastError_ =
            std::string(
                "Unsupported FireSmoke output datatype: "
            )
            +
            name;

        return false;
    }

    return true;
}

bool FireSmokeDetectionModel::setAnchors(
    const FireSmokeAnchors& anchors
)
{
    if (!validateAnchorSet(
            anchors.s4
        ) ||
        !validateAnchorSet(
            anchors.s8
        ) ||
        !validateAnchorSet(
            anchors.s16
        )) {

        lastError_ =
            "Every FireSmoke anchor width/height "
            "must be finite and > 0";

        anchorsConfigured_ =
            false;

        return false;
    }

    anchors_ =
        anchors;

    anchorsConfigured_ =
        true;

    lastError_.clear();

    return true;
}

bool FireSmokeDetectionModel::preprocess(
    const cv::Mat& bgrImage
)
{
    if (!ready()) {

        lastError_ =
            "FireSmokeDetectionModel is not initialized";

        return false;
    }

    if (bgrImage.empty()) {

        lastError_ =
            "FireSmoke input image is empty";

        return false;
    }

    if (bgrImage.type() !=
        CV_8UC3) {

        lastError_ =
            "FireSmokeDetectionModel expects CV_8UC3 BGR";

        return false;
    }

    const int originalWidth =
        bgrImage.cols;

    const int originalHeight =
        bgrImage.rows;

    if (originalWidth <= 0 ||
        originalHeight <= 0) {

        lastError_ =
            "FireSmoke image dimensions are invalid";

        return false;
    }

    const double scale =
        std::min(
            static_cast<double>(
                INPUT_WIDTH
            )
            /
            static_cast<double>(
                originalWidth
            ),

            static_cast<double>(
                INPUT_HEIGHT
            )
            /
            static_cast<double>(
                originalHeight
            )
        );

    const int resizedWidth =
        pythonRound(
            static_cast<double>(
                originalWidth
            )
            *
            scale
        );

    const int resizedHeight =
        pythonRound(
            static_cast<double>(
                originalHeight
            )
            *
            scale
        );

    if (resizedWidth <= 0 ||
        resizedHeight <= 0 ||
        resizedWidth > INPUT_WIDTH ||
        resizedHeight > INPUT_HEIGHT) {

        lastError_ =
            "FireSmoke calculated invalid resize";

        return false;
    }

    const double dw =
        (
            static_cast<double>(
                INPUT_WIDTH
            )
            -
            resizedWidth
        )
        /
        2.0;

    const double dh =
        (
            static_cast<double>(
                INPUT_HEIGHT
            )
            -
            resizedHeight
        )
        /
        2.0;

    const int padLeft =
        pythonRound(
            dw - 0.1
        );

    const int padRight =
        pythonRound(
            dw + 0.1
        );

    const int padTop =
        pythonRound(
            dh - 0.1
        );

    const int padBottom =
        pythonRound(
            dh + 0.1
        );

    if (resizedWidth +
            padLeft +
            padRight !=
        INPUT_WIDTH ||
        resizedHeight +
            padTop +
            padBottom !=
        INPUT_HEIGHT) {

        lastError_ =
            "FireSmoke letterbox geometry mismatch";

        return false;
    }

    cv::Mat resized;

    cv::resize(
        bgrImage,
        resized,
        cv::Size(
            resizedWidth,
            resizedHeight
        ),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    cv::Mat letterboxed;

    cv::copyMakeBorder(
        resized,
        letterboxed,
        padTop,
        padBottom,
        padLeft,
        padRight,
        cv::BORDER_CONSTANT,
        cv::Scalar(
            PAD_VALUE,
            PAD_VALUE,
            PAD_VALUE
        )
    );

    if (letterboxed.cols !=
            INPUT_WIDTH ||
        letterboxed.rows !=
            INPUT_HEIGHT) {

        lastError_ =
            "FireSmoke letterbox output is not 320x320";

        return false;
    }

    cv::Mat rgb;

    cv::cvtColor(
        letterboxed,
        rgb,
        cv::COLOR_BGR2RGB
    );

    auto& input =
        *inputBuffers_[0];

    const Qnn_DataType_t inputType =
        dataType(
            input.tensor()
        );

    if (inputType ==
        QNN_DATATYPE_FLOAT_32) {

        auto* destination =
            static_cast<float*>(
                input.data()
            );

        if (destination == nullptr) {

            lastError_ =
                "FireSmoke FLOAT32 input buffer is null";

            return false;
        }

        uint64_t index = 0;

        for (int y = 0;
             y < INPUT_HEIGHT;
             ++y) {

            const auto* row =
                rgb.ptr<cv::Vec3b>(y);

            for (int x = 0;
                 x < INPUT_WIDTH;
                 ++x) {

                for (int c = 0;
                     c < INPUT_CHANNELS;
                     ++c) {

                    destination[index++] =
                        static_cast<float>(
                            row[x][c]
                        )
                        /
                        255.0F;
                }
            }
        }

        if (index !=
            input.elementCount()) {

            lastError_ =
                "FireSmoke FLOAT32 preprocess "
                "element count mismatch";

            return false;
        }
    }
    else if (
        inputType ==
        QNN_DATATYPE_UFIXED_POINT_16
    ) {

        const auto* params =
            quantization(
                input.tensor()
            );

        if (params == nullptr ||
            params->quantizationEncoding !=
                QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

            lastError_ =
                "Invalid FireSmoke input quantization";

            return false;
        }

        const float quantScale =
            params
                ->scaleOffsetEncoding
                .scale;

        const int32_t quantOffset =
            params
                ->scaleOffsetEncoding
                .offset;

        if (!std::isfinite(
                quantScale
            ) ||
            quantScale <= 0.0F) {

            lastError_ =
                "Invalid FireSmoke input quantization scale";

            return false;
        }

        auto* destination =
            static_cast<uint16_t*>(
                input.data()
            );

        if (destination == nullptr) {

            lastError_ =
                "FireSmoke UFIXED16 input buffer is null";

            return false;
        }

        uint64_t index = 0;

        for (int y = 0;
             y < INPUT_HEIGHT;
             ++y) {

            const auto* row =
                rgb.ptr<cv::Vec3b>(y);

            for (int x = 0;
                 x < INPUT_WIDTH;
                 ++x) {

                for (int c = 0;
                     c < INPUT_CHANNELS;
                     ++c) {

                    const float value =
                        static_cast<float>(
                            row[x][c]
                        )
                        /
                        255.0F;

                    destination[index++] =
                        inference::
                        quantizeScaleOffsetToU16(
                            value,
                            quantScale,
                            quantOffset
                        );
                }
            }
        }

        if (index !=
            input.elementCount()) {

            lastError_ =
                "FireSmoke UFIXED16 preprocess "
                "element count mismatch";

            return false;
        }
    }
    else {

        lastError_ =
            "Unsupported FireSmoke input datatype";

        return false;
    }

    preprocessInfo_.originalWidth =
        originalWidth;

    preprocessInfo_.originalHeight =
        originalHeight;

    preprocessInfo_.resizedWidth =
        resizedWidth;

    preprocessInfo_.resizedHeight =
        resizedHeight;

    preprocessInfo_.scale =
        static_cast<float>(
            scale
        );

    preprocessInfo_.padLeft =
        padLeft;

    preprocessInfo_.padRight =
        padRight;

    preprocessInfo_.padTop =
        padTop;

    preprocessInfo_.padBottom =
        padBottom;

    lastError_.clear();

    return true;
}

bool FireSmokeDetectionModel::infer()
{
    if (!ready()) {

        lastError_ =
            "FireSmokeDetectionModel is not ready";

        return false;
    }

    if (!model_.executeGraph(
            0,

            executionInputTensors_.data(),
            static_cast<uint32_t>(
                executionInputTensors_.size()
            ),

            executionOutputTensors_.data(),
            static_cast<uint32_t>(
                executionOutputTensors_.size()
            )
        )) {

        lastError_ =
            "FireSmoke graph execution failed: "
            +
            model_.lastError();

        return false;
    }

    lastError_.clear();

    return true;
}

bool FireSmokeDetectionModel::decode(
    std::vector<FireSmokeDetectionProposal>& proposals,
    float confidenceThreshold
)
{
    proposals.clear();

    if (!ready()) {

        lastError_ =
            "FireSmokeDetectionModel is not ready";

        return false;
    }

    if (!anchorsConfigured_) {

        lastError_ =
            "FireSmoke anchors have not been configured";

        return false;
    }

    if (!std::isfinite(
            confidenceThreshold
        ) ||
        confidenceThreshold < 0.0F ||
        confidenceThreshold > 1.0F) {

        lastError_ =
            "FireSmoke confidence threshold must be in [0, 1]";

        return false;
    }

    if (!std::isfinite(
            preprocessInfo_.scale
        ) ||
        preprocessInfo_.scale <= 0.0F ||
        preprocessInfo_.originalWidth <= 0 ||
        preprocessInfo_.originalHeight <= 0) {

        lastError_ =
            "FireSmoke preprocess geometry is invalid";

        return false;
    }

    const auto* s4 =
        outputBufferByName(
            "output_s4"
        );

    const auto* s8 =
        outputBufferByName(
            "output_s8"
        );

    const auto* s16 =
        outputBufferByName(
            "output_s16"
        );

    if (s4 == nullptr ||
        s8 == nullptr ||
        s16 == nullptr) {

        lastError_ =
            "FireSmoke output tensors are missing";

        return false;
    }

    if (!decodeHead(
            *s4,
            80,
            80,
            4,
            anchors_.s4,
            confidenceThreshold,
            proposals
        )) {

        return false;
    }

    if (!decodeHead(
            *s8,
            40,
            40,
            8,
            anchors_.s8,
            confidenceThreshold,
            proposals
        )) {

        return false;
    }

    if (!decodeHead(
            *s16,
            20,
            20,
            16,
            anchors_.s16,
            confidenceThreshold,
            proposals
        )) {

        return false;
    }

    lastError_.clear();

    return true;
}

bool FireSmokeDetectionModel::decodeHead(
    const inference::QnnTensorBuffer& output,
    uint32_t height,
    uint32_t width,
    uint32_t stride,
    const std::array<FireSmokeAnchor, 3>& anchors,
    float confidenceThreshold,
    std::vector<FireSmokeDetectionProposal>& proposals
)
{
    const uint64_t expected =
        static_cast<uint64_t>(
            height
        )
        *
        static_cast<uint64_t>(
            width
        )
        *
        OUTPUT_CHANNELS;

    if (output.elementCount() !=
        expected) {

        lastError_ =
            "FireSmoke raw head element count mismatch";

        return false;
    }

    const float imageMaxX =
        static_cast<float>(
            preprocessInfo_.originalWidth - 1
        );

    const float imageMaxY =
        static_cast<float>(
            preprocessInfo_.originalHeight - 1
        );

    const float scale =
        preprocessInfo_.scale;

    const float padLeft =
        static_cast<float>(
            preprocessInfo_.padLeft
        );

    const float padTop =
        static_cast<float>(
            preprocessInfo_.padTop
        );

    for (uint32_t y = 0;
         y < height;
         ++y) {

        for (uint32_t x = 0;
             x < width;
             ++x) {

            const uint64_t cellBase =
                (
                    static_cast<uint64_t>(
                        y
                    )
                    *
                    width
                    +
                    x
                )
                *
                OUTPUT_CHANNELS;

            for (uint32_t anchorIndex = 0;
                 anchorIndex < NUM_ANCHORS;
                 ++anchorIndex) {

                const uint64_t base =
                    cellBase
                    +
                    static_cast<uint64_t>(
                        anchorIndex
                    )
                    *
                    VALUES_PER_ANCHOR;

                float rawX = 0.0F;
                float rawY = 0.0F;
                float rawW = 0.0F;
                float rawH = 0.0F;
                float rawObj = 0.0F;
                float rawClass0 = 0.0F;
                float rawClass1 = 0.0F;

                if (!readTensorValue(
                        output,
                        base + 0,
                        rawX
                    ) ||
                    !readTensorValue(
                        output,
                        base + 1,
                        rawY
                    ) ||
                    !readTensorValue(
                        output,
                        base + 2,
                        rawW
                    ) ||
                    !readTensorValue(
                        output,
                        base + 3,
                        rawH
                    ) ||
                    !readTensorValue(
                        output,
                        base + 4,
                        rawObj
                    ) ||
                    !readTensorValue(
                        output,
                        base + 5,
                        rawClass0
                    ) ||
                    !readTensorValue(
                        output,
                        base + 6,
                        rawClass1
                    )) {

                    lastError_ =
                        "Failed reading FireSmoke raw head";

                    return false;
                }

                const float px =
                    sigmoid(rawX);

                const float py =
                    sigmoid(rawY);

                const float pw =
                    sigmoid(rawW);

                const float ph =
                    sigmoid(rawH);

                const float objectness =
                    sigmoid(rawObj);

                const float class0 =
                    sigmoid(rawClass0);

                const float class1 =
                    sigmoid(rawClass1);

                int classId =
                    0;

                float classProbability =
                    class0;

                if (class1 >
                    class0) {

                    classId =
                        1;

                    classProbability =
                        class1;
                }

                const float score =
                    objectness
                    *
                    classProbability;

                if (!std::isfinite(
                        score
                    ) ||
                    score <
                        confidenceThreshold) {

                    continue;
                }

                // Python:
                //
                // xy =
                //   (p_xy * 2 - 0.5 + grid)
                //   * stride

                const float centerX =
                    (
                        px * 2.0F
                        -
                        0.5F
                        +
                        static_cast<float>(
                            x
                        )
                    )
                    *
                    static_cast<float>(
                        stride
                    );

                const float centerY =
                    (
                        py * 2.0F
                        -
                        0.5F
                        +
                        static_cast<float>(
                            y
                        )
                    )
                    *
                    static_cast<float>(
                        stride
                    );

                // Python:
                //
                // wh =
                //   (p_wh * 2)^2
                //   * anchor

                const float widthFactor =
                    pw * 2.0F;

                const float heightFactor =
                    ph * 2.0F;

                const float decodedWidth =
                    widthFactor
                    *
                    widthFactor
                    *
                    anchors[
                        anchorIndex
                    ].width;

                const float decodedHeight =
                    heightFactor
                    *
                    heightFactor
                    *
                    anchors[
                        anchorIndex
                    ].height;

                float x1 =
                    centerX
                    -
                    decodedWidth / 2.0F;

                float y1 =
                    centerY
                    -
                    decodedHeight / 2.0F;

                float x2 =
                    centerX
                    +
                    decodedWidth / 2.0F;

                float y2 =
                    centerY
                    +
                    decodedHeight / 2.0F;

                // Undo centered letterbox.

                x1 =
                    (
                        x1
                        -
                        padLeft
                    )
                    /
                    scale;

                x2 =
                    (
                        x2
                        -
                        padLeft
                    )
                    /
                    scale;

                y1 =
                    (
                        y1
                        -
                        padTop
                    )
                    /
                    scale;

                y2 =
                    (
                        y2
                        -
                        padTop
                    )
                    /
                    scale;

                x1 =
                    std::clamp(
                        x1,
                        0.0F,
                        imageMaxX
                    );

                x2 =
                    std::clamp(
                        x2,
                        0.0F,
                        imageMaxX
                    );

                y1 =
                    std::clamp(
                        y1,
                        0.0F,
                        imageMaxY
                    );

                y2 =
                    std::clamp(
                        y2,
                        0.0F,
                        imageMaxY
                    );

                if (!(x2 > x1) ||
                    !(y2 > y1)) {

                    continue;
                }

                FireSmokeDetectionProposal proposal;

                proposal.classId =
                    classId;

                proposal.score =
                    score;

                proposal.bbox = {
                    x1,
                    y1,
                    x2,
                    y2
                };

                proposals.push_back(
                    proposal
                );
            }
        }
    }

    return true;
}

bool FireSmokeDetectionModel::applyClassAwareNms(
    const std::vector<FireSmokeDetectionProposal>& proposals,
    std::vector<FireSmokeDetectionProposal>& kept,
    float nmsThreshold
)
{
    kept.clear();

    if (!std::isfinite(
            nmsThreshold
        ) ||
        nmsThreshold < 0.0F ||
        nmsThreshold > 1.0F) {

        lastError_ =
            "FireSmoke NMS threshold must be in [0, 1]";

        return false;
    }

    if (proposals.empty()) {

        lastError_.clear();

        return true;
    }

    std::vector<std::size_t>
        keptIndices;

    // Python class_aware_nms:
    //
    // for class_id in np.unique(class_ids):
    //     perform NMS only inside that class
    //
    // Current model has:
    //
    // class 0 = smoke
    // class 1 = fire

    for (int classId = 0;
         classId < static_cast<int>(
             NUM_CLASSES
         );
         ++classId) {

        std::vector<std::size_t>
            classIndices;

        for (std::size_t i = 0;
             i < proposals.size();
             ++i) {

            if (proposals[i].classId ==
                classId) {

                classIndices.push_back(
                    i
                );
            }
        }

        if (classIndices.empty()) {
            continue;
        }

        // Python nms:
        //
        // order = scores.argsort()[::-1]

        std::sort(
            classIndices.begin(),
            classIndices.end(),
            [&proposals](
                std::size_t lhs,
                std::size_t rhs
            ) {

                return
                    proposals[lhs].score
                    >
                    proposals[rhs].score;
            }
        );

        std::vector<uint8_t>
            suppressed(
                classIndices.size(),
                0
            );

        for (std::size_t i = 0;
             i < classIndices.size();
             ++i) {

            if (suppressed[i] != 0) {
                continue;
            }

            const std::size_t currentIndex =
                classIndices[i];

            keptIndices.push_back(
                currentIndex
            );

            for (std::size_t j = i + 1;
                 j < classIndices.size();
                 ++j) {

                if (suppressed[j] != 0) {
                    continue;
                }

                const std::size_t candidateIndex =
                    classIndices[j];

                const float iou =
                    calculateIoU(
                        proposals[
                            currentIndex
                        ].bbox,
                        proposals[
                            candidateIndex
                        ].bbox
                    );

                // Python keeps:
                //
                // iou <= threshold
                //
                // therefore suppress only:
                //
                // iou > threshold

                if (iou >
                    nmsThreshold) {

                    suppressed[j] =
                        1;
                }
            }
        }
    }

    // Python combines all class-specific kept indices,
    // then globally sorts them by score descending.

    std::sort(
        keptIndices.begin(),
        keptIndices.end(),
        [&proposals](
            std::size_t lhs,
            std::size_t rhs
        ) {

            return
                proposals[lhs].score
                >
                proposals[rhs].score;
        }
    );

    kept.reserve(
        keptIndices.size()
    );

    for (const std::size_t index :
         keptIndices) {

        kept.push_back(
            proposals[index]
        );
    }

    lastError_.clear();

    return true;
}

float FireSmokeDetectionModel::calculateIoU(
    const std::array<float, 4>& lhs,
    const std::array<float, 4>& rhs
) noexcept
{
    // Same convention as Python FireSmoke NMS.
    //
    // NO +1.

    const float lhsWidth =
        std::max(
            0.0F,
            lhs[2] - lhs[0]
        );

    const float lhsHeight =
        std::max(
            0.0F,
            lhs[3] - lhs[1]
        );

    const float rhsWidth =
        std::max(
            0.0F,
            rhs[2] - rhs[0]
        );

    const float rhsHeight =
        std::max(
            0.0F,
            rhs[3] - rhs[1]
        );

    const float lhsArea =
        lhsWidth
        *
        lhsHeight;

    const float rhsArea =
        rhsWidth
        *
        rhsHeight;

    const float xx1 =
        std::max(
            lhs[0],
            rhs[0]
        );

    const float yy1 =
        std::max(
            lhs[1],
            rhs[1]
        );

    const float xx2 =
        std::min(
            lhs[2],
            rhs[2]
        );

    const float yy2 =
        std::min(
            lhs[3],
            rhs[3]
        );

    const float intersectionWidth =
        std::max(
            0.0F,
            xx2 - xx1
        );

    const float intersectionHeight =
        std::max(
            0.0F,
            yy2 - yy1
        );

    const float intersection =
        intersectionWidth
        *
        intersectionHeight;

    const float unionArea =
        lhsArea
        +
        rhsArea
        -
        intersection;

    return
        intersection
        /
        std::max(
            unionArea,
            1e-9F
        );
}

bool FireSmokeDetectionModel::postprocess(
    std::vector<FireSmokeDetectionResult>& results,
    float confidenceThreshold,
    float nmsThreshold
)
{
    results.clear();

    std::vector<
        FireSmokeDetectionProposal
    > proposals;

    if (!decode(
            proposals,
            confidenceThreshold
        )) {

        return false;
    }

    std::vector<
        FireSmokeDetectionProposal
    > kept;

    if (!applyClassAwareNms(
            proposals,
            kept,
            nmsThreshold
        )) {

        return false;
    }

    results.reserve(
        kept.size()
    );

    for (const auto& proposal :
         kept) {

        FireSmokeDetectionResult result;

        result.classId =
            proposal.classId;

        result.score =
            proposal.score;

        result.bbox =
            proposal.bbox;

        results.push_back(
            result
        );
    }

    lastError_.clear();

    return true;
}

bool FireSmokeDetectionModel::detect(
    const cv::Mat& bgrImage,
    std::vector<FireSmokeDetectionResult>& results,
    float confidenceThreshold,
    float nmsThreshold
)
{
    results.clear();

    if (!anchorsConfigured_) {

        lastError_ =
            "FireSmoke anchors have not been configured";

        return false;
    }

    if (!preprocess(
            bgrImage
        )) {

        return false;
    }

    if (!infer()) {

        return false;
    }

    if (!postprocess(
            results,
            confidenceThreshold,
            nmsThreshold
        )) {

        return false;
    }

    lastError_.clear();

    return true;
}

cv::Mat FireSmokeDetectionModel::renderDetections(
    const cv::Mat& bgrImage,
    const std::vector<FireSmokeDetectionResult>& results
) const
{
    if (bgrImage.empty()) {
        return {};
    }

    cv::Mat rendered =
        bgrImage.clone();

    for (const auto& result :
         results) {

        const int x1 =
            pythonRound(
                result.bbox[0]
            );

        const int y1 =
            pythonRound(
                result.bbox[1]
            );

        const int x2 =
            pythonRound(
                result.bbox[2]
            );

        const int y2 =
            pythonRound(
                result.bbox[3]
            );

        cv::Scalar color;

        if (result.classId == 0) {

            // smoke
            //
            // Python:
            // (160,160,160)

            color =
                cv::Scalar(
                    160,
                    160,
                    160
                );
        }
        else {

            // fire
            //
            // Python:
            // (0,0,255)

            color =
                cv::Scalar(
                    0,
                    0,
                    255
                );
        }

        cv::rectangle(
            rendered,
            cv::Point(
                x1,
                y1
            ),
            cv::Point(
                x2,
                y2
            ),
            color,
            2
        );

        std::ostringstream label;

        label
            << className(
                   result.classId
               )
            << " "
            << std::fixed
            << std::setprecision(2)
            << result.score;

        cv::putText(
            rendered,
            label.str(),
            cv::Point(
                x1,
                std::max(
                    15,
                    y1 - 5
                )
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv::LINE_AA
        );
    }

    return rendered;
}

bool FireSmokeDetectionModel::validateAnchorSet(
    const std::array<FireSmokeAnchor, 3>& anchors
) noexcept
{
    for (const auto& anchor :
         anchors) {

        if (!std::isfinite(
                anchor.width
            ) ||
            !std::isfinite(
                anchor.height
            ) ||
            anchor.width <= 0.0F ||
            anchor.height <= 0.0F) {

            return false;
        }
    }

    return true;
}

bool FireSmokeDetectionModel::readTensorValue(
    const inference::QnnTensorBuffer& buffer,
    uint64_t index,
    float& value
) noexcept
{
    if (index >=
        buffer.elementCount()) {

        return false;
    }

    const Qnn_DataType_t type =
        dataType(
            buffer.tensor()
        );

    if (type ==
        QNN_DATATYPE_FLOAT_32) {

        const auto* raw =
            static_cast<const float*>(
                buffer.data()
            );

        if (raw == nullptr) {
            return false;
        }

        value =
            raw[index];

        return true;
    }

    if (type ==
        QNN_DATATYPE_UFIXED_POINT_16) {

        const auto* raw =
            static_cast<const uint16_t*>(
                buffer.data()
            );

        const auto* params =
            quantization(
                buffer.tensor()
            );

        if (raw == nullptr ||
            params == nullptr ||
            params->quantizationEncoding !=
                QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {

            return false;
        }

        value =
            inference::dequantizeScaleOffset(
                raw[index],
                params
                    ->scaleOffsetEncoding
                    .scale,
                params
                    ->scaleOffsetEncoding
                    .offset
            );

        return true;
    }

    return false;
}

float FireSmokeDetectionModel::sigmoid(
    float value
) noexcept
{
    const float clipped =
        std::clamp(
            value,
            -50.0F,
            50.0F
        );

    return
        1.0F
        /
        (
            1.0F
            +
            std::exp(
                -clipped
            )
        );
}

const char*
FireSmokeDetectionModel::className(
    int classId
) noexcept
{
    switch (classId) {

    case 0:
        return "smoke";

    case 1:
        return "fire";

    default:
        return "unknown";
    }
}

void FireSmokeDetectionModel::shutdown()
{
    executionInputTensors_.clear();

    executionOutputTensors_.clear();

    inputBuffers_.clear();

    outputBuffers_.clear();

    preprocessInfo_ = {};

    anchors_ = {};

    anchorsConfigured_ =
        false;

    model_.shutdown();
}

bool FireSmokeDetectionModel::ready() const noexcept
{
    return
        model_.graphsFinalized()
        &&
        inputBuffers_.size() == 1
        &&
        outputBuffers_.size() == 3
        &&
        executionInputTensors_.size() == 1
        &&
        executionOutputTensors_.size() == 3;
}

bool
FireSmokeDetectionModel::anchorsConfigured() const noexcept
{
    return
        anchorsConfigured_;
}

const FireSmokeAnchors&
FireSmokeDetectionModel::anchors() const noexcept
{
    return
        anchors_;
}

const FireSmokePreprocessInfo&
FireSmokeDetectionModel::preprocessInfo() const noexcept
{
    return
        preprocessInfo_;
}

const inference::QnnTensorBuffer*
FireSmokeDetectionModel::inputBuffer() const noexcept
{
    if (inputBuffers_.empty()) {
        return nullptr;
    }

    return
        inputBuffers_[0].get();
}

std::size_t
FireSmokeDetectionModel::outputCount() const noexcept
{
    return
        outputBuffers_.size();
}

const inference::QnnTensorBuffer*
FireSmokeDetectionModel::outputBuffer(
    std::size_t index
) const noexcept
{
    if (index >=
        outputBuffers_.size()) {

        return nullptr;
    }

    return
        outputBuffers_[index].get();
}

const inference::QnnTensorBuffer*
FireSmokeDetectionModel::outputBufferByName(
    const std::string& name
) const noexcept
{
    for (const auto& buffer :
         outputBuffers_) {

        if (buffer != nullptr &&
            buffer->name() == name) {

            return
                buffer.get();
        }
    }

    return nullptr;
}

const std::string&
FireSmokeDetectionModel::lastError() const noexcept
{
    return
        lastError_;
}

Qnn_DataType_t
FireSmokeDetectionModel::dataType(
    const Qnn_Tensor_t& tensor
) noexcept
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return
            tensor.v1.dataType;

    case QNN_TENSOR_VERSION_2:
        return
            tensor.v2.dataType;

    default:
        return
            QNN_DATATYPE_UNDEFINED;
    }
}

uint32_t
FireSmokeDetectionModel::rank(
    const Qnn_Tensor_t& tensor
) noexcept
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return
            tensor.v1.rank;

    case QNN_TENSOR_VERSION_2:
        return
            tensor.v2.rank;

    default:
        return 0;
    }
}

const uint32_t*
FireSmokeDetectionModel::dimensions(
    const Qnn_Tensor_t& tensor
) noexcept
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return
            tensor.v1.dimensions;

    case QNN_TENSOR_VERSION_2:
        return
            tensor.v2.dimensions;

    default:
        return nullptr;
    }
}

const Qnn_QuantizeParams_t*
FireSmokeDetectionModel::quantization(
    const Qnn_Tensor_t& tensor
) noexcept
{
    switch (tensor.version) {

    case QNN_TENSOR_VERSION_1:
        return
            &tensor.v1.quantizeParams;

    case QNN_TENSOR_VERSION_2:
        return
            &tensor.v2.quantizeParams;

    default:
        return nullptr;
    }
}

int FireSmokeDetectionModel::pythonRound(
    double value
) noexcept
{
    const double lower =
        std::floor(
            value
        );

    const double fraction =
        value
        -
        lower;

    if (fraction < 0.5) {

        return
            static_cast<int>(
                lower
            );
    }

    if (fraction > 0.5) {

        return
            static_cast<int>(
                lower + 1.0
            );
    }

    const auto lowerInteger =
        static_cast<int64_t>(
            lower
        );

    if (lowerInteger % 2 == 0) {

        return
            static_cast<int>(
                lowerInteger
            );
    }

    return
        static_cast<int>(
            lowerInteger + 1
        );
}

} // namespace models